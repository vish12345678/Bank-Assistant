"""RAG pipeline for the sales rep.

  1. Understand  — Claude turns the rep's free-text notes (whole conversation) into a
                   structured ClientProfile + several search queries (query expansion).
  2. Retrieve    — hybrid search over the product KB, re-scored with the profile.
  3. Recommend   — Claude streams a grounded, cited recommendation for the rep.

Without an API key the app still works: a rule-based profiler + a templated answer.
"""
from __future__ import annotations

import re
from typing import Iterator, Literal

import anthropic
from pydantic import BaseModel, Field

import config
from retriever import Hit, Retriever

SEGMENTS = [
    "individual", "salaried", "self_employed", "senior_citizen", "retired", "women", "minor", "student",
    "nri", "farmer", "msme", "startup", "corporate", "low_income", "hni", "new_to_credit", "govt_employee",
    "defence", "sc_st", "street_vendor", "artisan", "trader", "exporter", "young_professional",
    "first_time_home_buyer", "conservative", "frequent_traveller", "shg",
]
CATEGORIES = ["loan", "deposit", "investment", "insurance", "card", "account", "payments", "trade_finance"]


class ClientProfile(BaseModel):
    summary: str = Field(description="One-line description of the client and what they need")
    segments: list[str] = Field(description=f"Applicable tags from: {', '.join(SEGMENTS)}")
    needs: list[str] = Field(description=f"Product categories needed, from: {', '.join(CATEGORIES)}")
    age: int | None = Field(description="Client's age in years if stated, else null")
    income_or_turnover: str | None
    amount: str | None = Field(description="Amount to invest / borrow, if stated")
    horizon: str | None = Field(description="Time horizon or urgency, if stated")
    risk_appetite: Literal["conservative", "moderate", "aggressive"] | None
    goals: list[str]
    search_queries: list[str] = Field(
        description="3-5 short, diverse search queries for a bank-product knowledge base, "
        "covering each distinct need (e.g. 'collateral free loan for women entrepreneur')"
    )
    missing_info: list[str] = Field(description="Key facts the rep should still ask the client")


EXTRACT_SYSTEM = """You help Indian bank sales reps. Read the rep's notes about a client (the whole
conversation; later messages update earlier facts) and extract a structured client profile.
Only record facts that were stated or are clearly implied; use null when unknown. Use Indian
banking context (lakh/crore, 80C, CIBIL, KCC, Udyam, etc.)."""

ANSWER_SYSTEM = """You are "Product Advisor", a copilot for a sales representative at an Indian bank.
The rep describes a client's requirements; you recommend the best-fitting bank products and
government schemes so the rep can pitch confidently.

Ground every product fact in the SOURCES provided in the user turn and cite them like [S3].
If the sources don't cover something, say so briefly instead of inventing products, rates or limits.
Rates and limits change: when quoting a number from a source that says "verify", tell the rep to
confirm it on the current rate card. Respect eligibility: if a source carries an eligibility flag
or the client clearly doesn't qualify, don't recommend it (mention it only as "not eligible because...").
If the rep sells for one bank, recommend only that bank's products plus government schemes and
generic products that bank offers.

Write for a busy rep reading on a phone. Use this structure in Markdown:

**Client snapshot** — one line.

### Top recommendations
Numbered list of 2-5 products, best first. For each: **Product name** (bank) — why it fits this
client in one line, then 2-3 short bullets of key facts (limits, tenure, tax benefit, eligibility) with citations.

### Bundle / cross-sell
1-3 complementary products in one or two lines.

### Pitch line
One sentence the rep can say to the client.

### Ask the client next
Up to 4 questions that would sharpen the recommendation.

### Documents to collect
A short comma-separated list.

Add a one-line **Compliance note** only when relevant (market-linked risk, no forced bundling,
suitability for elderly clients, KFS for loans). Keep the whole answer under ~350 words.
Follow-up messages may refine the client's situation — update the recommendation accordingly."""


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, base_url=config.ANTHROPIC_BASE_URL)


# ---------------------------------------------------------------- understand
def _transcript(history: list[dict], message: str) -> str:
    lines = [f"{'Rep' if h['role'] == 'user' else 'Advisor'}: {h['content']}" for h in history]
    lines.append(f"Rep: {message}")
    return "\n\n".join(lines)


def extract_profile_llm(history: list[dict], message: str) -> ClientProfile:
    client = _client()
    resp = client.messages.parse(
        model=config.MODEL,
        max_tokens=4000,
        system=EXTRACT_SYSTEM,
        output_config={"effort": config.EFFORT_EXTRACT},
        messages=[{"role": "user", "content": _transcript(history, message)}],
        output_format=ClientProfile,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError("profile extraction declined")
    return resp.parsed_output


KEYWORD_SEGMENTS = {
    r"\b(retired|retiree|pensioner|senior)\b": ["senior_citizen", "retired"],
    r"\b(salaried|salary|employee|job|it professional|engineer)\b": ["salaried"],
    r"\b(govt|government|psu)\b": ["govt_employee"],
    r"\b(army|navy|air force|defence|defense)\b": ["defence"],
    r"\b(woman|women|lady|she|her|female|wife|homemaker)\b": ["women"],
    r"\b(daughter|son|child|kid|minor)\b": ["minor"],
    r"\b(student|college|university|study|admission)\b": ["student"],
    r"\b(nri|abroad|dubai|usa|uk|overseas|gulf)\b": ["nri"],
    r"\b(farmer|kisan|crop|agri|dairy|tractor)\b": ["farmer"],
    r"\b(business|shop|msme|sme|trader|manufactur|factory|boutique|restaurant|udyam|gst)\w*": ["msme", "self_employed"],
    r"\b(startup|founder)\b": ["startup"],
    r"\b(vendor|hawker)\b": ["street_vendor", "low_income"],
    r"\b(no credit history|new to credit|first credit card|no cibil)\b": ["new_to_credit"],
    r"\b(sc|st|scheduled caste|scheduled tribe|dalit)\b": ["sc_st"],
    r"\b(hni|crore|wealthy|high net)\b": ["hni"],
}
KEYWORD_NEEDS = {
    r"\b(loan|borrow|finance|emi|credit limit|working capital|funding|capital|collateral|mortgage|"
    r"needs?\s*(?:₹|rs\.?|inr)?\s*\d+(?:\.\d+)?\s*(?:lakh|lac|crore|cr|k))\b": "loan",
    r"\b(fd|fixed deposit|deposit|monthly income|interest income|park)\b": "deposit",
    r"\b(invest|sip|mutual fund|tax sav|80c|retirement|pension|corpus|wealth|save|saving)\w*": "investment",
    r"\b(insurance|cover|protect|term insurance|health|medical|hospital|mediclaim)\b": "insurance",
    r"\b(education|future|long.term plan)\b": "investment",
    r"\b(credit card|card|cashback|rewards|lounge)\b": "card",
    r"\b(account|zero balance|salary account|current account)\b": "account",
    r"\b(pos|qr|upi|payment gateway|collections)\b": "payments",
    r"\b(export|import|letter of credit|bank guarantee)\b": "trade_finance",
}


def extract_profile_rules(history: list[dict], message: str) -> ClientProfile:
    text = " ".join([h["content"] for h in history if h["role"] == "user"] + [message]).lower()
    segs, needs = ["individual"], []
    for pat, tags in KEYWORD_SEGMENTS.items():
        if re.search(pat, text):
            segs += tags
    for pat, cat in KEYWORD_NEEDS.items():
        if re.search(pat, text):
            needs.append(cat)
    age = None
    m = re.search(r"(\d{2})\s*(?:years?|yrs?|y/o|yo)\b", text) or re.search(r"\b(?:age|aged)\s*(\d{2})", text)
    if m:
        age = int(m.group(1))
        if age >= 60:
            segs.append("senior_citizen")
    amount = None
    m = re.search(r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(lakh|lac|crore|cr|k)\b", text)
    if m:
        amount = f"₹{m.group(1)} {m.group(2)}"
    # One query per clause, so "crop loan and a tractor" searches for each need separately.
    negated = r"\b(no|without|not|don'?t have|lacks?)\b.*\b(property|collateral|security|mortgage)\b"
    clauses = [c.strip() for c in re.split(r"\s+and\s+|[,;]|\balso\b", message) if len(c.split()) >= 3]
    # A clause like "no property to give" must not retrieve property loans.
    queries = [" ".join(re.sub(negated, " ", message, flags=re.I).split())]
    queries += [c for c in clauses if not re.search(negated, c, re.I)]
    if re.search(negated, text):
        queries.append("collateral free business loan without security credit guarantee")
    if "nri" in segs:
        queries += ["NRE account repatriable tax free interest NRI", "FCNR foreign currency deposit NRI no exchange risk"]
    if re.search(r"(maxed|exhausted|used up|full)\W+(?:\w+\W+)?80\s?c|80\s?c\W+(?:\w+\W+)?(maxed|exhausted|used up|full)", text):
        queries.append("additional tax deduction beyond 80C limit NPS 80CCD(1B) extra 50000")
    return ClientProfile(
        summary=message[:160], segments=sorted(set(segs)), needs=sorted(set(needs)), age=age,
        income_or_turnover=None, amount=amount, horizon=None, risk_appetite=None, goals=[],
        search_queries=queries, missing_info=[],
    )


# ---------------------------------------------------------------- retrieve
def retrieve(retriever: Retriever, profile: ClientProfile, message: str, bank: str | None) -> list[Hit]:
    # The rule-based profiler's first query is the message with negated clauses removed.
    queries = profile.search_queries if not config.LLM_ENABLED else [message, profile.summary, *profile.search_queries]
    return retriever.search(
        queries, segments=profile.segments, needs=profile.needs, bank=bank, age=profile.age, top_k=config.TOP_K
    )


def format_sources(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        p = h.product
        flag = f"\nELIGIBILITY FLAG: {h.eligibility_flag}" if h.eligibility_flag else ""
        parts.append(
            f'<source id="S{i}" title="{p.title}" bank="{p.bank}" category="{p.category}" url="{p.source}">'
            f"{flag}\n{p.body}\n</source>"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------- recommend
def stream_answer_llm(history: list[dict], message: str, profile: ClientProfile, hits: list[Hit],
                      bank: str | None) -> Iterator[str]:
    client = _client()
    bank_line = f"The rep sells for: {bank}." if bank and bank != "All banks" else "The rep can recommend any bank."
    user_turn = (
        f"<sources>\n{format_sources(hits)}\n</sources>\n\n"
        f"<client_profile>\n{profile.model_dump_json(indent=1)}\n</client_profile>\n\n"
        f"{bank_line}\n\nRep's latest message: {message}"
    )
    messages = [{"role": h["role"], "content": h["content"]} for h in history[-8:]]
    messages.append({"role": "user", "content": user_turn})
    with client.beta.messages.stream(
        model=config.MODEL,
        max_tokens=16000,
        system=[{"type": "text", "text": ANSWER_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": config.EFFORT_ANSWER},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text
        final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n_The model declined to answer this request._"


def answer_offline(profile: ClientProfile, hits: list[Hit]) -> str:
    """Templated answer used when no API key is configured (retrieval-only mode)."""
    products = [h for h in hits if h.product.category != "playbook" and not h.eligibility_flag][:4]
    lines = ["*Retrieval-only mode — add `ANTHROPIC_API_KEY` to `.env` for AI-written recommendations.*\n",
             f"**Client snapshot** — {profile.summary}\n", "### Top matches"]
    for i, h in enumerate(products, 1):
        p = h.product
        best_fit = next((ln for ln in p.body.splitlines() if ln.lower().startswith("best fit")), "")
        bullets = [ln for ln in p.body.splitlines() if ln.startswith("- ")][:3]
        why = f" — {', '.join(h.reasons)}" if h.reasons else ""
        lines.append(f"{i}. **{p.title}** ({p.bank}){why} [S{hits.index(h) + 1}]")
        lines += [f"   {b}" for b in bullets]
        if best_fit:
            lines.append(f"   - _{best_fit}_")
    flagged = [h for h in hits if h.eligibility_flag]
    if flagged:
        lines.append("\n**Not eligible:** " + "; ".join(f"{h.product.title} ({h.eligibility_flag})" for h in flagged))
    lines.append("\n_Verify current rates and eligibility on the bank's official rate card before quoting._")
    return "\n".join(lines)


# ---------------------------------------------------------------- orchestration
class Advisor:
    def __init__(self):
        self.retriever = Retriever()

    def run(self, message: str, history: list[dict], bank: str | None) -> Iterator[dict]:
        """Yields events: profile, sources, token*, done (or error)."""
        llm = config.LLM_ENABLED
        try:
            profile = extract_profile_llm(history, message) if llm else extract_profile_rules(history, message)
        except anthropic.APIError as exc:
            yield {"type": "notice", "text": f"Claude unavailable ({exc.__class__.__name__}); using offline mode."}
            llm, profile = False, extract_profile_rules(history, message)
        yield {"type": "profile", "profile": profile.model_dump()}

        hits = retrieve(self.retriever, profile, message, bank)
        yield {"type": "sources", "sources": [
            {"n": i, "id": h.product.id, "title": h.product.title, "bank": h.product.bank,
             "category": h.product.category, "score": h.score, "reasons": h.reasons,
             "flag": h.eligibility_flag, "url": h.product.source, "body": h.product.body}
            for i, h in enumerate(hits, 1)
        ]}

        if llm:
            try:
                for text in stream_answer_llm(history, message, profile, hits, bank):
                    yield {"type": "token", "text": text}
            except anthropic.APIError as exc:
                yield {"type": "notice", "text": f"Claude error ({exc.__class__.__name__}); showing offline answer."}
                yield {"type": "token", "text": answer_offline(profile, hits)}
        else:
            yield {"type": "token", "text": answer_offline(profile, hits)}
        yield {"type": "done", "llm": llm}
