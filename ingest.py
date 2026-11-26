"""Grow the knowledge base from content published by Indian banks.

Usage:
  py ingest.py https://sbi.bank.in/web/... https://www.hdfc.bank.in/...
  py ingest.py brochure.pdf --bank "Canara Bank"
  py ingest.py --list sources.txt          (one URL or file path per line; '#' comments allowed)
  py ingest.py --list sources.txt --llm    (use Claude to structure pages; needs ANTHROPIC_API_KEY)

Offline mode (default): each page is cleaned (menus, footers and text repeated across a bank's
pages are removed), split by section heading into chunks of about 1,200 characters, and tagged
with bank, category and client segments from the URL, title and text.

--llm mode: Claude turns each page into one structured record per product (key facts plus a
"Best fit" line).

Output: kb/ingested/<host>.json, one file per bank website. The server reads it on start, and
embeddings rebuild automatically when the KB changes.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

import config

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/130.0 Safari/537.36 BankProductAdvisor/1.0"}
MAX_CHARS = 60_000
CHUNK_CHARS = 1200

BANK_BY_HOST = {
    "sbi": "SBI", "hdfc": "HDFC Bank", "icici": "ICICI Bank", "axis": "Axis Bank", "kotak": "Kotak Mahindra Bank",
    "bankofbaroda": "Bank of Baroda", "canarabank": "Canara Bank", "pnb": "Punjab National Bank",
    "unionbank": "Union Bank of India", "idfcfirst": "IDFC FIRST Bank", "indusind": "IndusInd Bank",
    "standupmitra": "Govt Scheme", "mudra": "Govt Scheme", "jansuraksha": "Govt Scheme", "nsiindia": "Govt Scheme",
    "npstrust": "Govt Scheme", "pmjdy": "Govt Scheme", "kviconline": "Govt Scheme", "cgtmse": "Govt Scheme",
}

# Checked in order: the first match wins (e.g. "kisan credit card" is a loan, not a card).
CATEGORY_RULES = [
    ("loan", r"kisan|kcc|mudra|pmmy|pmegp|cgtmse|stand-?up|svanidhi|loan|finance|overdraft|working-capital|mortgage"),
    ("insurance", r"insurance|pmjjby|pmsby|bima|life-cover"),
    ("investment", r"ppf|provident|sukanya|nps|pension-system|atal|apy|mutual|bond|invest|demat|wealth|senior-citizens?-savings"),
    ("card", r"card"),
    ("deposit", r"deposit|\bfd\b|recurring|term"),
    ("trade_finance", r"trade|export|import|letter-of-credit|guarantee|forex"),
    ("payments", r"collect|merchant|\bpos\b|pay|remit|money-transfer|cash-management"),
    ("account", r"account|savings|saving|current|salary"),
]
SEGMENT_RULES = {
    "senior_citizen": r"senior|pensioner|retire", "women": r"wom[ae]n|mahila|lady|ladies|stree|silk|arise|her-",
    "nri": r"\bnri\b|nre|nro|fcnr|non-resident|overseas", "farmer": r"agri|kisan|farmer|krishi|tractor|crop|rural",
    "msme": r"msme|sme|business|trader|current-account|enterprise", "startup": r"startup|start-up",
    "student": r"student|education|scholar", "minor": r"kid|junior|minor|child|champ|future-stars",
    "salaried": r"salary|salaried", "defence": r"defence|military|army|police|forces|agniveer",
    "govt_employee": r"government-employee|central-government|govt", "hni": r"burgundy|privy|wealth|imperia|premium|priority",
    "corporate": r"corporate|wholesale", "low_income": r"basic-savings|bsbd|jan-dhan|pmjdy|small-account",
    "sc_st": r"sc-st|scheduled",
}
DROP_LINE = re.compile(r"^(apply now|know more|read more|click here|login|log in|sign in|calculate|download|"
                       r"view details|share|close|menu|search|skip to|home|explore|get started)\b|"
                       r"was this information helpful|terms (&|and) conditions apply|reserves the right to|"
                       r"service charges? schedule|please email us|call us|toll.free|sms .* to \d", re.I)
DROP_SECTION = re.compile(r"^(disclaimer|important links|need assistance|related|you may also|recently viewed|popular|"
                          r"quick links|other products|explore more|connect with us|follow us|downloads?)\b", re.I)


class IngestedProduct(BaseModel):
    id: str = Field(description="short kebab-case id, e.g. 'canara-home-loan'")
    title: str
    bank: str
    category: str = Field(description="one of: loan, deposit, investment, insurance, card, account, payments, trade_finance")
    segments: list[str] = Field(description="client segments, e.g. salaried, senior_citizen, msme, nri, farmer, women, student")
    tags: list[str] = Field(description="5-10 search keywords")
    body: str = Field(description="Key facts as '- ' bullets (eligibility, amount limits, tenure, rates as published, "
                                  "tax benefits, fees), ending with a 'Best fit:' line. Only facts from the page.")
    age_min: int | None
    age_max: int | None


class IngestResult(BaseModel):
    products: list[IngestedProduct]


# ------------------------------------------------------------------ fetch & parse
def fetch(src: str) -> tuple[str, bytes, bool]:
    """Returns (final_url, content, is_pdf)."""
    if Path(src).exists():
        return src, Path(src).read_bytes(), src.lower().endswith(".pdf")
    r = requests.get(src, headers=HEADERS, timeout=40)
    r.raise_for_status()
    is_pdf = "pdf" in r.headers.get("content-type", "") or src.lower().endswith(".pdf")
    return r.url, r.content, is_pdf


def parse_html(html: str) -> tuple[str, list[tuple[str, list[str]]]]:
    """Returns (page_title, [(section_heading, lines)])."""
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    title = (h1.get_text(" ", strip=True) if h1 else "") or (soup.title.get_text(strip=True) if soup.title else "")
    title = re.split(r"\s+[|–-]\s+", title)[0].strip()
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "svg", "form", "aside", "iframe", "button"]):
        tag.decompose()
    for tag in soup.find_all(attrs={"class": re.compile(r"menu|navbar|footer|header|breadcrumb|cookie|modal|popup|megamenu", re.I)}):
        tag.decompose()
    body = soup.find("main") or soup.body or soup
    sections: list[tuple[str, list[str]]] = [(title, [])]
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        if el.find(["p", "li", "td"]):  # only leaf-ish blocks, avoid duplicate text
            continue
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            sections.append((text[:120], []))
        else:
            sections[-1][1].append(text)
    return title, sections


def keep_line(line: str) -> bool:
    if DROP_LINE.match(line):
        return False
    words = len(line.split())
    # short lines survive only if they carry facts (numbers, ₹, %)
    return words >= 6 or bool(re.search(r"\d|₹|%", line)) and words >= 2


def infer_bank(host: str, hint: str | None) -> str:
    if hint:
        return hint
    for key, name in BANK_BY_HOST.items():
        if key in host:
            return name
    return "Generic"


def infer_category(text: str) -> str:
    t = text.lower().replace(" ", "-")
    for cat, pat in CATEGORY_RULES:
        if re.search(pat, t):
            return cat
    return "other"


def infer_segments(text: str) -> list[str]:
    t = text.lower().replace(" ", "-")
    segs = [s for s, pat in SEGMENT_RULES.items() if re.search(pat, t)]
    return sorted(set(segs or ["individual"]))


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# ------------------------------------------------------------------ offline structuring
def page_to_records(url: str, title: str, sections: list[tuple[str, list[str]]], boiler: set[str],
                    bank_hint: str | None) -> list[dict]:
    host = urlparse(url).netloc
    bank = infer_bank(host, bank_hint)
    path = urlparse(url).path
    path_words = " ".join(w for w in re.split(r"[/\-_.]+", path) if w and w not in {"web", "en", "html", "in", "pages"})
    category = infer_category(f"{path} {title}")
    page_segments = infer_segments(f"{path} {title}")
    tags = sorted({w for w in path_words.lower().split() if len(w) > 2})[:10]

    # merge sections into ~CHUNK_CHARS chunks, keeping headings inline
    chunks: list[tuple[str, str]] = []
    cur_head, cur = None, []
    for head, lines in sections:
        if head and DROP_SECTION.match(head):
            continue
        lines = [ln for ln in lines if ln not in boiler and keep_line(ln)]
        if not lines:
            continue
        block = (f"{head}:\n" if head and head != title else "") + "\n".join(f"- {ln}" for ln in lines)
        if cur and sum(len(c) for c in cur) + len(block) > CHUNK_CHARS:
            chunks.append((cur_head, "\n".join(cur)))
            cur_head, cur = None, []
        cur_head = cur_head or head
        cur.append(block)
    if cur:
        chunks.append((cur_head, "\n".join(cur)))

    records = []
    for i, (head, body) in enumerate(chunks, 1):
        if len(body) < 200:
            continue
        sub = f" — {head}" if head and head != title and len(chunks) > 1 else ""
        records.append({
            "id": f"{slugify(host)}-{slugify(path)[:70]}-{i}",
            "title": f"{title or path_words.title()}{sub} ({bank})",
            "bank": bank, "category": category,
            "segments": sorted(set(page_segments) | set(infer_segments(head or ""))),
            "tags": tags, "body": body[: CHUNK_CHARS + 600], "source": url, "origin": "ingested",
        })
    return records


def pdf_to_records(url: str, data: bytes, bank_hint: str | None) -> list[dict]:
    from pypdf import PdfReader
    text = "\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(data)).pages)
    text = re.sub(r"[ \t]+", " ", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[A-Z][^\n]{0,60}:)", text) if len(p.strip()) > 40]
    name = Path(urlparse(url).path).stem.replace("+", " ")
    sections = [(name, [re.sub(r"\s+", " ", p) for p in paras])]
    return page_to_records(url, name, sections, set(), bank_hint)


# ------------------------------------------------------------------ Claude structuring
def structure_with_claude(text: str, src: str, bank_hint: str | None) -> list[dict]:
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, base_url=config.ANTHROPIC_BASE_URL)
    resp = client.messages.parse(
        model=config.MODEL,
        max_tokens=16000,
        output_config={"effort": "low"},
        system="Extract bank product records from content published by an Indian bank. One record per distinct "
               "product or scheme. Use only facts present in the content; never invent rates or limits.",
        messages=[{"role": "user", "content": f"Source: {src}\nBank (hint): {bank_hint or 'infer from content'}\n\n"
                                              f"<content>\n{text[:MAX_CHARS]}\n</content>"}],
        output_format=IngestResult,
    )
    if resp.parsed_output is None:
        return []
    return [{**p.model_dump(), "source": src, "origin": "ingested"} for p in resp.parsed_output.products]


# ------------------------------------------------------------------ main
def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", help="URLs or local HTML/PDF files")
    ap.add_argument("--list", help="text file with one source per line")
    ap.add_argument("--bank", help="bank name hint, e.g. 'Canara Bank'")
    ap.add_argument("--llm", action="store_true", help="structure pages with Claude (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    sources = list(args.sources)
    if args.list:
        sources += [ln.split("#")[0].strip() for ln in Path(args.list).read_text(encoding="utf-8").splitlines()]
    sources = list(dict.fromkeys(s for s in sources if s))
    if not sources:
        ap.print_help()
        return
    if args.llm and not config.LLM_ENABLED:
        sys.exit("--llm needs ANTHROPIC_API_KEY in .env")

    def get(src):
        try:
            return src, fetch(src), None
        except Exception as exc:
            return src, None, exc

    with ThreadPoolExecutor(args.workers) as pool:
        fetched = list(pool.map(get, sources))

    pages = []  # (url, title, sections) for HTML; records for PDFs
    by_host: dict[str, list[dict]] = defaultdict(list)
    for src, res, err in fetched:
        if err:
            print(f"[fail] {src}: {str(err)[:100]}")
            continue
        url, data, is_pdf = res
        host = urlparse(url).netloc or "local"
        if is_pdf:
            recs = pdf_to_records(url, data, args.bank)
            by_host[host] += recs
            print(f"[ok]   {src} (pdf) -> {len(recs)} chunk(s)")
            continue
        title, sections = parse_html(data.decode("utf-8", "ignore"))
        if "not found" in title.lower() or "404" in title:
            print(f"[skip] {src}: page not found")
            continue
        pages.append((src, url, host, title, sections, data))

    # Lines that repeat on at least 30% of a host's pages (min 3) are site chrome, not product content.
    line_hosts: dict[str, Counter] = defaultdict(Counter)
    page_count = Counter(p[2] for p in pages)
    for _, _, host, _, sections, _ in pages:
        line_hosts[host].update({ln for _, lines in sections for ln in lines})
    boiler = {h: {ln for ln, c in cnt.items() if page_count[h] >= 3 and c >= max(3, 0.3 * page_count[h])}
              for h, cnt in line_hosts.items()}

    for src, url, host, title, sections, data in pages:
        if args.llm:
            text = "\n".join(f"## {h}\n" + "\n".join(lines) for h, lines in sections)
            recs = structure_with_claude(text, url, args.bank)
        else:
            recs = page_to_records(url, title, sections, boiler.get(host, set()), args.bank)
        if not recs:
            print(f"[skip] {src}: no product content found (page may render with JavaScript)")
            continue
        by_host[host] += recs
        print(f"[ok]   {src} -> {len(recs)} record(s)  [{recs[0]['category']}]")

    config.INGESTED_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for host, recs in by_host.items():
        out = config.INGESTED_DIR / f"{slugify(host)}.json"
        existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        fresh_sources = {r["source"] for r in recs}
        merged = [r for r in existing if r["source"] not in fresh_sources] + recs  # re-ingest replaces a page
        out.write_text(json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")
        total += len(recs)
        print(f"[save] {out.name}: {len(merged)} records")
    print(f"\nDone: {total} records from {len(pages)} pages. Restart the server to index them.")


if __name__ == "__main__":
    main(sys.argv[1:])
