"""Retrieval regression check: does each client scenario surface the expected products in the top 5?

  py eval_retrieval.py          rule-based profiles (no API key needed)
  py eval_retrieval.py --llm    Claude-extracted profiles (uses the API)

An expected product passes if it appears as the curated record OR as a bank's own page for that
product, so each expectation is a pattern matched against the record's id and title.
A scenario also fails if fewer than 5 products come back, or if an unsuitable product ranks in the
top 5 ("avoid"). Add a scenario whenever a rep reports a bad recommendation.
"""
import re
import sys

from advisor import Advisor, extract_profile_llm, extract_profile_rules, retrieve

SCSS = r"scss|senior.citizens?.sav"
STANDUP = r"stand.?up"
MUDRA = r"mudra|pmmy"
CGTMSE = r"cgtmse|credit guarantee"
NPS = r"\bnps\b|nps-|national.pension"
SSY = r"\bssy\b|sukanya"
NRE = r"\bnre\b|nre-"
FCNR = r"fcnr"
POS = r"merchant|pos-|\bpos\b|upi.collection|smarttrade"
WC = r"working.capital|cash.credit|overdraft"
KCC = r"\bkcc\b|kisan.credit|crop.loan|kisan.mitra|baroda.kisan"
TRACTOR = r"tractor|farm.mechani"
SECURED_CARD = r"secured.card|against.fd|credit.card.against|811"
SVANIDHI = r"svanidhi"

CASES = [
    ("62 yr old retired govt teacher, got 25 lakh PF + gratuity, wants safe monthly income, also worried about medical costs", [SCSS], []),
    ("Woman, 34, wants to open a boutique in Pune, needs 15 lakh, has no property to give as collateral", [STANDUP, f"{MUDRA}|{CGTMSE}"], [r"home.loan"]),
    ("Salaried IT employee, 29, 18 LPA, old tax regime, already maxed 80C, wants to save more tax and buy a flat in 2 years", [NPS], []),
    ("Parents of a 4-year-old daughter want a safe long-term plan for her education", [SSY], []),
    ("NRI in Dubai, sends 1.5L home every month, wants tax-free returns and may move back in 5 years", [NRE, FCNR], []),
    ("Small kirana shop owner, 40L annual turnover, wants to accept UPI/cards and needs working capital for stock", [POS, WC], []),
    ("Farmer with 4 acres, grows cotton, needs crop loan and wants to buy a tractor", [KCC, TRACTOR], []),
    ("22-year-old fresher, first job, no credit history, wants a credit card", [SECURED_CARD], []),
    ("45 year old wants a guaranteed pension scheme, works as a driver, doesn't pay income tax", [NPS], []),
    ("Street vendor selling vegetables needs small loan of 15000", [SVANIDHI], []),
]
BANKS = ["All banks", "SBI", "HDFC Bank", "ICICI Bank"]


def main():
    use_llm = "--llm" in sys.argv
    advisor = Advisor()
    passed = total = 0
    for bank in BANKS:
        for query, expected, avoid in CASES:
            profile = extract_profile_llm([], query) if use_llm else extract_profile_rules([], query)
            top = retrieve(advisor.retriever, profile, query, bank)[:5]
            labels = [f"{h.product.id} {h.product.title}".lower() for h in top]
            missing = [e for e in expected if not any(re.search(e, x) for x in labels)]
            bad = [a for a in avoid if any(re.search(a, x) for x in labels)]
            ok = not missing and not bad and len(top) == 5
            passed += ok
            total += 1
            print(f"{'PASS' if ok else 'MISS'}  [{bank}] {query[:60]}")
            if not ok:
                print(f"      missing={missing} unsuitable={bad} returned={len(top)}")
                print(f"      top5={[h.product.title[:45] for h in top]}")
    print(f"\n{passed}/{total} passed")


if __name__ == "__main__":
    main()
