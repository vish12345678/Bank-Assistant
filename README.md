# Product Advisor: a RAG copilot for bank sales reps

A rep types a client's needs in plain language, for example *"62 yr retired govt teacher, ₹25L PF, wants
safe monthly income"*. The advisor replies with ranked products and government schemes, why each one
fits, key facts with citations, cross-sells, a pitch line, follow-up questions and a document checklist.

## Architecture

```
Rep's note + chat history
        │
        ▼
1. UNDERSTAND   Claude (structured output) → ClientProfile
                {segments, needs, age, amount, risk, goals, search_queries[3-5], missing_info}
                (offline fallback: regex/keyword profiler)
        │
        ▼
2. RETRIEVE     Hybrid search over the product KB
                BM25 (+ Indian-banking synonyms) ⊕ bge-small embeddings → Reciprocal Rank Fusion
                → re-score: segment match, need/category match, "I sell for <bank>" filter,
                  age-eligibility flags (e.g. APY 18–40, SCSS 55+, PMJJBY 18–50)
        │
        ▼
3. RECOMMEND    Claude streams a grounded answer citing sources [S1..S8]
                (offline fallback: templated summary of the top matches)
```

| File | Role |
|---|---|
| `kb/*.md` | Curated knowledge base: 77 product/scheme records with metadata (bank, category, segments, tags, age limits, source URL) |
| `kb/ingested/*.json` | Records added by `ingest.py` from bank web pages/PDFs |
| `kb_loader.py` | Parses the KB into `Product` records |
| `retriever.py` | Hybrid BM25 + dense retriever with metadata re-ranking; caches embeddings in `index/` |
| `advisor.py` | The 3-step pipeline: profile extraction, retrieval, streamed recommendation |
| `server.py` | FastAPI app that streams events to the UI over SSE |
| `static/index.html` | Chat UI with a bank selector, extracted client profile, and clickable sources |
| `ingest.py` | Adds bank web pages or PDFs to the KB (Claude structures them into product records) |
| `eval_retrieval.py` | Retrieval check across 10 client scenarios |

## Knowledge base coverage

- **Govt schemes sold through banks:** PPF, SSY, SCSS, KVP, RBI Floating Rate Bonds, NPS/NPS Vatsalya,
  PMJDY, PMJJBY, PMSBY, APY, Mudra (incl. Tarun Plus), Stand-Up India, PMEGP, CGTMSE, PM SVANidhi,
  PM Vishwakarma, PM Vidyalaxmi, PMAY-U 2.0, Agri Infra Fund, KCC, SHG linkage
- **Bank products:** SBI, HDFC Bank, ICICI Bank, Axis Bank and Kotak home, car, personal, gold,
  education and LAP loans; FDs, tax-saver FD, RD, sweep-in, annuity deposit; premium, salary, minor,
  senior and current accounts; credit, secured, business and forex cards; NRE/NRO/FCNR, GIFT City
- **MSME/corporate:** CC/OD, term loans, TReDS, LC/BG and export credit, supply chain finance, POS/QR, CMS, startup banking
- **Sales playbook:** life-stage cross-sell map, discovery questions, compliance rules, objection handling, document checklists

Records were compiled from content that banks, NSI, PFRDA, RBI and scheme portals publish, and each one links
its source. **Rates and limits change** (small-savings rates change every quarter), so the advisor tells reps to
check the current rate card before quoting a number.

## Run

```bash
py -m pip install -r requirements.txt
copy .env.example .env        # then set ANTHROPIC_API_KEY=...
py server.py                  # http://127.0.0.1:8770
```

Without an API key the app runs in **offline mode**: the keyword profiler, hybrid retrieval and a templated answer.
With a key it uses `claude-opus-5-5` (change it with `ADVISOR_MODEL`): low effort for profile extraction and
medium effort for the recommendation. Server-side refusal fallback is turned on.

## Ingested bank websites

[sources.txt](sources.txt) lists 368 official product pages: SBI, HDFC Bank (including homeloans.hdfc), ICICI Bank,
Axis Bank, Kotak, Bank of Baroda, Canara Bank, and the Stand-Up Mitra portal. These pages were found by following
links from each bank's hub pages. Ingesting them gives about 1,700 chunks in `kb/ingested/` (one JSON file per website).

```bash
py ingest.py --list sources.txt                 # offline: clean, split into sections, auto-tag
py ingest.py --list sources.txt --llm           # Claude writes one structured record per product
py ingest.py https://www.axis.bank.in/loans/gold-loan
py ingest.py product_brochure.pdf --bank "Canara Bank"
```

- **Offline mode** removes menus, footers and lines repeated across a bank's pages, splits each page by heading
  into chunks of about 1,200 characters, and tags each chunk with bank, category and client segments from the
  URL and title.
- Re-ingesting a page replaces its old chunks. Restart the server afterwards. Only new or changed chunks are
  embedded, because vectors are cached per document in `index/`.
- Search returns at most one chunk per page, so a single long page can't fill the results.
- About 13 pages were skipped because they load their content with JavaScript (mostly SBI sections). For those,
  save the page from a browser or download the brochure PDF and ingest the file.
- The NSI, Mudra and Jan Suraksha portals timed out from here, so those schemes come from the curated KB and from
  the banks' own scheme pages.
- To add a product by hand, add a `=== PRODUCT` block to any `kb/*.md` file.

## Check retrieval quality

```bash
py eval_retrieval.py          # rule-based profiles, no key needed (currently 38/40)
py eval_retrieval.py --llm    # Claude-extracted profiles (not yet run)
```

The test runs 10 client scenarios under 4 bank filters. A scenario passes only if the expected products are in the
top 5, nothing marked unsuitable appears, and 5 results come back. The ranking was tuned against these same
scenarios, so treat the score as a regression check, not a measure of accuracy on new clients.

Known misses:
- "Crop loan and a tractor" with All banks: tractor pages crowd out KCC.
- NRI scenario with ICICI Bank: no FCNR result in the top 5.
