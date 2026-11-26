"""Hybrid retriever: BM25 (lexical) + bge-small embeddings (semantic), fused with
Reciprocal Rank Fusion, then re-scored with client-profile metadata (segment match,
product category, bank the rep sells for, age eligibility)."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from config import EMBED_MODEL, INDEX_DIR, MODEL_CACHE
from kb_loader import Product, load_products

STOP = set(
    "a an the and or of to for in on at by with is are was be been this that it its as from "
    "my his her their our your he she they we i me him them who which what wants want need needs "
    "looking client customer has have having can could would should will about some any also".split()
)

# Domain synonyms so short rep notes still hit the right products.
SYNONYMS = {
    "fd": ["fixed", "deposit"], "rd": ["recurring", "deposit"], "retired": ["senior", "retirement", "pension"],
    "retiree": ["senior", "retirement"], "pensioner": ["senior", "pension"], "daughter": ["girl", "child"],
    "son": ["child", "minor"], "kid": ["child", "minor"], "kids": ["child", "minor"], "farmer": ["kisan", "agri", "crop"],
    "shop": ["msme", "business", "trader"], "shopkeeper": ["msme", "business", "trader"], "business": ["msme"],
    "startup": ["startup", "msme"], "house": ["home"], "flat": ["home"], "apartment": ["home"],
    "tax": ["80c", "tax", "saving"], "abroad": ["international", "forex", "abroad"], "nri": ["nri", "nre", "nro"],
    "vendor": ["street", "vendor"], "woman": ["women"], "lady": ["women"], "wife": ["women"],
    "bike": ["two", "wheeler"], "scooter": ["two", "wheeler"], "ev": ["electric"], "lakh": [], "crore": [],
}


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    out: list[str] = []
    for t in toks:
        if t in STOP or len(t) < 2 and not t.isdigit():
            continue
        out.append(t)
        out.extend(SYNONYMS.get(t, []))
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            for t in query:
                if t in tf:
                    f = tf[t]
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return s


class Embedder:
    """Wraps fastembed. If it can't load (offline, no model), retrieval falls back to BM25 only."""

    def __init__(self):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=EMBED_MODEL, cache_dir=str(MODEL_CACHE))

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self._norm(np.array(list(self.model.passage_embed(texts, batch_size=16))))

    def embed_query(self, text: str) -> np.ndarray:
        return self._norm(np.array(list(self.model.query_embed(text))))[0]

    @staticmethod
    def _norm(m: np.ndarray) -> np.ndarray:
        return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9, None)


@dataclass
class Hit:
    product: Product
    score: float
    reasons: list[str]
    eligibility_flag: str | None = None


ALWAYS_ELIGIBLE_BANKS = {"Generic", "Govt Scheme", "Playbook"}


class Retriever:
    def __init__(self):
        self.products = load_products()
        texts = [p.search_text() for p in self.products]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.embedder = None
        self.doc_vecs = None
        try:
            self.embedder = Embedder()
            self.doc_vecs = self._load_or_build_vectors(texts)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[retriever] dense embeddings unavailable, using BM25 only: {exc}")

    EMBED_CHARS = 1000  # title + metadata + opening facts; long tails slow bge down a lot on CPU
    EMBED_SAVE_EVERY = 100
    SIMILARITY_FLOOR = 0.7  # cosine similarity above this counts as "the same product again"

    def _load_or_build_vectors(self, texts: list[str]) -> np.ndarray:
        """Per-document vector cache keyed by content hash, so re-ingesting only embeds what changed."""
        INDEX_DIR.mkdir(exist_ok=True)
        vec_file, keys_file = INDEX_DIR / "vectors.npy", INDEX_DIR / "keys.json"
        cache: dict[str, np.ndarray] = {}
        if vec_file.exists() and keys_file.exists():
            meta = json.loads(keys_file.read_text())
            if meta.get("model") == EMBED_MODEL:
                cache = dict(zip(meta["keys"], np.load(vec_file)))
        snippets = [t[: self.EMBED_CHARS] for t in texts]
        keys = [hashlib.sha1(s.encode()).hexdigest() for s in snippets]
        missing = list(dict.fromkeys(k for k in keys if k not in cache))
        if missing:
            print(f"[retriever] embedding {len(missing)} new/changed documents...", flush=True)
            snippet_by_key = dict(zip(keys, snippets))
            started = time.time()
            for start in range(0, len(missing), self.EMBED_SAVE_EVERY):
                batch = missing[start: start + self.EMBED_SAVE_EVERY]
                cache.update(zip(batch, self.embedder.embed_docs([snippet_by_key[k] for k in batch])))
                # Checkpoint, so an interrupted build resumes instead of starting over.
                saved = list(cache)
                np.save(vec_file, np.array([cache[k] for k in saved]))
                keys_file.write_text(json.dumps({"model": EMBED_MODEL, "keys": saved}))
                done = start + len(batch)
                print(f"[retriever] {done}/{len(missing)} embedded ({time.time() - started:.0f}s)", flush=True)
        return np.array([cache[k] for k in keys])

    @property
    def mode(self) -> str:
        return "hybrid (BM25 + bge-small embeddings)" if self.doc_vecs is not None else "BM25 only"

    def banks(self) -> list[str]:
        named = {"SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank", "Punjab National Bank",
                 "Bank of Baroda", "Canara Bank"}
        named |= {p.bank for p in self.products if p.bank not in ALWAYS_ELIGIBLE_BANKS}
        return sorted(named)

    def search(
        self,
        queries: list[str],
        segments: list[str] | None = None,
        needs: list[str] | None = None,
        bank: str | None = None,
        age: int | None = None,
        top_k: int = 8,
    ) -> list[Hit]:
        n = len(self.products)
        rrf = np.zeros(n)
        k = 60.0
        # Apply the bank filter BEFORE ranking: otherwise other banks' pages fill the candidate
        # cut-off and the rep's own bank is left with almost no results.
        best_per_query: set[int] = set()  # each need's top match, so one need can't crowd out another
        selling_one_bank = bool(bank) and bank != "All banks"
        allowed = np.array([
            not selling_one_bank or p.bank in ALWAYS_ELIGIBLE_BANKS or p.bank == bank for p in self.products
        ])
        for q in dict.fromkeys(q.strip() for q in queries if q and q.strip()):
            score_lists = [self.bm25.scores(tokenize(q))]
            if self.doc_vecs is not None:
                score_lists.append(self.doc_vecs @ self.embedder.embed_query(q))
            query_rrf = np.zeros(n)
            for scores in score_lists:
                ranking = np.argsort(-np.where(allowed, scores, -np.inf))[: min(50, int(allowed.sum()))]
                for rank, idx in enumerate(ranking):
                    query_rrf[idx] += 1.0 / (k + rank + 1)
            rrf += query_rrf
            best_per_query.add(int(np.argmax(query_rrf)))

        segs = {s.lower() for s in (segments or [])}
        need_set = {c.lower() for c in (needs or [])}
        hits: list[Hit] = []
        top_rrf = rrf.max() or 1.0
        for i, p in enumerate(self.products):
            if rrf[i] == 0:
                continue
            score = rrf[i] / top_rrf  # normalise to 0..1
            if i in best_per_query:
                score += 0.15
            reasons: list[str] = []
            if selling_one_bank:
                mentions_bank = bank.split()[0].lower() in (p.title + " " + p.body).lower()
                if p.bank == bank:
                    score += 0.10
                    reasons.append(f"{bank} product")
                elif p.bank == "Generic" and mentions_bank:
                    score += 0.05
                    reasons.append(f"offered by {bank}")
            overlap = segs & {s.lower() for s in p.segments}
            if overlap:
                score += 0.08 * min(len(overlap), 3)
                reasons.append("fits " + ", ".join(sorted(overlap)).replace("_", " "))
            if need_set and p.category in need_set:
                score += 0.12
                reasons.append(f"matches need: {p.category}")
            elif need_set and p.category != "playbook":
                score -= 0.08  # off-need products can still appear as cross-sell, just lower
            flag = None
            if age is not None:
                if p.age_min is not None and age < p.age_min:
                    flag = f"client age {age} is below minimum {p.age_min}"
                elif p.age_max is not None and age > p.age_max:
                    flag = f"client age {age} is above maximum {p.age_max}"
                if flag:
                    score -= 0.5
            if p.category == "playbook":
                score -= 0.15  # playbooks support the answer; they shouldn't outrank products
            hits.append(Hit(p, round(float(score), 4), reasons, flag))
        hits.sort(key=lambda h: h.score, reverse=True)
        # Diversity: one chunk per source page (ingested pages are split into several chunks).
        seen_sources: set[str] = set()
        unique: list[Hit] = []
        for h in hits:
            key = h.product.source if h.product.origin == "ingested" else h.product.id
            if key in seen_sources:
                continue
            seen_sources.add(key)
            unique.append(h)
        hits = unique
        eligible = self._diversify([h for h in hits if not h.eligibility_flag], top_k)
        cutoff = min((h.score for h in eligible), default=0.0)
        # Keep up to 2 ineligible products that would otherwise have ranked, so the rep sees why they're out.
        flagged = [h for h in hits if h.eligibility_flag and h.score + 0.5 >= cutoff][:2]
        return eligible[: top_k - len(flagged)] + flagged

    def _diversify(self, hits: list[Hit], top_k: int) -> list[Hit]:
        """Greedy MMR-style pick: near-duplicates (the same product on several banks' sites) are
        penalised, so a client with two needs gets both covered rather than one need three times."""
        if self.doc_vecs is None:
            return hits[:top_k]
        index = {id(p): i for i, p in enumerate(self.products)}
        pool, chosen = hits[:80], []
        while pool and len(chosen) < top_k:
            def adjusted(h: Hit) -> float:
                if not chosen:
                    return h.score
                vec = self.doc_vecs[index[id(h.product)]]
                closest = max(float(vec @ self.doc_vecs[index[id(c.product)]]) for c in chosen)
                return h.score - max(0.0, closest - self.SIMILARITY_FLOOR)
            best = max(pool, key=adjusted)
            chosen.append(best)
            pool.remove(best)
        return chosen


if __name__ == "__main__":
    r = Retriever()
    print(r.mode, len(r.products), "products")
    for q in ["retired teacher 61 years with 25 lakh PF money wants monthly income",
              "woman wants to start a boutique, no collateral, needs 15 lakh",
              "parents want to save for 4 year old daughter's education, tax saving"]:
        print("\nQ:", q)
        for h in r.search([q], top_k=5):
            print(f"  {h.score:.3f}  {h.product.title}  {h.reasons}")
