"""Loads the product knowledge base.

Sources:
  kb/*.md            curated product records ("=== PRODUCT" blocks with key: value metadata)
  kb/ingested/*.json records produced by ingest.py from bank web pages / PDFs
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from config import INGESTED_DIR, KB_DIR

LIST_FIELDS = {"segments", "tags"}
INT_FIELDS = {"age_min", "age_max", "max_beneficiary_age"}


@dataclass
class Product:
    id: str
    title: str
    bank: str
    category: str
    body: str
    segments: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source: str = ""
    age_min: int | None = None
    age_max: int | None = None
    max_beneficiary_age: int | None = None
    origin: str = "curated"  # curated | ingested

    def search_text(self) -> str:
        """Text used for both BM25 and embeddings: metadata is included so it is matchable."""
        return (
            f"{self.title}. Bank: {self.bank}. Category: {self.category}. "
            f"For: {', '.join(s.replace('_', ' ') for s in self.segments)}. "
            f"Keywords: {', '.join(self.tags)}.\n{self.body}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_block(block: str, origin_file: str) -> Product | None:
    if "\n---" not in block:
        return None
    header, body = block.split("\n---", 1)
    meta: dict = {}
    for line in header.strip().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if key in LIST_FIELDS:
            meta[key] = [v.strip() for v in val.split(",") if v.strip()]
        elif key in INT_FIELDS:
            meta[key] = int(val)
        else:
            meta[key] = val
    if "id" not in meta or "title" not in meta:
        return None
    return Product(
        id=meta["id"],
        title=meta["title"],
        bank=meta.get("bank", "Generic"),
        category=meta.get("category", "other"),
        body=body.strip(),
        segments=meta.get("segments", []),
        tags=meta.get("tags", []),
        source=meta.get("source", origin_file),
        age_min=meta.get("age_min"),
        age_max=meta.get("age_max"),
        max_beneficiary_age=meta.get("max_beneficiary_age"),
    )


def load_products() -> list[Product]:
    products: list[Product] = []
    for md in sorted(KB_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for block in text.split("=== PRODUCT")[1:]:
            p = _parse_block(block, md.name)
            if p:
                products.append(p)
    if INGESTED_DIR.exists():
        for js in sorted(INGESTED_DIR.glob("*.json")):
            data = json.loads(js.read_text(encoding="utf-8"))
            for rec in data if isinstance(data, list) else [data]:
                rec.setdefault("origin", "ingested")
                products.append(Product(**rec))
    seen: set[str] = set()
    unique = []
    for p in products:
        if p.id in seen:
            continue
        seen.add(p.id)
        unique.append(p)
    return unique


if __name__ == "__main__":
    items = load_products()
    print(f"{len(items)} products")
    from collections import Counter
    print(Counter(p.category for p in items))
    print(Counter(p.bank for p in items))
