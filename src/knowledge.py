"""Lightweight, explainable local retrieval; no downloads or API keys required.

Sections of rag_docs/*.md are indexed separately so citations name the source.
The optional semantic retriever remains available in src/rag.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import re

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "rag_docs"
WORD_RE = re.compile(r"[a-z0-9]+", re.I)
STOP = {"i", "you", "we", "he", "she", "it", "the", "a", "an", "and", "or", "to", "of", "for", "from", "on", "at", "in", "this", "that", "my", "your", "can", "do", "does", "is", "are", "how", "why", "what", "when", "with", "please", "me", "be", "was", "were", "could", "would", "have", "has", "about", "course", "courses"}


@dataclass(frozen=True)
class Passage:
    title: str
    text: str
    source: str
    score: float = 0.0


def _tokens(text: str) -> set[str]:
    words = {w.lower() for w in WORD_RE.findall(text) if len(w) > 2}
    return {w[:-1] if w.endswith("s") and len(w) > 4 else w for w in words - STOP}


@lru_cache(maxsize=1)
def load_passages() -> tuple[Passage, ...]:
    result = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        sections = re.split(r"(?=^## )", path.read_text(encoding="utf-8"), flags=re.M)
        for section in sections:
            first, _, rest = section.strip().partition("\n")
            if not first.startswith("## ") or not rest.strip():
                continue
            result.append(Passage(first[3:].strip(), rest.strip(), path.name))
    return tuple(result)


def search(query: str, intent: str | None = None, limit: int = 2) -> list[Passage]:
    """Return only meaningfully matching passages, no fabricated answer or sources."""
    terms = _tokens(query)
    if not terms:
        return []
    docs = load_passages()
    candidates = [p for p in docs if not intent or intent == "general" or
                  ("booking" if p.source == "booking_policy.md" else p.source.split("_")[0]) == intent]
    if not candidates:
        candidates = list(docs)
    doc_sets = [_tokens(p.title + " " + p.text) for p in candidates]
    df = {t: sum(t in s for s in doc_sets) for t in terms}
    ranked = []
    for p, docset in zip(candidates, doc_sets):
        overlap = terms & docset
        if not overlap:
            continue
        title_terms = _tokens(p.title)
        score = sum((1 + math.log((len(candidates) + 1) / (df[t] + 1))) * (2.0 if t in title_terms else 1.0) for t in overlap)
        score /= max(1.0, len(terms) ** 0.6)
        ranked.append(Passage(p.title, p.text, p.source, round(score, 3)))
    return sorted(ranked, key=lambda p: (-p.score, p.source, p.title))[:limit]
