"""
BM25 + fuzzy re-ranking retriever.

retrieve(query, top_k) → list of candidate dicts
get_relevant_notes(codes) → string of relevant tariff notes for LLM context
"""

import re
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rapidfuzz import fuzz
from build_index import load_index, tokenize, load_nodes, build_parent_map

_bm25 = None
_codes = None
_bm25_docs = None
_full_docs = None
_nodes = None
_parent = None


def _ensure_loaded():
    global _bm25, _codes, _bm25_docs, _full_docs, _nodes, _parent
    if _bm25 is None:
        _bm25, _codes, _bm25_docs, _full_docs = load_index()
        _nodes = load_nodes()
        _parent = build_parent_map(_nodes)


def get_nodes():
    _ensure_loaded()
    return _nodes


def _ancestor_chain(code: str) -> list[str]:
    _ensure_loaded()
    chain = []
    cur = _parent.get(code)
    while cur and cur != "ROOT":
        chain.append(cur)
        cur = _parent.get(cur)
    chain.reverse()
    return chain


def _breadcrumb(code: str) -> str:
    chain = _ancestor_chain(code)
    node = _nodes.get(code, {})
    parts = [_nodes[c]["description"] for c in chain if c in _nodes]
    parts.append(node.get("description", ""))
    return " > ".join(parts)


def retrieve(query: str, top_k: int = 15) -> list[dict]:
    _ensure_loaded()

    tokens = tokenize(query)
    bm25_scores = _bm25.get_scores(tokens)

    # BM25 pool: 5× top_k
    pool_size = min(top_k * 5, len(_codes))
    pool_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:pool_size]

    # Fuzzy re-rank against the raw (unstemmed) path text
    max_bm25 = max(bm25_scores[i] for i in pool_indices) or 1.0
    results = []
    for i in pool_indices:
        b_score = float(bm25_scores[i])
        fuzzy_s = fuzz.partial_ratio(query.lower(), _bm25_docs[i][:300].lower()) / 100.0
        combined = 0.65 * b_score + 0.35 * fuzzy_s * max_bm25
        results.append((combined, i))

    results.sort(reverse=True)

    out = []
    for _, i in results[:top_k]:
        code = _codes[i]
        node = _nodes.get(code, {})
        out.append({
            "code": code,
            "description": node.get("description", ""),
            "breadcrumb": _breadcrumb(code),
            "bm25_score": float(bm25_scores[i]),
        })
    return out


def get_relevant_notes(codes: list[str]) -> str:
    """Collect unique notes from ancestors of the candidate codes."""
    _ensure_loaded()
    seen = set()
    blocks = []
    for code in codes:
        chain = _ancestor_chain(code)
        for c in chain + [code]:
            if c in seen or c not in _nodes:
                continue
            seen.add(c)
            node = _nodes[c]
            for note in node.get("notes") or []:
                t = note.get("text", "").strip()
                if t:
                    label = f"[{c}] {node['description']}"
                    blocks.append(f"--- Notas de {label} ---\n{t}")
    return "\n\n".join(blocks)
