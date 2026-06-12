"""
Build enriched BM25 index from NODES tariff tree.

BM25 corpus = path descriptions only (section > chapter > heading > leaf).
Notes are stored separately for LLM context but excluded from BM25 scoring
to avoid false positives (e.g. "autobuses" appearing in tire chapter notes).

Tokenization uses SnowballStemmer('spanish').
"""

import json
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from nltk.stem import SnowballStemmer

NODES_PATH = Path(__file__).parent.parent / "nodes.json"
INDEX_PATH = Path(__file__).parent / "bm25_index.pkl"

_stemmer = SnowballStemmer("spanish")


def load_nodes() -> dict:
    with open(NODES_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_parent_map(nodes: dict) -> dict:
    parent = {}
    for code, node in nodes.items():
        for child in node.get("children") or []:
            parent[child] = code
    return parent


def ancestor_chain(code: str, nodes: dict, parent: dict) -> list[str]:
    chain = []
    cur = parent.get(code)
    while cur and cur != "ROOT":
        chain.append(cur)
        cur = parent.get(cur)
    chain.reverse()
    return chain


def notes_text(node: dict) -> str:
    return " ".join(n.get("text", "") for n in (node.get("notes") or []) if n.get("text"))


def clean(text: str) -> str:
    text = re.sub(r"[–—•·]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_enriched_docs(nodes: dict) -> tuple[list[str], list[str], list[str]]:
    """
    Returns (codes, bm25_docs, full_docs).
    bm25_docs: path descriptions only (no notes) — used for BM25 indexing.
    full_docs: path + notes — stored for display/debugging.
    """
    parent = build_parent_map(nodes)
    codes = []
    bm25_docs = []
    full_docs = []

    for code, node in nodes.items():
        if node.get("children"):
            continue

        chain = ancestor_chain(code, nodes, parent)
        path_parts = [nodes[c]["description"] for c in chain if c in nodes]
        path_parts.append(node["description"])

        # Path text only for BM25 — repeated 2x to boost weight vs IDF noise
        path_text = clean(" ".join(path_parts))
        bm25_text = path_text + " " + path_text  # boost

        # Full text (path + notes) for display / LLM context
        note_parts = []
        for c in chain:
            if c in nodes:
                t = notes_text(nodes[c])
                if t:
                    note_parts.append(t)
        self_notes = notes_text(node)
        if self_notes:
            note_parts.append(self_notes)
        full_text = clean(path_text + " " + " ".join(note_parts))

        codes.append(code)
        bm25_docs.append(bm25_text)
        full_docs.append(full_text)

    return codes, bm25_docs, full_docs


def tokenize(text: str) -> list[str]:
    """Stemmed tokenizer: lowercase + Spanish SnowballStemmer."""
    words = re.findall(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9]+", text.lower())
    return [_stemmer.stem(w) for w in words]


def build_and_save_index():
    print("Loading nodes…")
    nodes = load_nodes()

    print("Building enriched documents…")
    codes, bm25_docs, full_docs = build_enriched_docs(nodes)
    print(f"  {len(codes)} leaf nodes")

    print("Tokenizing with Spanish stemmer (path-only corpus)…")
    tokenized = [tokenize(d) for d in bm25_docs]

    print("Building BM25 index…")
    bm25 = BM25Okapi(tokenized)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "codes": codes,
            "bm25_docs": bm25_docs,
            "full_docs": full_docs,
        }, f)
    print(f"Index saved to {INDEX_PATH}")
    return bm25, codes, bm25_docs, full_docs


def load_index():
    if not INDEX_PATH.exists():
        build_and_save_index()
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    # backwards compat: old index had "docs" key
    full_docs = data.get("full_docs", data.get("docs", data.get("bm25_docs", [])))
    return data["bm25"], data["codes"], data.get("bm25_docs", full_docs), full_docs


if __name__ == "__main__":
    build_and_save_index()
