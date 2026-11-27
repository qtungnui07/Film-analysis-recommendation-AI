"""Compare the visible search results of two exported MiniLM models.

This is a qualitative inspection tool.  It uses the same catalog, query set,
and hybrid RRF setting for both models, then writes the top results so that a
reviewer can see which titles move in or out of the ranking.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

import numpy as np

RRF_K = 60


class BM25Index:
    def __init__(self, documents: list[str]) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE docs USING fts5(doc, tokenize='porter')")
        self.db.executemany("INSERT INTO docs(rowid, doc) VALUES (?, ?)", enumerate(documents, 1))

    def ranks(self, query: str, size: int) -> np.ndarray:
        ranks = np.zeros(size, dtype=np.int32)
        tokens = re.findall(r"\w+", query.lower(), re.UNICODE)
        if not tokens:
            return ranks
        expression = " OR ".join(f'"{token}"' for token in tokens)
        for rank, (row_id,) in enumerate(
            self.db.execute("SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs), rowid", (expression,)), 1
        ):
            ranks[row_id - 1] = rank
        return ranks

    def close(self) -> None:
        self.db.close()


def load_catalog(db_path: Path) -> list[dict[str, str]]:
    with sqlite3.connect(f"file:{db_path.resolve().as_uri()[5:]}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT tconst,title,doc FROM movies ORDER BY tconst")]


def rank_queries(model, documents: list[str], bm25: BM25Index, queries: list[str], *, semantic_weight: float, top_k: int, batch_size: int):
    vectors = np.asarray(model.encode(documents, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    query_vectors = np.asarray(model.encode(queries, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    results = {}
    for query, scores in zip(queries, query_vectors @ vectors.T):
        semantic_order = np.argsort(-scores, kind="stable")
        semantic_ranks = np.empty(len(documents), dtype=np.int32)
        semantic_ranks[semantic_order] = np.arange(1, len(documents) + 1)
        lexical_ranks = bm25.ranks(query, len(documents))
        fused = semantic_weight / (RRF_K + semantic_ranks.astype(np.float64))
        lexical_present = lexical_ranks > 0
        fused[lexical_present] += (1 - semantic_weight) / (RRF_K + lexical_ranks[lexical_present])
        ranking = np.argsort(-fused, kind="stable")[:top_k]
        results[query] = [
            {
                "rank": rank,
                "catalog_index": int(index),
                "semantic_rank": int(semantic_ranks[index]),
                "bm25_rank": int(lexical_ranks[index]) or None,
                "rrf_score": round(float(fused[index]), 8),
            }
            for rank, index in enumerate(ranking, 1)
        ]
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualitatively compare visible results from two exported search models.")
    parser.add_argument("--catalog-db", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--changed-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", default="cyberpunk,pixar,zombie,time travel,superhero,heist,martial arts,dystopia,artificial intelligence")
    parser.add_argument("--semantic-weight", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    if not (0 <= args.semantic_weight <= 1) or args.top_k < 1 or args.batch_size < 1:
        raise SystemExit("semantic weight must be 0..1; top-k and batch-size must be positive")
    if not args.catalog_db.is_file() or not args.base_model.is_dir() or not args.changed_model.is_dir():
        raise SystemExit("catalog DB and both model directories must exist")

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but PyTorch cannot see a CUDA GPU")
    queries = [query.strip() for query in args.queries.split(",") if query.strip()]
    if not queries:
        raise SystemExit("provide at least one query")
    catalog = load_catalog(args.catalog_db)
    documents = [movie["doc"] for movie in catalog]
    bm25 = BM25Index(documents)
    try:
        base = SentenceTransformer(str(args.base_model), device=device)
        base_results = rank_queries(base, documents, bm25, queries, semantic_weight=args.semantic_weight, top_k=args.top_k, batch_size=args.batch_size)
        changed = SentenceTransformer(str(args.changed_model), device=device)
        changed_results = rank_queries(changed, documents, bm25, queries, semantic_weight=args.semantic_weight, top_k=args.top_k, batch_size=args.batch_size)
    finally:
        bm25.close()

    for result_set in (base_results, changed_results):
        for items in result_set.values():
            for item in items:
                movie = catalog[item.pop("catalog_index")]
                item["title"] = movie["title"]
                item["tconst"] = movie["tconst"]
    payload = {
        "catalog_db": str(args.catalog_db),
        "base_model": str(args.base_model),
        "changed_model": str(args.changed_model),
        "hybrid": {"semantic_weight": args.semantic_weight, "bm25_weight": round(1 - args.semantic_weight, 4), "rrf_k": RRF_K},
        "queries": queries,
        "base": base_results,
        "changed": changed_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for query in queries:
        base_titles = [item["title"] for item in base_results[query][:3]]
        changed_titles = [item["title"] for item in changed_results[query][:3]]
        print(f"{query}:\n  base    {base_titles}\n  changed {changed_titles}")
    print(f"Saved qualitative comparison to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
