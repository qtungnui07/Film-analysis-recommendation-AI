"""Qualitatively rerank hybrid movie-search candidates using public metadata.

The first-stage model retrieves candidates with MiniLM + BM25 RRF.  This
second stage uses only application metadata that is available at search time:
TMDB keywords and IMDb rating/vote count.  It deliberately does not read
Genome labels, so it cannot leak the benchmark's held-out relevance labels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path

import numpy as np

RRF_K = 60


def tokens(value: str) -> set[str]:
    return set(re.findall(r"\w+", value.casefold(), re.UNICODE))


class BM25Index:
    def __init__(self, documents: list[str]) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE docs USING fts5(doc, tokenize='porter')")
        self.db.executemany("INSERT INTO docs(rowid, doc) VALUES (?, ?)", enumerate(documents, 1))

    def ranks(self, query: str, size: int) -> np.ndarray:
        rank_vector = np.zeros(size, dtype=np.int32)
        query_tokens = re.findall(r"\w+", query.lower(), re.UNICODE)
        if not query_tokens:
            return rank_vector
        expression = " OR ".join(f'"{token}"' for token in query_tokens)
        for rank, (row_id,) in enumerate(
            self.db.execute("SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs), rowid", (expression,)), 1
        ):
            rank_vector[row_id - 1] = rank
        return rank_vector

    def close(self) -> None:
        self.db.close()


def load_catalog(db_path: Path) -> list[dict[str, object]]:
    with sqlite3.connect(f"file:{db_path.resolve().as_uri()[5:]}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(
            "SELECT tconst,title,doc,coalesce(keywords, '') AS keywords,imdb_rating,coalesce(imdb_votes, 0) AS imdb_votes "
            "FROM movies ORDER BY tconst"
        )]


def keyword_match(query: str, movie_keywords: str) -> float:
    """Score exact phrase matches first, then coverage of query terms."""
    query_normalized = " ".join(re.findall(r"\w+", query.casefold(), re.UNICODE))
    keywords_normalized = " ".join(re.findall(r"\w+", movie_keywords.casefold(), re.UNICODE))
    if not query_normalized or not keywords_normalized:
        return 0.0
    if query_normalized in keywords_normalized:
        return 1.0
    query_tokens = tokens(query)
    return len(query_tokens & tokens(movie_keywords)) / len(query_tokens)


def minmax(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    return np.zeros_like(values) if high == low else (values - low) / (high - low)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerank hybrid movie candidates with keywords and IMDb quality.")
    parser.add_argument("--catalog-db", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", default="cyberpunk,pixar,zombie,time travel,superhero,heist,martial arts,dystopia,artificial intelligence")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--semantic-weight", type=float, default=0.8)
    parser.add_argument("--base-weight", type=float, default=0.60)
    parser.add_argument("--keyword-weight", type=float, default=0.25)
    parser.add_argument("--quality-weight", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    if args.candidate_k < args.top_k or args.top_k < 1 or args.batch_size < 1 or not 0 <= args.semantic_weight <= 1:
        raise SystemExit("candidate-k must be >= top-k; top-k/batch-size must be positive; semantic weight must be 0..1")
    if not math.isclose(args.base_weight + args.keyword_weight + args.quality_weight, 1.0, abs_tol=1e-9):
        raise SystemExit("base, keyword, and quality weights must sum to 1")
    if not args.catalog_db.is_file() or not args.model.is_dir():
        raise SystemExit("catalog DB and model directory must exist")

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but PyTorch cannot see a CUDA GPU")
    queries = [query.strip() for query in args.queries.split(",") if query.strip()]
    catalog = load_catalog(args.catalog_db)
    documents = [str(movie["doc"]) for movie in catalog]
    model = SentenceTransformer(str(args.model), device=device)
    document_vectors = np.asarray(model.encode(documents, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    query_vectors = np.asarray(model.encode(queries, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    bm25 = BM25Index(documents)
    results: dict[str, list[dict[str, object]]] = {}
    try:
        for query, vector in zip(queries, query_vectors):
            semantic_order = np.argsort(-(document_vectors @ vector), kind="stable")
            semantic_ranks = np.empty(len(catalog), dtype=np.int32)
            semantic_ranks[semantic_order] = np.arange(1, len(catalog) + 1)
            lexical_ranks = bm25.ranks(query, len(catalog))
            fused = args.semantic_weight / (RRF_K + semantic_ranks.astype(np.float64))
            present = lexical_ranks > 0
            fused[present] += (1 - args.semantic_weight) / (RRF_K + lexical_ranks[present])
            candidate_indices = np.argsort(-fused, kind="stable")[: args.candidate_k]
            base_scores = minmax(fused[candidate_indices])
            keyword_scores = np.asarray([keyword_match(query, str(catalog[index]["keywords"])) for index in candidate_indices])
            ratings = np.asarray([float(catalog[index]["imdb_rating"] or 0.0) for index in candidate_indices]) / 10.0
            votes = np.asarray([math.log1p(int(catalog[index]["imdb_votes"] or 0)) for index in candidate_indices])
            quality_scores = minmax(ratings * minmax(votes))
            final_scores = args.base_weight * base_scores + args.keyword_weight * keyword_scores + args.quality_weight * quality_scores
            final_order = np.argsort(-final_scores, kind="stable")[: args.top_k]
            rows = []
            for final_rank, position in enumerate(final_order, 1):
                index = int(candidate_indices[position])
                movie = catalog[index]
                rows.append({
                    "rerank": final_rank,
                    "base_rank": int(np.where(candidate_indices == index)[0][0]) + 1,
                    "title": movie["title"], "tconst": movie["tconst"],
                    "keyword_match": round(float(keyword_scores[position]), 3),
                    "imdb_rating": movie["imdb_rating"], "imdb_votes": movie["imdb_votes"],
                    "score": round(float(final_scores[position]), 4),
                })
            results[query] = rows
    finally:
        bm25.close()

    payload = {
        "model": str(args.model), "catalog_db": str(args.catalog_db), "queries": queries,
        "first_stage": {"semantic_weight": args.semantic_weight, "bm25_weight": round(1 - args.semantic_weight, 4), "candidate_k": args.candidate_k},
        "reranker": {"base_weight": args.base_weight, "keyword_weight": args.keyword_weight, "quality_weight": args.quality_weight},
        "warning": "Qualitative public-metadata reranker; it never uses Genome labels and has not been tuned on frozen test metrics.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for query, rows in results.items():
        print(f"{query}: {[row['title'] for row in rows[:5]]}")
    print(f"Saved reranked results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
