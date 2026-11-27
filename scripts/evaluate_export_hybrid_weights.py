"""Report a fixed MiniLM model across several semantic/BM25 RRF weights.

Use only after model selection is complete. This reads the private evaluation
database and writes aggregate metrics only; it never trains or changes a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

RRF_K = 60


def parse_weights(value: str) -> list[float]:
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not weights or any(not 0 <= weight <= 1 for weight in weights) or len(set(weights)) != len(weights):
        raise ValueError("--semantic-weights must contain distinct numbers from 0 through 1")
    return weights


class BM25Index:
    def __init__(self, documents: list[str]):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE docs USING fts5(doc, tokenize='porter')")
        self.db.executemany("INSERT INTO docs(rowid, doc) VALUES (?, ?)", enumerate(documents, 1))

    def ranks(self, query: str, size: int) -> np.ndarray:
        rank_vector = np.zeros(size, dtype=np.int32)
        tokens = re.findall(r"\w+", query.lower(), re.UNICODE)
        if not tokens:
            return rank_vector
        expression = " OR ".join(f'"{token}"' for token in tokens)
        for rank, (row_id,) in enumerate(
            self.db.execute("SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs), rowid", (expression,)), 1
        ):
            rank_vector[row_id - 1] = rank
        return rank_vector

    def close(self) -> None:
        self.db.close()


def load_evaluation(db_path: Path):
    with sqlite3.connect(f"file:{db_path.resolve().as_uri()[5:]}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        movies = [dict(row) for row in db.execute("SELECT tconst,title,doc,split,has_genome FROM movies ORDER BY tconst")]
        held_out = {key for (key,) in db.execute("SELECT tag_key FROM heldout_tags")}
        answers: defaultdict[int, set[str]] = defaultdict(set)
        for tag_id, tconst in db.execute(
            "SELECT gl.tagId,gl.tconst FROM genome_labels gl JOIN movies m ON m.tconst=gl.tconst "
            "WHERE m.split='test' AND m.has_genome=1"
        ):
            answers[tag_id].add(tconst)
        content_queries = []
        for tag_id, tag, key in db.execute("SELECT tagId,tag,tag_key FROM genome_tags ORDER BY tagId"):
            if 5 <= len(answers[tag_id]) <= 100:
                content_queries.append((tag, "unseen" if key in held_out else "seen", answers[tag_id]))

    content_movies = [movie for movie in movies if movie["split"] == "test" and movie["has_genome"]]
    titles: defaultdict[str, set[str]] = defaultdict(set)
    for movie in movies:
        titles[movie["title"].casefold()].add(movie["tconst"])
    title_queries = [(movie["title"], "title", titles[movie["title"].casefold()]) for movie in movies if movie["split"] == "test"]
    return movies, content_movies, content_queries, title_queries


def score_pool(model, documents: list[str], movie_ids: list[str], queries, *, limit: int, weights: list[float], batch_size: int, content: bool):
    document_vectors = np.asarray(
        model.encode(documents, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32
    )
    bm25 = BM25Index(documents)
    totals: dict[float, defaultdict[str, list[float]]] = {
        weight: defaultdict(lambda: [0.0, 0.0, 0.0]) for weight in weights
    }
    for start in range(0, len(queries), 32):
        batch = queries[start : start + 32]
        query_vectors = np.asarray(
            model.encode([query for query, _, _ in batch], batch_size=32, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
        for (query, group, truth), scores in zip(batch, query_vectors @ document_vectors.T):
            semantic_order = np.argsort(-scores, kind="stable")
            semantic_ranks = np.empty(len(movie_ids), dtype=np.int32)
            semantic_ranks[semantic_order] = np.arange(1, len(movie_ids) + 1)
            lexical_ranks = bm25.ranks(query, len(movie_ids))
            for weight in weights:
                if weight == 1:
                    ranked = semantic_order[:limit]
                else:
                    fused = weight / (RRF_K + semantic_ranks.astype(np.float64))
                    present = lexical_ranks > 0
                    fused[present] += (1 - weight) / (RRF_K + lexical_ranks[present])
                    ranked = np.argsort(-fused, kind="stable")[:limit]
                retrieved = [movie_ids[index] for index in ranked]
                bucket = totals[weight][group]
                if content:
                    bucket[0] += sum(movie_id in truth for movie_id in retrieved[:10]) / 10
                    bucket[1] += sum(movie_id in truth for movie_id in retrieved[:100]) / len(truth)
                else:
                    bucket[0] += int(retrieved[0] in truth)
                    bucket[1] += int(bool(set(retrieved[:10]) & truth))
                bucket[2] += 1
    bm25.close()
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description="Final aggregate evaluation of fixed hybrid RRF weights.")
    parser.add_argument("--evaluation-db", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON aggregate metrics output")
    parser.add_argument("--semantic-weights", default="1.0,0.8,0.7,0.6,0.5")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    if not args.evaluation_db.is_file() or not args.model.is_dir() or args.batch_size < 1:
        raise SystemExit("evaluation DB and model directory must exist; batch size must be positive")
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit(f"Missing evaluation dependency: {error}") from error
    weights = parse_weights(args.semantic_weights)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but PyTorch cannot see a CUDA GPU")

    movies, content_movies, content_queries, title_queries = load_evaluation(args.evaluation_db)
    model = SentenceTransformer(str(args.model), device=device)
    content = score_pool(
        model, [movie["doc"] for movie in content_movies], [movie["tconst"] for movie in content_movies],
        content_queries, limit=100, weights=weights, batch_size=args.batch_size, content=True,
    )
    title = score_pool(
        model, [movie["doc"] for movie in movies], [movie["tconst"] for movie in movies],
        title_queries, limit=10, weights=weights, batch_size=args.batch_size, content=False,
    )
    rows = []
    for weight in weights:
        seen_p, seen_r, seen_n = content[weight]["seen"]
        unseen_p, unseen_r, unseen_n = content[weight]["unseen"]
        hit1, hit10, title_n = title[weight]["title"]
        rows.append({
            "semantic_weight": weight,
            "bm25_weight": round(1 - weight, 4),
            "seen_queries": int(seen_n), "seen_p_at_10": round(seen_p / seen_n, 4), "seen_r_at_100": round(seen_r / seen_n, 4),
            "unseen_queries": int(unseen_n), "unseen_p_at_10": round(unseen_p / unseen_n, 4), "unseen_r_at_100": round(unseen_r / unseen_n, 4),
            "title_queries": int(title_n), "title_hit_at_1": round(hit1 / title_n, 4), "title_hit_at_10": round(hit10 / title_n, 4),
        })
    payload = {"model": str(args.model), "evaluation_db": str(args.evaluation_db), "weights": rows,
               "warning": "Frozen final-test sweep: report these metrics; do not train or tune from them."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print("Semantic/BM25  Seen P@10  Unseen P@10  Title hit@1  Title hit@10")
    for row in rows:
        print(f"{row['semantic_weight']:.0%}/{row['bm25_weight']:.0%}          {row['seen_p_at_10']:.4f}      {row['unseen_p_at_10']:.4f}        {row['title_hit_at_1']:.4f}      {row['title_hit_at_10']:.4f}")
    print(f"Saved aggregate weight sweep to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
