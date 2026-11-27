"""Compact a train-only Genome relevance CSV for hard-negative mining.

The input is intentionally a verbose audit export: one (movie, tag) row with
the movie text repeated.  This utility validates it and writes one compressed
NumPy artifact with a dense [movie, tag] float32 matrix.  It never accepts test
rows, so the output can be consumed by the hard-negative training stage.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {
    "tconst", "split", "has_genome", "tagId", "tag", "tag_key", "relevance", "movie_text"
}


def compact_relevance_csv(input_path: Path, output_path: Path) -> dict[str, int | float]:
    """Validate a sorted train-only relevance CSV and write a compact NPZ."""
    if not input_path.is_file():
        raise FileNotFoundError(f"input CSV not found: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    movie_ids: list[str] = []
    movie_texts: list[str] = []
    relevance_rows: list[np.ndarray] = []
    expected_tag_ids: tuple[int, ...] | None = None
    expected_tag_names: tuple[str, ...] | None = None
    expected_tag_keys: tuple[str, ...] | None = None
    current_movie: str | None = None
    current_text: str | None = None
    current_tag_ids: list[int] = []
    current_tag_names: list[str] = []
    current_tag_keys: list[str] = []
    current_scores: list[float] = []
    seen_movies: set[str] = set()
    low = high = rows = 0
    minimum, maximum = 1.0, 0.0

    def finish_movie() -> None:
        nonlocal expected_tag_ids, expected_tag_names, expected_tag_keys
        if current_movie is None or current_text is None:
            return
        tag_ids = tuple(current_tag_ids)
        tag_names = tuple(current_tag_names)
        tag_keys = tuple(current_tag_keys)
        if len(set(tag_ids)) != len(tag_ids):
            raise ValueError(f"movie {current_movie} has duplicate tag IDs")
        if expected_tag_ids is None:
            expected_tag_ids, expected_tag_names, expected_tag_keys = tag_ids, tag_names, tag_keys
        elif (tag_ids, tag_names, tag_keys) != (expected_tag_ids, expected_tag_names, expected_tag_keys):
            raise ValueError(f"movie {current_movie} does not have the same ordered tag matrix as earlier movies")
        if current_movie in seen_movies:
            raise ValueError(f"movie {current_movie} appears in non-contiguous duplicate rows")
        seen_movies.add(current_movie)
        movie_ids.append(current_movie)
        movie_texts.append(current_text)
        relevance_rows.append(np.asarray(current_scores, dtype=np.float32))

    with input_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not REQUIRED_COLUMNS <= set(reader.fieldnames):
            raise ValueError(f"{input_path} is missing required columns: {', '.join(sorted(REQUIRED_COLUMNS))}")
        for line_number, row in enumerate(reader, 2):
            movie_id = row["tconst"].strip()
            movie_text = row["movie_text"].strip()
            if row["split"].strip() != "train" or row["has_genome"].strip() != "1":
                raise ValueError(f"line {line_number} is not a Genome-labelled training movie")
            if not movie_id or not movie_text:
                raise ValueError(f"line {line_number} has an empty tconst or movie_text")
            try:
                tag_id = int(row["tagId"])
                score = float(row["relevance"])
            except ValueError as error:
                raise ValueError(f"line {line_number} has an invalid tagId or relevance") from error
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"line {line_number} relevance is outside [0, 1]")

            if current_movie is not None and movie_id != current_movie:
                finish_movie()
                current_tag_ids, current_tag_names, current_tag_keys, current_scores = [], [], [], []
            elif current_text is not None and movie_text != current_text:
                raise ValueError(f"line {line_number} changes movie_text within movie {movie_id}")
            current_movie, current_text = movie_id, movie_text
            current_tag_ids.append(tag_id)
            current_tag_names.append(row["tag"].strip())
            current_tag_keys.append(row["tag_key"].strip())
            current_scores.append(score)
            rows += 1
            low += score <= 0.1
            high += score >= 0.8
            minimum, maximum = min(minimum, score), max(maximum, score)
    finish_movie()

    if not relevance_rows or expected_tag_ids is None or len(movie_ids) < 2:
        raise ValueError("input contains too few train movies or no relevance rows")
    relevance = np.stack(relevance_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        movie_ids=np.asarray(movie_ids),
        movie_texts=np.asarray(movie_texts),
        tag_ids=np.asarray(expected_tag_ids, dtype=np.int32),
        tag_names=np.asarray(expected_tag_names),
        tag_keys=np.asarray(expected_tag_keys),
        relevance=relevance,
    )
    metadata: dict[str, int | float] = {
        "input_rows": rows,
        "movies": len(movie_ids),
        "tags": len(expected_tag_ids),
        "matrix_rows": int(relevance.shape[0]),
        "matrix_columns": int(relevance.shape[1]),
        "relevance_min": minimum,
        "relevance_max": maximum,
        "relevance_le_0_1": low,
        "relevance_ge_0_8": high,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact train-only Genome relevance rows into an NPZ matrix.")
    parser.add_argument("--input", type=Path, required=True, help="verbose all-score train-only relevance CSV")
    parser.add_argument("--output", type=Path, required=True, help="new .npz output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = compact_relevance_csv(args.input, args.output)
    print("Wrote compact train-only Genome relevance:")
    print(", ".join(f"{name}={value:,}" if isinstance(value, int) else f"{name}={value}" for name, value in metadata.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
