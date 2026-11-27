#!/usr/bin/env python
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "accelerate==1.15.0",
#     "datasets==5.2.0",
#     "numpy",
#     "sentence-transformers==6.2.0",
#     "torch",
#     "transformers==5.22.2",
# ]
# ///
"""Fine-tune the movie-search MiniLM model locally, without Google Colab.

The default configuration is the title-aware "second model" from
``notebooks/finetune_search_titles.ipynb``.  It uses MovieLens genome tags,
community tags, TMDB keywords and movie-title pairs.  It writes the trained
SentenceTransformer and a reproducible ``results.json`` to ``--output``.

Examples (PowerShell):

    # CPU-only quick start. Uses moviebot-drive/moviebot automatically when present.
    uv run --script scripts/train_search_local.py

    # Set up an NVIDIA CUDA environment, then verify it before training.
    uv venv .venv-train
    uv pip install --python .venv-train/Scripts/python.exe --torch-backend=auto torch \
      sentence-transformers==6.2.0 transformers==5.22.2 datasets==5.2.0 accelerate==1.15.0 numpy
    .venv-train/Scripts/python.exe scripts/train_search_local.py --check-device --device cuda

    # Train the first (tag-only) model, then use it as a comparison model.
    .venv-train/Scripts/python.exe scripts/train_search_local.py --no-title-pairs \
      --output data/minilm-movie-search --device cuda --batch-size 16
    .venv-train/Scripts/python.exe scripts/train_search_local.py \
      --previous-model data/minilm-movie-search --device cuda --batch-size 16

    # Mine hard negatives from the existing First model, then fine-tune it for one extra epoch.
    .venv-train/Scripts/python.exe scripts/train_search_local.py --hard-negatives \
      --hard-negative-base-model data/minilm-movie-search \
      --no-title-pairs --output data/minilm-movie-search-hard-negatives \
      --device cuda --batch-size 16

``uv run --script`` cannot choose a GPU PyTorch wheel automatically.  Use the
``uv pip --torch-backend=auto`` setup above: it detects the installed NVIDIA
driver and installs the matching CUDA wheel.  ``--device auto`` selects CUDA
when that PyTorch environment can see it, otherwise CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEED = 42
RRF_K = 60

# These are intentionally few, concrete search phrasings rather than generated
# paraphrases.  They are applied only to the train side of a proxy split and
# written to an audit CSV, so an experiment remains reproducible and reviewable.
QUERY_AUGMENTATION_ALIASES: dict[str, tuple[str, ...]] = {
    "sci-fi": ("science fiction",),
    "sci fi": ("science fiction",),
    "time travel": ("time-travel", "travelling through time"),
    "feel-good": ("feel good", "uplifting"),
    "coming-of-age": ("coming of age",),
    "coming of age": ("coming-of-age",),
    "found footage": ("found-footage",),
    "post-apocalyptic": ("post apocalyptic",),
    "superhero": ("super hero",),
    "zombies": ("zombie movie", "zombie films"),
}
VALID_TRAIN_SOURCES = frozenset({"genome", "tmdb_keywords", "user_tag", "title"})


def tag_key(tag: str) -> str:
    """Normalize only simple spelling variants, preserving meaningful synonyms."""
    return re.sub(r"[^a-z0-9]", "", tag.lower()).rstrip("s")


def movie_text(title: str, year: int, genres: str | None, overview: str) -> str:
    return ". ".join(part for part in (f"{title} ({year})", ", ".join((genres or "").split(",")), overview) if part)


def default_data_dir() -> Path:
    """Prefer regular project data, then the local Drive export included in this checkout."""
    for candidate in (REPO / "data", REPO / "moviebot-drive" / "moviebot"):
        if (candidate / "moviebot.db").exists():
            return candidate
    return REPO / "data"


def require_files(data_dir: Path, use_movielens_tags: bool) -> None:
    names = ["moviebot.db"] + (["tags.csv", "links.csv"] if use_movielens_tags else [])
    missing = [str(data_dir / name) for name in names if not (data_dir / name).is_file()]
    if missing:
        raise SystemExit("Missing input file(s):\n  " + "\n  ".join(missing))


def load_data(data_dir: Path, *, seed: int, test_share: float) -> tuple[list[str], dict[str, str], dict[str, str], dict[str, np.ndarray], set[str], set[str]]:
    """Load trainable movie text and make the fixed movie/tag hold-out split."""
    db_uri = f"{(data_dir / 'moviebot.db').resolve().as_uri()}?mode=ro"
    with sqlite3.connect(db_uri, uri=True) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(movies)")}
        if "keywords" not in columns:
            raise SystemExit("moviebot.db has no keywords column; run: python -m moviebot.import_tmdb")
        tag_names = [tag for (tag,) in db.execute("SELECT tag FROM genome_tags ORDER BY tag_id")]
        genome = {movie: np.frombuffer(blob, dtype="f4") for movie, blob in db.execute("SELECT movie_id, relevance FROM genome")}
        rows = db.execute(
            "SELECT movie_id, title, year, genres, overview FROM movies WHERE coalesce(overview, '') <> ''"
        )
        titles: dict[str, str] = {}
        texts: dict[str, str] = {}
        for movie, title, year, genres, overview in rows:
            titles[movie] = title
            texts[movie] = movie_text(title, year, genres, overview)

    rng = random.Random(seed)
    graded = sorted(movie for movie in genome if movie in texts)
    rng.shuffle(graded)
    test_movies = set(graded[: int(len(graded) * test_share)])
    train_movies = set(texts) - test_movies
    keys = sorted({tag_key(tag) for tag in tag_names})
    test_keys = set(rng.sample(keys, int(len(keys) * test_share)))
    return tag_names, titles, texts, genome, train_movies, test_keys


def validation_split(
    tag_names: list[str],
    outer_train_movies: set[str],
    outer_test_keys: set[str],
    *,
    seed: int,
    share: float,
) -> tuple[set[str], set[str]]:
    """Create a deterministic inner validation split without touching the frozen outer test.

    The returned movie and tag-key sets are removed from the inner-training
    pool. A separate seed prevents the validation sample from being correlated
    with the outer train/test shuffle.
    """
    if not 0 < share < 1:
        raise ValueError("validation share must be between 0 and 1")
    rng = random.Random(seed + 1)
    movies = sorted(outer_train_movies)
    keys = sorted({tag_key(tag) for tag in tag_names} - outer_test_keys)
    validation_movies = set(rng.sample(movies, int(len(movies) * share)))
    validation_keys = set(rng.sample(keys, int(len(keys) * share)))
    return validation_movies, validation_keys


def training_pairs(
    data_dir: Path,
    tag_names: list[str],
    titles: dict[str, str],
    texts: dict[str, str],
    genome: dict[str, np.ndarray],
    train_movies: set[str],
    test_keys: set[str],
    *,
    use_movielens_tags: bool,
    include_titles: bool,
    relevant: float,
    min_users: int,
    max_tag_chars: int,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Build (query phrase, movie id) pairs while keeping both hold-outs isolated."""
    sources: dict[str, set[tuple[str, str]]] = {}
    if use_movielens_tags:
        sources["genome"] = {
            (tag_names[index], movie)
            for movie, scores in genome.items()
            if movie in train_movies
            for index in np.flatnonzero(scores >= relevant)
            if tag_key(tag_names[index]) not in test_keys
        }
        with (data_dir / "links.csv").open(encoding="utf-8", newline="") as file:
            imdb_id = {row["movieId"]: "tt" + row["imdbId"] for row in csv.DictReader(file)}
        taggers: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        with (data_dir / "tags.csv").open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                tag = row["tag"].lower().strip()
                movie = imdb_id.get(row["movieId"])
                if movie in train_movies and 0 < len(tag) <= max_tag_chars and tag_key(tag) not in test_keys:
                    taggers[tag, movie].add(row["userId"])
        sources["user tags"] = {pair for pair, users in taggers.items() if len(users) >= min_users}

    db_uri = f"{(data_dir / 'moviebot.db').resolve().as_uri()}?mode=ro"
    with sqlite3.connect(db_uri, uri=True) as db:
        sources["TMDB keywords"] = {
            (keyword, movie)
            for movie, keywords in db.execute("SELECT movie_id, keywords FROM movies WHERE coalesce(keywords, '') <> ''")
            if movie in train_movies
            for keyword in map(str.strip, keywords.split("|"))
            if keyword and tag_key(keyword) not in test_keys
        }
    if include_titles:
        sources["titles"] = {(titles[movie], movie) for movie in train_movies if tag_key(titles[movie]) not in test_keys}

    pairs = sorted(set().union(*sources.values()))
    assert not {movie for _, movie in pairs} & (set(texts) - train_movies), "test movie leaked into training"
    assert not {tag_key(query) for query, _ in pairs} & test_keys, "held-out tag leaked into training"
    return pairs, {name: len(values) for name, values in sources.items()}


def balance_query_pairs(
    pairs: list[tuple[str, str]],
    *,
    min_pairs_per_query: int,
    max_pairs_per_query: int,
    seed: int,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Cap frequent queries and repeat rare queries for macro-style tag retrieval.

    The metric gives every query/tag equal weight, while the raw source pool
    contains a long tail of one-pair queries and a few very frequent queries.
    Sampling is deterministic so every experiment has an auditable data mix.
    """
    if min_pairs_per_query < 1 or max_pairs_per_query < min_pairs_per_query:
        raise ValueError("require 1 <= min_pairs_per_query <= max_pairs_per_query")
    grouped: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair[0]].append(pair)
    rng = random.Random(seed)
    balanced: list[tuple[str, str]] = []
    capped = oversampled = 0
    for query in sorted(grouped):
        group = sorted(grouped[query])
        if len(group) > max_pairs_per_query:
            group = rng.sample(group, max_pairs_per_query)
            capped += 1
        elif len(group) < min_pairs_per_query:
            group = [*group, *(rng.choice(group) for _ in range(min_pairs_per_query - len(group)))]
            oversampled += 1
        balanced.extend(group)
    rng.shuffle(balanced)
    return balanced, {
        "input_pairs": len(pairs),
        "output_pairs": len(balanced),
        "unique_queries": len(grouped),
        "capped_queries": capped,
        "oversampled_queries": oversampled,
    }


def query_variants(anchor: str) -> list[tuple[str, str]]:
    """Return small, auditable spelling and semantic variants for one query.

    This is deliberately conservative: it does not call an LLM or use data
    outside the supplied training CSV.  The caller is responsible for rejecting
    variants whose normalized key belongs to a proxy-validation holdout.
    """
    variants: list[tuple[str, str]] = []
    simplified = re.sub(r"\s*[-/]\s*", " ", anchor)
    simplified = re.sub(r"\s*&\s*", " and ", simplified)
    simplified = re.sub(r"(?<=\w)[’'](?=\w)", "", simplified)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    if simplified and simplified.casefold() != anchor.casefold():
        variants.append((simplified, "punctuation_normalization"))

    for alias in QUERY_AUGMENTATION_ALIASES.get(anchor.casefold().strip(), ()):
        variants.append((alias, "curated_alias"))

    # Preserve first occurrence/order while rejecting case-only duplicates.
    unique: list[tuple[str, str]] = []
    seen = {anchor.casefold()}
    for variant, strategy in variants:
        variant = re.sub(r"\s+", " ", variant).strip()
        if variant and variant.casefold() not in seen:
            seen.add(variant.casefold())
            unique.append((variant, strategy))
    return unique


def augment_query_pairs(
    pairs: list[tuple[str, str]],
    *,
    forbidden_keys: set[str],
    max_variants_per_query: int,
) -> tuple[list[tuple[str, str]], dict[str, int], list[tuple[str, str, str]]]:
    """Add train-only query variants without crossing a held-out anchor key."""
    if not 1 <= max_variants_per_query <= 2:
        raise ValueError("max_variants_per_query must be 1 or 2")
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for anchor, positive in pairs:
        grouped[anchor].append(positive)

    augmented = list(pairs)
    pair_set = set(pairs)
    audit: list[tuple[str, str, str]] = []
    skipped_heldout = skipped_duplicates = 0
    for anchor in sorted(grouped):
        accepted = 0
        for variant, strategy in query_variants(anchor):
            if tag_key(variant) in forbidden_keys:
                skipped_heldout += 1
                continue
            if accepted >= max_variants_per_query:
                break
            added_for_variant = False
            for positive in grouped[anchor]:
                pair = (variant, positive)
                if pair in pair_set:
                    skipped_duplicates += 1
                    continue
                pair_set.add(pair)
                augmented.append(pair)
                added_for_variant = True
            if added_for_variant:
                audit.append((anchor, variant, strategy))
                accepted += 1

    return augmented, {
        "input_pairs": len(pairs),
        "output_pairs": len(augmented),
        "source_queries": len(grouped),
        "generated_anchor_variants": len(audit),
        "augmented_pairs": len(augmented) - len(pairs),
        "skipped_heldout_variants": skipped_heldout,
        "skipped_duplicate_pairs": skipped_duplicates,
    }, audit


def load_export_pairs(path: Path) -> list[tuple[str, str]]:
    """Load the standalone handoff CSV without reading any evaluation database."""
    if not path.is_file():
        raise FileNotFoundError(f"training CSV not found: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["anchor", "positive"]:
            raise ValueError(f"{path} must have exactly these columns: anchor,positive")
        pairs = [(row["anchor"].strip(), row["positive"].strip()) for row in reader]
    if not pairs or any(not anchor or not positive for anchor, positive in pairs):
        raise ValueError(f"{path} contains no pairs or an empty anchor/positive")
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{path} contains duplicate (anchor, positive) pairs")
    return pairs


def parse_source_names(value: str | None) -> frozenset[str]:
    """Parse the comma-separated source selector used for source ablation."""
    if not value:
        return frozenset()
    names = frozenset(name.strip().lower() for name in value.split(",") if name.strip())
    unknown = names - VALID_TRAIN_SOURCES
    if unknown:
        raise ValueError(f"unknown source(s): {', '.join(sorted(unknown))}; allowed: {', '.join(sorted(VALID_TRAIN_SOURCES))}")
    return names


def load_export_source_pairs(path: Path) -> list[tuple[str, str, frozenset[str]]]:
    """Load the provenance-aware train CSV without opening a database."""
    if not path.is_file():
        raise FileNotFoundError(f"source training CSV not found: {path}")
    rows: list[tuple[str, str, frozenset[str]]] = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["anchor", "positive", "source"]:
            raise ValueError(f"{path} must have exactly these columns: anchor,positive,source")
        for row in reader:
            anchor, positive, raw_sources = row["anchor"].strip(), row["positive"].strip(), row["source"].strip()
            sources = frozenset(part.strip().lower() for part in raw_sources.split(";") if part.strip())
            if not anchor or not positive or not sources:
                raise ValueError(f"{path} contains an empty anchor, positive or source")
            unknown = sources - VALID_TRAIN_SOURCES
            if unknown:
                raise ValueError(f"{path} has unknown source(s): {', '.join(sorted(unknown))}")
            rows.append((anchor, positive, sources))
    pairs = [(anchor, positive) for anchor, positive, _ in rows]
    if not rows or len(set(pairs)) != len(pairs):
        raise ValueError(f"{path} contains no pairs or duplicate (anchor, positive) pairs")
    return rows


def select_source_pairs(
    rows: list[tuple[str, str, frozenset[str]]],
    *,
    include_sources: frozenset[str],
    exclude_sources: frozenset[str],
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Select source-only or leave-one-source-out pairs without double counting.

    A pair attributed to both Genome and TMDB remains in a leave-Genome-out
    run, because the TMDB label still independently supports that pair.
    """
    if include_sources & exclude_sources:
        overlap = ", ".join(sorted(include_sources & exclude_sources))
        raise ValueError(f"a source cannot be both included and excluded: {overlap}")
    selected: list[tuple[str, str]] = []
    source_counts: defaultdict[str, int] = defaultdict(int)
    for anchor, positive, sources in rows:
        remaining = sources - exclude_sources
        if remaining and (not include_sources or remaining & include_sources):
            selected.append((anchor, positive))
            for source in remaining:
                source_counts[source] += 1
    if not selected:
        raise ValueError("source filter selected zero training pairs")
    return selected, {
        "input_pairs": len(rows),
        "output_pairs": len(selected),
        **{f"pairs_with_{source}": source_counts[source] for source in sorted(VALID_TRAIN_SOURCES)},
    }


def load_compact_train_genome(path: Path) -> tuple[list[str], dict[str, str], dict[str, np.ndarray]]:
    """Load the prevalidated train-only Genome matrix for hard-negative mining."""
    if not path.is_file():
        raise FileNotFoundError(f"compact Genome relevance artifact not found: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        required = {"movie_ids", "movie_texts", "tag_names", "relevance"}
        if not required <= set(artifact.files):
            raise ValueError(f"{path} is missing required arrays: {', '.join(sorted(required))}")
        movie_ids = artifact["movie_ids"].astype(str).tolist()
        movie_texts = artifact["movie_texts"].astype(str).tolist()
        tag_names = artifact["tag_names"].astype(str).tolist()
        relevance = artifact["relevance"].astype(np.float32)
    if relevance.shape != (len(movie_ids), len(tag_names)) or len(movie_texts) != len(movie_ids) or not movie_ids or not tag_names:
        raise ValueError("compact Genome artifact has inconsistent matrix dimensions")
    if len(set(movie_ids)) != len(movie_ids) or len(set(tag_names)) != len(tag_names):
        raise ValueError("compact Genome artifact has duplicate movie IDs or tag names")
    if any(not text for text in movie_texts) or not np.isfinite(relevance).all() or not ((0 <= relevance) & (relevance <= 1)).all():
        raise ValueError("compact Genome artifact has empty text or invalid relevance values")
    return tag_names, dict(zip(movie_ids, movie_texts)), dict(zip(movie_ids, relevance, strict=True))


def export_proxy_split(
    pairs: list[tuple[str, str]], *, seed: int, share: float
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], set[str]]:
    """Hold out complete normalized anchor groups for train-only model selection."""
    if not 0 < share < 1:
        raise ValueError("proxy validation share must be between 0 and 1")
    keys = sorted({tag_key(anchor) for anchor, _ in pairs})
    rng = random.Random(seed + 2)
    held_keys = set(rng.sample(keys, int(len(keys) * share)))
    train_pairs = [pair for pair in pairs if tag_key(pair[0]) not in held_keys]
    validation_pairs = [pair for pair in pairs if tag_key(pair[0]) in held_keys]
    if not train_pairs or not validation_pairs:
        raise ValueError("proxy split produced an empty training or validation set")
    return train_pairs, validation_pairs, held_keys


def evaluate_export_proxy(model, validation_pairs: list[tuple[str, str]], all_pairs: list[tuple[str, str]], *, batch_size: int) -> dict[str, float | int]:
    """Evaluate unseen anchors using only relations contained in the handoff CSV.

    This is a model-selection proxy, not the project's final Genome evaluation.
    Candidate documents are all unique positive texts from the CSV; every
    validation anchor's known positive texts form its relevance set.
    """
    documents = sorted({positive for _, positive in all_pairs})
    document_index = {document: index for index, document in enumerate(documents)}
    relevant: defaultdict[str, set[int]] = defaultdict(set)
    for anchor, positive in validation_pairs:
        relevant[anchor].add(document_index[positive])
    queries = sorted(relevant)
    document_vectors = embed(model, documents, batch_size)
    query_vectors = embed(model, queries, batch_size)
    precision = recall = reciprocal_rank = 0.0
    top_k = min(10, len(documents))
    for start in range(0, len(queries), batch_size):
        scores = query_vectors[start:start + batch_size] @ document_vectors.T
        indices = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
        ordered = np.take_along_axis(indices, np.argsort(-np.take_along_axis(scores, indices, axis=1), axis=1), axis=1)
        for query, ranked in zip(queries[start:start + batch_size], ordered):
            truth = relevant[query]
            hits = [rank for rank, document_id in enumerate(ranked, 1) if document_id in truth]
            precision += len(hits) / top_k
            recall += len(hits) / len(truth)
            reciprocal_rank += 1 / hits[0] if hits else 0.0
    total = len(queries)
    return {
        "queries": total,
        "documents": len(documents),
        "proxy_precision_at_10": round(precision / total, 4),
        "proxy_recall_at_10": round(recall / total, 4),
        "proxy_mrr_at_10": round(reciprocal_rank / total, 4),
    }


def train_from_export_csv(args, *, device: str, Dataset, SentenceTransformer, SentenceTransformerTrainer, MultipleNegativesRankingLoss) -> int:
    """Train only from a clean handoff CSV; evaluation data is deliberately unavailable here."""
    source_selection: dict[str, object] | None = None
    source_rows: list[tuple[str, str, frozenset[str]]] = []
    genome_source_keys: set[str] = set()
    if args.train_with_source_csv:
        source_rows = load_export_source_pairs(args.train_with_source_csv)
        genome_source_keys = {
            tag_key(anchor) for anchor, _, sources in source_rows if "genome" in sources
        }
        include_sources = parse_source_names(args.include_sources)
        exclude_sources = parse_source_names(args.exclude_sources)
        selected_pairs, source_summary = select_source_pairs(
            source_rows,
            include_sources=include_sources,
            exclude_sources=exclude_sources,
        )
        # Keep the proxy target and document pool fixed across source ablations.
        # Only the training pairs change from one ablation to another.
        all_pairs = [(anchor, positive) for anchor, positive, _ in source_rows]
        source_selection = {
            "include_sources": sorted(include_sources),
            "exclude_sources": sorted(exclude_sources),
            "summary": source_summary,
        }
        print(
            "source selection: "
            + ", ".join(f"{value:,} {name.replace('_', ' ')}" for name, value in source_summary.items())
        )
    else:
        selected_pairs = load_export_pairs(args.train_csv)
        all_pairs = selected_pairs
    validation_pairs: list[tuple[str, str]] = []
    held_anchor_keys: set[str] = set()
    if args.proxy_validation_share:
        _, validation_pairs, held_anchor_keys = export_proxy_split(
            all_pairs, seed=args.seed, share=args.proxy_validation_share
        )
        pairs = [pair for pair in selected_pairs if tag_key(pair[0]) not in held_anchor_keys]
        if not pairs:
            raise ValueError("source selection contains no pairs outside the proxy-validation holdout")
        print(
            f"proxy validation: {len(validation_pairs):,} pairs across {len(held_anchor_keys):,} held-out anchor keys; "
            f"{len(pairs):,} train pairs before balancing"
        )
    else:
        pairs = selected_pairs
    augmentation_summary: dict[str, int] | None = None
    augmentation_audit: list[tuple[str, str, str]] = []
    if args.query_augmentation:
        pairs, augmentation_summary, augmentation_audit = augment_query_pairs(
            pairs,
            forbidden_keys=held_anchor_keys,
            max_variants_per_query=args.max_augmented_variants_per_query,
        )
        print(
            "query augmentation: "
            + ", ".join(f"{value:,} {name.replace('_', ' ')}" for name, value in augmentation_summary.items())
        )
    balance_summary: dict[str, int] | None = None
    if args.balanced_query_sampling:
        pairs, balance_summary = balance_query_pairs(
            pairs,
            min_pairs_per_query=args.min_pairs_per_query,
            max_pairs_per_query=args.max_pairs_per_query,
            seed=args.seed,
        )
        print(
            "balanced query sampling: "
            + ", ".join(f"{value:,} {name.replace('_', ' ')}" for name, value in balance_summary.items())
        )
    else:
        random.Random(args.seed).shuffle(pairs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    args.output.mkdir(parents=True, exist_ok=True)
    if augmentation_audit:
        with (args.output / "query_augmentation.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("generated_anchor", "original_anchor", "strategy"))
            writer.writerows((variant, anchor, strategy) for anchor, variant, strategy in augmentation_audit)
    train_source = args.train_with_source_csv or args.train_csv
    print(f"Using {device}; training only from export CSV: {train_source}; {len(pairs):,} pairs")
    hard_negative_summary: dict[str, int | float | str] | None = None
    if args.hard_negative_base_model:
        if not (args.hard_negative_base_model / "model.safetensors").exists():
            raise SystemExit(f"--hard-negative-base-model is not a saved SentenceTransformer: {args.hard_negative_base_model}")
        print(f"Loading base model for hard-negative fine-tuning: {args.hard_negative_base_model}")
        model = SentenceTransformer(str(args.hard_negative_base_model), device=device)
    else:
        model = SentenceTransformer(BASE_MODEL, device=device)
        train_data = Dataset.from_dict({"anchor": [anchor for anchor, _ in pairs], "positive": [positive for _, positive in pairs]})
        SentenceTransformerTrainer(
            model=model,
            args=training_arguments(
                args.output / "checkpoints-base",
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=device,
                seed=args.training_seed if args.training_seed is not None else args.seed,
            ),
            train_dataset=train_data,
            loss=MultipleNegativesRankingLoss(model),
        ).train()

    if args.hard_negatives:
        tag_names, genome_texts, genome = load_compact_train_genome(args.hard_negative_relevance)
        train_movies = set(genome)
        allowed_keys = genome_source_keys - held_anchor_keys
        if not allowed_keys:
            raise SystemExit("no train-only Genome anchor keys remain for hard-negative mining")
        examples = mine_hard_negatives(
            model,
            tag_names,
            genome_texts,
            genome,
            train_movies,
            held_anchor_keys,
            positive_threshold=0.8,
            negative_threshold=args.hard_negative_threshold,
            top_k=args.hard_negative_top_k,
            negatives_per_positive=args.hard_negatives_per_positive,
            batch_size=args.batch_size,
            allowed_tag_keys=allowed_keys,
            max_candidate_rank=args.hard_negative_max_rank,
        )
        if not examples:
            raise SystemExit("No hard negatives were mined. Raise --hard-negative-top-k or use a less strict --hard-negative-threshold.")
        assert not ({tag_key(example.tag) for example in examples} & held_anchor_keys)
        assert {tag_key(example.tag) for example in examples} <= allowed_keys
        assert all(example.negative_relevance <= args.hard_negative_threshold for example in examples)
        artifact = args.output / "hard_negatives.csv"
        write_hard_negative_artifact(artifact, examples)
        print(f"Mined {len(examples):,} train-only hard-negative triples; audit artifact: {artifact}")
        hard_data = Dataset.from_dict(
            {
                "anchor": [example.tag for example in examples],
                "positive": [genome_texts[example.positive_movie_id] for example in examples],
                "negative": [genome_texts[example.negative_movie_id] for example in examples],
            }
        )
        if args.hard_negative_mix_original:
            original_data = Dataset.from_dict(
                {"anchor": [anchor for anchor, _ in pairs], "positive": [positive for _, positive in pairs]}
            )
            hard_train_data: object = {"original": original_data, "hard_negative": hard_data}
            hard_loss: object = {
                "original": MultipleNegativesRankingLoss(model),
                "hard_negative": MultipleNegativesRankingLoss(model),
            }
        else:
            hard_train_data = hard_data
            hard_loss = MultipleNegativesRankingLoss(model)
        SentenceTransformerTrainer(
            model=model,
            args=training_arguments(
                args.output / "checkpoints-hard-negatives",
                epochs=args.hard_negative_epochs,
                batch_size=args.batch_size,
                learning_rate=args.hard_negative_learning_rate,
                device=device,
                seed=args.training_seed if args.training_seed is not None else args.seed,
            ),
            train_dataset=hard_train_data,
            loss=hard_loss,
        ).train()
        hard_negative_summary = {
            "base_model": str(args.hard_negative_base_model) if args.hard_negative_base_model else BASE_MODEL,
            "relevance_artifact": str(args.hard_negative_relevance),
            "triples": len(examples),
            "top_k": args.hard_negative_top_k,
            "max_candidate_rank": args.hard_negative_max_rank,
            "negative_threshold": args.hard_negative_threshold,
            "negatives_per_positive": args.hard_negatives_per_positive,
            "epochs": args.hard_negative_epochs,
            "learning_rate": args.hard_negative_learning_rate,
            "mixed_original_pairs": args.hard_negative_mix_original,
            "original_pairs": len(pairs) if args.hard_negative_mix_original else 0,
            "artifact": artifact.name,
        }
    model.save_pretrained(str(args.output))
    proxy_metrics: dict[str, float | int] | None = None
    if validation_pairs:
        proxy_metrics = evaluate_export_proxy(model, validation_pairs, all_pairs, batch_size=args.batch_size)
        print(
            "proxy validation: "
            + ", ".join(f"{name}={value}" for name, value in proxy_metrics.items())
        )
    metadata = {
        "base_model": BASE_MODEL,
        "device": device,
        "train_source": str(train_source),
        "train_pairs": len(pairs),
        "config": {
            "seed": args.seed,
            "training_seed": args.training_seed if args.training_seed is not None else args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "source_selection": source_selection,
            "query_augmentation": augmentation_summary,
            "balanced_query_sampling": balance_summary,
            "proxy_validation_share": args.proxy_validation_share,
            "proxy_held_anchor_keys": len(held_anchor_keys),
            "hard_negatives": hard_negative_summary,
        },
        "proxy_validation": proxy_metrics,
        "evaluation": "No private test data was read. Run the provided evaluation notebook only after selecting a final model.",
    }
    (args.output / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {args.output}. No evaluation data was read.")
    return 0


@dataclass(frozen=True)
class HardNegativeExample:
    """One explicit negative selected from the baseline model's difficult retrievals."""

    tag: str
    tag_id: int
    positive_movie_id: str
    negative_movie_id: str
    positive_relevance: float
    negative_relevance: float
    retrieval_rank: int


def mine_hard_negatives(
    model,
    tag_names: list[str],
    texts: dict[str, str],
    genome: dict[str, np.ndarray],
    train_movies: set[str],
    held_out_tag_keys: set[str],
    *,
    positive_threshold: float,
    negative_threshold: float,
    top_k: int,
    negatives_per_positive: int,
    batch_size: int,
    allowed_tag_keys: set[str] | None = None,
    max_candidate_rank: int | None = None,
) -> list[HardNegativeExample]:
    """Mine difficult train-only negatives without ever querying test movies or held-out tags.

    For each eligible Genome tag, retrieve its nearest train-only movie documents
    with the current model. A retrieved movie becomes a hard negative only when
    its Genome relevance is at or below ``negative_threshold``. Positives are
    assigned one or more high-ranked negatives in a round-robin pattern.
    """
    if not train_movies:
        raise ValueError("cannot mine hard negatives with no training movies")
    if top_k < 1 or negatives_per_positive < 1:
        raise ValueError("top_k and negatives_per_positive must be positive")
    if negative_threshold >= positive_threshold:
        raise ValueError("negative_threshold must be lower than positive_threshold")
    if max_candidate_rank is not None and not 1 <= max_candidate_rank <= top_k:
        raise ValueError("max_candidate_rank must be from 1 through top_k")

    # TMDB can add trainable overview text for movies outside MovieLens Genome.
    # They remain useful in the base pair training, but cannot be positives or
    # negatives here because no relevance score exists to verify their label.
    movie_ids = sorted(train_movies & genome.keys())
    if not movie_ids:
        raise ValueError("none of the training movies has a Genome relevance vector")
    documents = [texts[movie_id] for movie_id in movie_ids]
    document_vectors = embed(model, documents, batch_size)
    eligible_tags = [
        index
        for index, tag in enumerate(tag_names)
        if tag_key(tag) not in held_out_tag_keys
        and (allowed_tag_keys is None or tag_key(tag) in allowed_tag_keys)
    ]
    query_vectors = embed(model, (tag_names[index] for index in eligible_tags), batch_size)
    examples: list[HardNegativeExample] = []
    count = min(top_k, len(movie_ids))

    for tag_index, query_vector in zip(eligible_tags, query_vectors):
        relevance = {movie_id: float(genome[movie_id][tag_index]) for movie_id in movie_ids}
        positives = sorted(movie_id for movie_id, score in relevance.items() if score >= positive_threshold)
        if not positives:
            continue
        scores = document_vectors @ query_vector
        # argpartition avoids sorting every catalogue movie before sorting only the top-K candidates.
        top_indices = np.argpartition(-scores, count - 1)[:count]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        candidates = [
            (movie_ids[index], relevance[movie_ids[index]], rank)
            for rank, index in enumerate(top_indices, 1)
            if relevance[movie_ids[index]] <= negative_threshold
            and (max_candidate_rank is None or rank <= max_candidate_rank)
        ]
        if not candidates:
            continue
        for positive_offset, positive_movie_id in enumerate(positives):
            for negative_offset in range(negatives_per_positive):
                negative_movie_id, negative_relevance, rank = candidates[
                    (positive_offset * negatives_per_positive + negative_offset) % len(candidates)
                ]
                examples.append(
                    HardNegativeExample(
                        tag=tag_names[tag_index],
                        tag_id=tag_index + 1,
                        positive_movie_id=positive_movie_id,
                        negative_movie_id=negative_movie_id,
                        positive_relevance=relevance[positive_movie_id],
                        negative_relevance=negative_relevance,
                        retrieval_rank=rank,
                    )
                )
    return examples


def write_hard_negative_artifact(path: Path, examples: list[HardNegativeExample]) -> None:
    """Persist mined IDs and scores for an auditable training-data record."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(HardNegativeExample.__annotations__))
        writer.writeheader()
        writer.writerows(example.__dict__ for example in examples)


def embed(model, sentences: Iterable[str], batch_size: int) -> np.ndarray:
    return model.encode(list(sentences), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)


def semantic_ranks(model, documents: list[str], queries: list[str], batch_size: int, limit: int | None = None) -> np.ndarray:
    document_vectors = embed(model, documents, batch_size)
    scores = embed(model, queries, batch_size) @ document_vectors.T
    ranked = np.argsort(-scores, axis=1)
    return ranked if limit is None else ranked[:, :limit]


def bm25_ranks(documents: list[str], queries: list[str]) -> list[np.ndarray]:
    """Rank only lexical matches; no match gets no BM25 contribution in RRF."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE docs USING fts5(doc, tokenize='porter')")
    db.executemany("INSERT INTO docs(rowid, doc) VALUES (?, ?)", enumerate(documents, 1))
    ranks = []
    for query in queries:
        match = " OR ".join(f'"{word}"' for word in re.findall(r"\w+", query.lower()))
        rows = db.execute("SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs)", (match,)) if match else []
        ranks.append(np.array([row - 1 for (row,) in rows], dtype=int))
    db.close()
    return ranks


def weighted_rrf(vector_ranks: np.ndarray, lexical_ranks: list[np.ndarray], vector_weight: float, limit: int) -> np.ndarray:
    """Weighted reciprocal-rank fusion, identical to the live search site's scoring rule."""
    fused = np.empty((len(vector_ranks), limit), dtype=int)
    lexical_weight = 1.0 - vector_weight
    for row, (vector, lexical) in enumerate(zip(vector_ranks, lexical_ranks)):
        scores = np.zeros(len(vector), dtype=np.float64)
        scores[vector] += vector_weight / (RRF_K + np.arange(1, len(vector) + 1))
        if len(lexical):
            scores[lexical] += lexical_weight / (RRF_K + np.arange(1, len(lexical) + 1))
        fused[row] = np.argsort(-scores)[:limit]
    return fused


def report_metrics(
    name: str,
    tag_ranks: np.ndarray,
    title_ranks: np.ndarray,
    truth: np.ndarray,
    seen_tags: list[int],
    unseen_tags: list[int],
    all_titles: list[str],
    title_queries: list[str],
) -> dict[str, float | str]:
    def p10(start: int, tag_indices: list[int]) -> float:
        return float(
            np.mean([truth[tag_ranks[start + offset, :10], tag].mean() for offset, tag in enumerate(tag_indices)])
        )

    title_positions = [
        next((rank for rank, movie_index in enumerate(row, 1) if all_titles[movie_index].casefold() == query.casefold()), None)
        for query, row in zip(title_queries, title_ranks)
    ]
    metrics: dict[str, float | str] = {
        "setup": name,
        "seen_tags_p10": round(p10(0, seen_tags), 3),
        "unseen_tags_p10": round(p10(len(seen_tags), unseen_tags), 3),
        "title_hit1": round(sum(rank == 1 for rank in title_positions) / len(title_positions), 3),
        "title_hit10": round(sum(rank is not None and rank <= 10 for rank in title_positions) / len(title_positions), 3),
    }
    print(
        f"{name:34} {metrics['seen_tags_p10']:.3f}  {metrics['unseen_tags_p10']:.3f}  "
        f"{metrics['title_hit1']:.3f}  {metrics['title_hit10']:.3f}"
    )
    return metrics


def evaluate_split(
    models: list[tuple[str, object]],
    tag_names: list[str],
    titles: dict[str, str],
    texts: dict[str, str],
    genome: dict[str, np.ndarray],
    test_movies: set[str],
    test_keys: set[str],
    *,
    relevant: float,
    batch_size: int,
    device: str,
    excluded_tag_keys: set[str] = frozenset(),
) -> list[dict[str, float | str]]:
    """Evaluate model(s) and 80/20, 70/30 and 50/50 weighted-RRF variants."""
    movies = sorted(test_movies)
    truth = np.stack([genome[movie] for movie in movies]) >= relevant
    usable = [
        index
        for index, count in enumerate(truth.sum(axis=0))
        if 5 <= count <= 100 and tag_key(tag_names[index]) not in excluded_tag_keys
    ]
    seen = [index for index in usable if tag_key(tag_names[index]) not in test_keys]
    unseen = [index for index in usable if tag_key(tag_names[index]) in test_keys]
    queries = [tag_names[index] for index in [*seen, *unseen]]
    test_documents = [texts[movie] for movie in movies]

    all_movies = sorted(texts)
    all_titles = [titles[movie] for movie in all_movies]
    all_documents = [texts[movie] for movie in all_movies]
    title_queries = [titles[movie] for movie in movies]
    tag_bm25 = bm25_ranks(test_documents, queries)
    title_bm25 = bm25_ranks(all_documents, title_queries)

    print("\nSetup                              Seen P@10  Unseen P@10  Title hit@1  Title hit@10")
    results: list[dict[str, float | str]] = []
    for name, model in models:
        tag_vectors = semantic_ranks(model, test_documents, queries, batch_size)
        title_vectors = semantic_ranks(model, all_documents, title_queries, batch_size)
        results.append(report_metrics(name + " alone", tag_vectors, title_vectors, truth, seen, unseen, all_titles, title_queries))
        for weight in (0.8, 0.7, 0.5):
            tag_hybrid = weighted_rrf(tag_vectors, tag_bm25, weight, limit=100)
            title_hybrid = weighted_rrf(title_vectors, title_bm25, weight, limit=10)
            results.append(
                report_metrics(
                    f"{name} + BM25 {round(weight * 100)}/{round((1 - weight) * 100)}",
                    tag_hybrid,
                    title_hybrid,
                    truth,
                    seen,
                    unseen,
                    all_titles,
                    title_queries,
                )
            )
    # This is the base all-MiniLM-L6-v2 reference row shown in the comparison table.
    from sentence_transformers import SentenceTransformer

    stock = SentenceTransformer(BASE_MODEL, device=device)
    results.append(
        report_metrics(
            "Untuned MiniLM",
            semantic_ranks(stock, test_documents, queries, batch_size),
            semantic_ranks(stock, all_documents, title_queries, batch_size),
            truth,
            seen,
            unseen,
            all_titles,
            title_queries,
        )
    )
    return results


def training_arguments(
    output_dir: Path,
    *,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
):
    """Create consistent SentenceTransformer arguments for both training phases."""
    from sentence_transformers import SentenceTransformerTrainingArguments

    return SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=0.1,
        batch_sampler="no_duplicates",
        fp16=device == "cuda",
        save_strategy="no",
        logging_steps=200,
        report_to="none",
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate local movie search MiniLM.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(), help="contains moviebot.db, tags.csv and links.csv")
    parser.add_argument("--train-csv", type=Path, help="standalone anchor,positive training CSV; never reads an evaluation database")
    parser.add_argument("--train-with-source-csv", type=Path, help="standalone anchor,positive,source CSV for train-only source ablation")
    parser.add_argument("--include-sources", help="with --train-with-source-csv: comma-separated source-only filter (genome,tmdb_keywords,user_tag,title)")
    parser.add_argument("--exclude-sources", help="with --train-with-source-csv: comma-separated leave-one-source-out filter")
    parser.add_argument("--output", type=Path, default=REPO / "data" / "minilm-movie-search-titles-local")
    parser.add_argument("--previous-model", type=Path, help="optional first model to compare in the report")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64, help="64 matches the Colab experiment; use 16 for modest CPUs/GPUs")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--training-seed", type=int, help="separate trainer randomness from the fixed data split seed")
    parser.add_argument("--test-share", type=float, default=0.2)
    parser.add_argument("--validation-share", type=float, default=0.0, help="hold out this share of outer training movies and tag keys for model selection")
    parser.add_argument("--proxy-validation-share", type=float, default=0.0, help="for --train-csv only: hold out normalized anchor keys and report train-only retrieval metrics")
    parser.add_argument("--max-pairs", type=int, help="cap pairs for a quick smoke run")
    parser.add_argument("--balanced-query-sampling", action="store_true", help="cap frequent queries and oversample rare queries before base training")
    parser.add_argument("--min-pairs-per-query", type=int, default=3, help="minimum repeated pairs for a rare query when balancing (default 3)")
    parser.add_argument("--max-pairs-per-query", type=int, default=50, help="maximum pairs for one query when balancing (default 50)")
    parser.add_argument("--query-augmentation", action="store_true", help="for --train-csv only: add controlled spelling/alias variants from training anchors")
    parser.add_argument("--max-augmented-variants-per-query", type=int, default=1, help="with --query-augmentation: use one or two audited variants per anchor (default 1)")
    parser.add_argument("--no-title-pairs", action="store_true", help="train the first/tag-only model instead")
    parser.add_argument("--no-movielens-tags", action="store_true", help="train only from TMDB keywords and titles")
    parser.add_argument("--hard-negatives", action="store_true", help="mine train-only hard negatives and fine-tune with explicit negatives")
    parser.add_argument("--hard-negative-base-model", type=Path, help="reuse this trained model instead of repeating the initial training stage; requires --hard-negatives")
    parser.add_argument("--hard-negative-relevance", type=Path, help="compact train-only Genome relevance .npz; required for --hard-negatives with export CSV data")
    parser.add_argument("--hard-negative-top-k", type=int, default=100, help="retrieve this many train movies per tag while mining (default 100)")
    parser.add_argument("--hard-negative-max-rank", type=int, help="only retain negatives retrieved at this rank or better; default is all top-K")
    parser.add_argument("--hard-negative-mix-original", action="store_true", help="interleave original anchor-positive pairs with hard-negative triples to reduce forgetting")
    parser.add_argument("--hard-negative-threshold", type=float, default=0.10, help="maximum Genome relevance for a negative (default 0.10)")
    parser.add_argument("--hard-negatives-per-positive", type=int, default=1, help="mined negatives per positive pair (default 1)")
    parser.add_argument("--hard-negative-epochs", type=float, default=1, help="extra fine-tuning epochs after mining (default 1)")
    parser.add_argument("--hard-negative-learning-rate", type=float, default=5e-6, help="learning rate for hard-negative fine-tuning (default 5e-6)")
    parser.add_argument("--skip-evaluation", action="store_true", help="save time after training; still saves the model")
    parser.add_argument("--check-device", action="store_true", help="install dependencies and report CUDA availability, without reading data or training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    has_export_csv = bool(args.train_csv or args.train_with_source_csv)
    if not 0 < args.test_share < 1:
        raise SystemExit("--test-share must be between 0 and 1")
    if args.validation_share and not 0 < args.validation_share < 1:
        raise SystemExit("--validation-share must be between 0 and 1, or 0 to disable it")
    if args.proxy_validation_share and not 0 < args.proxy_validation_share < 1:
        raise SystemExit("--proxy-validation-share must be between 0 and 1, or 0 to disable it")
    if args.batch_size < 1 or args.epochs <= 0 or args.hard_negative_epochs <= 0:
        raise SystemExit("--batch-size, --epochs and --hard-negative-epochs must be positive")
    if args.hard_negative_base_model and not args.hard_negatives:
        raise SystemExit("--hard-negative-base-model requires --hard-negatives")
    if args.validation_share and (args.hard_negative_base_model or args.previous_model):
        raise SystemExit("validation mode trains an inner model from scratch; do not pass --hard-negative-base-model or --previous-model")
    if args.hard_negatives and (
        args.hard_negative_top_k < 1
        or args.hard_negatives_per_positive < 1
        or not 0 <= args.hard_negative_threshold < 0.8
        or (args.hard_negative_max_rank is not None and not 1 <= args.hard_negative_max_rank <= args.hard_negative_top_k)
    ):
        raise SystemExit("hard-negative ranks and count must be positive; max rank cannot exceed top-K; threshold must be from 0 (inclusive) to 0.8 (exclusive)")
    if args.balanced_query_sampling and (
        args.min_pairs_per_query < 1 or args.max_pairs_per_query < args.min_pairs_per_query
    ):
        raise SystemExit("balanced sampling requires 1 <= --min-pairs-per-query <= --max-pairs-per-query")
    if args.query_augmentation and not 1 <= args.max_augmented_variants_per_query <= 2:
        raise SystemExit("--max-augmented-variants-per-query must be 1 or 2")
    if args.train_csv and args.train_with_source_csv:
        raise SystemExit("choose only one of --train-csv or --train-with-source-csv")
    if (args.include_sources or args.exclude_sources) and not args.train_with_source_csv:
        raise SystemExit("--include-sources and --exclude-sources require --train-with-source-csv")
    try:
        included_sources = parse_source_names(args.include_sources)
        excluded_sources = parse_source_names(args.exclude_sources)
        if included_sources & excluded_sources:
            raise ValueError(f"a source cannot be both included and excluded: {', '.join(sorted(included_sources & excluded_sources))}")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if has_export_csv and (args.previous_model or args.validation_share):
        raise SystemExit("export CSV mode cannot use previous models or validation/test data")
    if args.hard_negative_relevance and not args.hard_negatives:
        raise SystemExit("--hard-negative-relevance requires --hard-negatives")
    if has_export_csv and args.hard_negatives and (not args.train_with_source_csv or not args.hard_negative_relevance):
        raise SystemExit("hard negatives with export CSV require --train-with-source-csv and --hard-negative-relevance")
    if args.query_augmentation and not has_export_csv:
        raise SystemExit("--query-augmentation requires an export CSV")
    if args.proxy_validation_share and not has_export_csv:
        raise SystemExit("--proxy-validation-share requires an export CSV")
    if not args.check_device:
        if args.output.exists() and any(args.output.iterdir()):
            raise SystemExit(f"Refusing to overwrite non-empty output directory: {args.output}")
        if not has_export_csv:
            require_files(args.data_dir, not args.no_movielens_tags)

    try:
        import torch
        from datasets import Dataset
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    except ImportError as error:
        raise SystemExit(f"Missing training dependency: {error}. Run this file through 'uv run --script'.") from error

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but PyTorch cannot see a CUDA GPU. Use --device cpu or install the matching CUDA PyTorch build.")
    if args.check_device:
        print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        return 0
    if has_export_csv:
        return train_from_export_csv(
            args,
            device=device,
            Dataset=Dataset,
            SentenceTransformer=SentenceTransformer,
            SentenceTransformerTrainer=SentenceTransformerTrainer,
            MultipleNegativesRankingLoss=MultipleNegativesRankingLoss,
        )
    print(f"Using {device}; data: {args.data_dir}; output: {args.output}")
    tag_names, titles, texts, genome, outer_train_movies, outer_test_keys = load_data(
        args.data_dir, seed=args.seed, test_share=args.test_share
    )
    outer_test_movies = set(texts) - outer_train_movies
    validation_movies: set[str] = set()
    validation_keys: set[str] = set()
    if args.validation_share:
        validation_movies, validation_keys = validation_split(
            tag_names,
            outer_train_movies & genome.keys(),
            outer_test_keys,
            seed=args.seed,
            share=args.validation_share,
        )
        train_movies = outer_train_movies - validation_movies
        held_out_keys = outer_test_keys | validation_keys
        report_movies, report_keys = validation_movies, validation_keys
        report_name = "Validation"
    else:
        train_movies = outer_train_movies
        held_out_keys = outer_test_keys
        report_movies, report_keys = outer_test_movies, outer_test_keys
        report_name = "Frozen test"
    pairs, source_counts = training_pairs(
        args.data_dir, tag_names, titles, texts, genome, train_movies, held_out_keys,
        use_movielens_tags=not args.no_movielens_tags,
        include_titles=not args.no_title_pairs,
        relevant=0.8,
        min_users=2,
        max_tag_chars=40,
    )
    balance_summary: dict[str, int] | None = None
    if args.balanced_query_sampling:
        pairs, balance_summary = balance_query_pairs(
            pairs,
            min_pairs_per_query=args.min_pairs_per_query,
            max_pairs_per_query=args.max_pairs_per_query,
            seed=args.seed,
        )
        print(
            "balanced query sampling: "
            + ", ".join(f"{value:,} {name.replace('_', ' ')}" for name, value in balance_summary.items())
        )
    else:
        random.Random(args.seed).shuffle(pairs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(
        f"{len(train_movies):,} inner-train movies, {len(validation_movies):,} validation movies, "
        f"{len(outer_test_movies):,} frozen-test movies; {len(validation_keys):,} validation tags, "
        f"{len(outer_test_keys):,} frozen-test tags"
    )
    print(", ".join(f"{count:,} {name}" for name, count in source_counts.items()) + f"; {len(pairs):,} pairs used")

    args.output.mkdir(parents=True, exist_ok=True)
    hard_negative_summary: dict[str, int | float | str] | None = None
    validation_baseline_path: Path | None = None
    if args.hard_negative_base_model:
        if not (args.hard_negative_base_model / "model.safetensors").exists():
            raise SystemExit(f"--hard-negative-base-model is not a saved SentenceTransformer: {args.hard_negative_base_model}")
        print(f"Loading base model for hard-negative fine-tuning: {args.hard_negative_base_model}")
        model = SentenceTransformer(str(args.hard_negative_base_model), device=device)
    else:
        model = SentenceTransformer(BASE_MODEL, device=device)
        train_data = Dataset.from_dict(
            {"anchor": [query for query, _ in pairs], "positive": [texts[movie] for _, movie in pairs]}
        )
        SentenceTransformerTrainer(
            model=model,
            args=training_arguments(
                args.output / "checkpoints-base",
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=device,
                seed=args.training_seed if args.training_seed is not None else args.seed,
            ),
            train_dataset=train_data,
            loss=MultipleNegativesRankingLoss(model),
        ).train()

    if args.hard_negatives:
        # Validation runs must compare against the exact inner-training model,
        # not the previously saved outer-training model (which saw validation data).
        if args.validation_share and not args.skip_evaluation:
            validation_baseline_path = args.output / "baseline_before_hard_negatives"
            model.save_pretrained(str(validation_baseline_path))
        examples = mine_hard_negatives(
            model,
            tag_names,
            texts,
            genome,
            train_movies,
            held_out_keys,
            positive_threshold=0.8,
            negative_threshold=args.hard_negative_threshold,
            top_k=args.hard_negative_top_k,
            negatives_per_positive=args.hard_negatives_per_positive,
            batch_size=args.batch_size,
            max_candidate_rank=args.hard_negative_max_rank,
        )
        if not examples:
            raise SystemExit("No hard negatives were mined. Raise --hard-negative-top-k or use a less strict --hard-negative-threshold.")
        # Assertions document the non-negotiable isolation and relevance constraints.
        all_non_train_movies = set(texts) - train_movies
        assert not (({example.positive_movie_id for example in examples} | {example.negative_movie_id for example in examples}) & all_non_train_movies)
        assert not ({tag_key(example.tag) for example in examples} & held_out_keys)
        assert all(example.negative_relevance <= args.hard_negative_threshold for example in examples)
        artifact = args.output / "hard_negatives.csv"
        write_hard_negative_artifact(artifact, examples)
        print(f"Mined {len(examples):,} hard-negative triples; audit artifact: {artifact}")
        hard_data = Dataset.from_dict(
            {
                "anchor": [example.tag for example in examples],
                "positive": [texts[example.positive_movie_id] for example in examples],
                "negative": [texts[example.negative_movie_id] for example in examples],
            }
        )
        SentenceTransformerTrainer(
            model=model,
            args=training_arguments(
                args.output / "checkpoints-hard-negatives",
                epochs=args.hard_negative_epochs,
                batch_size=args.batch_size,
                learning_rate=args.hard_negative_learning_rate,
                device=device,
                seed=args.training_seed if args.training_seed is not None else args.seed,
            ),
            train_dataset=hard_data,
            loss=MultipleNegativesRankingLoss(model),
        ).train()
        hard_negative_summary = {
            "base_model": str(args.hard_negative_base_model) if args.hard_negative_base_model else BASE_MODEL,
            "triples": len(examples),
            "top_k": args.hard_negative_top_k,
            "max_candidate_rank": args.hard_negative_max_rank,
            "negative_threshold": args.hard_negative_threshold,
            "negatives_per_positive": args.hard_negatives_per_positive,
            "epochs": args.hard_negative_epochs,
            "learning_rate": args.hard_negative_learning_rate,
            "artifact": artifact.name,
        }

    model.save_pretrained(str(args.output))

    results: list[dict[str, float | str]] = []
    if not args.skip_evaluation:
        models: list[tuple[str, object]] = []
        if validation_baseline_path:
            models.append((f"{report_name}: First model", SentenceTransformer(str(validation_baseline_path), device=device)))
        if args.previous_model:
            if not (args.previous_model / "model.safetensors").exists():
                raise SystemExit(f"--previous-model is not a saved SentenceTransformer: {args.previous_model}")
            models.append(("First model", SentenceTransformer(str(args.previous_model), device=device)))
        model_name = "Second model" if not args.no_title_pairs else "First model"
        if args.hard_negatives:
            model_name += " + hard negatives"
        model_name = f"{report_name}: {model_name}"
        models.append((model_name, model))
        results = evaluate_split(
            models,
            tag_names,
            titles,
            texts,
            genome,
            report_movies,
            report_keys,
            relevant=0.8,
            batch_size=args.batch_size,
            device=device,
            excluded_tag_keys=outer_test_keys if args.validation_share else set(),
        )

    metadata = {
        "base_model": BASE_MODEL,
        "device": device,
        "train_pairs": len(pairs),
        "source_pair_counts": source_counts,
        "config": {
            "seed": args.seed,
            "training_seed": args.training_seed if args.training_seed is not None else args.seed,
            "test_share": args.test_share,
            "validation_share": args.validation_share,
            "evaluation_split": report_name.lower(),
            "inner_train_movies": len(train_movies),
            "validation_movies": len(validation_movies),
            "frozen_test_movies": len(outer_test_movies),
            "validation_tag_keys": len(validation_keys),
            "frozen_test_tag_keys": len(outer_test_keys),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "title_pairs": not args.no_title_pairs,
            "movielens_tags": not args.no_movielens_tags,
            "balanced_query_sampling": balance_summary,
            "hard_negatives": hard_negative_summary,
            "validation_baseline": validation_baseline_path.name if validation_baseline_path else None,
        },
        "results": results,
    }
    (args.output / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved model and results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
