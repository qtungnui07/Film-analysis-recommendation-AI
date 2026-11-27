"""Unit tests for hard-negative mining without downloading or training a model."""

import importlib.util
import sys

import numpy as np

from moviebot.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location("train_search_local", REPO_ROOT / "scripts" / "train_search_local.py")
train = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = train
_spec.loader.exec_module(train)


class FakeModel:
    vectors = {
        "space": [1.0, 0.0],
        "safe positive": [0.8, 0.0],
        "deceptive negative": [1.0, 0.0],
        "not retrieved": [0.0, 1.0],
        "held out movie": [1.0, 0.0],
        "no genome movie": [1.0, 0.0],
    }

    def encode(self, sentences, **_kwargs):
        return np.array([self.vectors[sentence] for sentence in sentences], dtype=float)


def test_hard_negative_is_retrieved_but_genome_irrelevant():
    texts = {
        "positive": "safe positive",
        "negative": "deceptive negative",
        "other": "not retrieved",
        "held-out": "held out movie",
        "no-genome": "no genome movie",
    }
    genome = {
        "positive": np.array([0.9], dtype="f4"),
        "negative": np.array([0.0], dtype="f4"),
        "other": np.array([0.2], dtype="f4"),
        "held-out": np.array([0.0], dtype="f4"),
    }
    examples = train.mine_hard_negatives(
        FakeModel(),
        ["space"],
        texts,
        genome,
        {"positive", "negative", "other", "no-genome"},
        set(),
        positive_threshold=0.8,
        negative_threshold=0.1,
        top_k=2,
        negatives_per_positive=1,
        batch_size=3,
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.positive_movie_id == "positive"
    assert example.negative_movie_id == "negative"
    assert example.negative_relevance == 0.0 and example.retrieval_rank == 1
    assert {"held-out", "no-genome"}.isdisjoint({example.positive_movie_id, example.negative_movie_id})


def test_hard_negative_mining_never_uses_held_out_tag_key():
    examples = train.mine_hard_negatives(
        FakeModel(),
        ["space"],
        {"positive": "safe positive", "negative": "deceptive negative"},
        {"positive": np.array([0.9], dtype="f4"), "negative": np.array([0.0], dtype="f4")},
        {"positive", "negative"},
        {"space"},
        positive_threshold=0.8,
        negative_threshold=0.1,
        top_k=2,
        negatives_per_positive=1,
        batch_size=2,
    )
    assert examples == []


def test_hard_negative_mining_accepts_only_explicit_train_genome_anchor_keys():
    examples = train.mine_hard_negatives(
        FakeModel(),
        ["space"],
        {"positive": "safe positive", "negative": "deceptive negative"},
        {"positive": np.array([0.9], dtype="f4"), "negative": np.array([0.0], dtype="f4")},
        {"positive", "negative"},
        set(),
        positive_threshold=0.8,
        negative_threshold=0.1,
        top_k=2,
        negatives_per_positive=1,
        batch_size=2,
        allowed_tag_keys={"not-space"},
    )
    assert examples == []


def test_load_compact_train_genome_validates_dense_train_only_matrix(tmp_path):
    artifact = tmp_path / "genome.npz"
    np.savez_compressed(
        artifact,
        movie_ids=np.array(["movie-a", "movie-b"]),
        movie_texts=np.array(["Movie A", "Movie B"]),
        tag_names=np.array(["space", "romance"]),
        relevance=np.array([[0.9, 0.01], [0.02, 0.95]], dtype="f4"),
    )

    tags, texts, genome = train.load_compact_train_genome(artifact)

    assert tags == ["space", "romance"]
    assert texts == {"movie-a": "Movie A", "movie-b": "Movie B"}
    assert np.allclose(genome["movie-a"], [0.9, 0.01])


def test_validation_split_is_deterministic_and_disjoint_from_outer_test():
    tags = ["space", "romance", "action", "comedy", "drama", "horror", "crime", "music", "war", "western"]
    outer_train_movies = {f"movie-{index}" for index in range(10)}
    outer_test_keys = {"drama"}

    first = train.validation_split(tags, outer_train_movies, outer_test_keys, seed=42, share=0.2)
    second = train.validation_split(tags, outer_train_movies, outer_test_keys, seed=42, share=0.2)

    validation_movies, validation_keys = first
    assert first == second
    assert len(validation_movies) == 2 and validation_movies <= outer_train_movies
    assert len(validation_keys) == 1 and validation_keys.isdisjoint(outer_test_keys)


def test_balanced_query_sampling_caps_frequent_and_repeats_rare_queries():
    pairs = [("rare", "one"), ("common", "a"), ("common", "b"), ("common", "c"), ("common", "d")]

    balanced, summary = train.balance_query_pairs(
        pairs,
        min_pairs_per_query=3,
        max_pairs_per_query=3,
        seed=42,
    )

    counts = {query: sum(item[0] == query for item in balanced) for query in {query for query, _ in pairs}}
    assert counts == {"rare": 3, "common": 3}
    assert summary == {
        "input_pairs": 5,
        "output_pairs": 6,
        "unique_queries": 2,
        "capped_queries": 1,
        "oversampled_queries": 1,
    }


def test_export_proxy_split_and_metrics_use_only_csv_pairs():
    pairs = [(f"query-{index}", f"movie-{index}") for index in range(10)]
    train_pairs, validation_pairs, held_keys = train.export_proxy_split(pairs, seed=42, share=0.2)

    assert len(validation_pairs) == 2
    assert not {train.tag_key(anchor) for anchor, _ in train_pairs} & held_keys
    assert {train.tag_key(anchor) for anchor, _ in validation_pairs} <= held_keys

    class ProxyModel:
        vectors = {
            "space": [1.0, 0.0],
            "safe positive": [1.0, 0.0],
            "not retrieved": [0.0, 1.0],
        }

        def encode(self, sentences, **_kwargs):
            return np.array([self.vectors[sentence] for sentence in sentences], dtype=float)

    metrics = train.evaluate_export_proxy(
        ProxyModel(),
        [("space", "safe positive")],
        [("space", "safe positive"), ("other", "not retrieved")],
        batch_size=2,
    )
    assert metrics == {
        "queries": 1,
        "documents": 2,
        "proxy_precision_at_10": 0.5,
        "proxy_recall_at_10": 1.0,
        "proxy_mrr_at_10": 1.0,
    }


def test_query_augmentation_is_auditable_and_never_crosses_proxy_holdout():
    pairs = [("time travel", "movie one"), ("sci-fi", "movie two")]
    augmented, summary, audit = train.augment_query_pairs(
        pairs,
        forbidden_keys={train.tag_key("travelling through time"), train.tag_key("science fiction")},
        max_variants_per_query=2,
    )

    assert set(augmented) == {
        ("time travel", "movie one"),
        ("time-travel", "movie one"),
        ("sci-fi", "movie two"),
        ("sci fi", "movie two"),
    }
    assert audit == [
        ("sci-fi", "sci fi", "punctuation_normalization"),
        ("time travel", "time-travel", "curated_alias"),
    ]
    assert summary == {
        "input_pairs": 2,
        "output_pairs": 4,
        "source_queries": 2,
        "generated_anchor_variants": 2,
        "augmented_pairs": 2,
        "skipped_heldout_variants": 2,
        "skipped_duplicate_pairs": 0,
    }


def test_source_ablation_keeps_pairs_supported_by_another_source():
    rows = [
        ("cyberpunk", "movie genome", frozenset({"genome"})),
        ("cyberpunk", "movie shared", frozenset({"genome", "tmdb_keywords"})),
        ("Blade Runner", "movie title", frozenset({"title"})),
    ]

    genome_only, genome_summary = train.select_source_pairs(
        rows, include_sources=frozenset({"genome"}), exclude_sources=frozenset()
    )
    without_genome, without_summary = train.select_source_pairs(
        rows, include_sources=frozenset(), exclude_sources=frozenset({"genome"})
    )

    assert genome_only == [("cyberpunk", "movie genome"), ("cyberpunk", "movie shared")]
    assert without_genome == [("cyberpunk", "movie shared"), ("Blade Runner", "movie title")]
    assert genome_summary["output_pairs"] == 2 and genome_summary["pairs_with_genome"] == 2
    assert without_summary["output_pairs"] == 2 and without_summary["pairs_with_tmdb_keywords"] == 1
