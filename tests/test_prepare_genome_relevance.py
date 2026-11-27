"""Tests for compacting the verbose train-only Genome relevance export."""

import importlib.util
import sys

import numpy as np

from moviebot.config import REPO_ROOT


_spec = importlib.util.spec_from_file_location(
    "prepare_genome_relevance", REPO_ROOT / "scripts" / "prepare_genome_relevance.py"
)
prepare = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = prepare
_spec.loader.exec_module(prepare)


def test_compact_relevance_keeps_all_scores_and_one_copy_of_movie_text(tmp_path):
    input_path = tmp_path / "relevance.csv"
    input_path.write_text(
        "tconst,split,has_genome,tagId,tag,tag_key,relevance,movie_text\n"
        "movie-a,train,1,1,space,space,0.9,Movie A text\n"
        "movie-a,train,1,2,romance,romance,0.03,Movie A text\n"
        "movie-b,train,1,1,space,space,0.02,Movie B text\n"
        "movie-b,train,1,2,romance,romance,0.95,Movie B text\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "compact.npz"

    metadata = prepare.compact_relevance_csv(input_path, output_path)
    with np.load(output_path) as artifact:
        assert artifact["movie_ids"].tolist() == ["movie-a", "movie-b"]
        assert artifact["tag_names"].tolist() == ["space", "romance"]
        assert np.allclose(artifact["relevance"], [[0.9, 0.03], [0.02, 0.95]])
        assert artifact["relevance"].dtype == np.float32
    assert metadata["movies"] == 2
    assert metadata["tags"] == 2
    assert metadata["relevance_le_0_1"] == 2
    assert metadata["relevance_ge_0_8"] == 2


def test_compact_relevance_rejects_test_rows(tmp_path):
    input_path = tmp_path / "relevance.csv"
    input_path.write_text(
        "tconst,split,has_genome,tagId,tag,tag_key,relevance,movie_text\n"
        "movie-a,test,1,1,space,space,0.9,Movie A text\n",
        encoding="utf-8",
    )

    try:
        prepare.compact_relevance_csv(input_path, tmp_path / "compact.npz")
    except ValueError as error:
        assert "not a Genome-labelled training movie" in str(error)
    else:
        raise AssertionError("test row must be rejected")
