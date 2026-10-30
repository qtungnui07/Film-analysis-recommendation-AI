import json
import re
import sqlite3
import urllib.error
import zlib

import pytest
from chromadb.api.types import EmbeddingFunction

from moviebot import build_index
from moviebot.contracts import NotFoundError, ToolError
from moviebot.db_backend import SCHEMA, SqliteBackend
from moviebot.retrieval import VectorIndex

SPIDERS, DOG, ANIMAL = 165, 153, 189


class BagOfWords(EmbeddingFunction):
    """Deterministic stand-in for all-MiniLM-L6-v2: hashed word counts, no model download."""

    def __init__(self):
        pass

    def __call__(self, input):
        vectors = []
        for doc in input:
            vector = [0.0] * 256
            for word in re.findall(r"[a-z0-9]+", doc.lower()):
                vector[zlib.crc32(word.encode()) % 255] += 1.0
            vector[255] = 0.01  # never all zeros
            vectors.append(vector)
        return vectors

    @staticmethod
    def name():
        return "bag-of-words"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return BagOfWords()


def indexed_backend(db_path, tmp_path) -> SqliteBackend:
    index = VectorIndex(tmp_path / "chroma", embedding_function=BagOfWords())
    build_index.build(sqlite3.connect(db_path), index)
    return SqliteBackend(db_path, index=index)


def set_overviews(db_path, overviews: dict[str, str]):
    db = sqlite3.connect(db_path)
    with db:
        db.executemany("UPDATE movies SET overview = ? WHERE movie_id = ?", [(o, m) for m, o in overviews.items()])


class FakeDtdd:
    def __init__(self, votes: dict | None = None, fail: bool = False):
        self.calls: list[str] = []
        self._votes = votes or {}
        self._fail = fail

    def votes(self, imdb_id):
        self.calls.append(imdb_id)
        if self._fail:
            raise urllib.error.URLError("offline")
        return self._votes.get(imdb_id, {})


class FakeTmdb:
    def __init__(self, movies: list[dict]):
        self._movies = movies

    def search(self, title):
        return [m for m in self._movies if title.casefold() in m["title"].casefold()]

    def details(self, tmdb_id):
        return next(m for m in self._movies if m["id"] == tmdb_id)

    def find_by_imdb(self, imdb_id):
        return next((m for m in self._movies if m["imdb_id"] == imdb_id), None)


def tmdb_movie(tmdb_id, imdb_id, title, date, **extra):
    return {"id": tmdb_id, "imdb_id": imdb_id, "title": title, "release_date": date, "vote_average": 7.0,
            "genres": [], "overview": "", **extra}


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO topics VALUES (?, ?, ?, ?, ?, ?)", [
        (SPIDERS, "there are spiders", "Are there spiders", "Animal Phobia", "[]", '["spiders"]'),
        (DOG, "a dog dies", "Does the dog die", "Animal Death", "[]", '["dog dies"]'),
        (ANIMAL, "an animal dies", "Does an animal die", "Animal Death", f"[{DOG}]", '["animal dies"]'),
    ])
    db.executemany(
        "INSERT INTO movies (movie_id, title, year, rating, num_votes, genres, source) VALUES (?, ?, ?, ?, ?, ?, 'imdb')",
        [
            ("tt1", "Dataset Dog Film", 2001, 7.0, 50000, "Drama"),  # no animal dies, no dog label
            ("tt2", "Fully Labelled Film", 2002, 8.0, 60000, "Comedy"),  # every label an animal user needs
            ("tt3", "Unlabelled Film", 2020, 6.5, 20000, "Drama"),  # nothing in the dataset
        ],
    )
    db.executemany("INSERT INTO labels VALUES (?, ?, ?, 'ml-ddd')", [
        ("tt1", ANIMAL, "no"), ("tt1", SPIDERS, "no"),
        ("tt2", ANIMAL, "no"), ("tt2", DOG, "no"), ("tt2", SPIDERS, "no"),
    ])
    db.execute("INSERT INTO users VALUES ('animal_lover', 'Ana', ?)", (json.dumps({str(ANIMAL): "never"}),))
    db.commit()
    db.close()
    return path


def test_dtdd_fills_missing_labels_once(db_path):
    dtdd = FakeDtdd({"tt1": {DOG: (9, 1), 999: (5, 0)}})
    backend = SqliteBackend(db_path, dtdd=dtdd)
    titles = {m.title for m in backend.search_movies("animal_lover", "film", limit=10).results}
    assert "Dataset Dog Film" not in titles  # the dataset says no animal dies; DTDD now says a dog dies
    assert {"Fully Labelled Film", "Unlabelled Film"} <= titles
    assert sorted(dtdd.calls) == ["tt1", "tt3"]  # tt2 had every label it needed

    stored = sqlite3.connect(db_path).execute("SELECT topic_id, label FROM labels WHERE source = 'dtdd'").fetchall()
    assert stored == [(DOG, "yes")]  # topic 999 isn't in the topic table
    again = FakeDtdd()
    SqliteBackend(db_path, dtdd=again).search_movies("animal_lover", "film", limit=10)
    assert again.calls == []  # tt3 was asked too, even though DTDD had nothing


def test_dtdd_failure_keeps_unknown_and_stops_asking(db_path):
    dtdd = FakeDtdd(fail=True)
    found = SqliteBackend(db_path, dtdd=dtdd).search_movies("animal_lover", "film", limit=10)
    unlabelled = next(m for m in found.results if m.title == "Unlabelled Film")
    assert unlabelled.unknown_topics == ["an animal dies"]
    assert len(dtdd.calls) == 1
    assert sqlite3.connect(db_path).execute("SELECT count(dtdd_checked_at) FROM movies").fetchone() == (0,)


def test_tmdb_fallback_adds_a_movie_by_exact_title(db_path):
    tmdb = FakeTmdb([
        tmdb_movie(7, "tt7", "New Film", "2024-05-01", vote_average=7.26,
                   genres=[{"name": "Science Fiction"}, {"name": "TV Movie"}], overview="A new film."),
        tmdb_movie(8, "tt8", "New Film 2", "2025-01-01"),
    ])
    warnings = SqliteBackend(db_path, tmdb=tmdb).get_movie_warnings(title="new film")
    assert (warnings.movie_id, warnings.year) == ("tt7", 2024)
    movie = SqliteBackend(db_path)._movies["tt7"]  # stored, so a fresh backend has it
    assert (movie.genres, movie.rating, movie.overview) == (["Sci-Fi"], 7.3, "A new film.")


def test_tmdb_fallback_by_imdb_id(db_path):
    tmdb = FakeTmdb([tmdb_movie(7, "tt7", "New Film", "2024-05-01")])
    assert SqliteBackend(db_path, tmdb=tmdb).get_movie_warnings(movie_id="tt7").title == "New Film"


def test_tmdb_fallback_never_guesses(db_path):
    backend = SqliteBackend(db_path, tmdb=FakeTmdb([tmdb_movie(8, "tt8", "New Film 2", "2025-01-01")]))
    with pytest.raises(NotFoundError):
        backend.get_movie_warnings(title="New Film")

    things = [tmdb_movie(1, "tt82", "The Thing", "1982-06-25"), tmdb_movie(2, "tt11", "The Thing", "2011-10-14")]
    backend = SqliteBackend(db_path, tmdb=FakeTmdb(things))
    with pytest.raises(ToolError, match="several") as e:
        backend.get_movie_warnings(title="The Thing")
    assert "The Thing (1982)" in e.value.hint
    assert backend.get_movie_warnings(title="The Thing (1982)").movie_id == "tt82"


def test_duplicate_titles_ask_for_the_year(db_path):
    db = sqlite3.connect(db_path)
    with db:
        db.executemany(
            "INSERT INTO movies (movie_id, title, year, rating, num_votes, genres, source) VALUES (?, ?, ?, ?, 1, '', 'imdb')",
            [("tt82", "The Thing", 1982, 8.2), ("tt11", "The Thing", 2011, 6.2)],
        )
    backend = SqliteBackend(db_path)
    with pytest.raises(ToolError, match="several") as e:
        backend.get_movie_warnings(title="the thing")  # used to return whichever row came last
    assert "The Thing (1982, tt82)" in e.value.hint
    assert backend.get_movie_warnings(title="The Thing (2011)").movie_id == "tt11"
    assert backend.get_movie_warnings(title="Unlabeled Film").movie_id == "tt3"  # one sure fuzzy match


def test_index_build_embeds_only_what_changed(db_path, tmp_path):
    index = VectorIndex(tmp_path / "chroma", embedding_function=BagOfWords())
    db = sqlite3.connect(db_path)
    assert build_index.build(db, index) == (3, 6)  # 3 topics, each with its name and one synonym
    assert build_index.build(db, index) == (0, 0)
    with db:
        db.execute("UPDATE movies SET overview = 'Robots in space.' WHERE movie_id = 'tt2'")
        db.execute("DELETE FROM movies WHERE movie_id = 'tt1'")
    assert build_index.build(db, index) == (1, 0)
    assert index.movies.count() == 2
    assert index.document("tt2") == "Fully Labelled Film (2002). Comedy. Robots in space."


def test_semantic_search_still_excludes_never_topics(db_path, tmp_path):
    set_overviews(db_path, {"tt2": "A robot comedy set in space.", "tt3": "Robots in deep space."})
    db = sqlite3.connect(db_path)
    with db:
        db.execute("INSERT INTO labels VALUES ('tt3', ?, 'yes', 'ml-ddd')", (DOG,))  # so an animal dies
    found = indexed_backend(db_path, tmp_path).search_movies("animal_lover", "robots in deep space", limit=1)
    assert [m.title for m in found.results] == ["Fully Labelled Film"]
    assert found.removed_count == 1  # the best match, removed
    assert found.results[0].match_reasons == ["close match for your request"]


def test_semantic_search_title_and_like_queries(db_path, tmp_path):
    backend = indexed_backend(db_path, tmp_path)
    top = backend.search_movies("animal_lover", "Unlabeled Film", limit=1).results[0]
    assert (top.title, top.match_reasons) == ("Unlabelled Film", ["title match"])

    found = backend.search_movies("animal_lover", "something like Unlabelled Film", limit=3)
    assert [m.title for m in found.results] == ["Dataset Dog Film", "Fully Labelled Film"]  # shares "Drama"
    assert found.results[0].match_reasons == ["similar to Unlabelled Film (2020)"]


def test_phrase_mapping_adds_semantic_matches(db_path, tmp_path):
    backend = indexed_backend(db_path, tmp_path)
    assert SqliteBackend(db_path).map_phrase_to_topics("dies animal") == []  # word order defeats the synonym rules
    assert backend.map_phrase_to_topics("dies animal")[0].topic_id == ANIMAL
    assert backend.map_phrase_to_topics("dog dies")[0].score == 1.0
    assert backend.map_phrase_to_topics("nothing in particular") == []


def test_profile_edits_persist(db_path):
    SqliteBackend(db_path).update_user_preference("animal_lover", SPIDERS, "warn")
    assert SqliteBackend(db_path).get_user_profile("animal_lover").topic_ids("warn") == {SPIDERS}


def test_missing_database_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_db"):
        SqliteBackend(tmp_path / "nope.db")
