import importlib.util

import numpy as np
from fastapi.testclient import TestClient

from moviebot.config import REPO_ROOT
from moviebot.contracts import Movie
from moviebot.retrieval import movie_document

_spec = importlib.util.spec_from_file_location("search_site", REPO_ROOT / "notebooks" / "search_site.py")
site = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(site)

MOVIES = [
    {"movie_id": "tt1", "title": "First", "year": 2001, "rating": 7.5, "genres": "Drama,War", "overview": "Tom & Jerry"},
    {"movie_id": "tt2", "title": "Second", "year": None, "rating": None, "genres": None, "overview": None},
]


def client(tmp_path, calls=None, fetch_poster=None):
    def search(query):
        if calls is not None:
            calls.append(query)
        return [(1, 0.9), (0, 0.5)]

    return TestClient(site.create_app(MOVIES, search, fetch_poster, tmp_path))


def test_search_keeps_rank_order(tmp_path):
    assert [m["movie_id"] for m in client(tmp_path).get("/search", params={"q": "war"}).json()] == ["tt2", "tt1"]


def test_blank_query_does_not_search(tmp_path):
    calls = []
    assert client(tmp_path, calls).get("/search", params={"q": "   "}).json() == []
    assert calls == []


def test_overlong_query_is_rejected(tmp_path):
    assert client(tmp_path).get("/search", params={"q": "x" * (site.MAX_QUERY + 1)}).status_code == 422


def test_movie_details_and_unknown_movie(tmp_path):
    c = client(tmp_path)
    assert c.get("/movies/tt1").json() == MOVIES[0]
    assert c.get("/movies/tt9").status_code == 404


def test_poster_is_fetched_once_then_served_from_disk(tmp_path):
    calls = []

    def fetch(movie_id):
        calls.append(movie_id)
        return b"\xff\xd8jpeg"

    c = client(tmp_path, fetch_poster=fetch)
    for _ in range(2):
        r = c.get("/posters/tt1")
        assert r.content == b"\xff\xd8jpeg" and r.headers["content-type"] == "image/jpeg"
    assert calls == ["tt1"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tt1.jpg"]


def test_missing_poster_is_remembered(tmp_path):
    calls = []

    def fetch(movie_id):
        calls.append(movie_id)
        return None

    c = client(tmp_path, fetch_poster=fetch)
    for _ in range(2):
        assert c.get("/posters/tt1").text == site.NO_POSTER
    assert calls == ["tt1"]


def test_failed_fetch_is_not_cached(tmp_path):
    calls = []

    def fetch(movie_id):
        calls.append(movie_id)
        raise OSError("TMDB down")

    c = client(tmp_path, fetch_poster=fetch)
    for _ in range(2):
        r = c.get("/posters/tt1")
        assert r.text == site.NO_POSTER and r.headers["cache-control"] == "no-store"
    assert calls == ["tt1", "tt1"]
    assert list(tmp_path.iterdir()) == []


def test_unknown_poster_id_does_not_reach_tmdb(tmp_path):
    def fetch(movie_id):
        raise AssertionError("fetched")

    c = client(tmp_path, fetch_poster=fetch)
    assert c.get("/posters/tt9").status_code == 404
    assert c.get("/posters/..%2F..%2Fmoviebot").status_code == 404


def test_search_ranks_by_similarity_and_keeps_top_k():
    class Model:
        def encode(self, texts, **kwargs):
            if len(texts) == 1:
                return np.array([[1.0, 0.0]])
            return np.array([[i / 30, (1 - (i / 30) ** 2) ** 0.5] for i in range(len(texts))])

    hits = site.make_search(Model(), ["text"] * 30)("query")
    assert [i for i, _ in hits] == list(range(29, 9, -1))
    assert len(hits) == site.TOP_K


def test_movie_text_matches_the_app():
    m = MOVIES[0]
    movie = Movie(movie_id=m["movie_id"], title=m["title"], year=m["year"], rating=m["rating"], genres=m["genres"].split(","), overview=m["overview"])
    assert site.movie_text(m) == movie_document(movie, [])
