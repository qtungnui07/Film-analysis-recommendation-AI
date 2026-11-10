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
    {"movie_id": "tt1", "title": "<b>Bold</b>", "year": 2001, "rating": 7.5, "genres": "Drama,War", "overview": "Tom & Jerry"},
    {"movie_id": "tt2", "title": "Second", "year": None, "rating": None, "genres": None, "overview": None},
]


def client(calls=None):
    def search(query):
        if calls is not None:
            calls.append(query)
        return [(1, 0.9), (0, 0.5)]

    return TestClient(site.create_app(MOVIES, search))


def test_results_keep_search_order_and_are_escaped():
    body = client().get("/", params={"q": '"><script>'}).text
    assert body.index("Second") < body.index("&lt;b&gt;Bold&lt;/b&gt;")
    assert "Tom &amp; Jerry" in body
    assert "<script>" not in body and "<b>Bold" not in body


def test_blank_query_does_not_search():
    calls = []
    assert client(calls).get("/", params={"q": "   "}).status_code == 200
    assert calls == []


def test_overlong_query_is_rejected():
    assert client().get("/", params={"q": "x" * (site.MAX_QUERY + 1)}).status_code == 422


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
