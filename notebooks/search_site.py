# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "fastapi>=0.162.0",
#     "numpy",
#     "sentence-transformers==6.2.0",
#     "torch",
#     "transformers==5.22.2",
#     "uvicorn>=0.54.0",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Search API for the frontend, using the model trained by finetune_search.ipynb.

    uv run --env-file .env notebooks/search_site.py [--model data/minilm-movie-search] [--port 8002]

`--model` also takes a Hugging Face id, to compare another model on another port.
Posters come from TMDB (TMDB_KEY in .env) and are cached in data/posters/.
"""

import argparse
import os
import sqlite3
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = str(REPO / "data" / "minilm-movie-search")
DB_PATH = REPO / "data" / "moviebot.db"
POSTER_DIR = REPO / "data" / "posters"
POSTER_URL = "https://image.tmdb.org/t/p/w500"
TOP_K = 20
MAX_QUERY = 200
NO_POSTER = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 3"><rect width="2" height="3" fill="#d1d5db"/></svg>'


def load_movies(db_path: Path) -> list[dict]:
    db = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return [dict(r) for r in db.execute("SELECT movie_id, title, year, rating, genres, overview FROM movies")]


def movie_text(movie: dict) -> str:
    # moviebot.retrieval.movie_document with no tags: the text the model trained on.
    genres = ", ".join((movie["genres"] or "").split(","))
    return ". ".join(p for p in [f"{movie['title']} ({movie['year']})", genres, movie["overview"]] if p)


def make_search(model, texts: list[str]):
    # shortcut: embeds the catalogue on every start, cache the vectors if restarts get annoying.
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    def search(query: str) -> list[tuple[int, float]]:
        scores = vectors @ model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return [(int(i), float(scores[i])) for i in np.argsort(-scores)[:TOP_K]]

    return search


def make_fetch_poster(tmdb):
    def fetch_poster(movie_id: str) -> bytes | None:
        hit = tmdb.find_by_imdb(movie_id)
        if not (hit and hit.get("poster_path")):
            return None
        with urllib.request.urlopen(POSTER_URL + hit["poster_path"], timeout=15) as response:
            return response.read()

    return fetch_poster


def create_app(movies: list[dict], search, fetch_poster, poster_dir: Path = POSTER_DIR) -> FastAPI:
    app = FastAPI()
    by_id = {m["movie_id"]: m for m in movies}

    def placeholder(cache_control: str = "public, max-age=604800") -> Response:
        return Response(NO_POSTER, media_type="image/svg+xml", headers={"Cache-Control": cache_control})

    def known(movie_id: str) -> dict:
        if movie_id not in by_id:  # also keeps arbitrary ids out of poster_dir paths
            raise HTTPException(404, "movie not found")
        return by_id[movie_id]

    @app.get("/search")
    def search_movies(q: str = Query("", max_length=MAX_QUERY)) -> list[dict]:
        query = q.strip()
        return [movies[i] for i, _ in search(query)] if query else []

    @app.get("/movies/{movie_id}")
    def movie(movie_id: str) -> dict:
        return known(movie_id)

    @app.get("/posters/{movie_id}")
    def poster(movie_id: str) -> Response:
        known(movie_id)
        path, none = poster_dir / f"{movie_id}.jpg", poster_dir / f"{movie_id}.none"
        if not path.exists() and not none.exists():
            if fetch_poster is None:
                return placeholder("no-store")
            try:
                data = fetch_poster(movie_id)
            except (OSError, ValueError) as e:  # TMDB down or rate-limited: nothing cached, the next request retries
                print(f"poster {movie_id}: {e}", file=sys.stderr)
                return placeholder("no-store")
            poster_dir.mkdir(parents=True, exist_ok=True)
            if data is None:
                none.touch()  # TMDB has no poster: don't ask again
            else:
                part = path.with_suffix(f".{threading.get_ident()}.part")
                part.write_bytes(data)
                part.replace(path)  # atomic, so a parallel request never serves half a file
        if path.exists():
            return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800"})
        return placeholder()

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Search API for the fine-tuned model.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    if args.model == DEFAULT_MODEL and not Path(DEFAULT_MODEL).exists():
        print(f"{DEFAULT_MODEL} not found. Download minilm-movie-search from MyDrive/moviebot and unzip it into data/.", file=sys.stderr)
        return 2
    if not DB_PATH.exists():
        print(f"{DB_PATH} not found; build it with: python -m moviebot.build_db", file=sys.stderr)
        return 2

    from sentence_transformers import SentenceTransformer  # not installed in the project's test environment

    sys.path.insert(0, str(REPO))  # uv runs this script outside the project env; moviebot.sources is stdlib-only
    from moviebot.sources import TmdbClient

    key = os.environ.get("TMDB_KEY", "")
    if not key:
        print("TMDB_KEY is not set (run with --env-file .env); posters are off", file=sys.stderr)

    movies = load_movies(DB_PATH)
    print(f"embedding {len(movies):,} movies...", flush=True)
    search = make_search(SentenceTransformer(args.model), [movie_text(m) for m in movies])
    app = create_app(movies, search, make_fetch_poster(TmdbClient(key)) if key else None)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
