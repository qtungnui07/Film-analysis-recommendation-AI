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
"""Search page for testing the model trained by finetune_search.ipynb.

    uv run notebooks/search_site.py [--model data/minilm-movie-search] [--port 8002]

`--model` also takes a Hugging Face id, to compare another model on another port.
"""

import argparse
import html
import sqlite3
import sys
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = str(REPO / "data" / "minilm-movie-search")
DB_PATH = REPO / "data" / "moviebot.db"
TOP_K = 20
MAX_QUERY = 200

STYLE = """
:root { color-scheme: light dark; --muted: light-dark(#5b6168, #a9afb6); --line: light-dark(#dfe2e6, #383c42); }
body { font: 16px/1.5 system-ui, sans-serif; max-width: 44rem; margin: 0 auto; padding: 2rem 1rem; }
h1 { font-size: 1.25rem; margin: 0 0 1rem; }
form { display: flex; gap: .5rem; margin-bottom: 1.5rem; }
input, button { font: inherit; padding: .6rem .8rem; border: 1px solid var(--line); border-radius: .5rem; }
input { flex: 1; min-width: 0; }
button { cursor: pointer; }
ol { list-style: none; margin: 0; padding: 0; }
li { padding: 1rem 0; border-top: 1px solid var(--line); }
h2 { font-size: 1.05rem; margin: 0; }
h2 span, .meta { color: var(--muted); font-weight: 400; }
.meta { font-size: .875rem; margin: .1rem 0 .4rem; }
li p:last-child { margin: 0; }
"""


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


def result_item(movie: dict, score: float) -> str:
    details = [(movie["genres"] or "").replace(",", ", "), f"IMDb {movie['rating']}" if movie["rating"] else "", f"similarity {score:.2f}"]
    year = f" <span>{html.escape(str(movie['year']))}</span>" if movie["year"] else ""
    return (
        f"<li><h2>{html.escape(movie['title'])}{year}</h2>"
        f"<p class=meta>{html.escape(' · '.join(d for d in details if d))}</p>"
        f"<p>{html.escape(movie['overview'] or '')}</p></li>"
    )


def render(query: str, results: list[tuple[dict, float]]) -> str:
    items = "".join(result_item(movie, score) for movie, score in results)
    return (
        f"<!doctype html><html lang=en><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width, initial-scale=1'>"
        f"<title>Movie search</title><style>{STYLE}</style>"
        f"<h1>Movie search</h1>"
        f"<form><input type=search name=q aria-label='Search movies' maxlength={MAX_QUERY} autofocus value=\"{html.escape(query)}\">"
        f"<button>Search</button></form>"
        f"<ol>{items}</ol></html>"
    )


def create_app(movies: list[dict], search) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def page(q: str = Query("", max_length=MAX_QUERY)) -> str:
        query = q.strip()
        return render(query, [(movies[i], score) for i, score in search(query)] if query else [])

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Search page for the fine-tuned model.")
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

    movies = load_movies(DB_PATH)
    print(f"embedding {len(movies):,} movies...", flush=True)
    search = make_search(SentenceTransformer(args.model), [movie_text(m) for m in movies])
    uvicorn.run(create_app(movies, search), host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
