"""Fill movies.overview from TMDB, most-voted movies first.

    python -m moviebot.fetch_overviews --limit 200

Resumable: each run only asks about movies TMDB hasn't been asked about yet,
and saves after every movie. A movie TMDB has no overview for gets '' so it
isn't asked again.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

from moviebot.config import Settings
from moviebot.sources import TmdbClient


def fetch(db: sqlite3.Connection, tmdb: TmdbClient, limit: int) -> tuple[int, int]:
    """Fetch up to `limit` overviews. Returns (movies asked about, overviews found)."""
    todo = db.execute(
        "SELECT movie_id FROM movies WHERE overview IS NULL ORDER BY num_votes DESC LIMIT ?", (limit,)
    ).fetchall()
    found = 0
    for asked, (movie_id,) in enumerate(todo):
        try:
            hit = tmdb.find_by_imdb(movie_id)
        except (OSError, ValueError) as e:
            print(f"stopped at {movie_id}: {e}", file=sys.stderr)
            return asked, found
        overview = (hit or {}).get("overview") or ""
        with db:
            db.execute("UPDATE movies SET overview = ? WHERE movie_id = ?", (overview, movie_id))
        found += bool(overview)
    return len(todo), found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch plot overviews from TMDB into the movie database.")
    parser.add_argument("--limit", type=int, default=25, help="how many movies to ask TMDB about (default 25)")
    args = parser.parse_args(argv)
    settings = Settings()
    key = settings.tmdb_key.get_secret_value()
    if not key:
        print("TMDB_KEY is not set in .env", file=sys.stderr)
        return 2
    if not settings.db_path.exists():
        print(f"{settings.db_path} not found; build it with: python -m moviebot.build_db", file=sys.stderr)
        return 2
    db = sqlite3.connect(settings.db_path)
    asked, found = fetch(db, TmdbClient(key), args.limit)
    left = db.execute("SELECT count(*) FROM movies WHERE overview IS NULL").fetchone()[0]
    print(f"asked TMDB about {asked} movies, {found} had an overview; {left:,} movies not asked yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
