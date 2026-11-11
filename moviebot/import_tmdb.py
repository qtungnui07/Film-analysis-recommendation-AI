"""Fill overviews and keywords from the Kaggle TMDB dump, data/TMDB_all_movies.csv, without API calls.

    python -m moviebot.import_tmdb    # about a minute; rerun after downloading a newer dump

Overviews already in the database are kept: only empty ones are filled. Keywords are set for
every catalogue movie the dump covers. Movies still without an overview afterwards can be
asked about one by one with fetch_overviews.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

from moviebot.config import REPO_ROOT, Settings

DUMP = REPO_ROOT / "data" / "TMDB_all_movies.csv"


def import_dump(db: sqlite3.Connection, dump: Path) -> tuple[int, int]:
    """Returns (overviews filled, movies given keywords)."""
    if "keywords" not in {row[1] for row in db.execute("PRAGMA table_info(movies)")}:
        db.execute("ALTER TABLE movies ADD COLUMN keywords TEXT")  # databases built before the column existed
    catalogue = {m for (m,) in db.execute("SELECT movie_id FROM movies")}
    csv.field_size_limit(2**31 - 1)  # cast lists run long
    with open(dump, encoding="utf-8", newline="") as f:
        rows = [(r["imdb_id"], r["overview"].strip(), r["keywords"].strip()) for r in csv.DictReader(f) if r["imdb_id"] in catalogue]
    with db:
        filled = db.executemany(
            "UPDATE movies SET overview = ? WHERE movie_id = ? AND coalesce(overview, '') = ''",
            [(overview, m) for m, overview, _ in rows if overview],
        ).rowcount
        keyed = db.executemany(
            "UPDATE movies SET keywords = ? WHERE movie_id = ?", [(keywords, m) for m, _, keywords in rows if keywords]
        ).rowcount
    return filled, keyed


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=f"Fill overviews and keywords from {DUMP.name}.").parse_args(argv)
    settings = Settings()
    if not settings.db_path.exists():
        print(f"{settings.db_path} not found; build it with: python -m moviebot.build_db", file=sys.stderr)
        return 2
    if not DUMP.exists():
        print(f"{DUMP} not found; see data/README.md for where to get it", file=sys.stderr)
        return 2
    db = sqlite3.connect(settings.db_path)
    filled, keyed = import_dump(db, DUMP)
    left = db.execute("SELECT count(*) FROM movies WHERE overview IS NULL").fetchone()[0]
    print(f"filled {filled:,} overviews and set keywords on {keyed:,} movies; {left:,} movies still have no overview")
    return 0


if __name__ == "__main__":
    sys.exit(main())
