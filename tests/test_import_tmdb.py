import csv
import sqlite3

from moviebot.import_tmdb import import_dump

COLUMNS = ["id", "title", "imdb_id", "overview", "keywords", "cast"]


def test_import_fills_empty_overviews_and_sets_keywords(tmp_path):
    db = sqlite3.connect(tmp_path / "m.db")
    # The schema before the keywords column existed.
    db.execute("CREATE TABLE movies (movie_id TEXT PRIMARY KEY, overview TEXT)")
    db.executemany("INSERT INTO movies VALUES (?, ?)", [("tt1", "fetched"), ("tt2", None), ("tt3", ""), ("tt4", None)])
    with open(tmp_path / "dump.csv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f, quoting=csv.QUOTE_ALL).writerows([
            COLUMNS,
            ["1", "One", "tt1", "From the dump", "time travel|heist", "A, B"],
            ["2", "Two", "tt2", "Line one\nline two", "", ""],
            ["3", "Three", "tt3", " Filled ", "helsinki, finland|prison", ""],
            ["4", "Four", "tt4", "", "", ""],
            ["9", "Not in catalogue", "tt9", "Ignored", "ignored", ""],
            ["10", "No IMDb id", "", "Ignored", "ignored", ""],
        ])

    assert import_dump(db, tmp_path / "dump.csv") == (2, 2)
    assert db.execute("SELECT movie_id, overview, keywords FROM movies ORDER BY 1").fetchall() == [
        ("tt1", "fetched", "time travel|heist"),
        ("tt2", "Line one\nline two", None),
        ("tt3", "Filled", "helsinki, finland|prison"),
        ("tt4", None, None),
    ]
    assert import_dump(db, tmp_path / "dump.csv") == (0, 2)
