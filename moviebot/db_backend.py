"""ToolBackend over the SQLite database that `python -m moviebot.build_db` makes.

Search and filtering are MockBackend's (keyword overlap) until the retrieval
layer lands. What this adds is the live part of the data layer:

- DTDD on demand: when a movie lacks a label the caller needs, or has none at
  all, and DTDD hasn't been asked about it yet, its votes are fetched once,
  turned into labels with the dataset's own 75% rule, and stored. Fresher API
  labels replace dataset ones for the same topic.
- TMDB fallback: a movie the user asks for that isn't in the catalogue is
  looked up on TMDB by IMDb id or exact title, and added.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from moviebot.contracts import (
    DtddLabel,
    Movie,
    NotFoundError,
    ProfileUpdateStrictness,
    ToolError,
    UserProfile,
    vote_label,
)
from moviebot.mock_backend import MockBackend
from moviebot.sources import DtddClient, TmdbClient

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE movies (
    movie_id        TEXT PRIMARY KEY,  -- IMDb id, e.g. tt0099052
    title           TEXT NOT NULL,
    year            INTEGER NOT NULL,
    rating          REAL NOT NULL,     -- IMDb average; TMDB's vote average for rows added from TMDB
    num_votes       INTEGER,           -- IMDb votes; NULL for rows added from TMDB
    genres          TEXT NOT NULL,     -- comma-separated IMDb genre names
    overview        TEXT,              -- NULL until fetch_overviews asks TMDB; '' when TMDB has none
    source          TEXT NOT NULL,     -- 'imdb' (built catalogue) or 'tmdb' (added when a user asked for it)
    dtdd_checked_at TEXT               -- when DTDD was last asked about this movie; NULL = never
);
CREATE TABLE topics (
    topic_id INTEGER PRIMARY KEY,  -- DTDD topic id; 10001 and up for topics DTDD has retired
    name     TEXT NOT NULL,
    question TEXT NOT NULL,        -- the ML-DDD question, e.g. 'Does the dog die'
    category TEXT NOT NULL,
    covers   TEXT NOT NULL,        -- JSON list of the narrower topic ids this topic includes
    synonyms TEXT NOT NULL         -- JSON list, for map_phrase_to_topics
);
CREATE TABLE labels (
    movie_id TEXT NOT NULL,
    topic_id INTEGER NOT NULL,
    label    TEXT NOT NULL CHECK (label IN ('yes', 'no', 'unclear')),
    source   TEXT NOT NULL,        -- 'ml-ddd' (dataset) or 'dtdd' (fetched on demand)
    PRIMARY KEY (movie_id, topic_id)
) WITHOUT ROWID;
CREATE TABLE genome_tags (
    tag_id INTEGER PRIMARY KEY,
    tag    TEXT NOT NULL
);
CREATE TABLE genome (
    movie_id  TEXT PRIMARY KEY,
    relevance BLOB NOT NULL        -- float32 array, one value per genome_tags row in tag_id order
);
CREATE TABLE users (
    user_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    topic_prefs TEXT NOT NULL      -- JSON {"<topic_id>": "never" | "warn"}, as in data/mock/users.json
);
"""

# TMDB genre names that differ from IMDb's. TMDB's "TV Movie" has no IMDb genre and is dropped.
_TMDB_GENRES = {"Science Fiction": "Sci-Fi"}
_TITLE_YEAR = re.compile(r"(.+?)\s*\((\d{4})\)")


class SqliteBackend(MockBackend):
    def __init__(
        self,
        db_path: Path,
        *,
        exclude_unknown: bool = False,
        dtdd: DtddClient | None = None,
        tmdb: TmdbClient | None = None,
    ):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"{db_path} not found; build it with: python -m moviebot.build_db")
        self._db = sqlite3.connect(db_path)
        self._dtdd = dtdd
        self._tmdb = tmdb
        topics = [
            {"topic_id": t, "name": n, "category": c, "covers": json.loads(cv), "synonyms": json.loads(s)}
            for t, n, c, cv, s in self._db.execute("SELECT topic_id, name, category, covers, synonyms FROM topics")
        ]
        rows = self._db.execute(
            "SELECT movie_id, title, year, rating, genres, overview, dtdd_checked_at FROM movies"
        ).fetchall()
        self._dtdd_checked = {row[0] for row in rows if row[6]}
        users = {
            u: {"user_id": u, "name": n, "topic_prefs": json.loads(p)}
            for u, n, p in self._db.execute("SELECT user_id, name, topic_prefs FROM users")
        }
        movies = [
            Movie(movie_id=m, title=t, year=y, rating=r, genres=g.split(",") if g else [], overview=o or "")
            for m, t, y, r, g, o, _ in rows
        ]
        # Labels start empty and are read per movie when needed: the full table is about a million rows.
        self._setup(topics, movies, {}, users, exclude_unknown=exclude_unknown)

    def update_user_preference(
        self,
        user_id: str,
        topic_id: int,
        strictness: ProfileUpdateStrictness,
    ) -> UserProfile:
        profile = super().update_user_preference(user_id, topic_id, strictness)
        with self._db:
            self._db.execute(
                "UPDATE users SET topic_prefs = ? WHERE user_id = ?",
                (json.dumps(self._users[user_id]["topic_prefs"]), user_id),
            )
        return profile

    # Labels, with DTDD on demand

    def _labels_for(self, movie_id: str, needed: set[int] = frozenset()) -> Mapping[int, DtddLabel]:
        if movie_id not in self._labels:
            self._labels[movie_id] = dict(
                self._db.execute("SELECT topic_id, label FROM labels WHERE movie_id = ?", (movie_id,))
            )
        labels = self._labels[movie_id]
        if self._dtdd and movie_id not in self._dtdd_checked and not (labels and needed <= labels.keys()):
            self._fetch_dtdd(movie_id)
        return labels

    def _fetch_dtdd(self, movie_id: str) -> None:
        try:
            votes = self._dtdd.votes(movie_id)
        except (OSError, ValueError, KeyError) as e:
            # shortcut: one failure turns DTDD off for this session, so a search doesn't stall on every movie;
            # restart to retry. Labels stay as stored, so missing ones read as "unknown", never "safe".
            log.warning("DTDD lookup failed (%s); using stored labels only", e)
            self._dtdd = None
            return
        fetched = {t: vote_label(yes, no) for t, (yes, no) in votes.items() if t in self._topics}
        fetched = {t: label for t, label in fetched.items() if label}
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO labels VALUES (?, ?, ?, 'dtdd')",
                [(movie_id, t, label) for t, label in fetched.items()],
            )
            self._db.execute("UPDATE movies SET dtdd_checked_at = ? WHERE movie_id = ?", (now, movie_id))
        self._labels[movie_id].update(fetched)
        self._dtdd_checked.add(movie_id)

    # Movies, with the TMDB fallback

    def _find_movie(self, movie_id: str | None, title: str | None) -> Movie:
        try:
            return super()._find_movie(movie_id, title)
        except NotFoundError:
            movie = self._from_tmdb(movie_id, title) if self._tmdb else None
            if movie is None:
                raise
            return movie

    def _from_tmdb(self, movie_id: str | None, title: str | None) -> Movie | None:
        """Look a movie up on TMDB and add it to the catalogue; None when TMDB has no confident match."""
        try:
            hit = self._tmdb.find_by_imdb(movie_id) if movie_id else self._tmdb_title_match(title)
            details = self._tmdb.details(hit["id"]) if hit else None
        except (OSError, ValueError, KeyError) as e:
            log.warning("TMDB lookup failed: %s", e)
            return None
        if not details or not details.get("imdb_id") or not details.get("release_date"):
            return None
        if details["imdb_id"] in self._movies:  # the catalogue has it under another title
            return self._movies[details["imdb_id"]]
        movie = Movie(
            movie_id=details["imdb_id"],
            title=details["title"],
            year=int(details["release_date"][:4]),
            rating=round(details.get("vote_average") or 0.0, 1),
            genres=[_TMDB_GENRES.get(g["name"], g["name"]) for g in details.get("genres", []) if g["name"] != "TV Movie"],
            overview=details.get("overview") or "",
        )
        with self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO movies (movie_id, title, year, rating, genres, overview, source)"
                " VALUES (?, ?, ?, ?, ?, ?, 'tmdb')",
                (movie.movie_id, movie.title, movie.year, movie.rating, ",".join(movie.genres), movie.overview),
            )
        self._index_movie(movie)
        return movie

    def _tmdb_title_match(self, title: str) -> dict | None:
        # Exact titles only: a near miss would attach another movie's warnings.
        match = _TITLE_YEAR.fullmatch(title.strip())
        name, year = (match.group(1), match.group(2)) if match else (title.strip(), None)
        wanted = name.casefold()
        exact = [
            r for r in self._tmdb.search(name)
            if wanted in (r.get("title", "").casefold(), r.get("original_title", "").casefold())
            and (year is None or (r.get("release_date") or "").startswith(year))
        ]
        if len(exact) > 1:
            raise ToolError(
                f"{title!r} matches several movies",
                hint="add the year in brackets, one of: "
                + "; ".join(f"{r['title']} ({(r.get('release_date') or '????')[:4]})" for r in exact[:5]),
            )
        return exact[0] if exact else None
