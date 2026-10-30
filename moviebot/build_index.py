"""Embed data/moviebot.db into the Chroma index that semantic search uses. See data/README.md.

    python -m moviebot.build_index    # after build_db; rerun after fetch_overviews

Cheap to rerun: only texts that changed, such as a movie with a new overview,
are embedded again. The first run downloads all-MiniLM-L6-v2 (about 80 MB).
"""

from __future__ import annotations

import argparse
import heapq
import json
import sqlite3
import sys
from array import array
from collections import Counter

from moviebot.config import Settings
from moviebot.contracts import Movie
from moviebot.retrieval import VectorIndex, movie_document

TOP_TAGS = 10  # genome tags per movie text, most relevant first


def build(db: sqlite3.Connection, index: VectorIndex) -> tuple[int, int]:
    """Sync the index with the database. Returns (movie texts embedded, topic phrases embedded)."""
    tags = [t for (t,) in db.execute("SELECT tag FROM genome_tags ORDER BY tag_id")]
    top_tags = {}
    for movie_id, blob in db.execute("SELECT movie_id, relevance FROM genome"):
        vector = array("f", blob)
        top_tags[movie_id] = [tags[i] for i in heapq.nlargest(TOP_TAGS, range(len(vector)), key=vector.__getitem__)]
    movies = {
        m: movie_document(
            Movie(movie_id=m, title=t, year=y, rating=r, genres=g.split(",") if g else [], overview=o or ""),
            top_tags.get(m, []),
        )
        for m, t, y, r, g, o in db.execute("SELECT movie_id, title, year, rating, genres, overview FROM movies")
    }
    topic_phrases = {t: {n.lower(), *json.loads(s)} for t, n, s in db.execute("SELECT topic_id, name, synonyms FROM topics")}
    # DTDD keywords like "fear" (13 topics) or "death" (9) can't tell topics apart, and they tie unrelated
    # topics in the results. Embed only phrases unique to one topic; the synonym rules still use them all.
    shared = Counter(p for ps in topic_phrases.values() for p in ps)
    phrases, metadatas = {}, {}
    for topic_id, ps in topic_phrases.items():
        for i, phrase in enumerate(sorted(p for p in ps if shared[p] == 1)):
            phrases[f"{topic_id}/{i}"] = phrase
            metadatas[f"{topic_id}/{i}"] = {"topic_id": topic_id}
    return VectorIndex.sync(index.movies, movies), VectorIndex.sync(index.topics, phrases, metadatas)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Embed the movie database into the semantic search index.").parse_args(argv)
    settings = Settings()
    if not settings.db_path.exists():
        print(f"{settings.db_path} not found; build it with: python -m moviebot.build_db", file=sys.stderr)
        return 2
    print("embedding new and changed texts with all-MiniLM-L6-v2...", flush=True)
    movies, phrases = build(sqlite3.connect(settings.db_path), VectorIndex(settings.index_path))
    print(f"embedded {movies:,} movie texts and {phrases:,} topic phrases into {settings.index_path}; the rest were unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
