"""Semantic retrieval: a local Chroma store embedded with all-MiniLM-L6-v2, built by `python -m moviebot.build_index`.

Two collections:

- movies: one short text per movie (title, year, genres, top genome tags,
  overview). Only 3 movies have an overview so far, so the genome tags carry
  most of the meaning.
- topics: one text per topic name or synonym, so map_phrase_to_topics can
  match "eight legs" to spiders.

Retrieval only ranks. Phobia filtering stays deterministic in the backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from moviebot.contracts import Movie

_BATCH = 1000


def movie_document(movie: Movie, tags: Sequence[str]) -> str:
    parts = [f"{movie.title} ({movie.year})", ", ".join(movie.genres), ", ".join(tags), movie.overview]
    return ". ".join(p for p in parts if p)


class VectorIndex:
    def __init__(self, path: Path, *, min_similarity: float = 0.0, embedding_function=None):
        """`embedding_function` defaults to Chroma's own all-MiniLM-L6-v2 over ONNX (downloads ~80 MB once)."""
        client = chromadb.PersistentClient(path=str(path), settings=chromadb.Settings(anonymized_telemetry=False))
        ef = embedding_function or DefaultEmbeddingFunction()
        config = {"hnsw": {"space": "cosine"}}
        self.movies = client.get_or_create_collection("movies", embedding_function=ef, configuration=config)
        self.topics = client.get_or_create_collection("topics", embedding_function=ef, configuration=config)
        self.min_similarity = min_similarity

    def similar_movies(self, text: str) -> dict[str, float]:
        """Cosine similarity of every indexed movie to `text`, dropping those under `min_similarity`."""
        hits = _query(self.movies, text)
        return {m: s for m, s, _ in hits if s >= self.min_similarity}

    def similar_topics(self, phrase: str) -> dict[int, float]:
        """Each topic's best similarity over its name and synonyms."""
        best: dict[int, float] = {}
        for _, s, meta in _query(self.topics, phrase):
            best[meta["topic_id"]] = max(best.get(meta["topic_id"], -1.0), s)
        return best

    def document(self, movie_id: str) -> str | None:
        docs = self.movies.get(ids=[movie_id], include=["documents"])["documents"]
        return docs[0] if docs else None

    def add_movie(self, movie: Movie) -> None:
        """Index a movie added after the build (from TMDB). It has no genome tags."""
        self.movies.upsert(ids=[movie.movie_id], documents=[movie_document(movie, [])])

    @staticmethod
    def sync(collection, docs: dict[str, str], metadatas: dict[str, dict] | None = None) -> int:
        """Make `collection` hold exactly `docs`: embed new or changed ones, delete the rest. Returns how many were embedded."""
        have = collection.get(include=["documents"])
        current = dict(zip(have["ids"], have["documents"]))
        if stale := [i for i in current if i not in docs]:
            collection.delete(ids=stale)
        changed = [i for i, d in docs.items() if current.get(i) != d]
        for start in range(0, len(changed), _BATCH):
            ids = changed[start:start + _BATCH]
            collection.upsert(
                ids=ids,
                documents=[docs[i] for i in ids],
                metadatas=[metadatas[i] for i in ids] if metadatas else None,
            )
        return len(changed)


def _query(collection, text: str) -> list[tuple[str, float, dict]]:
    count = collection.count()
    if not count:
        return []
    hits = collection.query(query_texts=[text], n_results=count, include=["distances", "metadatas"])
    # Chroma's cosine distance is 1 - cosine similarity.
    return [(i, 1.0 - d, m) for i, d, m in zip(hits["ids"][0], hits["distances"][0], hits["metadatas"][0])]
