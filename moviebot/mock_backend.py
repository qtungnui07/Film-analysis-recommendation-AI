"""In-memory ToolBackend over data/mock/*.json, so the agent can run before the real data exists.

Search is keyword overlap, not semantic. Like a vector search, it always
returns the top-k. The phobia filtering, though, follows the real contract
exactly, because that is the part that has to be right.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Sequence

from moviebot.contracts import (
    Movie,
    MovieResult,
    MovieWarnings,
    NotFoundError,
    SearchResults,
    ToolError,
    Topic,
    TopicMatch,
    TopicPref,
    TopicWarning,
    UserProfile,
    ProfileUpdateStrictness,
    vote_status,
)

_STOPWORDS = {
    "a", "an", "and", "the", "of", "to", "in", "on", "for", "with", "about", "like",
    "is", "it", "me", "i", "my", "some", "something", "any", "movie", "movies", "film", "films",
    "good", "great", "want", "watch", "recommend", "please", "that", "this", "or",
}
_GENRE_ALIASES = {"sciencefiction": "scifi", "science": "scifi", "romcom": "romance", "animated": "animation"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _keywords(text: str) -> set[str]:
    return {_stem(w) for w in _words(text) if w not in _STOPWORDS}


def _genre_key(genre: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", genre.lower())
    return _GENRE_ALIASES.get(key, key)


class MockBackend:
    def __init__(self, data_dir: Path, *, exclude_unknown: bool = False):
        """`exclude_unknown=True` also drops movies whose excluded topics have no votes."""
        self.exclude_unknown = exclude_unknown

        def load(name: str) -> dict:
            return json.loads((Path(data_dir) / name).read_text(encoding="utf-8"))

        topics = load("topics.json")["topics"]
        self._topics = {t["topic_id"]: Topic(**t) for t in topics}
        self._synonyms = {
            t["topic_id"]: {" ".join(_words(s)) for s in [t["name"], *t["synonyms"]]} for t in topics
        }
        self._movies = {m["movie_id"]: Movie(**m) for m in load("movies.json")["movies"]}
        self._votes = {(v["movie_id"], v["topic_id"]): (v["yes"], v["no"]) for v in load("triggers.json")["votes"]}
        self._users = {u["user_id"]: u for u in load("users.json")["users"]}
        self._movie_keywords = {
            m.movie_id: _keywords(" ".join([m.title, *m.genres, m.overview])) for m in self._movies.values()
        }

    @property
    def user_ids(self) -> list[str]:
        return list(self._users)

    # ToolBackend

    def get_topic(self, topic_id: int) -> Topic:
        try:
            return self._topics[topic_id]
        except KeyError as exc:
            raise NotFoundError(f"no topic with id {topic_id}", hint="use map_phrase_to_topics to find a topic id") from exc

    def update_user_preference(
        self,
        user_id: str,
        topic_id: int,
        strictness: ProfileUpdateStrictness,
    ) -> UserProfile:
        self.get_topic(topic_id)
        user = self._users.get(user_id)
        if user is None:
            raise NotFoundError(f"no user {user_id!r}", hint=f"known users: {', '.join(self._users)}")
        prefs = user["topic_prefs"]
        if strictness == "fine":
            prefs.pop(str(topic_id), None)
        else:
            prefs[str(topic_id)] = strictness
        return self.get_user_profile(user_id)

    def search_movies(
        self,
        user_id: str,
        query: str,
        min_rating: float | None = None,
        genres: Sequence[str] | None = None,
        exclude_topic_ids: Sequence[int] = (),
        limit: int = 5,
    ) -> SearchResults:
        self._check_topics(exclude_topic_ids)
        profile = self.get_user_profile(user_id)
        excluded = profile.topic_ids("never") | set(exclude_topic_ids)
        reported = sorted(excluded | profile.topic_ids("warn"))
        wanted_genres = {_genre_key(g) for g in genres or []}
        query_words = _keywords(query)

        candidates = [
            m for m in self._movies.values()
            if (min_rating is None or m.rating >= min_rating)
            and (not wanted_genres or wanted_genres & {_genre_key(g) for g in m.genres})
        ]
        def relevance(movie: Movie) -> float:
            overlap = len(query_words & self._movie_keywords[movie.movie_id])
            exact_title = 1.0 if query.strip().casefold() == movie.title.casefold() else 0.0
            # A lightweight hybrid score for the mock: lexical relevance is
            # primary, while rating breaks ties and exact titles win strongly.
            return exact_title * 10.0 + overlap * 2.0 + movie.rating / 10.0

        candidates.sort(key=relevance, reverse=True)

        results: list[MovieResult] = []
        removed = 0
        for movie in candidates:
            if len(results) == limit:
                break
            statuses = {t: self._warning(movie.movie_id, t).status for t in excluded}
            unknown = sorted(self._topics[t].name for t, s in statuses.items() if s == "unknown")
            if "present" in statuses.values() or (self.exclude_unknown and unknown):
                removed += 1
                continue
            score = relevance(movie)
            reasons: list[str] = []
            if query.strip().casefold() == movie.title.casefold():
                reasons.append("exact title match")
            overlap = len(query_words & self._movie_keywords[movie.movie_id])
            if overlap:
                reasons.append(f"matches {overlap} query keyword{'s' if overlap != 1 else ''}")
            if min_rating is not None:
                reasons.append(f"rating {movie.rating:.1f} meets your minimum")
            if wanted_genres:
                reasons.append("matches the requested genre")
            if not reasons:
                reasons.append("best available match for the request")
            results.append(
                MovieResult(
                    **movie.model_dump(),
                    warnings=[self._warning(movie.movie_id, t) for t in reported],
                    unknown_topics=unknown,
                    relevance_score=round(score, 3),
                    match_reasons=reasons,
                )
            )
        return SearchResults(
            results=results,
            excluded_topics=sorted(self._topics[t].name for t in excluded),
            removed_count=removed,
        )

    def get_movie_warnings(
        self,
        movie_id: str | None = None,
        title: str | None = None,
        topic_ids: Sequence[int] | None = None,
    ) -> MovieWarnings:
        self._check_topics(topic_ids or ())
        movie = self._find_movie(movie_id, title)
        voted = {t for (m, t) in self._votes if m == movie.movie_id}
        return MovieWarnings(
            movie_id=movie.movie_id,
            title=movie.title,
            year=movie.year,
            warnings=[self._warning(movie.movie_id, t) for t in sorted(voted | set(topic_ids or ()))],
        )

    def map_phrase_to_topics(self, phrase: str) -> list[TopicMatch]:
        text = " ".join(_words(phrase))
        matches = []
        for topic_id, synonyms in self._synonyms.items():
            if text in synonyms:
                score = 1.0
            elif any(f" {s} " in f" {text} " for s in synonyms):
                score = 0.8
            else:
                continue
            matches.append(TopicMatch(topic_id=topic_id, name=self._topics[topic_id].name, score=score))
        return sorted(matches, key=lambda m: (-m.score, m.topic_id))

    def get_user_profile(self, user_id: str) -> UserProfile:
        user = self._users.get(user_id)
        if user is None:
            raise NotFoundError(f"no user {user_id!r}", hint=f"known users: {', '.join(self._users)}")
        prefs = [
            TopicPref(topic_id=int(t), name=self._topics[int(t)].name, strictness=s)
            for t, s in user["topic_prefs"].items()
        ]
        return UserProfile(user_id=user_id, name=user["name"], topic_prefs=prefs)

    # Helpers

    def _warning(self, movie_id: str, topic_id: int) -> TopicWarning:
        yes, no = self._votes.get((movie_id, topic_id), (0, 0))
        return TopicWarning(
            topic_id=topic_id,
            topic_name=self._topics[topic_id].name,
            yes_votes=yes,
            no_votes=no,
            status=vote_status(yes, no),
        )

    def _check_topics(self, topic_ids: Sequence[int]) -> None:
        unknown = sorted(set(topic_ids) - self._topics.keys())
        if unknown:
            raise ToolError(
                f"unknown topic id(s): {unknown}",
                hint="get topic ids from map_phrase_to_topics or get_user_profile",
            )

    def _find_movie(self, movie_id: str | None, title: str | None) -> Movie:
        if movie_id:
            if movie_id in self._movies:
                return self._movies[movie_id]
            if not title:
                raise NotFoundError(f"no movie with id {movie_id!r}", hint="use search_movies to find the movie_id")
        if not title:
            raise ToolError("give a movie_id or a title")

        wanted = title.strip().casefold()
        by_title = {m.title.casefold(): m for m in self._movies.values()}
        if wanted in by_title:
            return by_title[wanted]
        partial = [m for key, m in by_title.items() if wanted in key]
        if len(partial) == 1:
            return partial[0]
        if partial:
            raise ToolError(
                f"{title!r} matches several movies",
                hint="pick one: " + "; ".join(f"{m.title} ({m.year}, {m.movie_id})" for m in partial),
            )
        close = difflib.get_close_matches(wanted, by_title, n=3, cutoff=0.5)
        hint = "did you mean: " + "; ".join(by_title[c].title for c in close) if close else "try search_movies"
        raise NotFoundError(f"no movie titled {title!r}", hint=hint)
