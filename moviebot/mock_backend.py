"""In-memory ToolBackend over data/mock/*.json, so the agent can run before the real data exists.

Search is keyword overlap, not semantic. Like a vector search, it always
returns the top-k. The phobia filtering, though, follows the real contract
exactly, because that is the part that has to be right.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping, Sequence

from rapidfuzz import fuzz, process, utils

from moviebot.contracts import (
    DtddLabel,
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
    narrower_topics,
    topic_warning,
)

_STOPWORDS = {
    "a", "an", "and", "the", "of", "to", "in", "on", "for", "with", "about", "like",
    "is", "it", "me", "i", "my", "some", "something", "any", "movie", "movies", "film", "films",
    "good", "great", "want", "watch", "recommend", "please", "that", "this", "or",
}
_GENRE_ALIASES = {"sciencefiction": "scifi", "science": "scifi", "romcom": "romance", "animated": "animation"}
_TITLE_YEAR = re.compile(r"(.+?)\s*\((\d{4})\)")
# rapidfuzz WRatio: a title containing the query scores 90, typos score lower. Only "sure" matches resolve.
_TITLE_SURE, _TITLE_CLOSE = 90, 60


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
        """`exclude_unknown=True` also drops movies whose excluded topics have no label."""

        def load(name: str) -> dict:
            return json.loads((Path(data_dir) / name).read_text(encoding="utf-8"))

        labels: dict[str, dict[int, DtddLabel]] = {}
        for row in load("triggers.json")["labels"]:
            labels.setdefault(row["movie_id"], {})[row["topic_id"]] = row["label"]
        self._setup(
            load("topics.json")["topics"],
            [Movie(**m) for m in load("movies.json")["movies"]],
            labels,
            {u["user_id"]: u for u in load("users.json")["users"]},
            exclude_unknown=exclude_unknown,
        )

    def _setup(
        self,
        topics: list[dict],
        movies: list[Movie],
        labels: dict[str, dict[int, DtddLabel]],
        users: dict[str, dict],
        *,
        exclude_unknown: bool,
    ) -> None:
        """Shared by every loader. `topics` rows are Topic fields plus `synonyms`; `labels` is per movie."""
        self.exclude_unknown = exclude_unknown
        self._topics = {t["topic_id"]: Topic(**t) for t in topics}
        self._synonyms = {
            t["topic_id"]: {" ".join(_words(s)) for s in [t["name"], *t["synonyms"]]} for t in topics
        }
        self._labels = labels
        self._users = users
        self._movies: dict[str, Movie] = {}
        self._movie_keywords: dict[str, set[str]] = {}
        for movie in movies:
            self._index_movie(movie)

    def _index_movie(self, movie: Movie) -> None:
        self._movies[movie.movie_id] = movie
        self._movie_keywords[movie.movie_id] = _keywords(" ".join([movie.title, *movie.genres, movie.overview]))

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

        candidates = [
            m for m in self._movies.values()
            if (min_rating is None or m.rating >= min_rating)
            and (not wanted_genres or wanted_genres & {_genre_key(g) for g in m.genres})
        ]

        results: list[MovieResult] = []
        removed = 0
        for movie, score, reasons in self._rank(query, candidates):
            if len(results) == limit:
                break
            statuses = {t: self._warning(movie.movie_id, t).status for t in excluded}
            unknown = sorted(self._topics[t].name for t, s in statuses.items() if s == "unknown")
            if "present" in statuses.values() or (self.exclude_unknown and unknown):
                removed += 1
                continue
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
        voted = set(self._labels_for(movie.movie_id))
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

    def _rank(self, query: str, candidates: list[Movie]) -> list[tuple[Movie, float, list[str]]]:
        """Candidates best first, each with a score and match reasons. Keyword overlap here; SqliteBackend uses embeddings."""
        query_words = _keywords(query)
        ranked = []
        for movie in candidates:
            exact_title = query.strip().casefold() == movie.title.casefold()
            overlap = len(query_words & self._movie_keywords[movie.movie_id])
            reasons = ["exact title match"] if exact_title else []
            if overlap:
                reasons.append(f"matches {overlap} query keyword{'s' if overlap != 1 else ''}")
            # A lightweight hybrid score for the mock: lexical relevance is
            # primary, while rating breaks ties and exact titles win strongly.
            ranked.append((movie, exact_title * 10.0 + overlap * 2.0 + movie.rating / 10.0, reasons))
        return sorted(ranked, key=lambda r: r[1], reverse=True)

    def _warning(self, movie_id: str, topic_id: int) -> TopicWarning:
        needed = {topic_id} | narrower_topics(topic_id, self._topics)
        return topic_warning(topic_id, self._topics, self._labels_for(movie_id, needed))

    def _labels_for(self, movie_id: str, needed: set[int] = frozenset()) -> Mapping[int, DtddLabel]:
        """One movie's labels. `needed` lets a backend with a live source fetch the ones it lacks."""
        return self._labels.get(movie_id, {})

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
        return self._match_title(title)

    def _match_title(self, title: str) -> Movie:
        """The one catalogue movie `title` names; "Title (YYYY)" picks between remakes.

        A near miss would attach another movie's warnings, so it only resolves
        when exactly one title is an exact or sure match; typos get hints.
        """
        match = _TITLE_YEAR.fullmatch(title.strip())
        name, year = (match.group(1), int(match.group(2))) if match else (title.strip(), None)
        pool = {m.movie_id: m.title for m in self._movies.values() if year is None or m.year == year}
        sure = [i for i, t in pool.items() if t.casefold() == name.casefold()]
        close = []
        if not sure:
            close = process.extract(
                name, pool, scorer=fuzz.WRatio, processor=utils.default_process, limit=5, score_cutoff=_TITLE_CLOSE
            )
            sure = [i for _, score, i in close if score >= _TITLE_SURE]
        if len(sure) == 1:
            return self._movies[sure[0]]
        if sure:
            raise ToolError(
                f"{title!r} matches several movies",
                hint="use one movie_id, or add the year in brackets: "
                + "; ".join(f"{self._movies[i].title} ({self._movies[i].year}, {i})" for i in sure),
            )
        hint = (
            "did you mean: " + "; ".join(f"{self._movies[i].title} ({self._movies[i].year})" for _, _, i in close[:3])
            if close else "try search_movies"
        )
        raise NotFoundError(f"no movie titled {title!r}", hint=hint)
