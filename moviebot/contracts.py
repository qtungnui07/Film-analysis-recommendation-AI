"""Shared tool contracts: DRAFT for team sign-off (charter, week 1).

Every layer builds against this file. The data and retrieval leads implement
`ToolBackend` for real in week 2; the agent and UI only consume these models.
Changing a signature here is a team decision, not a local refactor.
"""

from __future__ import annotations

from typing import Literal, Protocol, Sequence

from pydantic import BaseModel

WarningStatus = Literal["present", "absent", "unknown"]
Strictness = Literal["never", "warn"]


def vote_status(yes_votes: int, no_votes: int) -> WarningStatus:
    """The one rule for turning DTDD votes into a status.

    No votes means "unknown", never "safe". A tie counts as present, so a
    disputed topic fails safe. Week 3 adds confidence scores on top of this.
    """
    if yes_votes == 0 and no_votes == 0:
        return "unknown"
    return "present" if yes_votes >= no_votes else "absent"


class Topic(BaseModel):
    topic_id: int
    name: str
    category: str


class TopicWarning(BaseModel):
    topic_id: int
    topic_name: str
    yes_votes: int
    no_votes: int
    status: WarningStatus


class Movie(BaseModel):
    movie_id: str  # IMDb id, e.g. "tt0099052"
    title: str
    year: int
    rating: float
    genres: list[str]
    overview: str


class MovieResult(Movie):
    # Warnings for the user's "never" and "warn" topics plus any extra exclusions.
    warnings: list[TopicWarning]
    # Names of excluded topics with no DTDD votes for this movie: the bot must say "no data".
    unknown_topics: list[str]


class SearchResults(BaseModel):
    results: list[MovieResult]
    excluded_topics: list[str]  # names of every topic that was filtered out
    removed_count: int  # matches that would have been shown but contained an excluded topic


class MovieWarnings(BaseModel):
    movie_id: str
    title: str
    year: int
    warnings: list[TopicWarning]


class TopicMatch(BaseModel):
    topic_id: int
    name: str
    score: float  # 0..1, higher is a closer match


class TopicPref(BaseModel):
    topic_id: int
    name: str
    strictness: Strictness


class UserProfile(BaseModel):
    user_id: str
    name: str
    topic_prefs: list[TopicPref]

    def topic_ids(self, strictness: Strictness | None = None) -> set[int]:
        return {p.topic_id for p in self.topic_prefs if strictness in (None, p.strictness)}


class ToolError(Exception):
    """An expected failure the LLM can recover from, e.g. a bad title or topic id."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint


class NotFoundError(ToolError):
    pass


class ToolBackend(Protocol):
    def search_movies(
        self,
        user_id: str,
        query: str,
        min_rating: float | None = None,
        genres: Sequence[str] | None = None,
        exclude_topic_ids: Sequence[int] = (),
        limit: int = 5,
    ) -> SearchResults:
        """Find movies for `query`.

        Implementations MUST remove every movie where one of the user's "never"
        topics, or any topic in `exclude_topic_ids`, has status "present". This
        must happen whatever the caller passes. Movies where such a topic is
        "unknown" are kept and listed in `unknown_topics`.
        """
        ...

    def get_movie_warnings(
        self,
        movie_id: str | None = None,
        title: str | None = None,
        topic_ids: Sequence[int] | None = None,
    ) -> MovieWarnings:
        """Warnings for one movie, looked up by IMDb id or title.

        Returns every topic with votes, plus each topic in `topic_ids` (as
        "unknown" if it has no votes). Raises NotFoundError for an unknown movie.
        """
        ...

    def map_phrase_to_topics(self, phrase: str) -> list[TopicMatch]:
        """Map a free-text fear ("eight legs", "shots at the doctor") to topics, best first."""
        ...

    def get_user_profile(self, user_id: str) -> UserProfile: ...
