"""Shared tool contracts: DRAFT for team sign-off (charter, week 1).

Every layer builds against this file. The data and retrieval leads implement
`ToolBackend` for real in week 2; the agent and UI only consume these models.
Changing a signature here is a team decision, not a local refactor.
"""

from __future__ import annotations

from typing import Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, Field

WarningStatus = Literal["present", "absent", "unknown"]
# The ML-DDD label for one topic on one movie: DTDD voters clearly said yes, clearly said no, or split.
DtddLabel = Literal["yes", "no", "unclear"]
Strictness = Literal["never", "warn"]
ProfileUpdateStrictness = Literal["never", "warn", "fine"]


def vote_label(yes_votes: int, no_votes: int) -> DtddLabel | None:
    """Turn DTDD votes into the label the ML-DDD dataset uses, so API data matches the dataset.

    A clear answer needs strictly more than 75% of the votes. The dataset's
    README says "at least", but its generator code uses ">", and that code made
    the data. No votes gives None.
    """
    total = yes_votes + no_votes
    if total == 0:
        return None
    if yes_votes > 0.75 * total:
        return "yes"
    if no_votes > 0.75 * total:
        return "no"
    return "unclear"


def label_status(label: DtddLabel | None) -> WarningStatus:
    """The one rule for turning a DTDD label into a status.

    No label means "unknown", never "safe". "unclear" (voters split) counts as
    present, so a disputed topic fails safe.
    """
    if label is None:
        return "unknown"
    return "absent" if label == "no" else "present"


class Topic(BaseModel):
    topic_id: int
    name: str
    category: str
    # Narrower topics this one includes, e.g. "an animal dies" covers "a dog dies".
    covers: list[int] = Field(default_factory=list)


class TopicWarning(BaseModel):
    topic_id: int
    topic_name: str
    label: DtddLabel | None  # this topic's own DTDD label; None when nobody voted on it
    status: WarningStatus
    # Narrower topics that are present and so make this one present, e.g. ["a dog dies"].
    present_via: list[str] = Field(default_factory=list)


def narrower_topics(topic_id: int, topics: Mapping[int, Topic]) -> set[int]:
    """Every topic that `topic_id` covers, directly or through another covered topic."""
    found: set[int] = set()
    stack = list(topics[topic_id].covers)
    while stack:
        narrower = stack.pop()
        if narrower not in found:
            found.add(narrower)
            stack.extend(topics[narrower].covers)
    return found


def topic_warning(topic_id: int, topics: Mapping[int, Topic], labels: Mapping[int, DtddLabel]) -> TopicWarning:
    """One topic's warning for one movie, given that movie's labels.

    A topic is present when it, or any narrower topic it covers, is present, so
    "never: an animal dies" also excludes a movie labelled animal=no, dog=yes.
    Otherwise its own label decides. Absent narrower topics never make a broader
    one absent, because some other animal could still die.
    """
    via = sorted(
        topics[t].name for t in narrower_topics(topic_id, topics) if label_status(labels.get(t)) == "present"
    )
    label = labels.get(topic_id)
    return TopicWarning(
        topic_id=topic_id,
        topic_name=topics[topic_id].name,
        label=label,
        status="present" if via else label_status(label),
        present_via=via,
    )


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
    # Human-readable, data-backed reasons for the ranking. These are optional so
    # existing backends remain compatible with the contract.
    relevance_score: float = 0.0
    match_reasons: list[str] = Field(default_factory=list)


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
    def get_topic(self, topic_id: int) -> Topic:
        """Return one topic or raise ``NotFoundError`` for an unknown id."""
        ...

    def update_user_preference(
        self,
        user_id: str,
        topic_id: int,
        strictness: ProfileUpdateStrictness,
    ) -> UserProfile:
        """Update one preference; ``fine`` removes the saved preference."""
        ...

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
        topics, or any topic in `exclude_topic_ids`, has status "present", as
        `topic_warning` computes it (a present narrower topic counts). This
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

        Returns every topic with a label, plus each topic in `topic_ids` (as
        "unknown" if it has no label). Raises NotFoundError for an unknown movie.
        """
        ...

    def map_phrase_to_topics(self, phrase: str) -> list[TopicMatch]:
        """Map a free-text fear ("eight legs", "shots at the doctor") to topics, best first."""
        ...

    def get_user_profile(self, user_id: str) -> UserProfile: ...
