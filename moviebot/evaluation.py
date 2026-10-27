"""Small, deterministic evaluation helpers for the project report.

The helpers operate on tool contracts rather than LLM text, so safety metrics
are reproducible and can be used for the RAG-only ablation as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from moviebot.contracts import SearchResults, TopicMatch, TopicWarning


@dataclass(frozen=True)
class SafetyMetrics:
    recommendations: int
    violations: int
    unknown_topic_movies: int

    @property
    def safety_recall(self) -> float:
        """Fraction of recommendations that did not violate a never topic."""
        if self.recommendations == 0:
            return 1.0
        return (self.recommendations - self.violations) / self.recommendations


def measure_safety(results: Iterable[SearchResults], never_topic_ids: Sequence[int]) -> SafetyMetrics:
    forbidden = set(never_topic_ids)
    recommendations = violations = unknown_movies = 0
    for batch in results:
        for movie in batch.results:
            recommendations += 1
            if any(w.topic_id in forbidden and w.status == "present" for w in movie.warnings):
                violations += 1
            if movie.unknown_topics:
                unknown_movies += 1
    return SafetyMetrics(recommendations, violations, unknown_movies)


def warning_accuracy(stated_topic_ids: Sequence[int], warnings: Sequence[TopicWarning]) -> float:
    """Measure whether every warning stated by the bot is data-supported."""
    stated = set(stated_topic_ids)
    if not stated:
        return 1.0
    supported = {w.topic_id for w in warnings if w.status == "present"}
    return len(stated & supported) / len(stated)


def phrase_mapping_accuracy(
    cases: Iterable[tuple[str, int]], mapper,
) -> float:
    """Evaluate free-text topic mapping against ``(phrase, expected_id)`` cases."""
    cases = list(cases)
    if not cases:
        return 1.0
    correct = 0
    for phrase, expected_id in cases:
        matches: list[TopicMatch] = mapper(phrase)
        if matches and matches[0].topic_id == expected_id:
            correct += 1
    return correct / len(cases)
