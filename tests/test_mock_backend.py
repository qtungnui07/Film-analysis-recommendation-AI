import pytest

from conftest import DATA_DIR, DOG_DIES, JUMP_SCARES, NEEDLES, SPIDERS, movies_with
from moviebot.contracts import NotFoundError, ToolError, vote_status
from moviebot.mock_backend import MockBackend


def test_vote_status_rule():
    assert vote_status(0, 0) == "unknown"
    assert vote_status(5, 5) == "present"  # ties fail safe
    assert vote_status(1, 9) == "absent"
    assert vote_status(9, 1) == "present"


def test_mock_data_is_consistent(backend):
    for movie_id, topic_id in backend._votes:
        assert movie_id in backend._movies
        assert topic_id in backend._topics
    for user_id in backend.user_ids:
        backend.get_user_profile(user_id)  # every pref points at a real topic


def test_fixture_has_the_cases_the_tests_rely_on(backend):
    assert len(movies_with(backend, SPIDERS, "present")) >= 3
    assert "Arrival" in movies_with(backend, SPIDERS, "unknown")


@pytest.mark.parametrize("query", ["spider movie", "giant spiders", "sci-fi", "fantasy adventure", ""])
def test_never_topics_are_always_excluded(backend, query):
    spider_movies = movies_with(backend, SPIDERS, "present")
    found = backend.search_movies("arachno", query, limit=10)  # note: no exclude_topic_ids
    assert found.results
    assert not {m.title for m in found.results} & spider_movies
    assert found.excluded_topics == ["spiders"]


def test_removed_count_reports_what_would_have_shown(backend):
    found = backend.search_movies("arachno", "spider movie", limit=3)
    assert found.removed_count >= 1
    assert backend.search_movies("fresh", "spider movie", limit=3).removed_count == 0


def test_unknown_is_flagged_not_treated_as_safe(backend):
    found = backend.search_movies("arachno", "sci-fi", genres=["Sci-Fi"], limit=10)
    by_title = {m.title: m for m in found.results}
    assert by_title["Arrival"].unknown_topics == ["spiders"]
    spiders = next(w for w in by_title["Arrival"].warnings if w.topic_id == SPIDERS)
    assert spiders.status == "unknown"
    assert by_title["Interstellar"].unknown_topics == []


def test_exclude_unknown_drops_movies_without_data():
    strict = MockBackend(DATA_DIR, exclude_unknown=True)
    titles = {m.title for m in strict.search_movies("arachno", "sci-fi", genres=["Sci-Fi"], limit=10).results}
    assert "Arrival" not in titles
    assert "Interstellar" in titles


def test_extra_exclusions_from_chat(backend):
    dog_movies = movies_with(backend, DOG_DIES, "present")
    found = backend.search_movies("fresh", "dog", exclude_topic_ids=[DOG_DIES], limit=10)
    assert not {m.title for m in found.results} & dog_movies
    assert found.excluded_topics == ["a dog dies"]


def test_unknown_topic_id_is_rejected(backend):
    with pytest.raises(ToolError, match="999"):
        backend.search_movies("fresh", "anything", exclude_topic_ids=[999])


def test_warn_topics_are_reported_on_results(backend):
    found = backend.search_movies("arachno", "alien", limit=3)
    alien = next(m for m in found.results if m.title == "Alien")
    jump = next(w for w in alien.warnings if w.topic_id == JUMP_SCARES)
    assert jump.status == "present"


def test_filters(backend):
    found = backend.search_movies("fresh", "", min_rating=8.5, genres=["science fiction"], limit=10)
    assert found.results
    assert all(m.rating >= 8.5 and "Sci-Fi" in m.genres for m in found.results)


def test_get_movie_warnings_by_title_is_case_insensitive(backend):
    warnings = backend.get_movie_warnings(title="arachnophobia")
    assert warnings.movie_id == "tt0099052"
    assert next(w for w in warnings.warnings if w.topic_id == SPIDERS).status == "present"


def test_get_movie_warnings_partial_title(backend):
    assert backend.get_movie_warnings(title="matrix").title == "The Matrix"


def test_get_movie_warnings_reports_requested_unknowns(backend):
    warnings = backend.get_movie_warnings(movie_id="tt2543164", topic_ids=[SPIDERS])
    assert next(w for w in warnings.warnings if w.topic_id == SPIDERS).status == "unknown"


def test_movie_with_no_data_at_all(backend):
    warnings = backend.get_movie_warnings(title="Primer", topic_ids=[SPIDERS, NEEDLES])
    assert [w.status for w in warnings.warnings] == ["unknown", "unknown"]


def test_not_found_suggests_close_titles(backend):
    with pytest.raises(NotFoundError) as e:
        backend.get_movie_warnings(title="Arachnofobia")
    assert "Arachnophobia" in e.value.hint


@pytest.mark.parametrize(
    "phrase, topic",
    [
        ("arachnophobia", "spiders"),
        ("I'm terrified of anything with eight legs", "spiders"),
        ("shots at the doctor", "needles"),
        ("I can't handle it when the dog dies", "a dog dies"),
        ("CLOWNS", "clowns"),
    ],
)
def test_map_phrase_to_topics(backend, phrase, topic):
    matches = backend.map_phrase_to_topics(phrase)
    assert matches and matches[0].name == topic


def test_map_phrase_no_match(backend):
    assert backend.map_phrase_to_topics("nothing in particular") == []


def test_profile(backend):
    profile = backend.get_user_profile("arachno")
    assert profile.topic_ids("never") == {SPIDERS}
    assert profile.topic_ids("warn") == {JUMP_SCARES}
    with pytest.raises(NotFoundError):
        backend.get_user_profile("nobody")
