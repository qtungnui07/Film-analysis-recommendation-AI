import pytest

from conftest import ANIMAL_DIES, DATA_DIR, DOG_DIES, JUMP_SCARES, NEEDLES, SPIDERS, movies_with
from moviebot.contracts import NotFoundError, Topic, ToolError, label_status, topic_warning, vote_label
from moviebot.mock_backend import MockBackend


def test_vote_label_uses_the_datasets_75_percent_rule():
    assert vote_label(0, 0) is None
    assert vote_label(76, 24) == "yes"
    assert vote_label(75, 25) == "unclear"  # strictly more than 75%, as the ML-DDD generator does
    assert vote_label(5, 5) == "unclear"
    assert vote_label(1, 9) == "no"


def test_label_status_rule():
    assert label_status(None) == "unknown"
    assert label_status("yes") == "present"
    assert label_status("unclear") == "present"  # split votes fail safe
    assert label_status("no") == "absent"


def test_broad_topic_covers_narrower_ones():
    topics = {
        1: Topic(topic_id=1, name="a non-human character dies", category="Death", covers=[2]),
        2: Topic(topic_id=2, name="an animal dies", category="Animal Death", covers=[3, 4]),
        3: Topic(topic_id=3, name="a dog dies", category="Animal Death"),
        4: Topic(topic_id=4, name="a cat dies", category="Animal Death"),
    }
    animal_no_dog_yes = topic_warning(2, topics, {2: "no", 3: "yes"})
    assert animal_no_dog_yes.status == "present"
    assert animal_no_dog_yes.label == "no"
    assert animal_no_dog_yes.present_via == ["a dog dies"]
    assert topic_warning(1, topics, {3: "unclear"}).present_via == ["a dog dies"]  # through two levels
    assert topic_warning(2, topics, {3: "no", 4: "no"}).status == "unknown"  # other animals can still die
    assert topic_warning(3, topics, {2: "yes"}).status == "unknown"  # covering only goes broad to narrow


def test_mock_data_is_consistent(backend):
    for movie_id, labels in backend._labels.items():
        assert movie_id in backend._movies
        assert labels.keys() <= backend._topics.keys()
        assert set(labels.values()) <= {"yes", "no", "unclear"}
    for user_id in backend.user_ids:
        backend.get_user_profile(user_id)  # every pref points at a real topic


def test_never_broad_topic_excludes_movies_with_a_narrower_one(backend):
    # Marley & Me is labelled "no animal dies" but "a dog dies"; John Wick has only the dog label.
    found = backend.search_movies("fresh", "dog", exclude_topic_ids=[ANIMAL_DIES], limit=10)
    assert not {m.title for m in found.results} & {"Marley & Me", "John Wick"}
    marley = backend.get_movie_warnings(title="Marley & Me", topic_ids=[ANIMAL_DIES])
    animal = next(w for w in marley.warnings if w.topic_id == ANIMAL_DIES)
    assert (animal.status, animal.present_via) == ("present", ["a dog dies"])


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
