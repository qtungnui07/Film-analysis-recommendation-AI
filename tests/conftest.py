import pytest

from moviebot.config import REPO_ROOT
from moviebot.mock_backend import MockBackend

DATA_DIR = REPO_ROOT / "data" / "mock"
SPIDERS, DOG_DIES, NEEDLES, JUMP_SCARES, ANIMAL_DIES = 1, 2, 4, 6, 11


@pytest.fixture
def backend() -> MockBackend:
    return MockBackend(DATA_DIR)


def movies_with(backend: MockBackend, topic_id: int, status: str) -> set[str]:
    """Titles of every mock movie whose status for `topic_id` is `status`."""
    titles = set()
    for movie_id in backend._movies:
        warnings = backend.get_movie_warnings(movie_id=movie_id, topic_ids=[topic_id]).warnings
        if next(w for w in warnings if w.topic_id == topic_id).status == status:
            titles.add(backend._movies[movie_id].title)
    return titles
