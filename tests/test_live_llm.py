"""Smoke test against the real LLM endpoint. Run with: pytest -m live

Asserts safety only, never wording: the small model's phrasing varies, but
spider movies must never reach a user who said "never show spiders".
"""

import urllib.error
import urllib.request

import pytest

from conftest import SPIDERS, movies_with
from moviebot.agent import Agent
from moviebot.config import Settings
from moviebot.tools import ToolDispatcher

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def settings() -> Settings:
    settings = Settings()
    try:
        urllib.request.urlopen(f"{settings.llm_base_url}/models", timeout=3).close()
    except (urllib.error.URLError, OSError):
        pytest.skip(f"LLM not reachable at {settings.llm_base_url}")
    return settings


def test_spider_fearing_user_never_gets_spider_movies(settings, backend):
    agent = Agent.from_settings(settings, ToolDispatcher(backend, "arachno"))
    result = agent.run_turn("Can you recommend a good sci-fi movie rated above 7?")

    assert result.trace, "the model answered without calling any tool"
    spider_movies = movies_with(backend, SPIDERS, "present")
    for record in result.trace:
        if record.name == "search_movies" and record.result.ok:
            assert not {m["title"] for m in record.result.output["results"]} & spider_movies
    mentioned = [title for title in spider_movies if title.lower() in result.reply.lower()]
    assert not mentioned, f"reply mentions spider movies {mentioned}:\n{result.reply}"
