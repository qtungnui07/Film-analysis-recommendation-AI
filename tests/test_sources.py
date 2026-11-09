import urllib.error

import pytest

from moviebot import sources
from moviebot.sources import DtddClient


def status(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("url", code, "error", {"Retry-After": retry_after} if retry_after else {}, None)


@pytest.fixture
def dtdd(monkeypatch):
    """A client whose responses are scripted; exposes the requests made and the waits."""
    client = DtddClient("key")
    client.responses, client.sleeps = [], []
    client.requests = 0

    def fake_get_json(url, headers=None, timeout=15):
        client.requests += 1
        response = client.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(sources, "_get_json", fake_get_json)
    monkeypatch.setattr(sources.time, "sleep", client.sleeps.append)
    return client


def test_waits_out_the_minute_limit(dtdd):
    dtdd.responses = [status(429, "7"), status(429), {"items": []}]
    assert dtdd.media("tt1") is None
    assert dtdd.sleeps == [7, 60]


def test_gives_up_when_the_limit_never_clears(dtdd):
    dtdd.responses = [status(429, "1"), status(429, "1"), status(429, "1")]
    with pytest.raises(urllib.error.HTTPError):
        dtdd.media("tt1")
    assert dtdd.requests == 3


def test_does_not_wait_for_the_monthly_quota(dtdd):
    dtdd.responses = [status(429, "86400")]
    with pytest.raises(urllib.error.HTTPError):
        dtdd.media("tt1")
    assert dtdd.sleeps == []


def test_other_errors_fail_at_once(dtdd):
    dtdd.responses = [status(500, "1")]
    with pytest.raises(urllib.error.HTTPError):
        dtdd.media("tt1")
    assert dtdd.sleeps == []
