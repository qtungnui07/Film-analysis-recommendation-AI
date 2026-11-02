"""API tests: the mock backend and a scripted fake LLM behind FastAPI's TestClient."""

from fastapi.testclient import TestClient

from conftest import SPIDERS
from moviebot.agent import Agent
from moviebot.server import create_app
from test_agent import FakeLLM, call, completion


def client(backend, *script) -> TestClient:
    llm = FakeLLM(*script)
    return TestClient(create_app(backend, lambda dispatcher: Agent(llm, dispatcher, model="fake")))


def test_onboarding_creates_a_user_and_saves_their_topics(backend):
    api = client(backend)
    topics = api.get("/topics").json()
    assert {"topic_id": SPIDERS, "name": "spiders", "category": "Animals", "covers": []} in topics

    created = api.post("/users", json={"name": "Alice"})
    assert created.status_code == 201
    user_id = created.json()["user_id"]
    assert created.json()["topic_prefs"] == []

    api.put(f"/users/{user_id}/preferences/{SPIDERS}", json={"strictness": "never"}).raise_for_status()
    prefs = api.get(f"/users/{user_id}").json()["topic_prefs"]
    assert prefs == [{"topic_id": SPIDERS, "name": "spiders", "strictness": "never"}]


def test_chat_returns_the_reply_and_safe_movies_with_warnings(backend):
    api = client(
        backend,
        completion(tool_calls=[call("search_movies", {"query": "spooky movie"})]),
        completion("Try one of these."),
    )
    session_id = api.post("/sessions", json={"user_id": "arachno"}).json()["session_id"]
    reply = api.post(f"/sessions/{session_id}/messages", json={"text": "Something spooky"}).json()

    assert reply["reply"] == "Try one of these."
    assert reply["movies"]
    for movie in reply["movies"]:
        assert not any(w["topic_id"] == SPIDERS and w["status"] == "present" for w in movie["warnings"])
    assert [t["name"] for t in reply["trace"]] == ["search_movies"]
    assert reply["pending_profile_update"] is None


def test_errors_say_what_went_wrong(backend):
    api = client(backend)
    unknown_user = api.post("/sessions", json={"user_id": "nobody"})
    assert unknown_user.status_code == 404
    assert "arachno" in unknown_user.json()["hint"]

    assert api.post("/sessions/nope/messages", json={"text": "hi"}).status_code == 404
    assert api.put(f"/users/arachno/preferences/{SPIDERS}", json={"strictness": "maybe"}).status_code == 422
    assert api.put("/users/arachno/preferences/999", json={"strictness": "warn"}).status_code == 404
