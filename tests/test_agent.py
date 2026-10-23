"""Agent loop tests against a scripted fake LLM: no network, fully deterministic."""

import copy
import json
from types import SimpleNamespace

from openai.types.chat import ChatCompletion

from conftest import SPIDERS, movies_with
from moviebot.agent import FALLBACK_REPLY, Agent
from moviebot.tools import TOOL_SPECS, ToolDispatcher


def completion(content=None, tool_calls=None) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "fake",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                }
            ],
        }
    )


def call(name, arguments, call_id="call_1") -> dict:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class FakeLLM:
    """Stands in for openai.OpenAI: returns scripted completions and records every request."""

    def __init__(self, *script: ChatCompletion):
        self.script = list(script)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs) -> ChatCompletion:
        self.requests.append(copy.deepcopy(kwargs))
        return self.script.pop(0)


def make_agent(backend, llm, user_id="arachno", **kwargs) -> Agent:
    return Agent(llm, ToolDispatcher(backend, user_id), model="fake", **kwargs)


def test_tool_call_then_answer(backend):
    llm = FakeLLM(
        completion(tool_calls=[call("search_movies", {"query": "sci-fi", "min_rating": 7})]),
        completion("Try Interstellar."),
    )
    result = make_agent(backend, llm).run_turn("Recommend a sci-fi movie")

    assert result.reply == "Try Interstellar."
    assert result.llm_calls == 2
    [record] = result.trace
    assert record.name == "search_movies" and record.result.ok
    titles = {m["title"] for m in record.result.output["results"]}
    assert titles and not titles & movies_with(backend, SPIDERS, "present")

    # The second request carries the assistant tool call and the matching tool result.
    second = llm.requests[1]["messages"]
    assert second[-2]["tool_calls"][0]["id"] == "call_1"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "call_1"
    assert json.loads(second[-1]["content"]) == record.result.output
    assert llm.requests[0]["tools"] == TOOL_SPECS and llm.requests[0]["tool_choice"] == "auto"


def test_several_tool_calls_in_one_round(backend):
    llm = FakeLLM(
        completion(
            tool_calls=[
                call("get_user_profile", {}, "a"),
                call("map_phrase_to_topics", {"phrase": "needles"}, "b"),
            ]
        ),
        completion("Done."),
    )
    result = make_agent(backend, llm).run_turn("hi")
    assert [r.name for r in result.trace] == ["get_user_profile", "map_phrase_to_topics"]
    tool_messages = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]


def test_bad_json_arguments_go_back_to_the_model(backend):
    llm = FakeLLM(completion(tool_calls=[call("search_movies", "{not json")]), completion("Sorry."))
    result = make_agent(backend, llm).run_turn("hi")
    [record] = result.trace
    assert not record.result.ok and "not valid JSON" in record.result.output["error"]
    assert result.reply == "Sorry."


def test_invalid_arguments_go_back_to_the_model(backend):
    llm = FakeLLM(
        completion(tool_calls=[call("search_movies", {"min_rating": 7})]),
        completion(tool_calls=[call("get_movie_warnings", {})]),
        completion("ok"),
    )
    result = make_agent(backend, llm).run_turn("hi")
    errors = [r.result.output["error"] for r in result.trace]
    assert "query" in errors[0]
    assert "movie_id or title" in errors[1]


def test_unknown_tool(backend):
    llm = FakeLLM(completion(tool_calls=[call("delete_everything", {})]), completion("ok"))
    [record] = make_agent(backend, llm).run_turn("hi").trace
    assert "unknown tool" in record.result.output["error"]
    assert "search_movies" in record.result.output["hint"]


def test_backend_errors_come_back_with_a_hint(backend):
    llm = FakeLLM(completion(tool_calls=[call("get_movie_warnings", {"title": "Arachnofobia"})]), completion("ok"))
    [record] = make_agent(backend, llm).run_turn("hi").trace
    assert "Arachnophobia" in record.result.output["hint"]


def test_user_id_comes_from_the_session_not_the_model(backend):
    llm = FakeLLM(
        completion(tool_calls=[call("get_user_profile", {"user_id": "dog_lover"})]),
        completion(tool_calls=[call("search_movies", {"query": "spiders", "user_id": "fresh", "limit": 10})]),
        completion("ok"),
    )
    profile, search = make_agent(backend, llm, user_id="arachno").run_turn("hi").trace
    assert profile.result.output["user_id"] == "arachno"
    assert search.result.output["excluded_topics"] == ["spiders"]


def test_movie_warnings_always_include_the_users_topics(backend):
    llm = FakeLLM(completion(tool_calls=[call("get_movie_warnings", {"title": "Primer"})]), completion("ok"))
    [record] = make_agent(backend, llm, user_id="arachno").run_turn("hi").trace
    statuses = {w["topic_name"]: w["status"] for w in record.result.output["warnings"]}
    assert statuses == {"spiders": "unknown", "jump scares": "unknown"}


def test_max_rounds_forces_an_answer(backend):
    looping = [completion(tool_calls=[call("get_user_profile", {}, f"c{i}")]) for i in range(2)]
    llm = FakeLLM(*looping, completion("Here's what I found."))
    result = make_agent(backend, llm, max_tool_rounds=2).run_turn("hi")
    assert [r["tool_choice"] for r in llm.requests] == ["auto", "auto", "none"]
    assert result.reply == "Here's what I found." and len(result.trace) == 2


def test_empty_answer_gets_a_fallback(backend):
    result = make_agent(backend, FakeLLM(completion(""))).run_turn("hi")
    assert result.reply == FALLBACK_REPLY


def test_history_persists_across_turns(backend):
    llm = FakeLLM(completion("First answer."), completion("Second answer."))
    agent = make_agent(backend, llm)
    agent.run_turn("first question")
    agent.run_turn("second question")
    roles_and_text = [(m["role"], m.get("content")) for m in llm.requests[1]["messages"][1:]]
    assert roles_and_text == [
        ("user", "first question"),
        ("assistant", "First answer."),
        ("user", "second question"),
    ]


def test_tool_specs_are_small_model_friendly():
    names = [spec["function"]["name"] for spec in TOOL_SPECS]
    assert names == [
        "get_user_profile",
        "request_profile_update",
        "confirm_profile_update",
        "map_phrase_to_topics",
        "search_movies",
        "get_movie_warnings",
    ]


def test_profile_update_requires_confirmation_in_a_later_turn(backend):
    llm = FakeLLM(
        completion(tool_calls=[call("request_profile_update", {"topic_id": SPIDERS, "strictness": "never"})]),
        completion("Bạn muốn tôi lưu thay đổi này không?"),
        completion("Đã lưu sở thích của bạn."),
    )
    agent = make_agent(backend, llm, user_id="fresh")

    agent.run_turn("Tôi muốn tránh nhện")
    assert backend.get_user_profile("fresh").topic_prefs == []
    result = agent.run_turn("xác nhận")

    assert result.trace[0].name == "confirm_profile_update"
    assert backend.get_user_profile("fresh").topic_ids("never") == {SPIDERS}


def test_model_cannot_confirm_profile_change_in_the_same_turn(backend):
    llm = FakeLLM(
        completion(
            tool_calls=[
                call("request_profile_update", {"topic_id": SPIDERS, "strictness": "never"}),
                call("confirm_profile_update", {"proposal_id": "profile-change-1"}),
            ]
        ),
        completion("Please confirm the change first."),
    )
    result = make_agent(backend, llm, user_id="fresh").run_turn("Set spiders to never")

    assert backend.get_user_profile("fresh").topic_prefs == []
    assert any(not record.result.ok for record in result.trace)


def test_explicit_avoidance_is_applied_even_when_model_skips_mapper(backend):
    llm = FakeLLM(
        completion(tool_calls=[call("search_movies", {"query": "sci-fi", "limit": 10})]),
        completion("I found some safe options."),
    )
    result = make_agent(backend, llm, user_id="fresh").run_turn("I don't want spiders; recommend sci-fi")

    search = next(record for record in result.trace if record.name == "search_movies")
    assert search.result.ok
    assert search.result.output["excluded_topics"] == ["spiders"]
    assert all(
        warning["status"] != "present"
        for movie in search.result.output["results"]
        for warning in movie["warnings"]
        if warning["topic_name"] == "spiders"
    )
    text = json.dumps(TOOL_SPECS)
    assert "anyOf" not in text and '"default": null' not in text
    warnings = next(s for s in TOOL_SPECS if s["function"]["name"] == "get_movie_warnings")
    # "title" is stripped as schema metadata but must survive as a real property.
    assert set(warnings["function"]["parameters"]["properties"]) == {"movie_id", "title", "topic_ids"}
    search = next(s for s in TOOL_SPECS if s["function"]["name"] == "search_movies")
    assert search["function"]["parameters"]["required"] == ["query"]
    assert search["function"]["parameters"]["properties"]["min_rating"]["type"] == "number"
