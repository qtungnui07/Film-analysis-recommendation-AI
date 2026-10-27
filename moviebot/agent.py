"""The tool-calling loop: ask the LLM, run the tools it calls, feed results back, repeat until it answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

from moviebot.config import Settings
from moviebot.prompts import SYSTEM_PROMPT
from moviebot.tools import TOOL_SPECS, ToolDispatcher, ToolResult

FALLBACK_REPLY = "Sorry, I couldn't put an answer together. Could you rephrase that?"


@dataclass
class ToolCallRecord:
    name: str
    raw_arguments: str  # exactly what the LLM sent
    result: ToolResult


@dataclass
class TurnResult:
    reply: str
    trace: list[ToolCallRecord] = field(default_factory=list)
    llm_calls: int = 0


@dataclass
class SessionState:
    """Small structured memory kept separately from the raw chat transcript."""

    recent_movies: list[dict] = field(default_factory=list)
    temporary_excluded_topic_ids: set[int] = field(default_factory=set)
    last_query: str | None = None


class Agent:
    """One conversation with one user. `messages` persists across turns."""

    def __init__(
        self,
        client: OpenAI,
        dispatcher: ToolDispatcher,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 500,
        max_tool_rounds: int = 6,
        memory_max_recommendations: int = 5,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.client = client
        self.dispatcher = dispatcher
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tool_rounds = max_tool_rounds
        self.memory_max_recommendations = memory_max_recommendations
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.session_state = SessionState()

    @classmethod
    def from_settings(cls, settings: Settings, dispatcher: ToolDispatcher) -> Agent:
        client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=settings.llm_timeout,
            max_retries=1,
        )
        return cls(
            client,
            dispatcher,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            max_tool_rounds=settings.agent_max_tool_rounds,
            memory_max_recommendations=settings.memory_max_recommendations,
        )

    def run_turn(self, user_text: str) -> TurnResult:
        if self.session_state.recent_movies:
            memory = "; ".join(
                movie["title"] for movie in self.session_state.recent_movies[: self.memory_max_recommendations]
            )
            self.messages.append(
                {
                    "role": "system",
                    "content": f"Recent recommendations in this conversation: {memory}. Use this context for follow-ups.",
                }
            )
        self.session_state.temporary_excluded_topic_ids = self.dispatcher.begin_turn(user_text)
        self.messages.append({"role": "user", "content": user_text})
        turn = TurnResult(reply="")

        # Profile writes are deliberately handled outside the model's judgment.
        # A proposal created in a previous turn is applied only after an explicit
        # confirmation in this turn; a rejection clears it without changing data.
        pending = self.dispatcher.pending_profile_update
        if pending and self._is_explicit_confirmation(user_text):
            result = self.dispatcher.confirm_pending_profile_update()
            turn.trace.append(
                ToolCallRecord(
                    name="confirm_profile_update",
                    raw_arguments=json.dumps({"proposal_id": pending["proposal_id"]}),
                    result=result,
                )
            )
            self.messages.append(
                {
                    "role": "system",
                    "content": "The pending profile change was explicitly confirmed and applied. Acknowledge it briefly.",
                }
            )
        elif pending and self._is_explicit_rejection(user_text):
            self.dispatcher.cancel_profile_update()
            self.messages.append(
                {
                    "role": "system",
                    "content": "The pending profile change was explicitly rejected and cancelled. Do not change the profile.",
                }
            )

        for _ in range(self.max_tool_rounds):
            message = self._complete(turn, tool_choice="auto")
            if not message.tool_calls:
                return self._finish(turn, message.content)
            self.messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id or f"call_{turn.llm_calls}_{i}",
                            "type": "function",
                            "function": {"name": call.function.name, "arguments": call.function.arguments},
                        }
                        for i, call in enumerate(message.tool_calls)
                    ],
                }
            )
            for sent in self.messages[-1]["tool_calls"]:
                name, raw_arguments = sent["function"]["name"], sent["function"]["arguments"]
                if name == "confirm_profile_update" and not self._is_explicit_confirmation(user_text):
                    result = self.dispatcher.reject_profile_confirmation()
                else:
                    result = self.dispatcher.call(name, raw_arguments)
                turn.trace.append(ToolCallRecord(name=name, raw_arguments=raw_arguments, result=result))
                if name == "search_movies" and result.ok:
                    self.session_state.last_query = (result.args or {}).get("query")
                    self.session_state.recent_movies = result.output.get("results", [])[: self.memory_max_recommendations]
                self.messages.append(
                    {"role": "tool", "tool_call_id": sent["id"], "content": json.dumps(result.output)}
                )

        # Out of tool rounds: make the model answer with what it has.
        message = self._complete(turn, tool_choice="none")
        return self._finish(turn, message.content)

    @staticmethod
    def _is_explicit_confirmation(text: str) -> bool:
        normalized = " ".join(re.sub(r"[^a-zA-ZÀ-ỹ0-9 ]", " ", text.casefold()).split())
        return bool(
            re.fullmatch(
                r"(yes|y|yeah|yep|yes please|ok|okay|sure|confirm|confirmed|correct|do it|apply|go ahead|đúng|đúng rồi|có|có lưu đi|xác nhận|đồng ý|lưu đi)",
                normalized,
            )
        )

    @staticmethod
    def _is_explicit_rejection(text: str) -> bool:
        normalized = " ".join(re.sub(r"[^a-zA-ZÀ-ỹ0-9 ]", " ", text.casefold()).split())
        return bool(
            re.fullmatch(
                r"(no|n|nope|cancel|stop|don't|do not|not now|sai rồi|không|không phải|hủy|huỷ|bỏ qua)",
                normalized,
            )
        )

    def _complete(self, turn: TurnResult, tool_choice: str):
        turn.llm_calls += 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOL_SPECS,
            tool_choice=tool_choice,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message

    def _finish(self, turn: TurnResult, content: str | None) -> TurnResult:
        turn.reply = (content or "").strip() or FALLBACK_REPLY
        self.messages.append({"role": "assistant", "content": turn.reply})
        return turn
