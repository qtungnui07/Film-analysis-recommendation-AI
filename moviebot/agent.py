"""The tool-calling loop: ask the LLM, run the tools it calls, feed results back, repeat until it answers."""

from __future__ import annotations

import json
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


class Agent:
    """One conversation with one user. `messages` persists across turns."""

    def __init__(
        self,
        client: OpenAI,
        dispatcher: ToolDispatcher,
        *,
        model: str,
        temperature: float = 0.2,
        max_tool_rounds: int = 6,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.client = client
        self.dispatcher = dispatcher
        self.model = model
        self.temperature = temperature
        self.max_tool_rounds = max_tool_rounds
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

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
            max_tool_rounds=settings.agent_max_tool_rounds,
        )

    def run_turn(self, user_text: str) -> TurnResult:
        self.messages.append({"role": "user", "content": user_text})
        turn = TurnResult(reply="")

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
                result = self.dispatcher.call(name, raw_arguments)
                turn.trace.append(ToolCallRecord(name=name, raw_arguments=raw_arguments, result=result))
                self.messages.append(
                    {"role": "tool", "tool_call_id": sent["id"], "content": json.dumps(result.output)}
                )

        # Out of tool rounds: make the model answer with what it has.
        message = self._complete(turn, tool_choice="none")
        return self._finish(turn, message.content)

    def _complete(self, turn: TurnResult, tool_choice: str):
        turn.llm_calls += 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOL_SPECS,
            tool_choice=tool_choice,
            temperature=self.temperature,
        )
        return response.choices[0].message

    def _finish(self, turn: TurnResult, content: str | None) -> TurnResult:
        turn.reply = (content or "").strip() or FALLBACK_REPLY
        self.messages.append({"role": "assistant", "content": turn.reply})
        return turn
