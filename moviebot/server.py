"""HTTP API for the UI: python -m moviebot.server [--db] [--port 8000], docs at /docs."""

from __future__ import annotations

import argparse
import sys
import uuid
from typing import Callable, Sequence

import openai
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from moviebot.agent import Agent
from moviebot.cli import make_backend
from moviebot.config import Settings
from moviebot.contracts import MovieResult, NotFoundError, ProfileUpdateStrictness, ToolError, Topic, UserProfile
from moviebot.mock_backend import MockBackend
from moviebot.tools import ToolDispatcher


class NewUser(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class PreferenceUpdate(BaseModel):
    strictness: ProfileUpdateStrictness  # "fine" removes the preference


class NewSession(BaseModel):
    user_id: str


class SessionCreated(BaseModel):
    session_id: str


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class ToolCall(BaseModel):
    name: str
    arguments: str  # exactly what the LLM sent
    ok: bool
    output: dict


class ChatReply(BaseModel):
    reply: str
    # Results of this turn's last search, with the user's warnings: what the UI shows as cards with badges.
    movies: list[MovieResult]
    # A profile change waiting for the user's "yes" or "no", sent as the next message.
    pending_profile_update: dict | None
    trace: list[ToolCall]


def create_app(
    backend: MockBackend,
    make_agent: Callable[[ToolDispatcher], Agent],
    cors_origins: Sequence[str] = (),
) -> FastAPI:
    app = FastAPI(title="Moviebot API")
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["*"], allow_headers=["*"])
    # shortcut: sessions live in memory and never expire, so a restart ends every chat; persist them if that matters.
    sessions: dict[str, Agent] = {}

    @app.exception_handler(ToolError)
    async def tool_error(request, e: ToolError) -> JSONResponse:
        status = 404 if isinstance(e, NotFoundError) else 400
        return JSONResponse({"error": e.message, "hint": e.hint}, status_code=status)

    @app.exception_handler(openai.APIConnectionError)
    async def llm_down(request, e: openai.APIConnectionError) -> JSONResponse:
        return JSONResponse({"error": "can't reach the LLM", "hint": "is the LLM server running?"}, status_code=503)

    # Handlers are async so every request runs on one thread: SqliteBackend's connection can't change threads.
    # shortcut: a chat turn blocks other requests until it ends; move to threads with a backend lock if several people chat at once.

    @app.get("/topics")
    async def list_topics() -> list[Topic]:
        return backend.topics

    @app.post("/users", status_code=201)
    async def create_user(body: NewUser) -> UserProfile:
        return backend.create_user(body.name)

    @app.get("/users/{user_id}")
    async def get_user(user_id: str) -> UserProfile:
        return backend.get_user_profile(user_id)

    @app.put("/users/{user_id}/preferences/{topic_id}")
    async def set_preference(user_id: str, topic_id: int, body: PreferenceUpdate) -> UserProfile:
        # No confirmation step: that guards against the LLM, and this is the user's own click.
        return backend.update_user_preference(user_id, topic_id, body.strictness)

    @app.post("/sessions", status_code=201)
    async def create_session(body: NewSession) -> SessionCreated:
        backend.get_user_profile(body.user_id)  # 404 for an unknown user
        session_id = uuid.uuid4().hex
        sessions[session_id] = make_agent(ToolDispatcher(backend, body.user_id))
        return SessionCreated(session_id=session_id)

    @app.post("/sessions/{session_id}/messages")
    async def send_message(session_id: str, body: Message) -> ChatReply:
        agent = sessions.get(session_id)
        if agent is None:
            raise NotFoundError(f"no session {session_id!r}", hint="start one with POST /sessions")
        result = agent.run_turn(body.text)
        searches = [r.result for r in result.trace if r.name == "search_movies" and r.result.ok]
        return ChatReply(
            reply=result.reply,
            movies=searches[-1].output["results"] if searches else [],
            pending_profile_update=agent.dispatcher.pending_profile_update,
            trace=[
                ToolCall(name=r.name, arguments=r.raw_arguments, ok=r.result.ok, output=r.result.output)
                for r in result.trace
            ],
        )

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the movie bot over HTTP.")
    parser.add_argument("--db", action="store_true", help="use the real database instead of data/mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    settings = Settings()
    try:
        backend = make_backend(settings, args.db)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    app = create_app(backend, lambda dispatcher: Agent.from_settings(settings, dispatcher), settings.api_cors_origins)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
