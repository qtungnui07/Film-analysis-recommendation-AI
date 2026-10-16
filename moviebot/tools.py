"""The LLM-facing side of the tools: JSON schemas for the model, and a dispatcher that runs its calls.

The argument models here are the single source for both the `tools` JSON sent
to the LLM and the validation of what the LLM sends back. The user id is never
an LLM argument: the dispatcher is bound to one session's user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError, model_validator

from moviebot.contracts import ToolBackend, ToolError


class SearchMoviesArgs(BaseModel):
    query: str = Field(description="What the user wants, e.g. 'funny space adventure' or 'something like Alien'.")
    min_rating: float | None = Field(None, ge=0, le=10, description="Minimum rating from 0 to 10.")
    genres: list[str] | None = Field(None, description="Only movies in these genres, e.g. ['Sci-Fi', 'Comedy'].")
    exclude_topic_ids: list[int] = Field(
        default_factory=list,
        description="Extra topic ids to avoid, from map_phrase_to_topics. "
        "The user's saved 'never show' topics are always excluded automatically.",
    )
    limit: int = Field(5, ge=1, le=10, description="How many movies to return.")


class GetMovieWarningsArgs(BaseModel):
    movie_id: str | None = Field(None, description="IMDb id from search_movies, e.g. 'tt0099052'. Preferred.")
    title: str | None = Field(None, description="Movie title, if you have no movie_id.")
    topic_ids: list[int] | None = Field(
        None, description="Topics you need an answer for; they are reported as 'unknown' if there is no data."
    )

    @model_validator(mode="after")
    def _needs_movie(self) -> GetMovieWarningsArgs:
        if not (self.movie_id or self.title):
            raise ValueError("give movie_id or title")
        return self


class MapPhraseArgs(BaseModel):
    phrase: str = Field(description="The user's own words for what they want to avoid, e.g. 'eight legs'.")


class GetUserProfileArgs(BaseModel):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]


SPECS = [
    ToolSpec(
        "get_user_profile",
        "Get the user's saved preferences: topics they never want to see and topics they want warnings about.",
        GetUserProfileArgs,
    ),
    ToolSpec(
        "map_phrase_to_topics",
        "Map a fear or sensitivity described in the user's own words to content-warning topic ids.",
        MapPhraseArgs,
    ),
    ToolSpec(
        "search_movies",
        "Search for movies to recommend. Movies with the user's 'never show' topics are removed automatically. "
        "Each result lists its warnings for the user's topics.",
        SearchMoviesArgs,
    ),
    ToolSpec(
        "get_movie_warnings",
        "Get the content warnings for one movie, with vote counts. Status is present, absent, or unknown (no data).",
        GetMovieWarningsArgs,
    ),
]


def _clean_schema(node: Any, in_properties: bool = False) -> Any:
    """Shrink Pydantic's schema for small models: drop titles, collapse Optional[X] to X."""
    if isinstance(node, list):
        return [_clean_schema(n) for n in node]
    if not isinstance(node, dict):
        return node
    out = {
        key: _clean_schema(value, in_properties=key == "properties" and not in_properties)
        for key, value in node.items()
        if in_properties or key != "title"
    }
    any_of = out.get("anyOf")
    if not in_properties and any_of and len(any_of) == 2 and {"type": "null"} in any_of:
        out.pop("anyOf")
        out.update(next(s for s in any_of if s != {"type": "null"}))
        if out.get("default", ...) is None:
            out.pop("default")
    return out


TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": _clean_schema(spec.args_model.model_json_schema()),
        },
    }
    for spec in SPECS
]


@dataclass
class ToolResult:
    name: str
    ok: bool
    args: dict | None  # validated arguments, or None if they never got that far
    output: dict  # what the LLM sees


def _error(name: str, message: str, hint: str | None = None, args: dict | None = None) -> ToolResult:
    output = {"error": message}
    if hint:
        output["hint"] = hint
    return ToolResult(name=name, ok=False, args=args, output=output)


class ToolDispatcher:
    def __init__(self, backend: ToolBackend, user_id: str):
        self.backend = backend
        self.user_id = user_id
        self._specs = {spec.name: spec for spec in SPECS}
        self._handlers: dict[str, Callable[[Any], dict]] = {
            "get_user_profile": self._get_user_profile,
            "map_phrase_to_topics": self._map_phrase_to_topics,
            "search_movies": self._search_movies,
            "get_movie_warnings": self._get_movie_warnings,
        }

    def call(self, name: str, raw_arguments: str | None) -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            return _error(name, f"unknown tool {name!r}", hint=f"available tools: {', '.join(self._specs)}")
        try:
            data = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as e:
            return _error(name, f"arguments are not valid JSON: {e.msg}", hint="send arguments as a JSON object")
        if not isinstance(data, dict):
            return _error(name, "arguments must be a JSON object")
        try:
            args = spec.args_model.model_validate(data)
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'arguments'}: {err['msg']}" for err in e.errors())
            return _error(name, f"invalid arguments: {problems}")
        try:
            output = self._handlers[name](args)
        except ToolError as e:
            return _error(name, e.message, hint=e.hint, args=args.model_dump())
        return ToolResult(name=name, ok=True, args=args.model_dump(), output=output)

    def _get_user_profile(self, args: GetUserProfileArgs) -> dict:
        return self.backend.get_user_profile(self.user_id).model_dump(mode="json")

    def _map_phrase_to_topics(self, args: MapPhraseArgs) -> dict:
        matches = self.backend.map_phrase_to_topics(args.phrase)
        output: dict = {"matches": [m.model_dump(mode="json") for m in matches]}
        if not matches:
            output["note"] = "No topic matched. Ask the user to describe it differently."
        return output

    def _search_movies(self, args: SearchMoviesArgs) -> dict:
        return self.backend.search_movies(user_id=self.user_id, **args.model_dump()).model_dump(mode="json")

    def _get_movie_warnings(self, args: GetMovieWarningsArgs) -> dict:
        # Always report the user's own topics, so a missing one shows up as "unknown" rather than silence.
        profile_topics = self.backend.get_user_profile(self.user_id).topic_ids()
        topic_ids = sorted(set(args.topic_ids or ()) | profile_topics)
        warnings = self.backend.get_movie_warnings(movie_id=args.movie_id, title=args.title, topic_ids=topic_ids)
        return warnings.model_dump(mode="json")
