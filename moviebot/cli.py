"""Terminal chat for trying the agent: python -m moviebot.cli --user arachno --verbose [--db]"""

from __future__ import annotations

import argparse
import json
import sys

import openai

from moviebot.agent import Agent
from moviebot.config import Settings
from moviebot.contracts import NotFoundError
from moviebot.db_backend import SqliteBackend
from moviebot.mock_backend import MockBackend
from moviebot.sources import DtddClient, TmdbClient
from moviebot.tools import ToolDispatcher


def _summarize(output: dict) -> str:
    if "error" in output:
        return f"ERROR {output['error']}" + (f" ({output['hint']})" if "hint" in output else "")
    if "results" in output:
        titles = ", ".join(m["title"] for m in output["results"]) or "none"
        return f"{len(output['results'])} results [{titles}], removed {output['removed_count']}"
    if "matches" in output:
        return "matches: " + (", ".join(f"{m['name']} ({m['score']})" for m in output["matches"]) or "none")
    if "warnings" in output:
        flagged = [f"{w['topic_name']}={w['status']}" for w in output["warnings"] if w["status"] != "absent"]
        return f"{output['title']}: " + (", ".join(flagged) or "nothing present")
    return json.dumps(output)[:200]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the movie bot.")
    parser.add_argument("--user", default="arachno", help="demo user id: arachno, dog_lover or fresh")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every tool call")
    parser.add_argument("--db", action="store_true", help="use the real database instead of data/mock")
    args = parser.parse_args(argv)

    settings = Settings()
    if args.db:
        from moviebot.retrieval import VectorIndex  # chromadb is slow to import; only --db needs it

        dtdd_key, tmdb_key = settings.dtdd_key.get_secret_value(), settings.tmdb_key.get_secret_value()
        index = VectorIndex(settings.index_path, min_similarity=settings.retrieval_min_similarity)
        try:
            backend = SqliteBackend(
                settings.db_path,
                exclude_unknown=settings.strict_unknown_warnings,
                dtdd=DtddClient(dtdd_key) if dtdd_key else None,
                tmdb=TmdbClient(tmdb_key) if tmdb_key else None,
                index=index,
            )
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            return 2
        if not index.movies.count():
            print(f"{settings.index_path} is empty; build it with: python -m moviebot.build_index", file=sys.stderr)
            return 2
    else:
        backend = MockBackend(settings.mock_data_dir, exclude_unknown=settings.strict_unknown_warnings)
    try:
        profile = backend.get_user_profile(args.user)
    except NotFoundError as e:
        print(f"{e.message}; {e.hint}", file=sys.stderr)
        return 2

    agent = Agent.from_settings(settings, ToolDispatcher(backend, args.user))
    prefs = ", ".join(f"{p.name}={p.strictness}" for p in profile.topic_prefs) or "none"
    print(f"Chatting as {profile.name} ({args.user}); saved topics: {prefs}")
    print(f"Model {settings.llm_model} at {settings.llm_base_url}. Type 'quit' to exit.\n")

    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            print()
            return 0
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            return 0
        if not sys.stdin.isatty():
            print(text)  # echo piped input so transcripts read naturally
        try:
            result = agent.run_turn(text)
        except openai.APIConnectionError:
            print(f"Can't reach the LLM at {settings.llm_base_url}. Is the server running?", file=sys.stderr)
            return 1
        if args.verbose:
            for record in result.trace:
                print(f"  [tool] {record.name}({record.raw_arguments}) -> {_summarize(record.result.output)}")
        print(f"bot> {result.reply}\n")


if __name__ == "__main__":
    sys.exit(main())
