# Changelog

## 2026-10-29 — Retrieval layer

### Added

- `python -m moviebot.build_index` embeds the database into a local Chroma store in `data/chroma`, using all-MiniLM-L6-v2. It runs locally through Chroma's ONNX runtime, with no torch, LangChain or LlamaIndex. Each movie's text is its title, year, genres, top 10 MovieLens genome tags and overview. Few movies have overviews yet, so the tags carry most of the meaning. Rerunning the build embeds only the texts that changed.
- With the index (`--db` now requires it), `search_movies` ranks by semantic similarity plus a small rating bonus, and a large boost for a query that is a movie's title. "Something like X" searches with X's own text and leaves X out. Phobia filtering is unchanged and still runs on every result.
- `map_phrase_to_topics` also matches topic names and DTDD keywords by meaning. On the real data, "people throwing up" finds vomiting and "I'm terrified of anything with eight legs" finds spiders. Bare "eight legs" still finds nothing: MiniLM doesn't link the phrase to spiders, so it would need a keyword.
- Title lookup uses rapidfuzz. A title resolves only when exactly one movie matches it exactly or surely; otherwise the error lists the candidates with their years and ids. "Title (YYYY)" picks between remakes.

### Fixed

- `get_movie_warnings(title=...)` silently picked one movie when several share a title (499 titles in the catalogue, e.g. The Thing 1982/2011), and could attach the wrong movie's warnings. It now asks which one.

### Verification

```text
59 passed, 1 deselected
```

## 2026-10-29 — Real data layer

### Changed (contract: affects every layer)

- `TopicWarning` carries the DTDD `label` (`yes` / `no` / `unclear`, or `None` when nobody voted) instead of `yes_votes` / `no_votes`. The ML-DDD dataset only has labels. `vote_status` is replaced by `vote_label` (votes to label, using the dataset's more-than-75% rule) and `label_status` (`unclear` counts as present, so it fails safe).
- `Topic.covers` lists the narrower topics a topic includes. `topic_warning` makes a topic present when any narrower topic is present, so "never: an animal dies" also hides a movie labelled animal = no, dog = yes. `TopicWarning.present_via` names the narrower topics responsible.
- The mock fixtures now store labels, converted with the same rule. Three mock warnings changed from absent to present: their votes were split, so they are now `unclear`. A new mock topic, "an animal dies", covers "a dog dies".
- The system prompt and the `get_movie_warnings` description explain `unclear` and `present_via`.

### Added

- `python -m moviebot.build_db` builds `data/moviebot.db` from the raw IMDb, MovieLens and ML-DDD downloads. It's reproducible. The catalogue is the 12,779 non-adult IMDb movies and TV movies with at least 10k votes. Topics use DTDD's ids; reworded and retired topics are mapped explicitly.
- `SqliteBackend` (`python -m moviebot.cli --db`) fetches DTDD labels on demand when the dataset lacks them, falls back to TMDB for movies outside the catalogue, and persists profile edits.
- `python -m moviebot.fetch_overviews --limit N` fills plot overviews from TMDB, most-voted first. It's resumable.
- [data/README.md](data/README.md): sources, terms, build steps, the label rules and the topic changes.
- `TMDB_KEY` in `.env.example`.

### Removed

- The LFS rules, and the raw data from git. The licences forbid redistribution, so everyone downloads and builds locally.

### Verification

```text
54 passed, 1 deselected
```

## 2026-10-22 — Profile safety and agent reliability

### Added

- Added a deterministic search safety gate. If a backend ever returns a movie with a present `never` topic, the result is rejected before the LLM can recommend it.
- Added turn-level phrase pre-processing for explicit avoidance language (`avoid`, `no`, `sợ`, `tránh`, `không muốn`, etc.). Matched topics are passed as temporary exclusions even if a small model forgets to call the mapper.
- Added structured `SessionState` memory for recent recommendations, the latest query, and temporary exclusions. Follow-up turns receive a compact, explicit context in addition to the transcript.
- Added explainability fields to every `MovieResult`: `relevance_score` and `match_reasons`.
- Added a lightweight hybrid mock ranking: exact-title matches first, then keyword overlap, with rating as a tie-breaker.
- Added [evaluation.py](moviebot/evaluation.py) helpers for safety recall, warning accuracy, and phrase-mapping accuracy.

- Added `request_profile_update` to create a pending preference change without writing it.
- Added `confirm_profile_update` to apply a pending change using a proposal id.
- Added support for three profile actions:
  - `never`: exclude a topic from recommendations.
  - `warn`: keep recommendations but report the topic when present.
  - `fine`: remove the saved preference.
- Added deterministic confirmation handling in `Agent`:
  - A change is applied only after an explicit confirmation in a later user turn.
  - Explicit rejection cancels the pending change.
  - Ambiguous replies do not change the profile.
  - A model cannot confirm a change during the same turn in which it requested it.
- Added `llm_max_tokens` configuration (default: `500`) and passed it to the chat completion request.
- Documented `LLM_MAX_TOKENS` in `.env.example`.
- Added retrieval/memory settings: `RETRIEVAL_TOP_K`, `RETRIEVAL_MIN_SIMILARITY`, `MEMORY_MAX_RECOMMENDATIONS`, and `STRICT_UNKNOWN_WARNINGS`.
- Added mock-backend support for reading topics and updating in-memory user preferences.

### Changed

- Updated the system prompt to describe the proposal/confirmation workflow.
- Updated the system prompt to require data-backed recommendation explanations and prohibit invented reasons.
- Expanded tool schemas with profile-update tools while keeping validation in Pydantic models.
- Added profile-update methods to the `ToolBackend` contract.
- Updated the tool schema test for the new tools.

### Notes

- Mock profile changes are in-memory and last for the current backend process only. A production backend should persist them in the user database.
- The mock ranking is intentionally dependency-free; production retrieval should replace it with hybrid BM25/vector search and a learned or tuned reranker.
- Evaluation helpers are library functions; the report/evaluation runner still needs to provide the project’s labelled test cases.
- Existing movie safety rules remain unchanged: `never` topics are filtered deterministically, and missing warning data remains `unknown` rather than safe.

### Verification

```text
42 passed, 1 deselected
```
