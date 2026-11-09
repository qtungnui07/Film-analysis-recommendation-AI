# Moviebot API

The HTTP API the frontend calls. It sits on the same agent and data as the terminal chat.

## Running it

```
uv run python -m moviebot.server          # mock data, http://127.0.0.1:8000
uv run python -m moviebot.server --db     # real database
uv run python -m moviebot.server --port 9000 --host 0.0.0.0
```

- **Interactive docs:** http://127.0.0.1:8000/docs lets you try every call in the browser.
- **Schema:** http://127.0.0.1:8000/openapi.json. You can generate TypeScript types from it, for example with `npx openapi-typescript http://127.0.0.1:8000/openapi.json -o api.d.ts`.
- **Calling from a browser on another origin** (e.g. a Vite dev server) needs that origin in `.env`, then a server restart:

  ```
  API_CORS_ORIGINS=["http://localhost:5173"]
  ```

- **Demo users** on the mock data: `arachno` (never spiders, warn on jump scares), `dog_lover` (never a dog dies, warn on blood and gore) and `fresh` (no preferences).

## The usual flow

1. **Onboarding.** `GET /topics` fills the picker. `POST /users` creates the user, then one `PUT .../preferences/{topic_id}` per topic they pick. Keep the returned `user_id`, for example in `localStorage`. There is no login.
2. **Chat.** `POST /sessions` once per conversation, then `POST /sessions/{session_id}/messages` for each message.
3. **Profile page.** `GET /users/{user_id}` to show it, the same `PUT` to change it.

## Calls

All bodies are JSON. All ids are strings except `topic_id`, which is an integer.

| Call | Body | Success |
| --- | --- | --- |
| `GET /topics` | | 200, list of [Topic](#topic) |
| `POST /users` | `{"name": "Alice"}` | 201, [UserProfile](#userprofile) |
| `GET /users/{user_id}` | | 200, [UserProfile](#userprofile) |
| `PUT /users/{user_id}/preferences/{topic_id}` | `{"strictness": "never"}` | 200, the updated [UserProfile](#userprofile) |
| `POST /sessions` | `{"user_id": "arachno"}` | 201, `{"session_id": "..."}` |
| `POST /sessions/{session_id}/messages` | `{"text": "a sci-fi movie please"}` | 200, [ChatReply](#chatreply) |
| `GET /sessions/{session_id}/progress` | | 200, `{"status": "..."}` |

### `GET /topics`

Every content-warning topic. Group them by `category` in the picker.

```json
[
  {"topic_id": 1, "name": "spiders", "category": "Animals", "covers": []},
  {"topic_id": 11, "name": "an animal dies", "category": "Animals", "covers": [2]}
]
```

The mock data has 11 topics in 8 categories. The real database has 137 topics in 37 categories, so plan for a long list, ideally with search.

### `POST /users`

`name` must be 1 to 100 characters. The server picks the `user_id`. A new user has no preferences.

```json
{"user_id": "acf46408", "name": "Alice", "topic_prefs": []}
```

### `GET /users/{user_id}`

```json
{
  "user_id": "acf46408",
  "name": "Alice",
  "topic_prefs": [{"topic_id": 1, "name": "spiders", "strictness": "never"}]
}
```

### `PUT /users/{user_id}/preferences/{topic_id}`

`strictness` is one of:

| Value | Meaning |
| --- | --- |
| `never` | Never recommend a movie with this topic |
| `warn` | Recommend it, but show the warning |
| `fine` | Remove the preference; the topic no longer appears in the profile |

The change applies straight away and returns the whole updated profile. The same call serves onboarding and the profile page.

### `POST /sessions`

Starts a conversation for one user. The bot remembers earlier messages in the same session. Start a new session for a fresh conversation.

```json
{"session_id": "44226fbe16e04431a3a9aa497668ec7a"}
```

### `POST /sessions/{session_id}/messages`

`text` must be 1 to 2000 characters. Returns a [ChatReply](#chatreply).

**This call is slow.** It returns when the LLM has finished, which takes about a minute on the local dev model. Show a loading state, poll [progress](#get-sessionssession_idprogress), don't put a short timeout on the request, and don't send the next message before the reply comes back.

### `GET /sessions/{session_id}/progress`

What the session's current turn is doing. Poll it while `POST .../messages` is outstanding.

```json
{"status": "Searching for spooky movie"}
```

`status` is `null` when no turn is running. While the POST is outstanding, `null` means the message is queued or just starting.

| `status` | When |
| --- | --- |
| `LLM is writing a response` | Waiting for the LLM |
| `Looking up the database` | Matching the message against avoid-topics |
| `Searching for {query}` | `search_movies` |
| `Getting {user}'s profile` | `get_user_profile` |
| `Mapping {phrase} to topics` | `map_phrase_to_topics` |
| `Requesting a warn/never/show for {topic}` | `request_profile_update`; `show` is the `fine` strictness |
| `Confirming a warn/never/show for {topic}` | `confirm_profile_update` |
| `Getting {title}'s warnings` | `get_movie_warnings` |
| `Fetching {title} from DTDD` | Live DTDD lookup, `--db` only |
| `Fetching {title or IMDb id} from TMDB` | Live TMDB lookup, `--db` only |

Show the text as it is. Tools other than search finish in milliseconds, so polling will rarely catch them.

## Shapes

### Topic

| Field | Type | Notes |
| --- | --- | --- |
| `topic_id` | int | |
| `name` | string | e.g. `"spiders"` |
| `category` | string | e.g. `"Animals"`, for grouping |
| `covers` | int[] | Narrower topics this one includes: "an animal dies" covers "a dog dies" |

### UserProfile

| Field | Type | Notes |
| --- | --- | --- |
| `user_id` | string | |
| `name` | string | |
| `topic_prefs` | list | Each item has `topic_id`, `name` and `strictness` (`never` or `warn`) |

### ChatReply

| Field | Type | Notes |
| --- | --- | --- |
| `reply` | string | The bot's answer. Show it as the chat message. |
| `movies` | [MovieResult](#movieresult)[] | The movies the bot searched for this turn, to show as cards. Empty when it didn't search. |
| `pending_profile_update` | object or null | Set when the bot offers to change the profile. See [Profile changes in chat](#profile-changes-in-chat). |
| `trace` | list | Every tool the bot called: `name`, `arguments` (the raw JSON string the LLM sent), `ok`, `output`. For a debug panel, not for the main UI. |

Example, with the trace shortened:

```json
{
  "reply": "You might enjoy The Matrix (1999)...",
  "movies": [{"movie_id": "tt0133093", "title": "The Matrix", "...": "see MovieResult"}],
  "pending_profile_update": null,
  "trace": [
    {
      "name": "search_movies",
      "arguments": "{\"query\": \"sci-fi\"}",
      "ok": true,
      "output": {"results": ["..."], "excluded_topics": ["spiders"], "removed_count": 0}
    }
  ]
}
```

### MovieResult

```json
{
  "movie_id": "tt0133093",
  "title": "The Matrix",
  "year": 1999,
  "rating": 8.7,
  "genres": ["Action", "Sci-Fi"],
  "overview": "A hacker learns that the world he lives in is a simulation...",
  "warnings": [
    {"topic_id": 1, "topic_name": "spiders", "label": "no", "status": "absent", "present_via": []},
    {"topic_id": 6, "topic_name": "jump scares", "label": "no", "status": "absent", "present_via": []}
  ],
  "unknown_topics": [],
  "relevance_score": 4.87,
  "match_reasons": ["matches 2 query keywords"]
}
```

| Field | Notes |
| --- | --- |
| `movie_id` | IMDb id |
| `rating` | 0–10 |
| `overview` | Can be `""` for the few movies with no plot summary |
| `warnings` | One [warning](#warnings-and-badges) for each of the user's `never` and `warn` topics, plus any topic they asked to avoid in this message |
| `unknown_topics` | Names of the topics being avoided (`never`, or asked in this message) that have no data for this movie |
| `match_reasons` | Short reasons the movie was picked, e.g. `"similar to Alien (1979)"` |

### Warnings and badges

| Field | Notes |
| --- | --- |
| `status` | `present`, `absent` or `unknown`. This is what the badge should show. |
| `label` | Raw vote result: `yes`, `no`, `unclear` (voters split), or `null` (nobody voted) |
| `present_via` | Narrower topics that make this one present, e.g. "an animal dies" present via `["a dog dies"]` |

Rules the UI must keep:

- **`unknown` is not safe.** It means there is no data. Show it as "no data", never as a green check.
- **`present` includes `unclear`.** A topic with split votes counts as present, so it fails safe. Use `label` if you want to word that badge differently, e.g. "maybe".
- **`never` topics never show as `present` in `movies`.** The server removes those movies before the bot sees them. They can still be `unknown`.

## Profile changes in chat

If the user says something like "add clowns to my never list", the bot doesn't change the profile straight away. It proposes the change, and `pending_profile_update` is set:

```json
{"proposal_id": "profile-change-1", "topic_id": 5, "topic_name": "clowns", "strictness": "never"}
```

Show confirm and cancel buttons. Confirm sends `"yes"` and cancel sends `"no"`, as ordinary messages to the same session. The server applies or drops the change itself; nothing else is needed. Changes made with `PUT .../preferences` skip this step, because a click on the profile page is already the user's own choice.

## Errors

Errors from the app have this body:

```json
{"error": "no user 'nobody'", "hint": "known users: arachno, dog_lover, fresh"}
```

| Status | When |
| --- | --- |
| 404 | Unknown user, topic or session. Sessions disappear when the server restarts, so on a 404 from `/messages`, start a new session. |
| 400 | Another request the backend refused |
| 422 | The body is invalid, e.g. a wrong `strictness` or empty `text`. This body is FastAPI's own: `{"detail": [...]}`. |
| 503 | The server can't reach the LLM. Chat is down; topics and profiles still work. |

## Limits

- **One chat turn at a time.** While a message is being answered, other chat messages and the user and profile calls wait. `GET /topics` and `GET .../progress` don't.
- **Sessions are in memory.** A server restart ends every conversation. Users and preferences are kept when the server runs with `--db`; on the mock data they are lost too.
- **No authentication.** Anyone who can reach the server can read or edit any user. Keep it on `127.0.0.1` unless you're running a test with others.
