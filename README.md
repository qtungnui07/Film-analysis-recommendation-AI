# Film-analysis-recommendation-AI

A chatbot that recommends movies and warns about each user's phobias and sensitivities. Filtering is deterministic code over [DoesTheDogDie](https://www.doesthedogdie.com/) (DTDD) content warnings, not an LLM judgement, and a movie with no data on a topic is "unknown", never "safe". The plan and the evaluation targets are in [PROJECT-CHARTER.md](PROJECT-CHARTER.md).

## Setup

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv sync --extra dev
cp .env.example .env
```

In `.env`:

- `LLM_*` points at any OpenAI-compatible chat endpoint. The default is a local model at `http://localhost:3636/v1`.
- `DTDD_KEY` and `TMDB_KEY` are only needed for the real database (below). The mock data works without them.

## Running the bot

```
uv run python -m moviebot.cli --user arachno -v        # mock data, no database needed
uv run python -m moviebot.cli --user arachno -v --db   # real database
```

- **Demo users:** `arachno` (never spiders, warn on jump scares), `dog_lover` (never a dog dies, warn on blood/gore) and `fresh` (no preferences).
- **`-v`** prints every tool call.

## Setting up the database

The datasets' licences don't allow redistribution, so the data is never committed: each of us downloads it and builds the database locally. Sources, terms and what ends up in the database are described in [data/README.md](data/README.md).

**1. ML-DDD and MovieLens.** The [Navigating-Sensitivity](https://github.com/sdean-group/Navigating-Sensitivity) script builds the content-warning table and downloads MovieLens 25M on the way. Run it next to this repo:

```
git clone https://github.com/sdean-group/Navigating-Sensitivity
cd Navigating-Sensitivity
uv run --with pandas --with requests --with huggingface_hub python ml-ddd/ml-ddd_data_generation.py
cp ml-ddd/data/ml-ddd_sensitivity_table.csv \
   ml-ddd/data/ml-25m/links.csv ml-ddd/data/ml-25m/genome-scores.csv ml-ddd/data/ml-25m/genome-tags.csv \
   ../Film-analysis-recommendation-AI/data/
cd ../Film-analysis-recommendation-AI
```

**2. IMDb.**

```
curl -o data/title.basics.tsv.gz https://datasets.imdbws.com/title.basics.tsv.gz
curl -o data/title.ratings.tsv.gz https://datasets.imdbws.com/title.ratings.tsv.gz
```

**3. Build.** This needs `DTDD_KEY` and takes about two minutes:

```
uv run python -m moviebot.build_db
```

It writes `data/moviebot.db` (about 75 MB) and refuses to overwrite an existing one. `--force` rebuilds from scratch, but loses the overviews and labels fetched since, and any profile edits.

**4. Plot overviews.** This is optional and needs `TMDB_KEY`. Most-voted movies go first, and each run continues where the last one stopped:

```
uv run python -m moviebot.fetch_overviews --limit 200
```

**5. Search index.** `--db` needs it. The first run downloads all-MiniLM-L6-v2 (80 MB) and takes about 10 minutes on a laptop CPU. Rerun it after fetching overviews; it only embeds what changed, so later runs take seconds:

```
uv run python -m moviebot.build_index
```

While the bot runs on the real database, it also fills gaps by itself:
- **Missing labels:** it fetches DTDD labels the dataset lacks, once per movie.
- **Missing movies:** it looks up movies outside the catalogue on TMDB.

Both need the keys in `.env`.

## Running the tests

```
uv run --extra dev pytest              # everything except the live LLM test
uv run --extra dev pytest -m live      # one end-to-end chat against the LLM in .env
```

- **The default suite** needs no network, API keys or downloaded data. It runs on the mock fixtures in `data/mock/` and on small databases the tests build themselves.
- **The live test** asks the real model for a recommendation and checks that no spider movie reaches the spider-fearing user. It skips itself when the LLM endpoint isn't reachable.

## Layout

| Path | What it is |
| --- | --- |
| [moviebot/contracts.py](moviebot/contracts.py) | The tool contracts every layer builds against, and the rules for turning DTDD labels into warnings |
| [moviebot/agent.py](moviebot/agent.py), [tools.py](moviebot/tools.py), [prompts.py](moviebot/prompts.py) | The LLM tool-calling loop, tool schemas and safety gate, and the system prompt |
| [moviebot/mock_backend.py](moviebot/mock_backend.py) | Backend over `data/mock/*.json`, used by the tests |
| [moviebot/db_backend.py](moviebot/db_backend.py) | Backend over `data/moviebot.db`, with semantic search, and DTDD and TMDB on demand |
| [moviebot/retrieval.py](moviebot/retrieval.py) | Chroma vector index (all-MiniLM-L6-v2, local) for movies and topic phrases |
| [moviebot/build_db.py](moviebot/build_db.py), [fetch_overviews.py](moviebot/fetch_overviews.py), [build_index.py](moviebot/build_index.py) | Database build, overview fetching, search index build |
| [moviebot/evaluation.py](moviebot/evaluation.py) | Safety recall, warning accuracy and phrase-mapping metrics |
