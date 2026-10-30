# Data

Only this README and `mock/` are committed. The dataset licences don't allow redistribution, so each of us downloads the raw files and builds the database locally. `.gitignore` keeps everything else in `data/` out of git.

## 1. Download

Put these files directly in `data/`:

| Files | Where from | Terms |
| --- | --- | --- |
| `title.basics.tsv.gz`, `title.ratings.tsv.gz` | [IMDb datasets](https://datasets.imdbws.com/) | Personal and non-commercial use only |
| `links.csv`, `genome-scores.csv`, `genome-tags.csv` | `ml-25m.zip` from [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) | Research use, no redistribution, cite (below) |
| `ml-ddd_sensitivity_table.csv` | Run `python3 ml-ddd/ml-ddd_data_generation.py` in [Navigating-Sensitivity](https://github.com/sdean-group/Navigating-Sensitivity) | Cite (below) |

Then set `DTDD_KEY` and `TMDB_KEY` in `.env` (see `.env.example`).

## 2. Build

```
python -m moviebot.build_db                    # about 1–2 minutes, about 75 MB
python -m moviebot.fetch_overviews --limit 200 # plot overviews from TMDB, most-voted first; rerun to continue
python -m moviebot.build_index                 # search index; about 10 minutes the first time, then seconds
python -m moviebot.cli --db --user arachno -v  # chat against the real data
```

Rerun `build_index` after `fetch_overviews`. It embeds only the texts that changed.

`build_db` refuses to overwrite `data/moviebot.db`. `--force` rebuilds it but loses fetched overviews, labels fetched on demand and profile edits.

## What's in `moviebot.db`

The schema, with a comment on every column, is in [db_backend.py](../moviebot/db_backend.py).

- **movies**: the catalogue. 12,779 IMDb movies and TV movies with at least 10,000 votes, adult titles excluded. Title, year, rating and genres are IMDb's, unchanged. `overview` stays empty until `fetch_overviews` fills it.
- **topics**: the 137 ML-DDD questions under DTDD's own topic ids, names and categories. Synonyms come from DTDD's keywords.
- **labels**: one `yes` / `no` / `unclear` per (movie, topic). The dataset covers 7,327 catalogue movies. The DTDD data stops in 2019 because it comes through MovieLens; newer movies get labels on demand.
- **genome**: a 1,128-tag relevance vector (atmospheric, dark, feel-good…) for 8,993 catalogue movies. It's for retrieval only, never for warnings: its scores are machine predictions, not votes.
- **users**: three demo profiles that match `mock/users.json`.

## What's in `chroma/`

The search index that `build_index` makes. It's a local Chroma store (about 45 MB), embedded with all-MiniLM-L6-v2 through Chroma's ONNX runtime. The model downloads once to `~/.cache/chroma`.

- **movies**: one text per catalogue movie: title, year, genres, its 10 most relevant genome tags, and the overview if TMDB has one. Few movies have overviews yet, so the genome tags carry most of the meaning.
- **topics**: each topic's name and DTDD keywords, for `map_phrase_to_topics`. Keywords that several topics share, like "fear" or "death", are left out because they can't tell topics apart.

The index only ranks. Warnings and exclusions always come from the labels, never from similarity.

## How DTDD votes become warnings

- **Label:** `yes` or `no` when more than 75% of votes agree, `unclear` otherwise, no label when nobody voted. This is ML-DDD's rule. Their README says "at least 75%", but their code uses "more than", and the code made the data. `contracts.vote_label` applies the same rule to votes fetched from the API.
- **Status:** `yes` and `unclear` are present (a split vote fails safe), `no` is absent, and no label is unknown, which never means safe.
- **Covering:** a broad topic includes narrower ones, so "never: an animal dies" also hides a movie labelled animal = no, dog = yes. When a narrower topic makes a broad one present, the warning's `present_via` names it. The list is `COVERS` in [build_db.py](../moviebot/build_db.py). Decisions so far:
  - Blood/gore covers excessive gore, but not audio gore, which is heard rather than seen.
  - Sexual assault (DTDD: "sexually assaulted or harassed", on or off screen) covers raped onscreen.
  - The full list includes judgment calls; change it there.

## Topics DTDD changed since the dataset

- **7 reworded:** mapped to the current topic (`RENAMED`). "Dog fighting" became the broader "animal fighting".
- **9 retired** (ids 10001–10009): raped onscreen, excessive gore, attempts suicide, overdoses, n-word, asphyxiates, farts or spits, "I'll kill myself", dead animal. They keep their dataset labels, but the API can't refresh them. Most are covered by a live broader topic.
- DTDD has 281 topics today. We use the 137 the dataset has.

## Live sources (`SqliteBackend`)

- **DTDD on demand:** when a movie lacks a label that a request needs, its DTDD votes are fetched once and stored. The fresher API labels replace the dataset's for the same topic. Each fetch takes about 3 s (a 1.5 MB response), so a first search over unseen movies is slow; later ones are cached. If DTDD fails, fetching turns off for the session and missing labels stay unknown.
- **TMDB fallback:** a movie a user asks for that isn't in the catalogue is looked up on TMDB, by IMDb id or by exact title, and added. Add the year in brackets to pick between remakes, e.g. "The Thing (1982)". These rows use TMDB's vote average as `rating`.

## Citations

- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TiiS 5, 4. https://doi.org/10.1145/2827872
- Jesse Vig, Shilad Sen, and John Riedl. 2012. The Tag Genome: Encoding Community Knowledge to Support Novel Interaction. ACM TiiS 2, 3. https://doi.org/10.1145/2362394.2362395
- Datasets for Navigating Sensitive Topics in Recommendation Systems. WWW '25 (resource track). https://github.com/sdean-group/Navigating-Sensitivity
- Content warnings from [DoesTheDogDie](https://www.doesthedogdie.com/); movie details from [TMDB](https://www.themoviedb.org/) (this product uses the TMDB API but is not endorsed or certified by TMDB).
