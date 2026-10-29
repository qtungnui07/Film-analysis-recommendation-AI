"""Build data/moviebot.db from the raw downloads in data/. See data/README.md for where to get them.

    python -m moviebot.build_db          # needs DTDD_KEY, for DTDD's topic ids, names and categories

- Catalogue: IMDb movies and TV movies with at least 10,000 votes, not adult.
- Labels: ML-DDD for every movie it covers, also outside the catalogue, so a
  movie added later from TMDB still finds its dataset labels.
- Topics: the 137 ML-DDD questions, under DTDD's own topic ids.
- Genome: MovieLens tag-genome vectors for catalogue movies.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
import sys
from array import array
from pathlib import Path

from moviebot.config import REPO_ROOT, Settings
from moviebot.db_backend import SCHEMA
from moviebot.sources import DtddClient

RAW_DIR = REPO_ROOT / "data"
MIN_VOTES = 10_000
# DTDD has no topic-list endpoint, but every movie's page lists every topic.
TOPIC_LIST_MOVIE = "tt0099052"

# ML-DDD questions DTDD has reworded since; values are the current wording (a topic's doesName).
RENAMED = {
    "Are there jumpscares": "Are there jump scares",
    "Does someone becomes unconscious": "Does someone become unconscious",
    "Is a minority is misrepresented": "Is a minority misrepresented",
    "Is the R word used": "Is the r-slur used",
    "Is there aphobia": "Is there Aphobia",
    "Is there dog fighting": "Is there animal fighting",  # DTDD broadened it
    "Does someone die by suicide": "Is there suicide",
}
# ML-DDD questions with no current DTDD topic: kept with dataset labels, never refreshed from the API.
RETIRED = {
    "Are there n-words": (10001, "the n-word is used", "Prejudice"),
    "Does someone asphyxiate": (10002, "someone asphyxiates", "Neck"),
    "Does someone attempt suicide": (10003, "someone attempts suicide", "Self Harm"),
    "Does someone fart or spit": (10004, "someone farts or spits", "Gross"),
    "Does someone overdose": (10005, "someone overdoses", "Addiction"),
    'Does someone say "I\'ll kill myself"': (10006, 'someone says "I\'ll kill myself"', "Self Harm"),
    "Is someone raped onscreen": (10007, "someone is raped onscreen", "Sexual Assault"),
    "Is there a dead animal": (10008, "there's a dead animal", "Animal Death"),
    "Is there excessive gore": (10009, "there's excessive gore", "Violence"),
}
# Broader topic -> the narrower topics it includes, by ML-DDD question. When a narrower topic is
# present, so is the broader one (contracts.topic_warning). Only clear inclusions belong here.
COVERS = {
    "Does a non-human character die": ["Does an animal die", "Does a dragon die"],
    "Does an animal die": ["Does a pet die", "Does a horse die", "Does a dragon die", "Is there a dead animal"],
    "Does a pet die": ["Does the dog die", "Does a cat die"],
    "Are animals abused": ["Is there dog fighting", "Is there bestiality"],
    "Is there sexual content": ["Is there bestiality", "Is someone raped onscreen"],
    # DTDD: "sexually assaulted or harassed", on or off screen.
    "Is someone sexually assaulted": ["Is someone raped onscreen"],
    # Not audio gore: that is gore you hear, not gore you see.
    "Is there blood/gore": ["Is there excessive gore"],
    "Does someone attempt suicide": ["Does someone die by suicide"],
    "Does someone use drugs": ["Does someone overdose"],
    "Does someone struggle to breathe": ["Does someone asphyxiate", "Does someone drown"],
    "Is there hate speech": ["Are there n-words", "Are there homophobic slurs"],
    "Is there ableist language or behavior": ["Is the R word used"],
    "Is someone kidnapped": ["Is an infant abducted"],
    "Are there babies or unborn children": ["Does a baby cry", "Is an infant abducted", "Is a baby stillborn", "Is there childbirth"],
}
# Demo profiles, matching data/mock/users.json, by ML-DDD question.
DEMO_USERS = [
    ("arachno", "Alex", {"Are there spiders": "never", "Are there jumpscares": "warn"}),
    ("dog_lover", "Sam", {"Does the dog die": "never", "Is there blood/gore": "warn"}),
    ("fresh", "Jordan", {}),
]
_LABEL_COLUMNS = {"Clear Yes": "yes", "Clear No": "no", "Unclear": "unclear"}  # "No Votes" means no label


def _tsv_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        for row in reader:
            yield dict(zip(header, row))


def _csv(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _topic_rows(questions: list[str], dtdd_topics: list[dict], covers: dict[str, list[str]]) -> tuple[list[tuple], dict[str, int]]:
    by_does = {t["doesName"]: t for t in dtdd_topics}
    rows, ids = [], {}
    for question in questions:
        if question in RETIRED:
            topic_id, name, category = RETIRED[question]
            synonyms = [name, question]
        elif RENAMED.get(question, question) in by_does:
            t = by_does[RENAMED.get(question, question)]
            topic_id, name, category = t["id"], t["name"], t["TopicCategory"]["name"]
            synonyms = [name, t["doesName"], *(t.get("keywords") or "").split(",")]
        else:
            raise ValueError(f"ML-DDD question {question!r} has no DTDD topic; add it to RENAMED or RETIRED")
        ids[question] = topic_id
        rows.append((topic_id, name, question, category, synonyms))
    unknown = {q for broad, narrower in covers.items() for q in [broad, *narrower]} - ids.keys()
    if unknown:
        raise ValueError(f"COVERS names questions that ML-DDD doesn't have: {sorted(unknown)}")
    return [
        (topic_id, name, question, category,
         json.dumps(sorted(ids[q] for q in covers.get(question, []))),
         json.dumps(sorted({s.strip().lower() for s in synonyms if s.strip()})))
        for topic_id, name, question, category, synonyms in rows
    ], ids


def build(raw: Path, out: Path, dtdd_topics: list[dict], covers: dict[str, list[str]] = COVERS) -> dict[str, int]:
    """Write a fresh database to `out` from the raw files in `raw`. Returns row counts."""
    csv.field_size_limit(2**31 - 1)  # ML-DDD's user_ratings column holds whole rating dicts
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript(SCHEMA)

    print("movies: IMDb ratings and titles...", flush=True)
    ratings = {
        r["tconst"]: (float(r["averageRating"]), int(r["numVotes"]))
        for r in _tsv_gz(raw / "title.ratings.tsv.gz")
        if int(r["numVotes"]) >= MIN_VOTES
    }
    db.executemany(
        "INSERT INTO movies (movie_id, title, year, rating, num_votes, genres, source) VALUES (?, ?, ?, ?, ?, ?, 'imdb')",
        (
            (r["tconst"], r["primaryTitle"], int(r["startYear"]), *ratings[r["tconst"]],
             "" if r["genres"] == "\\N" else r["genres"])
            for r in _tsv_gz(raw / "title.basics.tsv.gz")
            if r["tconst"] in ratings and r["titleType"] in ("movie", "tvMovie")
            and r["isAdult"] == "0" and r["startYear"] != "\\N"
        ),
    )
    catalogue = {m for (m,) in db.execute("SELECT movie_id FROM movies")}
    imdb_id = {int(r["movieId"]): "tt" + r["imdbId"] for r in _csv(raw / "links.csv")}

    print("topics and labels: ML-DDD...", flush=True)
    with open(raw / "ml-ddd_sensitivity_table.csv", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        questions = sorted({c.split(": ", 1)[1] for c in header if c.startswith("Clear Yes: ")})
        topic_rows, topic_ids = _topic_rows(questions, dtdd_topics, covers)
        db.executemany("INSERT INTO topics VALUES (?, ?, ?, ?, ?, ?)", topic_rows)
        columns = [
            (i, topic_ids[col.split(": ", 1)[1]], _LABEL_COLUMNS[col.split(": ", 1)[0]])
            for i, col in enumerate(header) if col.split(": ", 1)[0] in _LABEL_COLUMNS
        ]
        work_id = header.index("work_id")
        db.executemany(
            "INSERT INTO labels VALUES (?, ?, ?, 'ml-ddd')",
            (
                (imdb_id[int(row[work_id])], topic_id, label)
                for row in reader if int(row[work_id]) in imdb_id
                for i, topic_id, label in columns if row[i] == "1"
            ),
        )

    print("genome: MovieLens tag genome...", flush=True)
    tags = [(int(r["tagId"]), r["tag"]) for r in _csv(raw / "genome-tags.csv")]
    assert [t for t, _ in tags] == list(range(1, len(tags) + 1)), "genome tag ids are not 1..n"
    db.executemany("INSERT INTO genome_tags VALUES (?, ?)", tags)
    vectors, current, done = [], None, set()
    for r in _csv(raw / "genome-scores.csv"):  # sorted by movieId, then tagId
        movie = int(r["movieId"])
        if movie != current:
            assert movie not in done, "genome-scores.csv is not sorted by movieId"
            done.add(movie)
            current, vector = movie, array("f", bytes(4 * len(tags)))
            if imdb_id.get(movie) in catalogue:
                vectors.append((imdb_id[movie], vector))
        vector[int(r["tagId"]) - 1] = float(r["relevance"])
    db.executemany("INSERT INTO genome VALUES (?, ?)", ((m, v.tobytes()) for m, v in vectors))

    db.executemany(
        "INSERT INTO users VALUES (?, ?, ?)",
        [(u, n, json.dumps({str(topic_ids[q]): s for q, s in prefs.items()})) for u, n, prefs in DEMO_USERS],
    )
    db.commit()
    counts = {t: db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("movies", "topics", "labels", "genome")}
    counts["catalogue movies with labels"] = db.execute(
        "SELECT count(DISTINCT movie_id) FROM labels WHERE movie_id IN (SELECT movie_id FROM movies)"
    ).fetchone()[0]
    db.close()
    tmp.replace(out)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/moviebot.db from the raw downloads.")
    parser.add_argument(
        "--force", action="store_true",
        help="replace an existing database, losing fetched overviews, on-demand labels and profile edits",
    )
    args = parser.parse_args(argv)
    settings = Settings()
    if settings.db_path.exists() and not args.force:
        print(f"{settings.db_path} exists; pass --force to rebuild it from scratch", file=sys.stderr)
        return 2
    key = settings.dtdd_key.get_secret_value()
    if not key:
        print("DTDD_KEY is not set in .env; the build needs it for DTDD's topic list", file=sys.stderr)
        return 2
    media = DtddClient(key).media(TOPIC_LIST_MOVIE)
    counts = build(RAW_DIR, settings.db_path, [s["topic"] for s in media["topicItemStats"]])
    print(f"wrote {settings.db_path}: " + ", ".join(f"{v:,} {k}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
