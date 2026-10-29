import csv
import gzip
import json
import sqlite3
from array import array

import pytest

from moviebot.build_db import build

QUESTIONS = [
    "Does the dog die", "Does an animal die", "Is there dog fighting", "Is there excessive gore",
    "Are there spiders", "Are there jumpscares", "Is there blood/gore",  # used by the demo users
]
DTDD_TOPICS = [
    {"id": 153, "name": "a dog dies", "doesName": "Does the dog die", "keywords": "Dog death, puppy dies",
     "TopicCategory": {"name": "Animal Death"}},
    {"id": 189, "name": "an animal dies", "doesName": "Does an animal die", "keywords": None,
     "TopicCategory": {"name": "Animal Death"}},
    {"id": 231, "name": "there's animal fighting", "doesName": "Is there animal fighting", "keywords": "",
     "TopicCategory": {"name": "Animal Distress"}},
    {"id": 165, "name": "there are spiders", "doesName": "Are there spiders", "TopicCategory": {"name": "Animal Phobia"}},
    {"id": 161, "name": "there are jump scares", "doesName": "Are there jump scares", "TopicCategory": {"name": "Fear"}},
    {"id": 188, "name": "there's blood/gore", "doesName": "Is there blood/gore", "TopicCategory": {"name": "Violence"}},
]


def write(path, rows, *, tsv=False):
    opener = gzip.open if tsv else open
    with opener(path, "wt", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t" if tsv else ",", quoting=csv.QUOTE_NONE if tsv else csv.QUOTE_MINIMAL,
                   escapechar="\\" if tsv else None).writerows(rows)


def sensitivity_row(work_id, answers):
    """One ML-DDD row; `answers` maps question -> 'Clear Yes' | 'Clear No' | 'Unclear', the rest get 'No Votes'."""
    kinds = ("Clear Yes", "Clear No", "Unclear", "No Votes")
    cells = [int(answers.get(q, "No Votes") == k) for q in QUESTIONS for k in kinds]
    return [work_id, *cells, 3, 4.0, "{1: 4.0, 2: 3.5}"]


@pytest.fixture
def raw(tmp_path):
    write(tmp_path / "title.ratings.tsv.gz", [
        ["tconst", "averageRating", "numVotes"],
        ["tt0000001", "8.1", "20000"], ["tt0000002", "7.0", "9999"], ["tt0000003", "6.0", "50000"],
        ["tt0000004", "5.0", "50000"], ["tt0000005", "9.0", "12000"],
    ], tsv=True)
    with gzip.open(tmp_path / "title.basics.tsv.gz", "wt", encoding="utf-8") as f:  # IMDb never quotes
        f.write("tconst\ttitleType\tprimaryTitle\toriginalTitle\tisAdult\tstartYear\tendYear\truntimeMinutes\tgenres\n"
                'tt0000001\tmovie\tA "Quoted" Film\tx\t0\t1999\t\\N\t100\tDrama,Sci-Fi\n'
                "tt0000002\tmovie\tToo Few Votes\tx\t0\t2000\t\\N\t90\tDrama\n"
                "tt0000003\tmovie\tAdult Film\tx\t1\t2001\t\\N\t90\tAdult\n"
                "tt0000004\ttvEpisode\tAn Episode\tx\t0\t2002\t\\N\t40\tDrama\n"
                "tt0000005\ttvMovie\tA TV Movie\tx\t0\t2003\t\\N\t80\t\\N\n")
    write(tmp_path / "links.csv", [["movieId", "imdbId", "tmdbId"], ["1", "0000001", "11"], ["2", "0000002", "22"]])
    header = ["work_id", *(f"{k}: {q}" for q in QUESTIONS for k in ("Clear Yes", "Clear No", "Unclear", "No Votes")),
              "n_ratings", "av_rating", "user_ratings"]
    write(tmp_path / "ml-ddd_sensitivity_table.csv", [
        header,
        sensitivity_row(1, {"Does the dog die": "Clear Yes", "Does an animal die": "Clear No",
                            "Is there excessive gore": "Unclear"}),
        sensitivity_row(2, {"Does the dog die": "Clear No"}),
        sensitivity_row(9, {"Does the dog die": "Clear Yes"}),  # not in links: dropped
    ])
    write(tmp_path / "genome-tags.csv", [["tagId", "tag"], ["1", "dark"], ["2", "funny"]])
    write(tmp_path / "genome-scores.csv", [
        ["movieId", "tagId", "relevance"], ["1", "1", "0.9"], ["1", "2", "0.1"], ["2", "1", "0.5"], ["2", "2", "0.5"],
    ])
    return tmp_path


def test_build_from_raw_files(raw):
    out = raw / "out.db"
    counts = build(raw, out, DTDD_TOPICS, covers={"Does an animal die": ["Does the dog die"]})
    db = sqlite3.connect(out)

    assert db.execute("SELECT movie_id, title, year, genres FROM movies ORDER BY 1").fetchall() == [
        ("tt0000001", 'A "Quoted" Film', 1999, "Drama,Sci-Fi"),
        ("tt0000005", "A TV Movie", 2003, ""),
    ]
    topics = {q: (t, json.loads(c), json.loads(s)) for t, q, c, s in db.execute("SELECT topic_id, question, covers, synonyms FROM topics")}
    assert topics["Is there dog fighting"][0] == 231  # renamed on DTDD
    assert topics["Is there excessive gore"][0] == 10009  # retired on DTDD
    assert topics["Does an animal die"][1] == [153]
    assert "puppy dies" in topics["Does the dog die"][2]
    assert sorted(db.execute("SELECT movie_id, topic_id, label FROM labels")) == [
        ("tt0000001", 153, "yes"), ("tt0000001", 189, "no"), ("tt0000001", 10009, "unclear"),
        ("tt0000002", 153, "no"),  # outside the catalogue, kept for a later TMDB fallback
    ]
    (movie_id, blob), = db.execute("SELECT movie_id, relevance FROM genome")
    assert movie_id == "tt0000001" and array("f", blob).tolist() == pytest.approx([0.9, 0.1])
    assert json.loads(db.execute("SELECT topic_prefs FROM users WHERE user_id = 'arachno'").fetchone()[0]) == {
        "165": "never", "161": "warn",
    }
    assert counts["catalogue movies with labels"] == 1


def test_build_refuses_unmapped_questions(raw):
    with pytest.raises(ValueError, match="no DTDD topic"):
        build(raw, raw / "out.db", DTDD_TOPICS[1:], covers={})
    with pytest.raises(ValueError, match="COVERS"):
        build(raw, raw / "out.db", DTDD_TOPICS, covers={"Does a unicorn die": ["Does the dog die"]})
