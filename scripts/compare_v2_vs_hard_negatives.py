"""Side-by-side search table: second fine-tune (title pairs) vs first fine-tune + hard negatives.

Both models are fused with SQLite-FTS5 BM25 at 80/20, the same rule as the demo site.
Searches the whole catalogue with overviews (training movies included), so the
tag hit counts are illustrative examples, not a held-out evaluation.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from train_search_local import bm25_ranks, movie_text, weighted_rrf  # noqa: E402

TAG_QUERIES = ['twist ending', 'heist', 'time travel', 'cyberpunk', 'feel-good', 'zombies',
               'dystopia', 'artificial intelligence', 'serial killer', 'tearjerker']
TITLE_QUERIES = ['the godfather', 'alien', 'young frankenstein', 'heat', 'jane eyre',
                 'inception', 'up', 'the matrix']
MODELS = {'v2': ('Tinh chỉnh lần 2 + BM25', ROOT / 'moviebot-db-old' / 'minilm-movie-search-titles'),
          'hn': ('Lần 1 + mẫu âm khó + BM25', ROOT / 'data' / 'old-hn-five-seeds' / 'seed-42')}
TOP_SHOW, TOP_COUNT, RELEVANT = 5, 10, 0.8


def tex(text: str) -> str:
    for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'), ('#', r'\#'),
                 ('_', r'\_'), ('{', r'\{'), ('}', r'\}'), ('~', r'\textasciitilde{}'), ('^', r'\^{}')]:
        text = text.replace(a, b)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, default=ROOT / 'moviebot-db-old' / 'moviebot.db')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, default=ROOT / 'temp' / 'compare_v2_vs_hn')
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    with sqlite3.connect(f'{args.db.resolve().as_uri()}?mode=ro', uri=True) as db:
        movies = [dict(zip(('id', 'title', 'year', 'genres', 'overview'), row)) for row in db.execute(
            "SELECT movie_id,title,year,genres,overview FROM movies WHERE coalesce(overview,'') <> '' ORDER BY movie_id")]
        tag_index = {tag.lower(): i for i, (tag,) in enumerate(db.execute('SELECT tag FROM genome_tags ORDER BY tag_id'))}
        genome = {m: np.frombuffer(b, dtype='f4') for m, b in db.execute('SELECT movie_id, relevance FROM genome')}
    documents = [movie_text(m['title'], m['year'], m['genres'], m['overview']) for m in movies]
    queries = TAG_QUERIES + TITLE_QUERIES
    lexical = bm25_ranks(documents, queries)

    results = {}
    for key, (_, path) in MODELS.items():
        print(f'Encoding catalogue with {path}', flush=True)
        model = SentenceTransformer(str(path), device=args.device)
        docs = model.encode(documents, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
        qv = model.encode(queries, normalize_embeddings=True)
        ranks = np.argsort(-(qv @ docs.T), axis=1)
        results[key] = weighted_rrf(ranks, lexical, 0.8, limit=len(documents))

    def relevant(query: str, idx: int) -> bool:
        tag = tag_index.get(query)
        vec = genome.get(movies[idx]['id'])
        return tag is not None and vec is not None and float(vec[tag]) >= RELEVANT

    def title_rank(query: str, order: np.ndarray) -> str:
        hits = [r for r, i in enumerate(order, 1) if movies[i]['title'].lower() == query]
        return str(hits[0]) if hits else '>' + str(len(order))

    md = ['# So sánh tìm kiếm: lần 2 + BM25 vs lần 1 + mẫu âm khó + BM25', '',
          f'Tìm trong {len(movies):,} phim có tóm tắt (gồm cả phim huấn luyện). BM25 FTS5, trọng số 80/20.', '']
    latex = []
    for kind, qs in [('tag', TAG_QUERIES), ('title', TITLE_QUERIES)]:
        head = 'Đúng/10' if kind == 'tag' else 'Hạng phim đúng'
        md += [f'## Truy vấn {"nội dung" if kind == "tag" else "tên phim"}', '',
               f'| Truy vấn | {MODELS["v2"][0]} | {head} | {MODELS["hn"][0]} | {head} |', '| --- | --- | ---: | --- | ---: |']
        for q in qs:
            qi = queries.index(q)
            cells_md, cells_tex = [], []
            for key in MODELS:
                order = results[key][qi]
                shown = []
                for i in order[:TOP_SHOW]:
                    m = movies[i]
                    name = f"{m['title']} ({m['year']})"
                    good = relevant(q, i) if kind == 'tag' else m['title'].lower() == q
                    shown.append((name, good))
                if kind == 'tag':
                    unlabeled = sum(movies[i]['id'] not in genome for i in order[:TOP_COUNT])
                    score = f'{sum(relevant(q, i) for i in order[:TOP_COUNT])}/10'
                    score += f' ({unlabeled} chưa nhãn)' if unlabeled else ''
                else:
                    score = title_rank(q, order)
                cells_md += ['; '.join(f'**{n}**' if g else n for n, g in shown), score]
                cells_tex += ['; '.join(rf'\textbf{{{tex(n)}}}' if g else tex(n) for n, g in shown), score]
            md.append(f'| {q} | ' + ' | '.join(cells_md) + ' |')
            latex.append(f'{tex(q)} & ' + ' & '.join(cells_tex) + r' \\')
        md.append('')
        latex.append('% ----')
    md.append(f'In đậm: phim có điểm tag genome ≥ {RELEVANT} với thẻ (truy vấn nội dung) hoặc đúng tên (truy vấn tên phim).')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    args.out.with_suffix('.tex').write_text('\n'.join(latex) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
