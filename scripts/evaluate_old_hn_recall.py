"""Compute P@10 and R@100 for the existing old-data hard-negative runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from train_search_local import bm25_ranks, load_data, semantic_ranks, tag_key, weighted_rrf


def main() -> None:
    output = Path('data/old-hn-five-seeds/recall-summary.json')
    if output.exists():
        raise SystemExit(f'Refusing to overwrite {output}')
    names, _, texts, genome, train_movies, test_keys = load_data(Path('moviebot-db-old'), seed=42, test_share=0.2)
    movies = sorted(set(texts) - train_movies)
    truth = np.stack([genome[movie] for movie in movies]) >= 0.8
    usable = [i for i, n in enumerate(truth.sum(axis=0)) if 5 <= n <= 100]
    seen = [i for i in usable if tag_key(names[i]) not in test_keys]
    unseen = [i for i in usable if tag_key(names[i]) in test_keys]
    indices = seen + unseen
    queries = [names[i] for i in indices]
    documents = [texts[movie] for movie in movies]
    lexical = bm25_ranks(documents, queries)

    def score(ranking):
        groups = {}
        for label, positions in [('seen', range(len(seen))), ('unseen', range(len(seen), len(indices)))]:
            p = [truth[ranking[j, :10], indices[j]].sum() / 10 for j in positions]
            r = [truth[ranking[j, :100], indices[j]].sum() / truth[:, indices[j]].sum() for j in positions]
            groups[label] = {'p10': round(float(np.mean(p)), 6), 'r100': round(float(np.mean(r)), 6)}
        return groups

    paths = [('baseline', Path('moviebot-db-old/minilm-movie-search')),
             ('hn_single', Path('data/old-original-hn-recheck'))]
    paths += [(f'hn_seed_{seed}', Path(f'data/old-hn-five-seeds/seed-{seed}')) for seed in range(42, 47)]
    results = {}
    for label, path in paths:
        print(f'Evaluating {label}', flush=True)
        model = SentenceTransformer(str(path), device='cuda')
        ranking = semantic_ranks(model, documents, queries, 64)
        results[label] = {'semantic': score(ranking), 'hybrid_80_20': score(weighted_rrf(ranking, lexical, 0.8, limit=100))}
    payload = {'seen_queries': len(seen), 'unseen_queries': len(unseen), 'results': results,
               'note': 'Old split previously inspected; 5 runs vary hard-negative trainer seed only.'}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
