"""Descriptive comparison of a fixed hybrid model with cross-encoder reranking.

This reads the previously inspected Genome test split for aggregate reporting.
Do not use these results as an untouched final test after model selection.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_export_hybrid_weights import BM25Index, load_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-db', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cross-encoder', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite {args.output}')
    if args.batch_size < 1:
        raise SystemExit('batch size must be positive')
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    _, movies, queries, _ = load_evaluation(args.evaluation_db)
    documents = [movie['doc'] for movie in movies]
    ids = [movie['tconst'] for movie in movies]
    cross = CrossEncoder(args.cross_encoder, device=args.device, max_length=256)
    model = SentenceTransformer(str(args.model), device=args.device)
    document_vectors = model.encode(documents, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    query_vectors = model.encode([q for q, _, _ in queries], batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    index = BM25Index(documents)
    totals = defaultdict(list)
    try:
        for start in range(0, len(queries), 16):
            batch = queries[start:start + 16]
            scores = query_vectors[start:start + 16] @ document_vectors.T
            baselines, candidates = [], []
            for (query, _, _), similarities in zip(batch, scores):
                semantic = np.argsort(-similarities, kind='stable')
                ranks = np.empty(len(documents), dtype=np.int32)
                ranks[semantic] = np.arange(1, len(documents) + 1)
                lexical = index.ranks(query, len(documents))
                fused = 0.8 / (60 + ranks.astype(float))
                present = lexical > 0
                fused[present] += 0.2 / (60 + lexical[present])
                ordered = np.argsort(-fused, kind='stable')
                baselines.append(ordered)
                candidates.append(ordered[:100])
            cross_pairs = [(query, documents[int(i)]) for (query, _, _), top in zip(batch, candidates) for i in top]
            cross_scores = np.asarray(cross.predict(cross_pairs, batch_size=args.batch_size, show_progress_bar=False)).reshape(len(batch), 100)
            for (_, group, truth), baseline, top, scores_for_query in zip(batch, baselines, candidates, cross_scores):
                reranked = top[np.argsort(-scores_for_query, kind='stable')]
                for name, ranking in [('hybrid_80_20', baseline), ('cross_encoder_top100', reranked)]:
                    hits = sum(ids[int(i)] in truth for i in ranking[:10])
                    recall100 = sum(ids[int(i)] in truth for i in ranking[:100]) / len(truth)
                    totals[(name, group)].append((hits / 10, recall100))
            print(f'Processed {min(start + 16, len(queries))}/{len(queries)} content queries', flush=True)
    finally:
        index.close()
    rows = []
    for name in ('hybrid_80_20', 'cross_encoder_top100'):
        row = {'setup': name}
        for group in ('seen', 'unseen'):
            values = np.asarray(totals[(name, group)])
            row[f'{group}_queries'] = len(values)
            row[f'{group}_p_at_10'] = round(float(values[:, 0].mean()), 4)
            row[f'{group}_recall_at_100'] = round(float(values[:, 1].mean()), 4)
        rows.append(row)
    payload = {'model': str(args.model), 'cross_encoder': args.cross_encoder, 'evaluation_db': str(args.evaluation_db), 'candidate_count': len(movies), 'rows': rows, 'warning': 'Previously inspected test split: descriptive comparison, not an untouched final evaluation.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
