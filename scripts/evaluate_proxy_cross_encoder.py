"""Evaluate a cross-encoder reranker on held-out export proxy queries.

This uses only the export CSV and a proxy model trained without the held-out
anchor groups. It never accesses the Genome test database.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_export_hybrid_weights import BM25Index
from evaluate_proxy_relevance_rerank import metrics, terms
from train_search_local import export_proxy_split, load_export_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cross-encoder', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-queries', type=int, default=0, help='0 evaluates all proxy queries; otherwise use a seeded sample')
    parser.add_argument('--share', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite: {args.output}')
    if args.batch_size < 1 or args.max_queries < 0:
        raise SystemExit('batch size must be positive and max-queries nonnegative')
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    pairs = load_export_pairs(args.train_csv)
    _, validation, _ = export_proxy_split(pairs, seed=args.seed, share=args.share)
    documents = sorted({document for _, document in pairs})
    document_index = {document: index for index, document in enumerate(documents)}
    relevant = defaultdict(set)
    for anchor, document in validation:
        relevant[anchor].add(document_index[document])
    queries = sorted(relevant)
    if args.max_queries and args.max_queries < len(queries):
        queries = sorted(random.Random(args.seed).sample(queries, args.max_queries))
    # Load the cross-encoder first, so a missing model fails before expensive encoding.
    cross = CrossEncoder(args.cross_encoder, device=args.device, max_length=256)
    model = SentenceTransformer(str(args.model), device=args.device)
    doc_vectors = model.encode(documents, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    query_vectors = model.encode(queries, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    doc_terms = [terms(doc) for doc in documents]
    index = BM25Index(documents)
    totals = defaultdict(list)
    try:
        for start in range(0, len(queries), 16):
            batch_queries = queries[start:start + 16]
            scores = query_vectors[start:start + 16] @ doc_vectors.T
            candidates = []
            baselines = []
            for query, similarities in zip(batch_queries, scores):
                semantic_order = np.argsort(-similarities, kind='stable')
                semantic_rank = np.empty(len(documents), dtype=np.int32)
                semantic_rank[semantic_order] = np.arange(1, len(documents) + 1)
                lexical_rank = index.ranks(query, len(documents))
                fused = 0.8 / (60 + semantic_rank.astype(float))
                present = lexical_rank > 0
                fused[present] += 0.2 / (60 + lexical_rank[present])
                baseline = np.argsort(-fused, kind='stable')
                baselines.append(baseline)
                candidates.append(baseline[:100])
            cross_pairs = [(query, documents[int(i)]) for query, top in zip(batch_queries, candidates) for i in top]
            cross_scores = np.asarray(cross.predict(cross_pairs, batch_size=args.batch_size, show_progress_bar=False)).reshape(len(batch_queries), 100)
            for query, baseline, top, scores_for_query in zip(batch_queries, baselines, candidates, cross_scores):
                query_terms = terms(query)
                coverage = np.array([len(query_terms & doc_terms[i]) / len(query_terms) if query_terms else 0 for i in top])
                original = 1 / (60 + np.arange(1, len(top) + 1, dtype=float))
                normalized = (original - original.min()) / (original.max() - original.min())
                lexical = top[np.argsort(-(0.75 * normalized + 0.25 * coverage), kind='stable')]
                cross_order = top[np.argsort(-scores_for_query, kind='stable')]
                for name, ranking in [('hybrid_80_20', baseline), ('lexical_rerank', lexical), ('cross_encoder_rerank', cross_order)]:
                    totals[name].append(metrics(ranking, relevant[query]))
            print(f'Processed {min(start + 16, len(queries))}/{len(queries)} queries', flush=True)
    finally:
        index.close()
    rows = []
    for name, values in totals.items():
        mean = np.mean(values, axis=0)
        rows.append({'setup': name, 'p_at_10': round(float(mean[0]), 4), 'recall_at_10': round(float(mean[1]), 4), 'mrr_at_10': round(float(mean[2]), 4)})
    result = {'model': str(args.model), 'cross_encoder': args.cross_encoder, 'train_csv': str(args.train_csv), 'proxy_share': args.share, 'seed': args.seed, 'queries': len(queries), 'documents': len(documents), 'rows': rows, 'note': 'Anchor-held-out proxy; incomplete positive labels. Sample is exploratory and not the Genome final test.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
