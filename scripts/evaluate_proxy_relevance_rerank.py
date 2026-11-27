"""Compare hybrid retrieval and an experimental lexical reranker on held-out proxy queries.

The proxy model must have been trained with the matching anchor-level split.
Only the export CSV is read; this script does not read the Genome test DB.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_export_hybrid_weights import BM25Index
from train_search_local import export_proxy_split, load_export_pairs


def terms(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold(), re.UNICODE))


def metrics(order: np.ndarray, truth: set[int]) -> tuple[float, float, float]:
    top = order[:10]
    hits = [int(int(index) in truth) for index in top]
    first = next((i for i, hit in enumerate(hits, 1) if hit), None)
    return sum(hits) / 10, sum(hits) / len(truth), 1 / first if first else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--train-csv', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--share', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite: {args.output}')
    from sentence_transformers import SentenceTransformer
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    pairs = load_export_pairs(args.train_csv)
    _, validation, _ = export_proxy_split(pairs, seed=args.seed, share=args.share)
    documents = sorted({document for _, document in pairs})
    doc_index = {document: i for i, document in enumerate(documents)}
    relevant = defaultdict(set)
    for anchor, document in validation:
        relevant[anchor].add(doc_index[document])
    queries = sorted(relevant)
    model = SentenceTransformer(str(args.model), device=args.device)
    document_vectors = model.encode(documents, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)
    query_vectors = model.encode(queries, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)
    doc_terms = [terms(doc) for doc in documents]
    bm25 = BM25Index(documents)
    totals = defaultdict(list)
    for start in range(0, len(queries), 32):
        score_matrix = query_vectors[start:start + 32] @ document_vectors.T
        for query, scores in zip(queries[start:start + 32], score_matrix):
            semantic_order = np.argsort(-scores, kind='stable')
            semantic_rank = np.empty(len(documents), dtype=np.int32)
            semantic_rank[semantic_order] = np.arange(1, len(documents) + 1)
            lexical_rank = bm25.ranks(query, len(documents))
            fused = 0.8 / (60 + semantic_rank.astype(float))
            mask = lexical_rank > 0
            fused[mask] += 0.2 / (60 + lexical_rank[mask])
            baseline = np.argsort(-fused, kind='stable')
            top100 = baseline[:100]
            query_terms = terms(query)
            # Fixed blend: first-stage ranking remains dominant. No rating or labels enter the score.
            coverage = np.array([len(query_terms & doc_terms[i]) / len(query_terms) if query_terms else 0 for i in top100])
            original = 1 / (60 + np.arange(1, len(top100) + 1, dtype=float))
            normalized = (original - original.min()) / (original.max() - original.min())
            order = top100[np.argsort(-(0.75 * normalized + 0.25 * coverage), kind='stable')]
            for name, ranking in [('hybrid_80_20', baseline), ('hybrid_plus_lexical_rerank', order)]:
                totals[name].append(metrics(ranking, relevant[query]))
        if (start // 32) % 20 == 0:
            print(f'Processed {min(start + 32, len(queries))}/{len(queries)} queries', flush=True)
    bm25.close()
    rows = []
    for name, values in totals.items():
        mean = np.mean(values, axis=0)
        rows.append({'setup': name, 'p_at_10': round(float(mean[0]), 4), 'recall_at_10': round(float(mean[1]), 4), 'mrr_at_10': round(float(mean[2]), 4)})
    result = {'model': str(args.model), 'train_csv': str(args.train_csv), 'proxy_share': args.share, 'seed': args.seed, 'queries': len(queries), 'documents': len(documents), 'rows': rows, 'note': 'Anchor-held-out proxy; incomplete positive labels. Not Genome final test.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
