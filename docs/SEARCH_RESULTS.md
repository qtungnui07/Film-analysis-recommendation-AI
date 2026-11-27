# Tag Search Optimization Results

Local training run on NVIDIA GeForce RTX 2060. The test split holds out 20% of movies and 20% of tag keys. `P@10` is the proportion of relevant movies among the top 10 returned results.

| Setup | Seen tags P@10 | Unseen tags P@10 |
| --- | ---: | ---: |
| First model alone | 0.390 | 0.327 |
| **First model + BM25 80/20** | **0.395** | **0.330** |
| First model + BM25 70/30 | 0.389 | 0.326 |
| First model + BM25 50/50 | 0.338 | 0.279 |
| Second model alone | 0.387 | 0.317 |
| Second model + BM25 80/20 | 0.393 | 0.322 |
| Second model + BM25 70/30 | 0.386 | 0.312 |
| Second model + BM25 50/50 | 0.334 | 0.274 |
| Untuned MiniLM | 0.197 | 0.175 |

## Conclusion

**First model + BM25 80/20** is the best configuration for tag-based movie search, reaching **0.395 Seen P@10** and **0.330 Unseen P@10**. It outperforms the untuned MiniLM baseline by `+0.198` and `+0.155` respectively.

The First model is trained with MovieLens tag signals and TMDB keywords. The Second model adds title-to-movie pairs, which improves title lookup but causes a small trade-off in tag-search performance.
