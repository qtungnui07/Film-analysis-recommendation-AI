# Movie Search Optimization Plan

> Vietnamese version: [SEARCH_OPTIMIZATION_PLAN_VI.md](SEARCH_OPTIMIZATION_PLAN_VI.md).

## Goal and current baseline

Improve tag-based retrieval while preserving a valid held-out evaluation. The current best tag-search setup is **First model + BM25 80/20**:

| Metric | Current result |
| --- | ---: |
| Seen tags P@10 | 0.395 |
| Unseen tags P@10 | 0.330 |

`Seen` and `unseen` tags, and 20% of movies, are held out by the fixed seed-42 split. This test split is frozen: it must never supply training pairs, augmented phrases, or mined negatives.

## Phase 0 — Establish a validation split

Before tuning, reserve a validation set *inside the current training data*.

- Keep the current test movies and test tag keys unchanged for final reporting.
- From the remaining training data, hold out 10% of movies and 10% of tag keys with a fixed second seed.
- Mine and train only on the remaining inner-training subset.
- Choose thresholds, number of negatives, epochs and learning rate from validation P@10.
- Run the frozen test exactly once for the selected configuration.

This prevents repeatedly selecting settings that happen to score well on the final test split.

## Phase 1 — Hard-negative mining (first implementation)

### Objective

Teach the embedding model that a film can look semantically plausible for a tag but still be irrelevant. Random negatives are usually too easy and give little ranking signal.

### Inputs and rules

For every train-only tag query:

| Item | Rule |
| --- | --- |
| Positive movie | Genome relevance `>= 0.80` |
| Candidate negative | Genome relevance `<= 0.10` |
| Hard negative | Candidate retrieved in the baseline model's top 100 for that tag |
| Scope | Only inner-training movies and non-held-out tag keys |
| Quantity | Begin with 1 negative per positive; evaluate 2 and 3 only after validation |

The top-100 retrieval condition makes the negative difficult: it is a film the baseline model considers close to the tag even though the MovieLens relevance signal says it is not relevant.

### Pipeline

1. Train or load the existing **First model**.
2. Embed only the inner-training movie documents and eligible tag queries.
3. Retrieve each query's top 100 movies with the First model.
4. Remove positives and any movie whose relevance is above `0.10`.
5. Select the highest-ranked remaining candidates as hard negatives; keep distinct negatives when a query has several positives.
6. Build triples: `(tag query, positive movie text, hard-negative movie text)`.
7. Fine-tune the First model for one additional epoch with a lower learning rate (`5e-6`) using a contrastive loss that consumes explicit negatives.
8. Evaluate on validation; keep the checkpoint only if both tag metrics improve or the trade-off is explicitly justified.
9. Evaluate the chosen checkpoint once on the frozen test and append its results to the report.

### Initial experiment matrix

Run these experiments in this order, selecting from validation results only:

| Run | Top-K mining | Negative cutoff | Negatives / positive | Epochs | Learning rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| HN-0 | — | — | 0 | — | — |
| HN-1 | 100 | `<= 0.10` | 1 | 1 | `5e-6` |
| HN-2 | 100 | `<= 0.10` | 2 | 1 | `5e-6` |
| HN-3 | 50 | `<= 0.10` | 1 | 1 | `5e-6` |

`HN-0` is the current First-model baseline. Stop the matrix if validation regresses consistently; this suggests noisy negatives rather than a need for more epochs.

### Implementation deliverables

- Extend `scripts/train_search_local.py` with a `--hard-negatives` phase.
- Add parameters for mining model/checkpoint, top-K, cutoff, negatives per positive, and fine-tuning epochs.
- Save mined triples and their source relevance scores as a CSV/JSON artifact for inspection.
- Save the post-mining model separately, for example `data/minilm-movie-search-hard-negatives/`.
- Add deterministic tests for split isolation and negative selection.

### Success criteria

- No validation or frozen-test item appears in a mined triple.
- Every selected hard negative has Genome relevance `<= 0.10`.
- The selected configuration improves validation `Seen P@10` without a material `Unseen P@10` regression.
- Report final frozen-test metrics alongside the current `0.395 / 0.330` baseline.

## Phase 2 — Balanced query sampling

If hard negatives help or are neutral, balance the number of pairs per tag query.

- Cap extremely frequent tags at a chosen maximum (start with 30–50 positives/tag).
- Oversample rare train-only tags to a minimum count.
- Evaluate separately by tag frequency, not only the overall mean.

This matters because P@10 averages tags equally, while the current pair pool is dominated by frequent tags.

## Phase 3 — Query augmentation

Add train-only phrase variants for tags: singular/plural forms, hyphenation, spelling variants and carefully reviewed paraphrases.

- Generate variants after creating the split.
- Never augment held-out tag keys or use test queries.
- Keep an audit file from each generated phrase to its source tag.

This phase is expected to help `Unseen tags P@10` more than simple pair balancing.

## Phase 4 — Source ablation and deployment decision

Compare the value and noise of each positive-pair source:

1. Genome relevance only
2. Genome + MovieLens user tags
3. Genome + TMDB keywords
4. All sources

For tag search, retain the configuration with the best frozen-test P@10. For a live application that also needs title lookup, evaluate the title-aware Second model separately and document its tag-versus-title trade-off.
