# Phobia-Aware Movie Chatbot — 5-Week Plan

Oct 15, 2026 · @Ernest

## Overview

We will ship a working chatbot that recommends movies and warns about each user's phobias, plus an evaluation showing it never recommends what users said to avoid. The deadline is about 5 weeks out, in the week of Nov 16, 2026.

One design rule shapes the build: phobia filtering is deterministic code over structured DoesTheDogDie (DTDD) data, not something the LLM judges from retrieved text. RAG handles the fuzzy parts, such as plot similarity, mood and "something like X". The LLM is the conversational layer that calls tools and phrases accurate warnings.

A movie with no DTDD votes on a topic is "unknown", never "safe".

Deliverables:

- A working web app with onboarding and chat
- Evaluation results, including an ablation against a naive RAG-only design
- Written report, slides and a demo video

## Team roles

Each person owns one layer end to end, so work runs in parallel from week 1 against agreed interfaces. Fill in names below.

| Role | Owns | Name |
| --- | --- | --- |
| Data lead | Kaggle cleanup, DTDD API ingestion and caching, movie–warning join, database schema |  |
| Retrieval lead | Embeddings, vector store, filtered search, mapping free-text phobias to DTDD topics |  |
| Agent lead | LLM tool calling, system prompt, multi-turn flow, user profile memory |  |
| UI lead | Onboarding topic picker, chat interface, warning display, profile editing |  |
| Eval lead and PM | Task board, integration, test sets, metrics, ablation, report assembly |  |

In week 1, the data, retrieval and agent leads agree on the tool signatures (`search_movies`, `get_movie_warnings`, `map_phrase_to_topics`, `get_user_profile`) so everyone can build against mocks.

## Week-by-week schedule

Core features freeze at the end of week 3, so weeks 4 and 5 go to evaluation, fixes and the write-up. Each week ends with a milestone the whole team checks together.

### Week 1 (Oct 15–21): foundations

Milestone: data loads into the database, and a mocked conversation runs end to end.

- [ ] Data: request the DTDD API key on day 1; pull the DTDD topic list; pick a 3–5k movie subset by vote count; set up the SQLite schema (`movies`, `topics`, `triggers`)
- [ ] Retrieval: choose an embedding model; prototype a Chroma index over plot summaries
- [ ] Agent: pick the LLM; build the tool-calling skeleton with mock tools
- [ ] UI: Streamlit skeleton; wireframe the onboarding flow
- [ ] Eval/PM: repo and task board; fix tool signatures with the team; draft the metric definitions

### Week 2 (Oct 22–28): real data, real tools

Milestone: a query like "no spiders, rating above 7" returns correct results, and filtered semantic search works.

- [ ] Data: fetch DTDD warnings for every movie in the subset and cache them; join on IMDb ID, falling back to title plus year; spot-check 50 joins
- [ ] Retrieval: filtered search (vector results, then trigger exclusion); free-text-to-topic mapper
- [ ] Agent: wire the real tools; write the system prompt with the warning rules (unknown is not safe; use vote counts)
- [ ] UI: onboarding topic picker grouped by category, with strictness levels (never show, warn me, fine)
- [ ] Eval/PM: write 50 phrase-mapping test cases and 30 test user profiles

### Week 3 (Oct 29–Nov 4): full conversations, then feature freeze

Milestone: a classmate completes onboarding and a multi-turn chat with no help.

- [ ] Data: confidence scores from vote counts; DTDD coverage stats; fix bad joins
- [ ] Retrieval: tune ranking (similarity plus rating); add keyword search for exact titles
- [ ] Agent: multi-turn context; profile edits in chat; explain exclusions; spoiler-light warnings with details on request
- [ ] UI: chat interface with warning badges; profile page
- [ ] Eval/PM: automated safety-recall harness; first full eval run

### Week 4 (Nov 5–11): evaluate and fix

Milestone: all evaluation numbers are in, and the failures they found are fixed.

- [ ] Data: support eval fixes; freeze the dataset
- [ ] Retrieval: build the ablation baseline (warnings embedded in RAG, LLM decides)
- [ ] Agent: fix failure cases from the eval; harden the prompt
- [ ] UI: user test with 5–8 classmates; polish from their feedback
- [ ] Eval/PM: run the full eval and ablation; collect relevance ratings

### Week 5 (Nov 12–18): ship

Milestone: submission. Code freezes on Nov 15.

- [ ] Everyone: write your own layer's report section
- [ ] UI and Agent: record the demo video
- [ ] Eval/PM: assemble the report and slides; run a rehearsal
- [ ] Everyone: final rehearsal and submit

## Tech stack and budget

The whole project should cost well under $50 if a small LLM handles development and a stronger one is kept for the final eval and demo. The figures below are approximate; check current provider pricing before choosing.

| Layer | Choice | Approx. cost |
| --- | --- | --- |
| Language | Python | Free |
| Structured data | SQLite (Postgres only if you deploy) | Free |
| Vector store | Chroma or FAISS, local | Free |
| Embeddings | Local sentence-transformers model, or a hosted small embedding model | Free locally; a few cents hosted for about 5k plots |
| LLM for development | A small, fast model with tool calling (e.g. Claude Haiku or Gemini Flash class) | About $5–20 over 4 weeks |
| LLM for final eval and demo | A stronger model, same tool interface | About $10–25 |
| UI | Streamlit or Gradio | Free |
| Hosting for the demo | Streamlit Community Cloud or Hugging Face Spaces | Free |

Cost controls:

- Put one shared API key behind a monthly spending cap
- Cache LLM responses during eval reruns
- Keep the model name in config so swapping models is a one-line change

## Evaluation plan

Safety recall is the headline metric: zero recommended movies should contain a confidently flagged "never show" topic. The ablation against a naive RAG design is the result most likely to impress graders.

| Metric | How it is measured | Target |
| --- | --- | --- |
| Safety recall | 30 test profiles × 10 queries; count recommendations that contain a "never show" topic with confident DTDD data | 0 violations |
| Warning accuracy | Check every warning the bot states against the database | 100% supported by data |
| Unknown handling | Queries about movies with no DTDD votes; the bot must say "no data" | 100% |
| Phrase mapping | 50 free-text phobia descriptions mapped to DTDD topics | 90% or higher |
| Recommendation relevance | 5–8 classmates rate suggestions from 1 to 5 | Average 3.5 or higher |
| Ablation | The same safety test on a baseline that puts warnings in RAG and lets the LLM decide | Show the baseline leaks; ours does not |

## Risks and mitigations

The biggest schedule risk is DTDD access, so request the API key on day 1.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| DTDD API key is slow to arrive, or rate limits are tight | Blocks the data layer | Request the key on day 1; read the terms; cache every response; build against a hand-made sample of 50 movies meanwhile |
| Sparse DTDD coverage for less popular movies | Many "unknown" results | Restrict the subset to well-known movies; report coverage as a stat |
| Wrong title matches attach the wrong warnings | Silent safety failures | Join on IMDb ID where possible; spot-check a random sample |
| LLM states warnings not in the data | Hallucinated or missing warnings | Warnings come only from tool output; check this in the warning-accuracy eval |
| Integration slips into week 4 | No time to evaluate | Agree on tool signatures in week 1; integration milestone at the end of week 3 |
| Scope creep (TV, books, extra features) | Core features unfinished | Movies only; new ideas go in a "future work" list for the report |
| API costs run over | Budget overrun | Spending cap; cheap model for development; cache eval runs |
