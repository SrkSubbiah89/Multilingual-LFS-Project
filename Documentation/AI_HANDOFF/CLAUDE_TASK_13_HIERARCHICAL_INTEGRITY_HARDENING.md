# Task 13 — Eliminate Silent Hierarchical Fallbacks and Add Bounded Evaluation Guards

## Why this task is required

Task 12 stopped correctly. Its full WISCO run is unusable as benchmark evidence because 23 of 18,747 hierarchical rows were actually `flat_semantic` fallbacks.

Direct code inspection identifies a reproducible code-path risk for 21 fast fallbacks:

1. `ISCOClassifier` supplies a keyword-derived `major_hint` by default.
2. `HierarchicalISCOStore._hierarchical_search()` treats that hint as a one-branch `SeedSpec` and skips semantic stage 1.
3. If that single anchored branch reaches no stage-4 unit-group result, the generic engine returns `None`.
4. `HierarchicalISCOStore.search()` silently calls `_flat_search()` and returns a successful `flat_semantic` result with duplicated candidate data under stages 1–4.

This is a real integrity problem for a hierarchical benchmark, regardless of whether the hint itself is wrong or a collection/filter edge case produced no child hit. The full-run report also exposed multi-minute-to-multi-hour individual stage stalls. The exact root cause of those stalls is not proven, but the current ISCO Qdrant client has no explicit request timeout and the evaluator has no strict fail-fast guard.

## Starting point

1. Fetch and verify the exact remote tip:
   - Base branch: `reviewer2-wisco-tier1-results-20260808`
   - Required base SHA: `6710192ea255c2cfb00599db9e55f65f9a53ca53`
2. Begin with a clean working tree.
3. Create a new feature branch:
   - `reviewer2-hierarchical-integrity-hardening-20260808`
4. Preserve the failed Task 12 raw output and report unchanged. Do not modify or delete any local ignored artifact.
5. Do not merge, rebase, reset, clean, stash, pull, force-push, create a PR, or alter protected/prior branches.

## A. Recover hierarchical retrieval after a failed keyword anchor

In `backend/rag/hierarchical_store.py` and only the minimum connected code:

1. Keep the existing keyword-major-hint fast path when it produces a genuine hierarchical result. Default production behavior for a successful hint must stay unchanged.
2. When `major_hint` was used and the seeded engine call returns `None`, retry the **same existing generic hierarchy engine** exactly once with no seed and normal semantic stage-1 retrieval before considering flat fallback.
3. Do not add a new retrieval algorithm, fabricate candidates, broaden filters, or replace parent-filtered retrieval.
4. The final trace after a successful retry must show genuine final-path stage 1–4 evidence. It must not retain the failed seeded stage as if it were the winning retrieval path.
5. Add transparent trace metadata documenting that a keyword-anchor retry happened and the original hint, without exposing any sensitive input. If the existing CSV schema needs an additive field to preserve that metadata, add it with an empty/default value and document it. Do not overload `stage1_source` misleadingly.
6. If the unseeded retry also cannot produce a complete hierarchical path, retain the existing production flat fallback behavior. The fallback must remain explicitly identifiable through `fallback_used=True` and method label, never silently relabelled as hierarchical.

## B. Bound Qdrant request waiting without claiming the Task 12 root cause

1. Add an explicit configurable Qdrant-client request timeout for `HierarchicalISCOStore`.
2. The default must be finite and conservative, such as 30 seconds, with a documented `QDRANT_TIMEOUT_SECONDS` environment override. Validate malformed/non-positive values fail safely to the documented default rather than crashing startup.
3. Preserve dependency injection/testability. Do not make a live Qdrant call in tests.
4. Preserve existing query-exception behavior: a timed-out or failed stage query returns no hits, follows the explicit fallback/retry logic, and is never represented as genuine hierarchical success.
5. Do not state that the timeout definitively fixes the Task 12 multi-hour stalls. State only that it bounds a class of blocked Qdrant requests and makes a future run diagnosable.

## C. Add a strict evaluation-only guard

In `eval/run_eval.py`, add an opt-in flag named clearly, for example `--require-genuine-hierarchical`.

1. It is valid only with `--system hierarchical`; reject invalid combinations at argument validation.
2. When enabled, any `fallback_used=True`, non-hierarchical `pred_method`, missing stage evidence, or stage latency above an explicit opt-in threshold must fail the run immediately and non-zero. Do not write a normal successful result CSV that can be mistaken for valid evidence.
3. Add an opt-in `--max-stage-latency-ms` argument. It has no behavior unless supplied. It must be a positive numeric value; reject zero/negative/invalid input.
4. Check every recorded stage 1–4 latency after classification. If a configured maximum is exceeded, stop with a clear error naming the stage and observed value. This is a guard against unrepresentative stalled timing data; it does not prove causation.
5. Default runs without these flags must retain current behavior, including explicit fallback labelling. Do not change dry-run behavior.
6. The strict guard should be evaluative only. It must not hide, overwrite, retry, or reinterpret a fallback as a successful hierarchical prediction.

## D. Tests

Add focused hermetic tests, using fake Qdrant clients, fake embedders, and fake classifier/store outputs only. Do not use the WISCO dataset, live Qdrant, SentenceTransformer download, Ollama, LLM/API, or any benchmark command.

At minimum prove:

1. A successful keyword seed performs one engine search and preserves prior successful behavior.
2. A seed that produces no complete path retries exactly once without a seed, then returns a genuine hierarchical result when the unseeded engine succeeds.
3. The retry trace contains only the final winning stage evidence in stage 1–4 fields, plus separate truthful retry metadata.
4. If both seeded and unseeded paths fail, flat fallback remains explicitly labelled, with duplicated flat trace candidates still distinguishable as fallback evidence.
5. The Qdrant timeout default, valid environment override, invalid override fallback, and client construction are covered without a live connection.
6. Timeout/query exceptions do not appear as hierarchical success.
7. `--require-genuine-hierarchical` rejects non-hierarchical system selections.
8. A strict hierarchical run fails non-zero on a fallback, missing stage evidence, or exceeded configured stage-latency threshold.
9. A valid genuine four-stage result passes the strict guard.
10. Ordinary runs without strict flags preserve current fallback-labelling and dry-run behavior.
11. Task 09 model-free behavior remains intact: reranker-off mode does not initialize an LLM, and WISCO-style rows do not construct ISIC/ISCED/SRE.

Do not weaken, skip, xfail, or delete existing tests merely to make the suite pass.

## E. Documentation

Update only directly affected documentation:

- `Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`
- the hierarchical retrieval implementation documentation if it explains fallback behavior

Document the difference between:

- a successful keyword-anchor hierarchical route;
- a recovered semantic-stage hierarchical route after a failed keyword anchor;
- an explicit flat fallback;
- strict evaluation mode, which refuses to treat fallback/stalled rows as valid hierarchical benchmark evidence.

Create:

- `Documentation/AI_HANDOFF/CLAUDE_TASK_13_FINAL_REPORT.md`

The report must include the starting/final SHA and branch; exact changed files; a clear distinction between the verified silent-fallback code path and the unconfirmed environment-stall hypothesis; test commands and actual results; confirmation that Task 12 raw artifacts were not modified; confirmation of zero live Qdrant/model/LLM/benchmark use; protected-branch status; working-tree status; and residual manuscript limits.

## Required verification

Run focused tests first, including the new tests and existing hierarchy/evaluator tests. Then run:

```bash
pytest backend/tests eval/ -q
```

Report the exact result. Do not call the suite green unless the full command has zero failures.

## Git boundary and stop condition

Commit only scoped source, tests, docs, and the final report to `reviewer2-hierarchical-integrity-hardening-20260808`, then push only that branch. Do not create a PR or merge it.

Stop after the pushed branch, final report, full-suite verification, and clean-tree check. Do not rerun WISCO, build collections, or produce any new benchmark metric. A separate task will independently audit and integrate this fix before a stricter preflight and rerun are considered.
