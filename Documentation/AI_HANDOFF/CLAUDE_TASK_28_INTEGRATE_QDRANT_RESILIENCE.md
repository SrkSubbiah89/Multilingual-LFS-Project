# Task 28 — Integrate Bounded Qdrant Resilience Into the Official WISCO Evidence Line

## Purpose

Integrate the audited Task 27 bounded, opt-in Qdrant timeout resilience and hierarchical telemetry work into the Task 26 official WISCO evidence line. This is a Git integration and test-verification task only. It must not run a preflight, smoke test, WISCO evaluation, benchmark rerun, analysis, collection build, or manuscript work.

This integration does not make Task 24 or Task 26 valid benchmark-comparison evidence. Task 24 remains invalid, and Task 26 remains `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no` because no valid full hierarchical CSV exists.

## Verified sources

1. Canonical evidence base:
   - branch: `reviewer2-wisco-official-tier1-telemetry-rerun-20260809`
   - SHA: `6d0453a310d44c91dd911f746bfc25b5db54dffd`
2. Audited feature source:
   - branch: `reviewer2-qdrant-timeout-resilience-20260809`
   - SHA: `e0b0649d43fa01dddd3c707bc7f4d2400b230ad2`

Fetch origin. Verify both SHAs locally and on origin, and start with a clean working tree. Stop and report if either SHA differs.

## Branch and merge rules

1. Create exactly one new branch from the canonical evidence base:
   - `reviewer2-wisco-timeout-resilience-baseline-20260809`
2. Merge the feature source using exactly an explicit no-fast-forward merge commit.
3. Do not cherry-pick, squash, reimplement, or change Task 27 logic during integration.
4. If there is a merge conflict, stop immediately. Do not resolve it. Record the conflict files and commit/push only a factual report if possible without changing source files.
5. Do not merge, rebase, reset, clean, stash, pull, force-push, open a PR, modify protected branches, or change historical output artifacts.

## Required preservation checks

Before and after the merge, verify all of the following with direct evidence:

1. Task 24's four raw artifacts remain byte-identical to their documented checksums.
2. Task 25 remains the diagnostic record for the original flat timeout and Task 25 telemetry remains present.
3. Task 26 remains exactly `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no`; its flat CSV/logs and failed hierarchical log remain untouched.
4. Task 27 retry configuration has these semantics exactly:
   - default `QDRANT_QUERY_MAX_ATTEMPTS=1` and zero backoff;
   - hard maximum three attempts;
   - hard maximum two-second backoff per retry;
   - retry only direct `httpx.TimeoutException` or `ResponseHandlingException` wrapping an `httpx.TimeoutException`;
   - no string-matching retry decision;
   - unknown, non-timeout, validation, authentication, authorization, rate-limit, schema, and generic errors remain non-retryable and fail closed.
5. Task 27 flat telemetry remains backward-compatible, and separate hierarchical-stage telemetry is present.
6. `--require-genuine-hierarchical` remains unchanged and fail-closed; it must still reject fallback, retry exhaustion, unavailable output, missing stage evidence, and excessive total stage latency.
7. `--max-stage-latency-ms` continues to cover every retry attempt and any retry backoff.
8. Official ILO catalogue/profile code and Task 23-built official collections are not changed by the merge.
9. B1 quarantine and B2 safety files remain byte-identical to the canonical base.

## Important safety clarification for future work

Record, but do not change in this task, the following configuration fact: with the default `QDRANT_TIMEOUT_SECONDS=30`, a retry-enabled stage could take longer than the 30,000 ms strict stage-latency threshold. This is not a flaw to hide or repair here. Any later live preflight must select and validate a lower per-attempt Qdrant timeout together with retry count/backoff so that the full stage duration, including retries, remains under the strict cap.

Do not select or test a future runtime configuration in this task. Do not make a live Qdrant call.

## Verification

Run focused tests covering:

- Task 27 retry and serialization tests;
- Task 25 flat telemetry tests;
- hierarchy engine/store tests;
- official ILO profile tests;
- strict hierarchical guard tests;
- `eval/run_eval.py` evaluator tests;
- WISCO leakage and existing fail-closed analysis tests.

Then run:

```text
pytest backend/tests eval/ -q
```

Expected result is a fully green suite with the Task 27 baseline of 2,129 passed, subject only to clearly explained environment-neutral differences. Do not weaken, skip, modify, or repair unrelated tests.

## Prohibited live operations

Do not perform any of these:

- Qdrant connection, mutation, query, or health check;
- SentenceTransformer/model load;
- Ollama, CrewAI, LLM, or paid API call;
- WISCO file export/build/read-write operation;
- benchmark/preflight/smoke/evaluation run;
- official or legacy collection build, deletion, overwrite, or mutation;
- accuracy, Wilson, McNemar, latency comparison, cost, or manuscript analysis.

## Required final report

Create `Documentation/AI_HANDOFF/CLAUDE_TASK_28_FINAL_REPORT.md` and include:

- both verified source branches and exact SHAs;
- integration branch, merge SHA, final report SHA, and push confirmation;
- conflict status and changed-file list from base to merge;
- every preservation check above with direct evidence;
- explicit confirmation of Task 27 retry allowlist/default/hard limits and strict-guard semantics;
- the future-configuration safety clarification, stated as a constraint rather than a measured result;
- focused and full test commands/results;
- confirmation of zero live operations;
- confirmation that all protected branches and historical raw artifacts are unchanged;
- mandatory limitation: no new benchmark evidence, accuracy/statistics, or paper claim was produced; WISCO remains controlled multilingual ISCO-08 evidence and not real Labour Force Survey validation.

Commit the merge commit and final report only on the new branch, push only that branch, and stop.
