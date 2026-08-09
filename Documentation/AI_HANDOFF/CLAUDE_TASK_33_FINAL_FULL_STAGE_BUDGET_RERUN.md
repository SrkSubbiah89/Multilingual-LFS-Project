# Task 33 — Final Full Official WISCO Run With Strict Stage Budget

## Purpose

Run one final fresh full controlled WISCO ISCO-08 evaluation using the official ILO 2021 profile, Task 27 bounded retry telemetry, and Task 31 shared strict stage-budget enforcement as operationally validated in Task 32. This task creates raw output evidence only. It must not calculate accuracy, confidence intervals, statistical tests, latency or cost comparisons, or manuscript claims.

Task 33 is authorized only because Task 32 completed with `STAGE_BUDGET_PREFLIGHT_READY: yes`. It is not a retry, replacement, or reinterpretation of Tasks 24, 26, or 30. Those outputs remain preserved and ineligible for flat-versus-hierarchical analysis.

## Starting point

1. Fetch origin and verify exactly:
   - base branch: `reviewer2-stage-budget-live-preflight-20260809`
   - base SHA: `f2ba8ccb03bab177616f3b39e596f050761fc0d4`
2. Start with a clean working tree.
3. Create exactly one branch:
   - `reviewer2-wisco-official-tier1-stage-budget-rerun-20260809`
4. Do not merge, rebase, reset, clean, stash, pull, force-push, open a PR, change protected branches, modify source/test code, alter WISCO/catalogue data, or mutate Qdrant collections.
5. If any condition fails, preserve the new output/logs, commit only the factual report, push the new branch, and stop. Do not rerun a process, case, selection, or full benchmark.

## Historical preservation

Before and after work, verify unchanged:

- Task 24 four raw artifacts.
- Task 25 diagnostic record.
- Task 26 invalid/empty hierarchical output evidence.
- Task 29 preflight artifacts.
- Task 30 invalid full-run raw evidence, including its complete flat CSV, hierarchical log, and empty hierarchical output directory.
- Task 31 source/report and Task 32 preflight artifacts.

Historical data must not be edited, reclassified, or used for accuracy or flat-versus-hierarchical analysis.

## Exact runtime configuration

Use exactly these environment variables for both evaluator processes:

```text
QDRANT_TIMEOUT_SECONDS=8
QDRANT_QUERY_MAX_ATTEMPTS=3
QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5
```

For strict hierarchy use exactly:

```text
--require-genuine-hierarchical
--max-stage-latency-ms 30000
```

Record these factual limitations:

- `query_points(timeout=...)` is a server-side Qdrant operation-timeout hint, not a guaranteed client-side cancellation mechanism.
- The shared stage budget covers all branch queries, retries, and backoffs, and blocks new work after the deadline.
- A query already in flight cannot be forcibly cancelled by the current code.
- This task must fail closed if an in-flight overrun ultimately causes a strict-guard or integrity failure.

## Full preflight

Before launching evaluators:

1. Validate WISCO v2 without rebuilding:
   - hash `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c`
   - 20,760 total / 2,013 dev / 18,747 heldout
   - zero leakage, zero cross-split duplicates, zero malformed ISCO codes.
2. Export a fresh full heldout CSV and verify exactly 18,747 unique ISCO-only rows, no industry/education columns, and 100% blank ISIC/ISCED gold columns.
3. Verify official normalized ILO catalogue hash `29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3`, counts 10/43/130/436.
4. Read Qdrant only and verify official counts 10/43/130/436/436 and legacy counts. Confirm the Task 23 build-success manifest exists.
5. Confirm Task 31 explicit timeout/stage-budget code and Task 32 preflight status are present.
6. Record before-run official and legacy collection counts. No mutation is allowed.

## Durable execution

For each evaluator, launch one durable background process from its first invocation. Record original PID, exact command, start/end time, exit code, and log path.

- No foreground evaluator invocation.
- No preliminary probe, duplicate process, alternative command, or post-failure relaunch.
- Monitor only the original PID until it exits.
- If a process is interrupted, killed, exits non-zero, or produces invalid output, preserve evidence and stop. Do not start another process.
- Use one unique UTC-timestamped, Git-ignored `eval/local_runs/` root. Never edit raw outputs/logs after they are produced.

## Full flat run

Run exactly once across all 18,747 rows with:

- `--system flat`
- `--use-llm-reranker off`
- `--isco-catalogue-profile official_ilo2021_v1`
- no record limit
- unique config/run ID/output directory.

### Flat integrity gate

All must pass:

1. Exit zero and 18,747 unique rows.
2. Every row has no error, valid four-digit ISCO code, exact method `flat_isco08_official_ilo2021_v1`.
3. No LLM/reranker/API/cost/token/ISIC/ISCED/SRE activity.
4. Every flat telemetry record is parseable and consistent: outcome, duration, attempts, and attempt duration list.
5. `success_after_retry` is allowed only with genuine final output and complete telemetry; report all such cases transparently.
6. Any final exception, retry exhaustion, unavailable result, coarse/missing code, malformed telemetry, output-cardinality issue, or row error fails the run.
7. Qdrant official and legacy counts remain unchanged.

If flat fails, set `OFFICIAL_TIER1_STAGE_BUDGET_RERUN_COMPLETED: no`, report, push, and stop without hierarchy.

## Full strict hierarchical run

Run only if flat passes, exactly once across the same 18,747 rows with:

- `--system hierarchical`
- `--use-llm-reranker off`
- `--isco-catalogue-profile official_ilo2021_v1`
- `--require-genuine-hierarchical`
- `--max-stage-latency-ms 30000`
- no record limit
- unique config/run ID/output directory.

### Strict hierarchical integrity gate

All must pass:

1. Exit zero and 18,747 unique rows.
2. Every row has no error, valid four-digit ISCO code, and exact method `hierarchical_isco08_official_ilo2021_v1`.
3. Every row has genuine complete four-stage evidence. No fallback, flat/unavailable method, missing/empty evidence, or stage latency at/above 30,000 ms.
4. `stage{i}_query_telemetry` is parseable on every relevant row/stage. Every successful row reports `initial_stage_budget_ms=30000.0` and `stage_budget_exhausted=false` in every stage.
5. Any stage budget exhaustion, final exception, retry exhaustion, malformed stage telemetry, or strict-guard abort fails the run.
6. Successful retries may pass only with genuine final evidence, complete telemetry, and full stage latency under 30,000 ms. Report every retry transparently with case ID, stage, collection, attempts, safe exception type, final outcome, and total stage latency.
7. `flat_query_*` fields are blank for all hierarchical rows.
8. No LLM/reranker/API/cost/token/ISIC/ISCED/SRE activity.
9. Qdrant official and legacy counts remain unchanged after both processes.

If hierarchy fails, preserve evidence, set `OFFICIAL_TIER1_STAGE_BUDGET_RERUN_COMPLETED: no`, report, push, and stop. Do not run analysis.

## Success boundary

Set `OFFICIAL_TIER1_STAGE_BUDGET_RERUN_COMPLETED: yes` only if both output gates pass. Even if yes:

- Do not calculate accuracy, confidence intervals, McNemar or any significance test, latency/cost comparison, or result manifest.
- Do not update paper/reviewer/guide documents.
- Do not claim real Labour Force Survey validation, ISIC/ISCED/SRE/reranking performance, or real-world effectiveness.
- WISCO remains controlled multilingual ISCO-08 benchmark evidence only.
- A separate independent analysis task must validate the raw output before any statistics or manuscript wording can be eligible.

## Tests

Run focused stage-budget, retry, telemetry, strict guard, official-profile, evaluator, and WISCO integrity tests. Then run:

```text
pytest backend/tests eval/ -q
```

Expected baseline is 2,156 passed. Report deviation factually and do not repair unrelated tests.

## Required final report

Create `Documentation/AI_HANDOFF/CLAUDE_TASK_33_FINAL_REPORT.md`. Include:

- base branch/SHA, new branch, and final pushed tip in Claude's final response;
- complete preflight hashes, data/collection integrity, and preservation checks;
- exact runtime configuration and limitations;
- exact one-time process launches with PID, timing, exit code, and logs;
- flat/hierarchical output paths, rows, method/error/code checks, retry and telemetry distributions, stage-budget data, maximum stage latency, and all anomalies;
- official/legacy Qdrant count diffs;
- exact `OFFICIAL_TIER1_STAGE_BUDGET_RERUN_COMPLETED: yes|no`;
- focused/full test results;
- confirmation of no collection mutation, protected-branch change, rerun, analysis, or manuscript claim;
- mandatory limitation: raw WISCO ISCO-08 pipeline evidence only, not statistics, real LFS validation, or ISIC/ISCED/SRE/reranking evidence.

Commit only the report, never local-run artifacts. Push only the new branch and stop.
