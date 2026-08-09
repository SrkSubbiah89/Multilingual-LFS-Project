# Task 30 — Full Official WISCO ISCO-08 Rerun With Bounded Retry Telemetry

## Purpose

Run one fresh full controlled WISCO ISCO-08 evaluation using the official ILO 2021 catalogue profile and the Task 29 validated bounded retry configuration. This task produces raw benchmark evidence only. It must not compute accuracy, confidence intervals, significance tests, latency/cost comparison tables, or manuscript claims.

This is the first full rerun authorized after Task 29 established `RETRY_CONFIGURATION_PREFLIGHT_READY: yes`. It is not a retry or modification of Task 24 or Task 26. Those historical outputs remain preserved and invalid for comparison.

## Starting point

1. Fetch origin and verify exactly:
   - base branch: `reviewer2-wisco-retry-config-preflight-20260809`
   - base SHA: `10c5f1a8755a375e373ca38ab49cbe269d9426d1`
2. Begin with a clean working tree.
3. Create exactly one new branch:
   - `reviewer2-wisco-official-tier1-retry-rerun-20260809`
4. Do not merge, rebase, reset, clean, stash, pull, force-push, open a PR, change protected branches, edit source/test code, alter the official/WISCO data, or change Qdrant collections.
5. If any preflight or integrity gate fails, preserve new output, commit only a factual final report, push the new branch, and stop. Do not rerun a case, a system, or the full benchmark.

## Historical evidence preservation

Before and after the new evaluation, verify and record:

- Task 24's four raw artifacts match their documented checksums.
- Task 25 diagnostic evidence is unchanged.
- Task 26 still reads `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no` and its failed hierarchical output directory remains empty.
- Task 29 report remains unchanged; its 525-case selected export and its output artifacts must not be edited.
- The Task 27 retry semantics are present: default one attempt/zero backoff; hard limit three attempts/two-second backoff; direct or wrapped `httpx.TimeoutException` only; no string matching; unknown/non-timeout errors fail closed.

## Exact runtime configuration

For every evaluator process in this task, use exactly:

```text
QDRANT_TIMEOUT_SECONDS=8
QDRANT_QUERY_MAX_ATTEMPTS=3
QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5
```

Use Python to record the configuration calculation:

```text
3 × 8 seconds + 2 × 0.5 seconds = 25,000 ms
```

The strict hierarchical cap remains 30,000 ms. Record that the calculation is a configuration bound and strict runtime guarding is the empirical enforcement.

## Full dataset and collection preflight

Before any evaluator process:

1. Validate the WISCO v2 benchmark package without rebuilding it:
   - dataset hash `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c`
   - 20,760 records total, 2,013 dev, 18,747 heldout
   - zero source-family leakage, zero cross-split duplicates, zero malformed ISCO codes
   - fresh heldout export exactly 18,747 rows, unique IDs, ISCO-only, no industry/education text, and 100% blank ISIC/ISCED gold fields.
2. Verify the official normalized ILO catalogue hash:
   - `29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3`
   - counts 10 / 43 / 130 / 436.
3. Read local Qdrant only. Verify official counts exactly:
   - major 10
   - submajor 43
   - minor 130
   - unit 436
   - unit-flat 436
4. Record legacy counts and confirm the Task 23 build-success manifest exists.
5. Record before-run official and legacy counts. No mutation is allowed.
6. Verify Task 29's preflight artifacts remain unchanged before beginning.

## Durable execution requirement

Use a durable background process from the first invocation of each evaluator command. Record the original process identifier, start time, end time, exit code, and log path.

- Do not start a foreground evaluator command that could be killed by a tool/session timeout.
- Do not launch a preliminary probe, duplicate command, or alternate evaluator command.
- Monitor the same original process until completion.
- If the original process is interrupted, killed, exits non-zero, or fails to write a valid output, preserve its log and stop. Do not launch another process.

Every new output must be written under a new, Git-ignored `eval/local_runs/` root with a unique UTC timestamp. Do not edit raw CSVs or logs after they are written.

## Full flat run

Run exactly once on all 18,747 heldout rows using only supported CLI flags:

- `--system flat`
- `--use-llm-reranker off`
- `--isco-catalogue-profile official_ilo2021_v1`
- no record limit
- unique config/run identifier and output directory

### Flat integrity gate

All conditions are mandatory:

1. Original process exits zero and exactly 18,747 unique output rows exist.
2. Every row has no row-level error, a valid four-digit ISCO-08 code, and exactly `flat_isco08_official_ilo2021_v1` as its method.
3. No reranker/LLM/API activity, cost, token use, ISIC, ISCED, or SRE activity occurs.
4. Every row has well-formed flat telemetry: outcome, non-negative duration, positive attempt count, and attempt-duration list matching the count.
5. A `success_after_retry` row is permitted only if its final valid code/method is present and all retry telemetry is retained. Report every such row count, collection/stage if available, exception type, and duration distribution. Do not hide it or classify it as a zero-hit result.
6. Any `exception`, `retry_exhausted`, unavailable method, invalid/missing/coarse code, malformed telemetry, duplicate/missing/extra row, or row-level error fails the entire flat gate.
7. Official and legacy Qdrant counts after the flat run must exactly match the before-run inventory.

If the flat gate fails, set `OFFICIAL_TIER1_RETRY_RERUN_COMPLETED: no`, preserve output/logs, report the exact failure, commit only the report, push, and stop. Do not run hierarchical evaluation.

## Full strict hierarchical run

Run only if the flat gate passes. Run exactly once on the same 18,747 heldout rows using:

- `--system hierarchical`
- `--use-llm-reranker off`
- `--isco-catalogue-profile official_ilo2021_v1`
- `--require-genuine-hierarchical`
- `--max-stage-latency-ms 30000`
- no record limit
- unique config/run identifier and output directory

### Hierarchical integrity gate

All conditions are mandatory:

1. Original process exits zero and exactly 18,747 unique output rows exist.
2. Every row has no row-level error, a valid four-digit ISCO-08 code, and exactly `hierarchical_isco08_official_ilo2021_v1` as its method.
3. Every row has genuine valid four-stage evidence. No fallback, flat/unavailable method, missing/empty evidence, or stage latency at or above 30,000 ms is allowed.
4. No reranker/LLM/API activity, cost, token use, ISIC, ISCED, or SRE activity occurs.
5. Hierarchical stage telemetry is parseable and summarized for every relevant query. A successful retry is permitted only when final genuine stage evidence exists and full recorded stage latency stays below 30,000 ms.
6. `flat_query_*` telemetry fields are blank on all hierarchical output rows.
7. Report every retry transparently: stage, collection, case ID, attempt count, safe exception type, full stage latency, and final outcome. Do not suppress retries.
8. Official and legacy Qdrant counts after both runs exactly match before-run values.

If any condition fails, preserve output/logs, set `OFFICIAL_TIER1_RETRY_RERUN_COMPLETED: no`, report exact evidence, commit only the final report, push, and stop. Do not rerun any case or system, and do not analyze.

## Success boundary

Set `OFFICIAL_TIER1_RETRY_RERUN_COMPLETED: yes` only if both full raw output gates pass. Even then:

- Do not calculate accuracy, Wilson intervals, McNemar tests, latency comparison tables, costs, or any other comparisons.
- Do not create a result manifest or modify reviewer/paper documents.
- Do not make any ISIC, ISCED, SRE, reranking, real-world, or real Labour Force Survey claim.
- WISCO remains a controlled multilingual ISCO-08 benchmark, not real Labour Force Survey validation.
- A separate independent analysis task must inspect the raw outputs before any statistic or manuscript statement is eligible.

## Tests

Run focused tests for retry resilience, flat/hierarchical telemetry serialization, strict guard, official profile, evaluator handling, WISCO integrity, and analysis gates. Then run:

```text
pytest backend/tests eval/ -q
```

Expected baseline is 2,129 passed. Do not repair unrelated tests. Report every deviation factually.

## Required final report

Create `Documentation/AI_HANDOFF/CLAUDE_TASK_30_FINAL_REPORT.md` with:

- base branch/SHA, new branch, and push confirmation. Report the final pushed branch tip in Claude's final response, not as a self-referential SHA inside the committed report.
- all preflight hashes, data integrity checks, collection inventories, and before/after mutation checks;
- exact runtime variables and Python calculation;
- exact launch commands, original process IDs, timing, exit code, and logs;
- flat and hierarchical raw output paths, row counts, method distributions, code-format/error checks, and telemetry/retry distributions;
- every retry or failure in full detail;
- maximum observed stage latency and strict-cap check;
- a clear `OFFICIAL_TIER1_RETRY_RERUN_COMPLETED: yes|no`;
- checks confirming Task 24/25/26/29 evidence remained unchanged;
- focused/full test commands and results;
- confirmation that no protected branch, PR, merge/rebase/reset/clean/stash/pull/force-push, collection mutation, raw-result editing, whole-run retry, or analysis occurred;
- mandatory limitation: this task reports raw controlled WISCO ISCO-08 pipeline evidence only, not accuracy/statistics, real Labour Force Survey validation, or claims for ISIC/ISCED/SRE/reranking.

Commit only the final report, never `eval/local_runs/` artifacts. Push only the new branch and stop.
