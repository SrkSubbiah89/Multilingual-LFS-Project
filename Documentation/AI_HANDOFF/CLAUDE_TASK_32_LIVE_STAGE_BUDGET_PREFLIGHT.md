# Task 32 — Live Strict Preflight of Stage-Budget Enforcement

## Purpose

Validate Task 31's stage-budget enforcement under controlled live conditions before any final full official WISCO rerun. This is a small, read-only Qdrant preflight. It is not a full benchmark, not a rerun of Task 30, not analysis, and not manuscript evidence.

The purpose is operational: establish whether the official flat and strict hierarchical pipelines can complete a deterministic risk-enriched subset while respecting genuine hierarchical evidence, retry telemetry, and the 30,000 ms shared stage deadline. It cannot prove an in-flight query is forcibly cancelled; that limitation remains explicit.

## Starting point

1. Fetch origin and verify exactly:
   - base branch: `reviewer2-qdrant-stage-budget-enforcement-20260809`
   - base SHA: `bfa02c508054fedd7886a1a43e85db816298ed14`
2. Begin with a clean working tree and create exactly:
   - `reviewer2-stage-budget-live-preflight-20260809`
3. Do not merge, rebase, reset, clean, stash, pull, force-push, open a PR, change source/test code, modify protected branches, alter collections, or edit historical outputs.
4. If any gate fails, preserve the new logs/output, commit only a factual report, push the new branch, and stop. Do not repair code, rerun a case, rerun a system, or run the full benchmark.

## Historical preservation

Before and after live commands, verify unchanged:

- Task 24 four raw artifacts.
- Task 25 diagnostic record.
- Task 26 `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no` and empty failed hierarchical output directory.
- Task 29 preflight artifacts.
- Task 30 `OFFICIAL_TIER1_RETRY_RERUN_COMPLETED: no`, its complete flat CSV, strict hierarchical log, and empty hierarchical output directory.
- Task 31 report and source behavior.

Do not modify, reinterpret, or use any historical output for flat-versus-hierarchical accuracy/statistical analysis.

## Exact runtime configuration

For every evaluator process, use exactly:

```text
QDRANT_TIMEOUT_SECONDS=8
QDRANT_QUERY_MAX_ATTEMPTS=3
QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5
```

Use `--require-genuine-hierarchical --max-stage-latency-ms 30000` for the hierarchical run. Confirm from code/test-visible behavior that strict evaluation now passes the 30,000 ms value as the shared monotonic stage budget.

Record these limitations before running:

- `query_points(timeout=...)` is a Qdrant server-side operation-timeout hint.
- The client-side HTTP stack cannot be claimed to forcibly cancel an already-started request.
- The shared stage budget prevents subsequent branches/retries/backoffs from starting after the deadline, but a single in-flight query can still overrun before control returns.

## Deterministic risk-enriched selection

Create a new temporary, Git-ignored selection without changing the canonical WISCO package:

1. Start with the established deterministic 500-record stratified heldout subset with seed 42.
2. Add all 24 Task 12 known-risk IDs.
3. Add these three exact cases:
   - `WISCO-3521002100018-ur` — Task 26 stage-1 timeout case.
   - `WISCO-7123000400018-ur` — Task 30 flat `success_after_retry` case.
   - `WISCO-7212080000000-hi` — Task 30 stage-4 budget/latency anomaly case.
4. Deduplicate. Do not assert a count in advance. Record base count, added-risk count, overlap IDs/count, final count, selection hash, language distribution, and ISCO-major-group distribution.
5. Verify the selection is ISCO-only, has unique case IDs, no industry/education text, and blank ISIC/ISCED gold columns.

No accuracy, language-performance, group-performance, or classification-quality measure may be calculated from this selection.

## Preflight checks

Before evaluator execution, verify:

- WISCO v2 hash `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c`, totals 20,760/2,013/18,747, zero leakage, zero cross-split duplicates, zero malformed codes.
- Official ILO normalized-catalogue hash `29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3`, counts 10/43/130/436.
- Read-only Qdrant official collection counts: 10/43/130/436/436, plus legacy inventory.
- Task 23 official build-success manifest exists.
- Before-run counts are recorded. No collection mutation is allowed.
- Task 31 `stage_budget_exhausted` telemetry fields and strict-budget pass-through are present in source.

## Durable execution

Launch each evaluator as one durable background process from its first invocation. Record PID, command, start/end time, exit code, and log path. Do not use foreground execution, a preliminary probe, an alternative command, or a duplicate invocation.

If an original process is interrupted, killed, exits non-zero, or has invalid output, preserve evidence and stop. Do not launch it again.

Use one new unique Git-ignored `eval/local_runs/` root. Do not edit raw logs or CSVs after writing.

## Flat run

Run exactly once on the selected export using supported CLI flags:

- `--system flat`
- `--use-llm-reranker off`
- `--isco-catalogue-profile official_ilo2021_v1`
- no record limit

Flat pass conditions:

1. Exit zero and exact selected number of unique output rows.
2. No row error; all valid four-digit codes; exact method `flat_isco08_official_ilo2021_v1`.
3. No LLM/reranker/API/ISIC/ISCED/SRE activity.
4. Every flat telemetry record is parseable and internally consistent.
5. Successful retries may pass only with transparent, valid telemetry and genuine final output. Final exceptions, retry exhaustion, unavailable/coarse code, malformed telemetry, or output cardinality issue fail.
6. Official/legacy collection counts remain unchanged.

If flat fails, stop without hierarchy.

## Strict hierarchical run

Run only if flat passes, exactly once on the same selection:

- `--system hierarchical`
- `--use-llm-reranker off`
- `--isco-catalogue-profile official_ilo2021_v1`
- `--require-genuine-hierarchical`
- `--max-stage-latency-ms 30000`
- no record limit

Hierarchical pass conditions:

1. Exit zero and exact selected number of unique output rows.
2. No row error; all valid four-digit codes; exact method `hierarchical_isco08_official_ilo2021_v1`.
3. Every row has genuine complete stage evidence. No fallback, flat/unavailable method, missing/empty evidence, or stage latency at/above 30,000 ms.
4. Each `stage{i}_query_telemetry` is parseable. Report stage query counts, retry outcomes, budget-exhaustion counts, and the stage maximum latency.
5. `stage_budget_exhausted` must be false on every successful row. Any budget exhaustion, retry exhaustion, final exception, or strict-guard abort fails the preflight.
6. `flat_query_*` fields are blank on all hierarchical rows.
7. No LLM/reranker/API/ISIC/ISCED/SRE activity.
8. Official and legacy collection counts remain unchanged after both runs.

If hierarchy fails, preserve evidence, report exactly what happened, and stop. Do not rerun the selection or any individual anomaly case.

## Prohibited work

- No full 18,747-case evaluation.
- No collection mutation/build/delete/overwrite.
- No code/test modification.
- No model download, LLM/API/Ollama/CrewAI use.
- No accuracy, Wilson, McNemar, latency comparison, cost table, manifest, paper drafting, or reviewer-document edit.

## Tests

Run focused Task 31 timeout/stage-budget, strict guard, retry, telemetry, official-profile, and WISCO-integrity tests. Then run:

```text
pytest backend/tests eval/ -q
```

Expected baseline: 2,156 passed. Report deviations factually and do not repair unrelated tests.

## Required final report

Create `Documentation/AI_HANDOFF/CLAUDE_TASK_32_FINAL_REPORT.md`. Include:

- base SHA, branch, and final pushed tip in Claude's final response;
- preflight hashes, collection inventories, source-preservation checks, and selection details;
- exact runtime configuration and explicit in-flight-query limitation;
- exact one-time process launches and durable execution evidence;
- flat/hierarchical output integrity results, all retries/budget-exhaustion events, maximum stage latency, and exact failure evidence if any;
- before/after Qdrant counts;
- exact `STAGE_BUDGET_PREFLIGHT_READY: yes|no`;
- focused/full tests;
- confirmation of no collection mutation, full run, analysis, or paper claim;
- mandatory limitation: a passing result is a small operational preflight only, not full controlled-benchmark evidence or real Labour Force Survey validation.

Commit only the final report, never local-run artifacts. Push only the new branch and stop.
