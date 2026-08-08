# Task 26 — Clean Full Official WISCO ISCO-08 Rerun With Exception Telemetry

## Purpose

Run one fresh controlled WISCO ISCO-08 benchmark on the official ILO 2021 catalogue profile after Task 25 added read-only flat-query exception telemetry. This task can produce raw benchmark evidence only. It must not compute accuracy, confidence intervals, significance tests, comparisons, or manuscript results.

Task 24 remains invalid and must never be reinterpreted. Its raw artifacts, Task 25 diagnostic artifacts, and all protected branches must remain unchanged.

## Starting point

1. Fetch origin and verify the exact base exists locally and on origin:
   - Branch: `reviewer2-official-flat-unavailability-diagnostics-20260809`
   - SHA: `0590781c4a7562bb29e83da32bfb16b3fb01ca7d`
2. Start with a clean working tree. Do not continue if the base or origin SHA differs.
3. Create exactly one new branch:
   - `reviewer2-wisco-official-tier1-telemetry-rerun-20260809`
4. Do not merge, rebase, reset, clean, stash, pull, force-push, open a PR, modify protected branches, or alter historical raw evaluation outputs.

## Preconditions

Before any evaluation command, record a preflight report and fail closed if any condition fails:

1. Confirm Task 25 telemetry is present in the checked-out code and existing relevant tests pass. The flat-path CSV fields must be present:
   - `flat_query_outcome`
   - `flat_query_duration_ms`
   - `flat_query_exception_type`
   - `flat_query_exception_message`
2. Validate the WISCO v2 package without rebuilding it:
   - dataset hash `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c`
   - 20,760 records total
   - 2,013 dev and 18,747 heldout
   - zero source-family leakage, zero cross-split duplicates, and zero malformed ISCO codes
   - heldout export has 18,747 rows and is ISCO-only. Do not introduce industry or education text.
3. Verify the official ILO normalized catalogue and its metadata:
   - catalogue hash `29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3`
   - official levels 10 / 43 / 130 / 436
4. Read local Qdrant only. Confirm exactly these official collections are present and non-empty:
   - `isco08_major_groups_ilo2021_v1`: 10 points
   - `isco08_submajor_groups_ilo2021_v1`: 43 points
   - `isco08_minor_groups_ilo2021_v1`: 130 points
   - `isco08_unit_groups_ilo2021_v1`: 436 points
   - `isco08_unit_groups_flat_ilo2021_v1`: 436 points
5. Record legacy collection counts before the run. The benchmark must not mutate either official or legacy collections.
6. Confirm Task 23's official collection-build success manifest is present. Do not rebuild or overwrite any collection.
7. Verify Task 24's four raw artifacts and Task 25's diagnostic evidence are untouched before this task.

If any preflight gate fails, write a factual report, commit only the report, push the new branch, and stop. Do not start the benchmark.

## Execution rules

- This is one fresh rerun, not a retry of Task 24. Use a new Git-ignored output root under `eval/local_runs/`, with a unique UTC timestamp.
- Set `QDRANT_TIMEOUT_SECONDS=30` for every evaluation process.
- Use only documented CLI flags. Inspect `eval/run_eval.py --help` first and record the exact commands actually run.
- Reranking must be off. No Ollama, external LLM, paid API, ISIC, ISCED, or SRE activity is allowed.
- Use the official profile `official_ilo2021_v1`.
- Preserve every raw CSV and process log unchanged after it is written. Do not edit any result row.
- No automatic retry of any full run or any failed case is allowed.

## Flat run

Run the complete 18,747-row heldout export exactly once with:

- `--system flat`
- `--use-llm-reranker off`
- `--isco-catalogue-profile official_ilo2021_v1`
- a new, unique run identifier and output directory

Use the repository's supported configuration and output flags only. Do not use a record limit.

### Flat integrity gate

The flat run may pass only if all conditions hold:

1. Process exits zero and exactly 18,747 output rows exist.
2. Every row has no row-level error and a valid four-digit ISCO-08 predicted code.
3. Every row has exactly `flat_isco08_official_ilo2021_v1` as its method label.
4. No reranker model, trace, token count, or cost is present. ISIC, ISCED, and SRE are not applicable and were not constructed.
5. Every row has `flat_query_outcome=success` and a non-negative numeric `flat_query_duration_ms`.
6. For every success row, `flat_query_exception_type` and `flat_query_exception_message` are blank.
7. Any `flat_query_outcome=exception`, unavailable method label, missing or coarse code, malformed telemetry, missing row, extra row, or row-level error fails the gate.
8. Recheck official and legacy Qdrant collection point counts. They must be unchanged.

If the flat gate fails:

- Set `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no`.
- Report exact failing IDs, values, telemetry, and process/log evidence.
- Do not run hierarchical evaluation, do not retry, do not alter data or code, do not analyze results.
- Commit only the final report and push the new branch, then stop.

## Strict hierarchical run

Run this only if the flat gate passes. Run the complete heldout export exactly once with:

- `--system hierarchical`
- `--use-llm-reranker off`
- `--isco-catalogue-profile official_ilo2021_v1`
- `--require-genuine-hierarchical`
- `--max-stage-latency-ms 30000`
- a new, unique run identifier and output directory

### Hierarchical integrity gate

The hierarchical run may pass only if all conditions hold:

1. Process exits zero and exactly 18,747 rows exist.
2. Every row has no row-level error, a valid four-digit ISCO-08 code, and exactly `hierarchical_isco08_official_ilo2021_v1` as the method label.
3. Every row has genuine, valid four-stage evidence. There is no fallback, flat method label, unavailable result, missing stage evidence, or stage latency above 30,000 ms.
4. Reranking, ISIC, ISCED, and SRE remain absent/not applicable.
5. `flat_query_outcome`, `flat_query_duration_ms`, `flat_query_exception_type`, and `flat_query_exception_message` are blank on hierarchical records. They are flat-path telemetry, not hierarchical evidence.
6. Official and legacy Qdrant collection counts after both runs exactly match pre-run values.

If this gate fails, preserve outputs, set `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no`, report the evidence, commit only the report, push, and stop. Do not retry or analyze.

## Pass condition and evidence boundary

Set `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: yes` only if both complete runs pass every gate above.

Even if they pass:

- Do not calculate accuracy, Wilson intervals, McNemar tests, latency comparison tables, cost comparisons, or manuscript text.
- Do not create a manifest or a comparison artifact.
- Do not claim real Labour Force Survey validation.
- Do not make any ISIC, ISCED, SRE, reranking, or real-world effectiveness claim.
- WISCO remains a controlled multilingual ISCO-08 benchmark, not real LFS validation.
- The next task, if needed, will independently audit raw outputs and decide whether analysis is eligible.

## Tests

Run focused tests covering Task 25 telemetry, official catalogue/profile behaviour, strict hierarchical guard, and relevant evaluation serialisation. Then run:

```text
pytest backend/tests eval/ -q
```

Do not repair unrelated failures. If tests fail, report exact failures and stop according to the evidence boundary.

## Required final report

Create `Documentation/AI_HANDOFF/CLAUDE_TASK_26_FINAL_REPORT.md`. Include:

- branch, base SHA, final SHA, and whether pushed
- exact commands actually run
- every preflight result and all relevant hashes/counts
- flat and hierarchical wall times and output paths, without changing raw results
- row counts, method distributions, error counts, code-format checks, telemetry distributions, and maximum stage latency
- exact information for every failure if any
- before/after Qdrant counts
- checks that Task 24/25 artifacts remained unchanged
- focused and full test commands/results
- a clear `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: yes|no`
- mandatory limitation: WISCO is controlled multilingual ISCO-08 evidence, not real Labour Force Survey validation; no accuracy or statistical result is reported in this task
- confirmation that no protected branch, PR, merge/rebase/reset/clean/stash/pull/force-push, collection rebuild, overwrite, result editing, or automatic retry occurred.

Commit only the final report and any intentional source/test changes approved by this task. Never commit `eval/local_runs/` raw outputs. Push only the new task branch and stop.
