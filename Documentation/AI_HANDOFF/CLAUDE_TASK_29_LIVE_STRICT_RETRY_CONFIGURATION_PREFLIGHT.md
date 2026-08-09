# Task 29 — Live Strict Preflight for Bounded Qdrant Retry Configuration

## Purpose

Validate one explicit, bounded runtime configuration for the Task 27 retry mechanism before any further full official WISCO benchmark is considered. This is a small live, read-only Qdrant preflight only. It is not a full benchmark, not a rerun of Task 24 or Task 26, not an accuracy experiment, and not manuscript evidence.

The preflight must use both official flat and strict official hierarchical retrieval against a deterministic 525-case WISCO ISCO-08 subset. It must include the case that stopped Task 26, but it must not execute that case by itself or make any automatic retry of an entire evaluation. The engine-level bounded retry is the only retry permitted.

## Starting point

1. Fetch origin and verify the base exactly:
   - branch: `reviewer2-wisco-timeout-resilience-baseline-20260809`
   - SHA: `1aad2d4c9b0f497eaa75b3c54bb769ee63a049b7`
2. Begin from a clean working tree and create exactly:
   - `reviewer2-wisco-retry-config-preflight-20260809`
3. Do not merge, rebase, reset, clean, stash, pull, force-push, open a PR, modify protected branches, edit historical raw outputs, or change source/test code.
4. If a code/configuration defect is found, preserve evidence, commit only the factual report, push the new branch, and stop. Do not repair it in this task.

## Historical-evidence preservation

Before and after all live commands, verify:

- Task 24's four raw artifacts match their documented checksums.
- Task 25 diagnostic evidence is unchanged.
- Task 26 remains `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no`; its flat outputs and failed hierarchical log are untouched.
- Task 27/28 retry code remains present with default one attempt, maximum three attempts, maximum two-second backoff, and its strict timeout-only type allowlist.
- No historical CSV, log, JSON, WISCO package file, or catalogue file is edited.

## Runtime configuration to validate

Use exactly these environment variables for every evaluator process in this task:

```text
QDRANT_TIMEOUT_SECONDS=8
QDRANT_QUERY_MAX_ATTEMPTS=3
QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5
```

Use Python, not mental arithmetic, to record the configuration's theoretical upper bound:

```text
3 attempts × 8 seconds + 2 backoffs × 0.5 seconds = 25 seconds
```

This is below the strict `--max-stage-latency-ms 30000` limit. Record that this is a configuration bound, not proof that each external call exactly obeys that wall-time bound. The strict guard remains the empirical enforcement mechanism.

Do not change the code-level defaults or hard caps. Do not use a higher timeout, more than three attempts, more than two seconds of backoff, an unbounded retry, a whole-case retry, or an automatic rerun.

## Deterministic selection

Create the preflight selection without changing the canonical WISCO package:

1. Use the existing deterministic 500-record stratified WISCO heldout subset with seed 42, using the repository's established selection tooling where available.
2. Add all 24 Task 12 known-risk IDs.
3. Add `WISCO-3521002100018-ur`, the Task 26 stage-1 timeout case.
4. Deduplicate IDs. The required selection size is exactly 525 if the 25 risk IDs are not already present; if overlap exists, report the exact overlap and final size rather than fabricating 525.
5. Record the selection hash, number of base records, risk IDs, overlap count, final count, language distribution, and ISCO-major-group distribution.
6. The selected export must remain ISCO-only, with 100% blank gold ISIC/ISCED fields and no industry/education text.

The selection is a diagnostic preflight only. Do not calculate or report accuracy, precision, recall, language performance, or any model comparison statistic.

## Live Qdrant preflight gates

Before evaluator execution, read Qdrant only and verify exact official collection counts:

- `isco08_major_groups_ilo2021_v1`: 10
- `isco08_submajor_groups_ilo2021_v1`: 43
- `isco08_minor_groups_ilo2021_v1`: 130
- `isco08_unit_groups_ilo2021_v1`: 436
- `isco08_unit_groups_flat_ilo2021_v1`: 436

Also record legacy collection counts. Confirm the Task 23 success manifest exists. Do not build, delete, overwrite, optimize, or mutate any collection.

## Flat preflight run

Run the deterministic selected export exactly once using only supported CLI flags:

- `--system flat`
- `--use-llm-reranker off`
- `--isco-catalogue-profile official_ilo2021_v1`
- no record limit
- unique Git-ignored output root/run identifier

### Flat pass gate

The flat run passes only if:

1. Process exits zero and output contains exactly the selected number of unique cases.
2. Every row has no row-level error, a valid four-digit ISCO-08 code, and method exactly `flat_isco08_official_ilo2021_v1`.
3. No LLM/reranker, ISIC, ISCED, or SRE activity occurs.
4. Every row has `flat_query_outcome` equal to `success` or `success_after_retry`, non-negative total duration, a valid positive attempt count, and a serializable attempt-duration list whose length equals the attempt count.
5. A `success_after_retry` outcome is permitted and must be reported transparently. Its initial transient timeout is not a zero-hit response and must not be hidden.
6. Any `exception`, `retry_exhausted`, unavailable method, missing/coarse code, malformed telemetry, duplicate/missing/extra row, or row error fails the gate.
7. Qdrant official and legacy counts remain unchanged.

If flat fails, preserve outputs, write the report, commit only the report, push, and stop. Do not run hierarchical evaluation or retry the preflight.

## Strict hierarchical preflight run

Run only if the flat gate passes. Run the same selected export exactly once, with:

- `--system hierarchical`
- `--use-llm-reranker off`
- `--isco-catalogue-profile official_ilo2021_v1`
- `--require-genuine-hierarchical`
- `--max-stage-latency-ms 30000`
- no record limit
- unique Git-ignored output root/run identifier

### Hierarchical pass gate

The hierarchical run passes only if:

1. Process exits zero and output contains exactly the selected number of unique cases.
2. Every row has no error, valid four-digit ISCO code, and method exactly `hierarchical_isco08_official_ilo2021_v1`.
3. Every row has genuine valid stage evidence. No fallback, flat method, unavailable result, missing/empty stage evidence, or stage latency at/above 30,000 ms is allowed.
4. No LLM/reranker, ISIC, ISCED, or SRE activity occurs.
5. `hier_stage_query_telemetry` is parseable structured JSON whenever stages issue Qdrant queries; retry presence/counts and exception types must be summarized exactly.
6. A successfully retried stage may pass only if final genuine stage evidence exists and full recorded stage latency remains below 30,000 ms. It must be disclosed in the report.
7. `flat_query_*` fields remain blank on hierarchical rows.
8. Qdrant official and legacy counts remain unchanged after both runs.

If hierarchical fails, preserve all outputs/logs, report exact evidence, commit only the report, push, and stop. Do not rerun the selection, individual cases, or the full benchmark.

## Prohibited work

- No full 18,747-case evaluation.
- No benchmark analysis, accuracy, Wilson intervals, McNemar tests, latency/cost comparison, manifest, or manuscript drafting.
- No WISCO rebuild, split change, label change, catalogue change, or collection mutation.
- No Ollama, CrewAI, external LLM, paid API, or model download.
- No code/test change, except creating the required final report.

## Tests

Run focused retry, telemetry, strict-guard, official-profile, evaluator, and WISCO-integrity tests. Then run:

```text
pytest backend/tests eval/ -q
```

Expected baseline is 2,129 passed. Report any deviation factually. Do not repair unrelated tests.

## Required final report

Create `Documentation/AI_HANDOFF/CLAUDE_TASK_29_FINAL_REPORT.md`. Include:

- base branch/SHA, new branch, and push confirmation. Report the final pushed branch tip in Claude's final response rather than trying to place a self-referential final commit SHA inside the committed report.
- all preflight checks, hashes, selection details, and runtime configuration;
- Python-calculated worst-case configuration bound and the distinction between that bound and observed wall time;
- exact flat and hierarchical commands, output roots, exit status, wall times, row/method/error distributions, code-format checks, telemetry distributions, retry counts, maximum stage latency, and every failure if one occurs;
- before/after official and legacy Qdrant counts;
- exact status `RETRY_CONFIGURATION_PREFLIGHT_READY: yes|no`;
- confirmation that Task 24/25/26 artifacts remain untouched;
- focused/full test commands and results;
- mandatory limitation: this is a small controlled operational preflight, not a full benchmark or real Labour Force Survey validation. It reports no accuracy, statistical comparison, or paper result.

Commit only the final report, never `eval/local_runs/` output. Push only the new branch and stop.
