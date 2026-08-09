# Task 35: Precise Client-Deadline Live Preflight

## Purpose

Run one tightly controlled, read-only live preflight after Task 34.1's client-side deadline precision correction. This task decides only whether the repaired official ILO 2021 flat and strict hierarchical paths are operationally eligible for one future full WISCO rerun.

This is not a benchmark, not an accuracy analysis, and not a paper-update task.

## Source and branch

- Base branch: `reviewer2-qdrant-client-deadline-precision-20260809`
- Required base SHA: `656bbe144f9ab635766fd77ecbc549b498c9431e`
- New branch: `reviewer2-precise-deadline-live-preflight-20260809`
- Fetch `origin` and verify both the local base and `origin/reviewer2-qdrant-client-deadline-precision-20260809` equal the required SHA before branching.
- Do not merge, rebase, reset, clean, stash, pull, force-push, or open a PR.
- Protected and prior-task branches remain untouched.

## Scope boundary

This task explicitly authorizes only the following live operations:

1. Read the already-existing local WISCO Tier-1 package and produce a deterministic temporary/preflight input subset.
2. Read local Qdrant collection metadata and execute read-only searches through the evaluator.
3. Run exactly one official flat preflight and exactly one official strict-hierarchical preflight on the deterministic subset.

Do not create, delete, rebuild, populate, compact, alias, or otherwise mutate any Qdrant collection. Do not modify the WISCO source package, heldout export, gold labels, prior raw output, official catalogue, B1/B2 files, or any historical artifact. Do not call Ollama, any paid API, an LLM, a reranker, or load a SentenceTransformer outside the existing evaluator path. Keep reranking off.

No source-code, test, dependency, configuration, or manuscript modification is permitted. The only committed file must be the final Task 35 report.

## Preserve historical evidence first

Before any live read:

1. Record cryptographic hashes for Task 24, Task 25, Task 26, Task 29, Task 30, Task 32, and Task 33 raw artifacts and reports that are available locally.
2. Record the relevant official ILO catalogue hash and the B1/B2 safety-file hashes.
3. Record point counts for all five official `ilo2021_v1` collections:
   - `isco08_major_groups_ilo2021_v1`
   - `isco08_submajor_groups_ilo2021_v1`
   - `isco08_minor_groups_ilo2021_v1`
   - `isco08_unit_groups_ilo2021_v1`
   - `isco08_unit_groups_flat_ilo2021_v1`
4. Repeat all applicable hashes and the five point counts after the two preflight runs. Any unexpected difference is a blocker.

The local Qdrant metadata reads above are explicitly authorized for this task and must be reported as read-only.

## Deterministic selection

Build one deterministic preflight selection from the heldout WISCO data:

1. Reuse the canonical 500-record stratified, seed-42 selection from Task 32 when it is available and can be reproduced exactly.
2. Add every Task 12 known-risk ID.
3. Add the Task 26 stage-1 timeout case.
4. Add both named Task 30 anomaly cases.
5. Add the Task 33 failure case: `WISCO-2423002100018-ur`.
6. Deduplicate by case ID. Do not fabricate an expected total: calculate it and report the exact count, component counts, and every included exceptional ID.
7. Verify every selected ID is heldout-only, unique, has a valid four-digit gold ISCO-08 code, and has no paired industry or education text.

Write the selection and evaluator outputs only beneath the existing Git-ignored local-run/preflight output root. Do not commit them.

## Required runtime configuration

Use the following exact environment settings for both evaluator commands:

```text
QDRANT_TIMEOUT_SECONDS=8
QDRANT_QUERY_MAX_ATTEMPTS=3
QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5
```

Before running, calculate the worst-case single-query bound using Python and record it. Also record that Task 34.1's minimum practical client deadline is 1.0 second and that a stage budget is 30,000 ms.

Use:

- `--isco-catalogue-profile official_ilo2021_v1`
- `--use-llm-reranker off`
- the official flat system for the first run
- `--system hierarchical --require-genuine-hierarchical --max-stage-latency-ms 30000` for the second run

Use the correct existing evaluator CLI names as defined by the checked-out code. Do not invent flags. Run each command once only. Start each evaluator process directly as a durable background job from its first invocation, capture stdout/stderr and exit status, and monitor it using the process’s own log/file timestamps. Never treat a controller or foreground tool timeout as an evaluator retry.

## Flat preflight gate

The single official-flat run must satisfy every condition:

- exit code 0;
- exact selected-row count;
- zero row-level errors;
- every row has `pred_method=flat_isco08_official_ilo2021_v1`;
- every predicted ISCO code is a valid official four-digit code;
- no unavailable/fallback/empty prediction;
- telemetry is present for every row;
- every flat query outcome is `success` or transparently recorded `success_after_retry`;
- every retryable exception, if any, is disclosed through the telemetry;
- no Qdrant mutation and no forbidden model/LLM/reranker activity.

## Strict hierarchical preflight gate

The single strict-hierarchical run must satisfy every condition:

- exit code 0;
- exact selected-row count;
- zero row-level errors;
- every row has a genuine hierarchical method label, never flat, fallback, unavailable, or synthetic;
- complete valid stage-1 through stage-4 evidence for every row;
- no stage-budget exhaustion;
- no missing stage telemetry;
- each stage’s observed latency is at or below 30,000 ms;
- per-stage deadline telemetry shows the budget was established and propagated;
- all exceptional IDs listed above are present and clean;
- retries are allowed only when their telemetry is explicit, policy-compliant, and the final stage latency remains within the strict cap;
- no Qdrant mutation and no forbidden model/LLM/reranker activity.

## Stop conditions

Immediately stop after documenting the failure if any pre-run, preservation, flat, or strict-hierarchical gate fails. In particular, stop if a row is unavailable, falls back, is flat-labelled in the hierarchical run, lacks stage evidence, exhausts a budget, exceeds the latency cap, contains an invalid code, or if any historical hash/point count changes.

On failure, do not retry the evaluator, alter the data, tune timeouts, change code, rebuild collections, run the full heldout evaluation, compute accuracy/statistics, update manuscript text, or make any performance claim.

If and only if both runs pass, stop after writing the report. Do not start the full 18,747-case rerun. That requires a separate task and approval.

## Required verification

Run:

```bash
python -m pytest backend/tests eval/ -q
```

Expected baseline: `2174 passed, 1 deselected, 1 warning`. If the result differs, report the exact result and stop; do not repair unrelated failures.

Also run the smallest relevant preflight/evaluator validation tests already present in the repository before live execution, and record exact commands and outputs.

## Final report

Create and commit only:

`Documentation/AI_HANDOFF/CLAUDE_TASK_35_FINAL_REPORT.md`

The report must include:

1. branch name, base SHA, final SHA, push confirmation, and clean working tree;
2. `PRECISE_DEADLINE_PREFLIGHT_READY: yes` or `no`;
3. the exact deterministic-selection method, exact count, and exceptional IDs;
4. exact command lines, environment values, process/log timestamps, exit statuses, and output locations;
5. all pre-run and post-run hashes and Qdrant point-count comparisons;
6. flat and hierarchical gate results field by field, including method-label distributions, code validity, error count, retry count, query/stage latency maxima, budget-exhaustion count, and exceptional-case outcomes;
7. confirmation that the permitted live operations were read-only and confirmation of no Qdrant mutation, LLM, reranker, API, benchmark, analysis, or manuscript activity beyond this bounded preflight;
8. exact test commands and outputs;
9. protected-branch preservation;
10. a conservative evidence statement:
    - passing this task is an operational preflight only;
    - it is not an accuracy, latency, cost, coverage, real-LFS, ISIC, ISCED, SRE, or manuscript claim;
    - WISCO remains a controlled multilingual ISCO-08 benchmark, not real Labour Force Survey validation;
    - B1 remains stale/quarantined.

Push only the new branch. Stop after reporting.
