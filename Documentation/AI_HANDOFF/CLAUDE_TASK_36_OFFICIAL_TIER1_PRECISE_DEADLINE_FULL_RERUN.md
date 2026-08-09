# Task 36: Official Tier-1 Precise-Deadline Full Rerun

## Purpose

Run one controlled, reproducible, evidence-producing full official ILO 2021 WISCO Tier-1 evaluation after Task 35 passed the 528-case precise-client-deadline preflight.

This task produces raw evaluator output only. It does not analyze accuracy, produce confidence intervals or significance tests, update the paper, or make a manuscript claim.

## Required evidence basis

Task 35 is the direct prerequisite:

- Task 35 branch: `reviewer2-precise-deadline-live-preflight-20260809`
- Task 35 final SHA: `280ad76c7cafa3d6ff8ffc55199fbe7efd447306`
- Task 35 result: `PRECISE_DEADLINE_PREFLIGHT_READY: yes`
- Task 35 demonstrated 528/528 clean official-flat and 528/528 clean strict-hierarchical cases, including all known exceptional cases, with zero retry, exception, fallback, or budget-exhaustion event.

Do not reinterpret Task 35 as accuracy evidence. It is operational eligibility only.

## Source and branch

- Base branch: `reviewer2-precise-deadline-live-preflight-20260809`
- Required base SHA: `280ad76c7cafa3d6ff8ffc55199fbe7efd447306`
- New branch: `reviewer2-wisco-official-tier1-precise-deadline-full-results-20260809`
- Fetch `origin` and verify both local base and `origin/reviewer2-precise-deadline-live-preflight-20260809` equal the required SHA before branching.
- Do not merge, rebase, reset, clean, stash, pull, force-push, or open a PR.
- Do not touch protected or prior-task branches.

## Explicitly authorized live operations

This task is explicitly authorized to:

1. Read the frozen WISCO Tier-1 heldout input package.
2. Read local Qdrant metadata and execute read-only local Qdrant searches.
3. Run exactly one full official-flat evaluator process and, only after the flat gate passes, exactly one full official strict-hierarchical evaluator process.
4. Write raw logs and raw evaluator CSV output only beneath a new, Git-ignored `eval/local_runs/` directory.

No Qdrant collection mutation is allowed. Do not create, delete, rebuild, populate, compact, alias, or otherwise alter a collection. Do not modify the WISCO source package, heldout export, gold labels, official ILO catalogue, B1/B2 files, previous raw evidence, or production code.

No Ollama, paid API, LLM, reranker, or external API call is allowed. Keep reranking off.

No source-code, test, dependency, configuration, document, manuscript, figure, or README modification is allowed. The only committed file must be the final report.

## Preservation and pre-run gates

Before any evaluator process starts:

1. Verify the frozen WISCO package and heldout export using the repository's existing integrity checks:
   - exact package hash;
   - 20,760 total records;
   - 2,013 dev records;
   - 18,747 heldout records;
   - no source-family leakage;
   - no cross-split duplicate;
   - no malformed ISCO code;
   - heldout schema has no paired `industry_text` or `education_text`;
   - heldout `gold_isic` and `gold_isced` are blank.
2. Verify the official catalogue hash and official counts: 10/43/130/436.
3. Read and record pre-run point counts for:
   - `isco08_major_groups_ilo2021_v1` = 10;
   - `isco08_submajor_groups_ilo2021_v1` = 43;
   - `isco08_minor_groups_ilo2021_v1` = 130;
   - `isco08_unit_groups_ilo2021_v1` = 436;
   - `isco08_unit_groups_flat_ilo2021_v1` = 436.
4. Hash every available historical raw output and report from Tasks 24 through 35, along with required official-catalogue and B1/B2 safety files. Re-check after each evaluator run and at task completion.
5. Run the full test suite before any live evaluation:

```bash
python -m pytest backend/tests eval/ -q
```

Expected baseline: `2174 passed, 1 deselected, 1 warning`. Any different result is a blocker. Record the exact output and stop without evaluation.

## Required runtime configuration

Use these exact environment settings for both evaluator processes:

```text
QDRANT_TIMEOUT_SECONDS=8
QDRANT_QUERY_MAX_ATTEMPTS=3
QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5
```

Before running, calculate and record with Python:

```text
3 * 8 + 2 * 0.5 = 25.0 seconds = 25,000 ms
```

Record that:

- the strict stage budget is 30,000 ms;
- Task 34.1 enforces a minimum practical client deadline of 1.0 second;
- a bounded client deadline applies to the caller wait, but it does not claim to cancel server-side work already in flight.

Use the exact existing evaluator CLI syntax. Do not invent flags.

Both evaluator processes must start directly as durable background jobs from their first invocation, with separate stdout/stderr logs, explicit PID capture, exit-status capture, and process/log/output timestamps. Do not treat any controller or foreground tool timeout as an evaluator retry.

## Full official-flat run

Run exactly once on the full frozen 18,747-case heldout export:

- official flat system;
- `--isco-catalogue-profile official_ilo2021_v1`;
- `--use-llm-reranker off`;
- a unique clear `--config`, `--run-id`, and Git-ignored output directory for this task.

Do not begin the hierarchical process until all flat gate checks pass.

### Flat gate

The flat output must meet every condition:

- evaluator exit code 0;
- exactly 18,747 output rows with unique case IDs matching heldout exactly;
- zero row-level error;
- every `pred_method` is exactly `flat_isco08_official_ilo2021_v1`;
- every predicted code is a valid official four-digit ISCO-08 code;
- no unavailable, fallback, empty prediction, missing telemetry, or method-label drift;
- flat telemetry present for every row;
- every `flat_query_outcome` is `success` or transparent `success_after_retry`;
- every retry/exception is fully represented by the existing telemetry;
- no forbidden model, LLM, reranker, cost, token, ISIC, ISCED, or SRE activity;
- all Qdrant point counts and all protected historical hashes remain unchanged.

A transparent, policy-compliant `success_after_retry` is not automatically a failure. It must be reported precisely, including case ID, attempts, exception type, first-attempt timing, final timing, and predicted code. Any exhausted retry, non-timeout exception, unavailable result, or invalid output is a failure.

If any flat gate fails, stop immediately. Do not launch hierarchy, retry, alter data, tune configuration, repair code, rebuild collections, analyze output, or update manuscript text.

## Full official strict-hierarchical run

Only after the flat gate passes, run exactly once on the same frozen 18,747-case heldout export:

- `--system hierarchical`;
- `--isco-catalogue-profile official_ilo2021_v1`;
- `--use-llm-reranker off`;
- `--require-genuine-hierarchical`;
- `--max-stage-latency-ms 30000`;
- a unique clear `--config`, `--run-id`, and Git-ignored output directory for this task.

### Strict hierarchical gate

The hierarchical output must meet every condition:

- evaluator exit code 0;
- exactly 18,747 output rows with unique case IDs matching heldout exactly;
- zero row-level error;
- every `pred_method` is exactly `hierarchical_isco08_official_ilo2021_v1`;
- no flat method, fallback, unavailable, synthetic, missing, or mixed label;
- every row has complete, valid, distinct stage-1 through stage-4 evidence;
- stage telemetry is JSON-parseable and present on every row;
- every stage query has deadline/budget telemetry;
- zero stage-budget exhaustion;
- every row’s stage latency is at or below 30,000 ms;
- all Task 12 known-risk IDs plus the Task 26, Task 30, and Task 33 exceptional IDs appear and pass;
- retries are permitted only for policy-allowed direct/wrapped `httpx.TimeoutException` failures and only when fully disclosed in telemetry and the final stage latency remains under 30,000 ms;
- zero exhausted retries, non-timeout exception outcomes, or hidden failure;
- no forbidden model, LLM, reranker, cost, token, ISIC, ISCED, or SRE activity;
- all Qdrant point counts and all protected historical hashes remain unchanged.

If any hierarchical gate fails, stop immediately. Do not retry, repair, tune, alter data, rebuild collections, run analysis, compute statistics, or make an evidence claim.

## Required post-run validation

For each completed evaluator CSV, independently validate:

1. row count and unique/matching case IDs;
2. method-label distribution;
3. predicted-code validity against the official catalogue;
4. error, fallback, unavailable, retry, exception, and budget-exhaustion distributions;
5. flat and per-stage telemetry completeness;
6. query/stage latency distributions and maxima;
7. reranker, token, cost, ISIC, ISCED, and SRE fields;
8. known-risk/exceptional-case outcomes;
9. pre/post point-count and historical-hash equality.

Do not invoke any accuracy, Wilson-interval, McNemar, paired-comparison, or other analysis script. This task ends with validated raw data only.

## Stop conditions

Stop and report `OFFICIAL_TIER1_PRECISE_DEADLINE_FULL_RERUN_COMPLETED: no` if any required pre-run, flat, hierarchical, preservation, or integrity gate fails.

If both full runs pass every gate, report `OFFICIAL_TIER1_PRECISE_DEADLINE_FULL_RERUN_COMPLETED: yes` and stop. Do not analyze the results, update the manuscript, or begin a follow-on task.

In either outcome, this run remains a controlled multilingual ISCO-08 benchmark. It is not real Labour Force Survey validation and cannot support ISIC, ISCED, SRE, real-field, cost, or coverage claims. B1 remains stale/quarantined.

## Final report

Create and commit only:

`Documentation/AI_HANDOFF/CLAUDE_TASK_36_FINAL_REPORT.md`

The report must include:

1. final status (`yes` or `no`), branch, verified base SHA, final SHA, push confirmation, and clean working tree;
2. the exact full commands, environment variables, process IDs, logs, timestamps, exit statuses, output locations, and output hashes;
3. pre-run and post-run asset/package/catalogue/hash/point-count checks;
4. full flat gate table and full strict-hierarchical gate table;
5. exact method-label, retry, exception, fallback, budget-exhaustion, telemetry, code-validity, and latency summaries;
6. all known-risk and exceptional-case outcomes;
7. confirmation of all read-only live operations and zero Qdrant mutation;
8. confirmation of no LLM, reranker, external API, ISIC, ISCED, SRE, analysis, manuscript, or document activity;
9. exact full-suite result;
10. protected-branch preservation;
11. a conservative evidence boundary stating that raw output is not yet analyzed and is not manuscript-ready.

Push only the new branch. Stop after the report.
