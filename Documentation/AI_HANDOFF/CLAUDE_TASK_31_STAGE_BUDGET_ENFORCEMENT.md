# Task 31 — Enforce Per-Request Transport Timeouts and Stage-Level Deadline Budgets

## Purpose

Task 30 exposed a correctness gap in the previous retry configuration reasoning. The calculation `3 × 8s + 2 × 0.5s = 25s` applies to one Qdrant query, but the hierarchical engine performs multiple beam-branch queries within one classification stage and records stage latency as their accumulated total. The Task 30 stage-4 strict-guard failure at `1,155,613.62 ms` therefore cannot be treated as a single 19-minute request, but it does prove that the existing configuration does not bound total stage time below the 30,000 ms strict cap.

This task must correct that engineering gap with transparent, fail-closed code and hermetic proof. It performs no live Qdrant operation, no WISCO preflight or rerun, no collection build, no analysis, and no manuscript work.

## Starting point

1. Fetch origin and verify exactly:
   - base branch: `reviewer2-wisco-official-tier1-retry-rerun-20260809`
   - base SHA: `d3d46ac95992ad7699c4a18eabbb342139698056`
2. Begin with a clean working tree and create exactly:
   - `reviewer2-qdrant-stage-budget-enforcement-20260809`
3. Do not merge, rebase, reset, clean, stash, pull, force-push, open a PR, modify protected branches, edit historical raw output, or start a live evaluator.

## Historical evidence preservation

Verify before and after work:

- Task 24's four raw artifacts remain byte-identical.
- Task 25 diagnostic records remain unchanged.
- Task 26 remains `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no`.
- Task 29 remains `RETRY_CONFIGURATION_PREFLIGHT_READY: yes`.
- Task 30 remains `OFFICIAL_TIER1_RETRY_RERUN_COMPLETED: no`; its complete flat CSV, strict hierarchical log, and empty hierarchical output directory remain unchanged.
- No historical artifact may be edited, regenerated, reclassified, or analyzed as a flat-versus-hierarchical comparison.

## Required forensic audit

Document exact, observed code/package facts before changing behavior:

1. Trace every configuration step from `QDRANT_TIMEOUT_SECONDS` to the concrete client request used by the Qdrant query path.
2. Inspect the installed `qdrant-client` and `httpx` APIs/source signatures used by this repository. Determine whether the configured timeout is:
   - a client-wide default;
   - a per-request timeout;
   - passed through to an underlying transport;
   - overridden or ignored by a call-site/default; or
   - otherwise insufficient to bound observed synchronous wall time.
3. Trace `HierarchyBeamSearchEngine.search()` stage traversal and prove how many Qdrant branch queries can contribute to a single stage's accumulated latency.
4. Explain, with direct source evidence, why Task 30's per-query calculation did not guarantee the stage-level strict cap.
5. Keep a hard distinction between observed code facts and unproven operational causes. Do not claim a server/network root cause.

## Required implementation

Implement a fail-closed correction only if its semantics can be proven with hermetic tests and the installed client API. Do not invent a timeout mechanism or claim a thread/future cancellation stops an underlying HTTP request unless the implementation actually proves it.

### Reliable per-request timeout propagation

1. Ensure each Qdrant query uses an explicit, finite per-request transport timeout derived from the configured timeout value.
2. Use the documented installed client/HTTP API, including all required timeout components where applicable. Do not depend only on a constructor setting if the API supports a safer explicit request-level setting.
3. Preserve default behavior where no explicit strict-stage budget is requested.
4. The retry allowlist remains unchanged: only direct `httpx.TimeoutException` or Qdrant `ResponseHandlingException` wrapping that exact cause is retryable. No substring matching and no broad exception retry.
5. The requested timeout must be visible in safe telemetry or a test-observable call argument, without exposing credentials, vectors, raw respondent text, or tracebacks.

### Strict stage-level deadline budget

1. Add a clearly named, opt-in stage deadline/budget mechanism for strict evaluation. It must be threaded from the evaluator's strict `--max-stage-latency-ms` configuration into hierarchical search.
2. For every hierarchical stage, establish one monotonic deadline covering:
   - all beam-branch Qdrant queries;
   - each retry; and
   - any retry backoff.
3. Before every branch query and every retry/backoff, compute remaining stage budget. Do not start a query or sleep when no budget remains.
4. Every actual per-request timeout must be no greater than the smaller of the configured query timeout and the remaining stage budget.
5. On stage-budget exhaustion, return an explicit structured failure state with `stage_budget_exhausted`, never a fabricated candidate or successful stage. It must lead to a strict fail-closed outcome under `--require-genuine-hierarchical`.
6. The strict guard must reject stage-budget exhaustion, fallback, unavailable output, missing evidence, retry exhaustion, and any stage that reaches/exceeds its cap.
7. Do not reset, hide, or overwrite elapsed time after a retry. Stage latency remains the full accumulated wall time.
8. In non-strict/default classification calls, preserve existing behavior unless an explicit budget is supplied. Do not silently impose a new deadline on production/default use.

### Telemetry and output discipline

1. Extend hierarchical telemetry additively to report safe structured facts needed to audit the correction:
   - stage query count;
   - configured per-query timeout;
   - initial stage budget;
   - remaining budget before each query/retry where feasible;
   - actual attempt count/durations;
   - retry classification;
   - whether a stage budget was exhausted;
   - bounded exception type/message only when a final error remains.
2. Keep flat Task 25 telemetry backward-compatible.
3. Never expose raw query vectors, raw respondent text, full tracebacks, credentials, or unbounded exception strings.
4. Do not use telemetry as fake stage evidence. Genuine hierarchy still requires actual candidates and stage provenance.

## Tests

All tests must be hermetic. Use fake clients, fake transport, controllable monotonic clocks, and mocked sleeps. Do not connect to Qdrant, load models, call an LLM/API, read/rebuild WISCO, or run `eval/run_eval.py` on a real dataset.

At minimum, prove:

1. Explicit per-request timeout is passed at the real query call boundary according to the installed API.
2. The timeout is capped by remaining stage budget.
3. Default/non-strict calls retain the pre-task no-stage-budget behavior.
4. A stage with multiple branch queries stops before starting another branch when its shared budget is exhausted.
5. Retry and backoff both consume the same stage budget.
6. A retry is not started when insufficient budget remains.
7. `stage_budget_exhausted` is explicit, safe, and never produces fabricated candidates.
8. `--require-genuine-hierarchical` rejects budget exhaustion and no CSV is written in evaluator-level hermetic tests.
9. A success after an allowed timeout retry remains genuine only with real final stage evidence and within budget.
10. Non-retryable/unknown errors still fail on first attempt.
11. Genuine no-hit results are not retried.
12. Direct/wrapped timeout taxonomy from Task 27 remains correct.
13. Telemetry is bounded, sanitized, additive, parseable, and distinct from flat telemetry.
14. Existing official profile, flat telemetry, strict guard, B1/B2 safety, ISIC/ISCED, and default hierarchy tests remain green.

Run focused tests, then:

```text
pytest backend/tests eval/ -q
```

Do not weaken unrelated tests. If a sound per-request timeout cannot be implemented with the installed API, stop with a factual report explaining the blocker. Do not substitute an unproven pseudo-timeout.

## Required final report

Create `Documentation/AI_HANDOFF/CLAUDE_TASK_31_FINAL_REPORT.md` with:

- base branch/SHA, new branch, and final pushed tip in Claude's final response;
- the exact audited transport/client configuration path and package API facts;
- direct explanation of the Task 30 stage-accumulation gap;
- exact changed files and the new semantics;
- retry allowlist preservation;
- stage-budget and per-request timeout semantics;
- default/non-strict backward-compatibility statement;
- telemetry schema/privacy limits;
- focused/full test commands and results;
- confirmation of zero live operations;
- confirmation that Task 24/25/26/29/30 historical artifacts and all protected branches are unchanged;
- an explicit `STAGE_BUDGET_ENFORCEMENT_READY: yes|no`;
- mandatory limitation: no new benchmark evidence, accuracy/statistics, or paper result exists; WISCO remains controlled multilingual ISCO-08 evidence and not real Labour Force Survey validation.

Commit only intentional source/test/documentation changes and the final report on the new branch, push only that branch, and stop. Do not start a preflight or rerun.
