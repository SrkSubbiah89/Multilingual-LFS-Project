# Task 27 — Bounded Qdrant Timeout Resilience and Hierarchical Query Telemetry

## Purpose

Task 26 establishes a reliability blocker, not an accuracy result. The official flat WISCO run passed cleanly, but the strict hierarchical run aborted because one stage-1 Qdrant query timed out and the existing fallback was correctly rejected. A valid flat-versus-hierarchical comparison therefore does not exist and must not be analyzed.

This task makes transient Qdrant-query failures explicit and auditable and implements a narrowly bounded, opt-in retry mechanism for clearly identified transient transport/time-out exceptions only. It performs no benchmark, no WISCO rerun, no collection build, and no results analysis.

## Starting point

1. Fetch origin and verify the exact base exists locally and on origin:
   - branch: `reviewer2-wisco-official-tier1-telemetry-rerun-20260809`
   - SHA: `6d0453a310d44c91dd911f746bfc25b5db54dffd`
2. Begin with a clean working tree.
3. Create exactly one new branch:
   - `reviewer2-qdrant-timeout-resilience-20260809`
4. Do not merge, rebase, reset, clean, stash, pull, force-push, open a PR, modify protected branches, or edit historical raw outputs.

## Evidence that must be preserved

Before editing, verify and record:

- Task 24 remains invalid and its four raw artifacts are byte-identical to their recorded checksums.
- Task 25 remains the flat exception diagnostic record.
- Task 26 remains `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no`.
- Task 26 observed one strict-guard failure at `WISCO-3521002100018-ur`, at stage 1 on `isco08_major_groups_ilo2021_v1`, after a logged timeout. It wrote no hierarchical CSV.
- Neither Task 24 nor Task 26 must be treated as accuracy or comparison evidence.

Do not change, delete, regenerate, or re-run any historical output.

## Scope

### Required code audit

Trace the exact Qdrant call path from `HierarchyBeamSearchEngine._query()` through both the flat and hierarchical stores, including:

- where Qdrant client timeout configuration is applied;
- the concrete exception classes produced by the installed Qdrant/HTTP stack, including wrapped causes where applicable;
- current behavior for a real no-hit response versus an exception;
- where Task 25 flat telemetry is attached; and
- how hierarchical stage search currently loses or preserves exception information.

Document observed code facts separately from unproven operational hypotheses. Do not claim to know the server or network root cause of the two historical timeouts.

### Required implementation

Implement only a narrow, transparent retry facility for read-only Qdrant query calls.

#### Opt-in and backward-compatible

- Default behavior remains one attempt, with no retry, so existing normal classification behavior is byte-for-byte compatible unless explicitly enabled.
- Introduce a clearly named configuration mechanism for retry attempts, documented with its default and upper bound.
- The future benchmark must be able to explicitly enable it through configuration, without modifying code.

#### Strict eligibility for retry

- Retry only a concrete, explicitly allowlisted class of transient timeout/transport failures supported by the installed Qdrant client stack.
- Inspect and handle wrapper causes explicitly if the library uses them.
- Do not use message-substring matching to decide retry eligibility.
- Do not retry programmer errors, validation errors, schema errors, malformed vectors, authorization errors, generic `Exception`, genuine no-hit responses, or unknown exceptions.
- Any non-allowlisted exception must retain the previous fail-closed outcome and be recorded as such.

#### Bounded execution

- Use a small fixed maximum number of attempts with a hard upper bound enforced in code. The exact value must be justified in the report and tested.
- Use a bounded deterministic backoff only if needed. Do not use unbounded sleeps, exponential sequences without a cap, or automatic reruns of cases.
- Each retry is only a repeat of the same idempotent read query. It must never mutate, rebuild, delete, or overwrite Qdrant collections.
- Do not silently turn a repeated failure into a no-hit result or a fabricated prediction.

#### Per-query telemetry

Extend the existing optional query telemetry so it can distinguish at least:

- success on first attempt;
- success after retry;
- non-retryable exception;
- retryable exception that exhausted its attempt limit;
- genuine successful zero-hit response, if that is a valid engine outcome.

Record only safe structured diagnostics: number of attempts, total query duration, per-attempt duration, retry classification, safe exception type, and bounded/sanitized exception message if an exception occurs.

- Do not record raw user text, query vectors, full tracebacks, credentials, or unbounded exception strings.
- Flat-path CSV fields from Task 25 must remain backward-compatible. Extend them only additively if necessary.
- Add separate, clearly named hierarchical-stage telemetry. Do not overload or misuse flat telemetry as stage evidence.
- A hierarchical result is genuine only when every actual stage result is genuine. A retry that subsequently returns a valid stage result may be reported as a genuine stage result with retry telemetry; it must not conceal the retry.

#### Strict guard preservation

- `--require-genuine-hierarchical` must remain fail-closed on every fallback, unavailable result, missing stage evidence, non-retryable exception, or retry-exhausted error.
- `--max-stage-latency-ms` must continue to measure the full stage time, including retries and backoff. Do not reset or hide latency after a retry.
- Do not weaken, remove, bypass, or relabel the strict guard.

#### No broad behavior change

Do not change scoring, candidate ranking, hierarchy traversal, official catalogue data, WISCO data, output rows, default timeout, B1/B2 gates, ISIC, ISCED, SRE, or LLM behavior. Do not alter the official or legacy Qdrant collections.

## Tests

Create hermetic tests using fakes/mocks only. No live Qdrant, SentenceTransformer, Ollama, LLM/API call, WISCO operation, or evaluation run is allowed in this task.

Cover at least:

1. Default configuration has one attempt and preserves existing behavior.
2. A concrete allowlisted transient timeout is retried and succeeds, with correct telemetry.
3. A retryable timeout that exhausts its limit returns a transparent failed/no-code path, with correct telemetry.
4. A non-allowlisted exception is not retried and retains the fail-closed behavior.
5. A genuine no-hit response is not retried merely because it has no candidates.
6. Attempt limit and configuration validation fail closed on malformed, zero, negative, or excessive values.
7. Sanitization bounds messages and excludes vectors, input text, and tracebacks.
8. Flat telemetry remains backward-compatible and serializes correctly.
9. Hierarchical stage telemetry is distinct and preserves stage provenance.
10. The strict genuine-hierarchical guard still rejects fallback, retry exhaustion, missing evidence, and excessive total stage latency.
11. Legacy/default profile behavior, official-profile methods, and Task 25 flat telemetry tests remain green.

Run focused tests, then:

```text
pytest backend/tests eval/ -q
```

Do not repair unrelated tests or make any live operation to force a result.

## Required final report

Create `Documentation/AI_HANDOFF/CLAUDE_TASK_27_FINAL_REPORT.md` with:

- base branch/SHA, new branch, final SHA, and push confirmation;
- exact changed files and diff scope;
- the audited Qdrant call path and installed exception taxonomy;
- what is observed from Task 24/26 versus what remains unproven about the operational root cause;
- default retry behavior, opt-in configuration, upper limit, allowlisted exception categories, and why each is safe;
- explicit statement that unknown and non-retryable exceptions still fail closed;
- telemetry schema and privacy/sanitization limits;
- strict-guard and latency semantics after retry;
- all focused and full test commands/results;
- confirmation of zero live Qdrant/model/LLM/benchmark/data operation;
- confirmation that Task 24/25/26 historical artifacts and all protected branches are unchanged;
- limitations: this task produces no benchmark evidence and no paper claim. WISCO remains controlled multilingual ISCO-08 evidence, not real Labour Force Survey validation.

Commit the intended source/test/documentation changes and final report on the new branch only, push it, and stop. Do not start a smoke test, preflight, rerun, analysis, or manuscript work.
