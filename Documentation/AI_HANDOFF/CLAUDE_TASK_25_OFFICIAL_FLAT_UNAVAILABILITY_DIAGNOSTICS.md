# Task 25 — Diagnose the Official Flat Unavailability and Add Fail-Closed Retrieval Telemetry

## Purpose

Task 24 correctly failed closed. Its official flat output contains one
unavailable result:

```text
case_id: WISCO-8131001300018-ar
input_language: ar
gold_isco_4digit: 8131
pred_method: unavailable_isco08_official_ilo2021_v1
pred_isco_4digit: ""
pred_reasoning: No candidates returned by the vector store.
end_to_end_latency_ms: 60743.26
```

This task must diagnose that event without rerunning, repairing, replacing, or
scoring the failed Task 24 benchmark. It also adds minimal, additive
observability so a future evaluation can distinguish:

1. a real successful Qdrant response with zero candidates;
2. a Qdrant query exception, including a timeout-shaped exception; and
3. the duration of a flat-only Qdrant query.

The purpose is evidence quality and fail-closed debugging. It is not a
performance improvement, benchmark rerun, or paper-result task.

## Exact base and branch requirements

1. Fetch `origin` and verify this exact base:

   ```text
   reviewer2-wisco-official-tier1-raw-results-20260809
   2e6647c5413b9d8a4ca14e84aa8104f43f5a488d
   ```

2. Confirm a clean working tree before branching.
3. Create exactly:

   ```text
   reviewer2-official-flat-unavailability-diagnostics-20260809
   ```

4. Do not use `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
   `git pull`, force-push, or a PR.
5. Do not modify any protected or prior-task branch.

## Verified starting facts and required epistemic discipline

The following facts are verified from Task 24 and current source inspection:

- Task 24 is `OFFICIAL_TIER1_COMPLETED: no` and is ineligible for downstream
  analysis.
- Its flat run wrote 18,747 rows; 18,746 had the expected official four-digit
  method/code and the one named row was unavailable.
- The hierarchical Task 24 command was never started, correctly following the
  flat-gate stop condition.
- `backend/rag/hierarchy_engine.py` currently catches any exception in
  `HierarchyBeamSearchEngine._query()` and returns `[]`.
- `backend/rag/hierarchical_store.py` therefore maps an exception and a
  successful zero-hit flat response to the same empty result without retaining
  the exception in the evaluation trace.
- The 60.743-second end-to-end time is observed. It does **not** by itself
  prove a Qdrant timeout, Arabic-language defect, Qdrant server bug, collection
  defect, retry, or any other root cause.

State every conclusion using one of these exact categories:

```text
confirmed_exception
confirmed_zero_hits
intermittent_unresolved
another evidence-backed category (define it precisely)
```

Never label a hypothesis as a confirmed cause.

## Absolute restrictions

- Do not rerun Task 24 or any full benchmark.
- Do not calculate accuracy, precision, recall, confidence intervals,
  McNemar tests, latency comparisons, or any score.
- Do not edit, replace, delete, truncate, normalize, or otherwise modify:
  - Task 24's raw output CSV;
  - Task 24's final report;
  - the WISCO package or split manifest;
  - the official ILO normalized catalogue or metadata;
  - official or legacy Qdrant collections;
  - previous local-run artefacts.
- No Qdrant mutation, builder execution, collection rebuild, deletion,
  recreation, or repair.
- No LLM, reranker, Ollama, paid API, ISIC, ISCED, SRE, B1 re-freeze, or B2
  sweep.
- Do not weaken or bypass the Task 24 integrity gate. An unavailable prediction
  remains a gate failure for a full future comparator run.

## Part A — Immutable artefact and source-path audit

1. Identify the Task 24 raw output CSV, `flat_stdout.log`,
   `flat_integrity_gate_report.json`, and `failing_row_detail.json`.
2. Compute and record SHA-256 checksums for all four before this task makes any
   code or diagnostic change.
3. Inspect the exact raw row and the Task 24 flat log for a Qdrant warning,
   exception class, timeout text, retry text, or any other failure evidence.
4. Inspect the complete live code path from:

   ```text
   eval/run_eval.py
   backend/agents/isco_classifier.py
   backend/rag/hierarchical_store.py
   backend/rag/hierarchy_engine.py
   ```

   Identify precisely where a query exception is collapsed into a no-candidate
   result and where flat-query duration fails to become an explicit trace value.
5. If Task 24's existing log contains no specific exception evidence, state
   this explicitly. Do not infer an exception merely because the output was
   unavailable.

## Part B — Controlled single-case diagnostics

This is a small read-only diagnosis, not an evaluation rerun.

1. Create a new ignored root:

   ```text
   eval/local_runs/official_flat_unavailability_diagnostics_<UTC timestamp>/
   ```

2. Verify it is Git-ignored.
3. Materialise a one-row temporary input containing only
   `WISCO-8131001300018-ar`, without altering the source WISCO package.
4. Set `QDRANT_TIMEOUT_SECONDS=30`.
5. Use the current official profile and no reranker.
6. Run at most **three** independent, separately logged single-case diagnostic
   attempts. Do not run more than three even if results differ.
7. Each attempt must capture:
   - command and supported CLI flags;
   - wall-clock duration;
   - exit code;
   - `pred_method`, predicted code, reasoning, and row error;
   - flat query telemetry after Part C is implemented;
   - any Qdrant warning/exception evidence;
   - Qdrant point counts before and after, proving no mutation.

If the result succeeds in one attempt and fails in another, classify it as
`intermittent_unresolved` unless the newly captured exception telemetry proves
a narrower diagnosis. A diagnostic success never validates or repairs Task 24.

## Part C — Narrow additive observability implementation

Implement the smallest design that preserves existing retrieval outcomes and
adds evaluation-visible telemetry. Do not invent candidate results or code.

### Required behavior

1. Distinguish a genuine successful zero-hit Qdrant response from an exception
   in `HierarchyBeamSearchEngine._query()` and its callers.
2. Record a sanitized exception type and message only when an actual exception
   occurs. The message must be bounded, must not contain raw query text,
   credentials, endpoints with secrets, or stack traces.
3. Record flat-query duration explicitly. Flat-only retrieval must no longer
   appear as `stage1_latency_ms` through `stage4_latency_ms` all being zero
   merely because it has no hierarchy stages.
4. Thread the telemetry through the official flat classifier/evaluator path
   using additive, clearly named fields. Existing CSV consumers must remain
   compatible; do not remove, rename, or reinterpret old columns.
5. Preserve all classifier decisions:
   - on exception, return the existing explicit unavailable/no-code result;
   - on a genuine zero-hit response, also return the existing explicit
     unavailable/no-code result;
   - never fabricate a fallback candidate or code;
   - successful official and legacy retrieval output must remain unchanged.
6. Preserve strict-hierarchical guard behavior. It must continue to reject
   flat fallback/unavailability and must not treat telemetry as stage evidence.

Choose the smallest maintainable representation after inspecting the current
trace-to-`CaseResult`/CSV schema. Document the precise schema and whether a
field is blank, false, or populated in each outcome.

## Part D — Hermetic regression tests

Use fake Qdrant clients and fake embedders only. No live Qdrant/model/LLM
dependency is permitted in unit tests.

Add focused tests covering at least:

1. a successful Qdrant response with zero points, proving no exception
   telemetry is recorded;
2. a raised generic Qdrant/transport-style exception;
3. a timeout-shaped raised exception;
4. flat-query duration telemetry;
5. serialization of the additive telemetry to evaluation output;
6. successful official flat retrieval remains a valid four-digit result;
7. strict-hierarchical guard behavior is unchanged;
8. legacy-profile compatibility remains unchanged;
9. no raw query text or secret-like text appears in telemetry.

Run all relevant focused tests and then:

```bash
pytest backend/tests eval/ -q
```

Do not repair or silence unrelated failures.

## Part E — Final report, commit, and stop

Commit only the minimal source/test/documentation files required by the
implementation, plus:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_25_FINAL_REPORT.md
```

The final report must include:

1. branch, base SHA, final SHA, push confirmation, and clean-tree status;
2. exact changed files and why each was necessary;
3. the before/after SHA-256 checksums of all four Task 24 artefacts, proving
   they remain byte-identical;
4. Task 24 log/raw artefact evidence, clearly distinguishing observed fact
   from hypothesis;
5. the exact diagnostic classification from the allowed categories and the
   evidence that supports it;
6. all one-case diagnostic attempts, including per-attempt duration/outcome,
   with an explicit statement that they were not a Task 24 retry;
7. Qdrant before/after inventories confirming zero mutation;
8. exact additive telemetry schema and its privacy/sanitization behavior;
9. focused and full test output;
10. protected-branch status and no-PR confirmation;
11. a clear technical decision:
    - whether the cause is diagnosed sufficiently to propose a narrow fix;
    - whether a future clean full official-tier rerun is technically justified;
    - and why Task 24 remains ineligible regardless of the diagnostic result.

Include this exact evidence boundary:

```text
Task 24 remains an incomplete, fail-closed raw run and is not eligible for
accuracy analysis or manuscript claims. Task 25 single-case probes are
diagnostics only, not a benchmark retry or validation result. No conclusion
about real Labour Force Survey performance, ISIC, ISCED, SRE, reranking, cost,
or real-world accuracy is supported.
```

Stop after pushing the Task 25 final report. Do not run a new full benchmark,
analysis, paper drafting, B1 re-freeze, or B2 sweep.
