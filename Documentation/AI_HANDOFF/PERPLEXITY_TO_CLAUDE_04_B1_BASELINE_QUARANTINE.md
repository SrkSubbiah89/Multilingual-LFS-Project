# Task 04: Quarantine the Historical B1 Baseline Without Re-running It

## Purpose

Make the existing B1 frozen baseline's status explicit and mechanically enforced after the verified hierarchy-engine refactor changed its implementation fingerprint.

This is an evidence-governance and test-correctness task. It is **not** a benchmark re-run, a classifier-performance task, or a paper-editing task.

The historical B1 result must be retained exactly as recorded. The repository must make it impossible to mistake that result for current, sweep-ready, or manuscript-eligible evidence.

## Required starting point

Work only on:

```text
reviewer2-b2-integration-20260807
expected starting HEAD: 1561f6742b04aaa965dda0f982b0c6330c1a3bae
```

Before editing, verify:

```bash
git fetch origin
git switch reviewer2-b2-integration-20260807
git status --short
git rev-parse HEAD
```

The working tree must be clean and `HEAD` must equal the expected commit above. If not, stop and report the actual state.

## Protected branches

Do not modify, merge into, rebase, reset, clean, or switch to any of these branches:

```text
master
conference1-b2-evaluation
reviewer2-wip-snapshot-20260807
reviewer2-enhancement
```

Do not run `git pull`, `git merge`, `git rebase`, `git reset`, `git clean`, or `git stash`.

## Known facts to preserve

- `eval/configs/b1_frozen.json` records the historical B1/full130 source result of **54/130**.
- That result and all of its historical source details, hashes, timestamp, and recorded implementation fingerprint are evidence from the original run and must remain unchanged.
- The Reviewer #2 hierarchy-engine refactor changed the current implementation source fingerprint. The current fingerprint mismatch is therefore expected.
- The `full130` set has **unknown provenance** and is not manuscript-eligible evaluation evidence.
- `eval/dev_sweep.py::assert_baseline_matches_codebase()` must continue to reject a stale fingerprint before any B2 K-sweep can run.
- A real B1 re-freeze needs separate explicit authorization and must not happen in this task.

## Required implementation

### Add structured baseline-validity metadata

Update `eval/configs/b1_frozen.json` with a structured `baseline_validity` object. Use clear, machine-checkable fields equivalent to:

```json
{
  "status": "historical_stale_requires_rerun",
  "reason": "The historical B1 implementation fingerprint no longer matches the current source tree after the hierarchy-engine refactor. The original run evidence is retained unchanged.",
  "permitted_use": "Historical engineering evidence only; not manuscript-eligible performance evidence and not permitted to seed a B2 sweep.",
  "b2_sweep_permitted": false,
  "re_freeze_requires": "Separate explicit approval, a fresh documented B1 run, and a new frozen baseline."
}
```

The exact wording may be improved, but retain the meaning and stable machine-readable status.

### Validate the metadata without weakening the safety gate

Update `eval/dev_sweep.py` schema validation so a frozen baseline requires valid `baseline_validity` metadata and accepts only a deliberately small allowed-status enum. Ensure malformed status, missing status, or `b2_sweep_permitted: true` for `historical_stale_requires_rerun` fails validation.

Do **not** change `assert_baseline_matches_codebase()` to accept or bypass a mismatch. It must still raise `BaselineMismatchError` when a current-code fingerprint differs from the frozen one. A stale baseline must never enable a B2 sweep.

If the command path can reach a stale baseline before the fingerprint check, make it emit a clear reason that this baseline is historical/stale and a re-freeze is required. It must still fail closed.

### Replace the misleading shipped-baseline test with an explicit safety-state test

The current failing test in `eval/test_dev_sweep.py` is:

```text
test_the_actual_shipped_b1_frozen_json_passes_shape_and_codebase_checks
```

Replace or rename it so it verifies the correct, intentional state:

1. The shipped JSON passes shape/metadata validation.
2. `baseline_validity.status` is `historical_stale_requires_rerun`.
3. `baseline_validity.b2_sweep_permitted` is false.
4. `assert_baseline_matches_codebase()` raises `BaselineMismatchError`.
5. The exception clearly identifies the implementation-fingerprint mismatch and does not silently permit the B2 sweep.

Do not skip, xfail, delete, or weaken the safety test. Keep/add a separate test showing that a synthetically current, valid baseline can pass the fingerprint gate, if one does not already exist.

### Correct the Task 03 test-count arithmetic

In `Documentation/AI_HANDOFF/CLAUDE_B2_INTEGRATION_REPORT.md`, correct the test-count arithmetic only:

```text
14 shared + 3 Reviewer #2 + 4 B2 = 21
```

Do not rewrite the historical integration report beyond this factual correction.

### Document the decision

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_B1_BASELINE_STATUS_REPORT.md
```

It must state:

- the historical 54/130 B1 result remains unchanged;
- the stale fingerprint arose after a code refactor and does not demonstrate a newly measured regression;
- B1/full130 is engineering-only and not manuscript eligible because provenance is unknown;
- B2 K-sweeps remain blocked until separately approved re-freeze work;
- no benchmark, classifier, LLM, Ollama, network, or dataset operation was performed in this task;
- the recommended path for new research evidence is the externally sourced WISCO ISCO benchmark plus explicitly labelled synthetic/public ISIC and ISCED-F resources, not full130.

Update `eval/PRE_RUN_B2_CHECKLIST.md` with the same gate in concise operational form.

## Tests

Run at minimum:

```bash
pytest eval/test_dev_sweep.py eval/test_pre_run_check.py -q
pytest eval/test_run_eval_b2.py -q
pytest backend/tests eval/ -q
```

Expected: the former B1 fingerprint mismatch must no longer appear as a failing test because it is tested as an expected fail-closed safety state. The known unrelated mock-signature failure in:

```text
backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity
```

may remain. If any other failure occurs, stop and report it precisely.

## Prohibited actions

Do not:

- run any benchmark, `eval/run_eval.py`, `eval/ablation_runner.py`, B1, B2, WISCO, full130, LLM, Ollama, Qdrant, or network operation;
- alter the 54/130 historical B1 result or any historical run hash/fingerprint/source result field;
- modify datasets, frozen evaluation outputs, paper/manuscript files, reviewer response evidence, or protected branches;
- suppress an error with a skip, xfail, broad exception handler, or hard-coded “pass”;
- merge/rebase/reset/clean/stash/pull.

## Commit and report

After successful tests:

1. Commit only the intended Task 04 changes on `reviewer2-b2-integration-20260807`.
2. Push that branch only.
3. Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_04_FINAL_REPORT.md
```

4. Commit and push the report, if it was not included in the implementation commit.

The final report must include:

- starting and final commit SHAs;
- changed files;
- exact historical B1 fields confirmed unchanged;
- status and enforcement behavior;
- test commands and exact results;
- confirmation that no benchmark/inference/data operation occurred;
- confirmation that protected branches remain unchanged;
- any remaining known failure.

Stop when finished.
