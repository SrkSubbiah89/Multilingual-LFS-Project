# Task 04.1: Make the Historical-B1 Quarantine Test Hermetic

## Purpose

Remove the local Ollama `/api/tags` call from the Task 04 shipped-B1 quarantine test, while preserving the B1 evidence, all production safety checks, and the fail-closed test assertions.

Task 04 completed the intended evidence-governance change. Its final report accurately disclosed one metadata-only Ollama request during testing. That request did not perform inference, but it conflicts with Task 04's no-network restriction and makes the unit test depend on a local environment. This task corrects that test isolation issue only.

## Required starting state

Work only on:

```text
reviewer2-b2-integration-20260807
expected HEAD: ec36aa0a1d9fc413e5d1ad939f80c1eae1573293
```

Before modifying anything:

```bash
git fetch origin
git switch reviewer2-b2-integration-20260807
git status --short
git rev-parse HEAD
```

The tree must be clean and the commit must match exactly. Otherwise stop and report.

## Protected branches

Do not modify, merge into, rebase, reset, clean, stash, pull, or switch to:

```text
master
conference1-b2-evaluation
reviewer2-wip-snapshot-20260807
reviewer2-enhancement
```

Do not run `git pull`, `git merge`, `git rebase`, `git reset`, `git clean`, or `git stash`.

## Required change

In `eval/test_dev_sweep.py`, change:

```text
test_the_actual_shipped_b1_frozen_json_is_correctly_quarantined
```

so it uses the existing `fake_ollama_identity` fixture (or an equivalent fully local monkeypatch) for `resolve_ollama_model_identity`.

The test must make no `/api/tags` request and must not require Ollama to be installed, running, or holding any model.

Keep all of these assertions:

1. The shipped `eval/configs/b1_frozen.json` passes structural validation.
2. `baseline_validity.status == "historical_stale_requires_rerun"`.
3. `baseline_validity.b2_sweep_permitted is False`.
4. `assert_baseline_matches_codebase()` raises `BaselineMismatchError`.
5. Its message exposes both:
   - the stale/not-permitted baseline-validity reason, and
   - the `implementation_fingerprint` mismatch.
6. The shipped historical `_source_top1_accuracy == "54/130"` remains asserted unchanged.

Remove/update the test docstring language that says the test should use a real local Ollama environment. State clearly that production `check_ollama_model_identity()` remains a live fail-closed safety check when a real B2 run is attempted; only the unit test substitutes an identity fixture.

## Guardrails

Do not:

- alter `eval/configs/b1_frozen.json`, including its historical result, source CSV, timestamp, hashes, fingerprint, or Task 04 `baseline_validity`;
- weaken, remove, or bypass `check_ollama_model_identity()`, `check_implementation_fingerprint()`, `check_baseline_validity_permits_sweep()`, or `assert_baseline_matches_codebase()` in production;
- remove any model-identity unit-test coverage. Keep/add mocked coverage for confirmed, unknown, and digest-mismatch behaviour as appropriate;
- change the manuscript, datasets, WISCO, full130, benchmark outputs, or reviewer evidence;
- run `eval/run_eval.py`, `eval/ablation_runner.py`, a benchmark, a classifier, LLM/Ollama inference, Qdrant, network call, or dataset operation.

## Documentation

The Task 04 final report already accurately says the prior run made a metadata-only local `/api/tags` request and no inference. Do not rewrite historical facts.

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_04_1_FINAL_REPORT.md
```

It must state:

- why a test using local Ollama was not hermetic;
- the exact test isolation fix;
- confirmation that production model-identity safety checks remain live/fail-closed for actual B2 execution;
- that the historical B1 result and Task 04 validity metadata were not touched;
- test commands/results;
- explicit confirmation that Task 04.1 itself made no network, inference, benchmark, or dataset operation;
- protected branch SHA check.

## Tests

Run:

```bash
pytest eval/test_dev_sweep.py eval/test_pre_run_check.py -q
pytest eval/test_run_eval_b2.py -q
pytest backend/tests eval/ -q
```

Expected full suite: only the pre-existing unrelated mock-signature failure may remain:

```text
backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity
```

If any other failure occurs, stop and report it precisely.

## Commit and stop

Commit only the intended Task 04.1 files on `reviewer2-b2-integration-20260807`, push that branch, then stop. Return the final commit SHA, changed files, exact test results, and confirmations above.
