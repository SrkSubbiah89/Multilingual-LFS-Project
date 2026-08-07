# Task 04.1 Final Report — Hermetic B1 Quarantine Test

Produced in response to
`Documentation/AI_HANDOFF/PERPLEXITY_TO_CLAUDE_04_1_HERMETIC_QUARANTINE_TEST.md`
(task ID `04.1-hermetic-quarantine-test`), executed on
`reviewer2-b2-integration-20260807` after explicit user instruction in the
current session.

Start-state check passed exactly: working tree clean, branch
`reviewer2-b2-integration-20260807`, `HEAD` equal to the expected
`ec36aa0a1d9fc413e5d1ad939f80c1eae1573293` before any edit was made.

## Why the test was not hermetic

`test_the_actual_shipped_b1_frozen_json_is_correctly_quarantined()`
(added by Task 04) called `eval/dev_sweep.py::assert_baseline_matches_codebase()`
without mocking `resolve_ollama_model_identity()`. That function makes a
real `GET http://localhost:11434/api/tags` call to resolve the shipped
baseline's pinned Ollama model digest. The request is metadata-only (never
`/api/generate` or `/api/chat` — no inference was ever performed), but it
still meant the unit test's outcome depended on a local environment
(Ollama installed, running, and holding the exact `llama3.2:latest` tag)
rather than being a pure, self-contained assertion of the quarantine
logic — and it made a real network call from what should be an offline
unit test suite.

## The exact test isolation fix

`test_the_actual_shipped_b1_frozen_json_is_correctly_quarantined` now takes
the existing `fake_ollama_identity` fixture as a parameter — the same
hermetic stand-in already used by every other
`test_assert_baseline_matches_codebase_*` test in this file
(`monkeypatch.setattr(ds, "resolve_ollama_model_identity", ...)`, returning
a fixed, fake digest, never touching the network). No new fixture was
created; no existing fixture was modified. The test's docstring was
rewritten to state plainly that the test is hermetic and why (the test's
purpose — proving the `baseline_validity`/`implementation_fingerprint`
quarantine state — has nothing to do with Ollama model identity), and to
state explicitly that `check_ollama_model_identity()` remains completely
unmodified in production.

All six required assertions remain in the test, unchanged in substance:

1. `ds.validate_baseline_shape(baseline) == []`
2. `baseline["baseline_validity"]["status"] == "historical_stale_requires_rerun"`
3. `baseline["baseline_validity"]["b2_sweep_permitted"] is False`
4. `pytest.raises(ds.BaselineMismatchError)` around
   `assert_baseline_matches_codebase(baseline)`
5. The raised message contains `"baseline_validity"`, `"historical/stale"`,
   and `"implementation_fingerprint"`
6. `baseline["_source_top1_accuracy"] == "54/130"`

(With `fake_ollama_identity` active, the fake digest also happens not to
match the shipped file's real digest, so the exception message now
additionally contains an `ollama_model_identity` mismatch reason alongside
the two required ones — harmless, since the test only asserts the three
required substrings are *present*, never that they are the *only* content.)

## Confirmation: production model-identity safety checks remain live/fail-closed

`eval/dev_sweep.py::check_ollama_model_identity()`,
`resolve_ollama_model_identity()`, `check_implementation_fingerprint()`,
`check_baseline_validity_permits_sweep()`, and
`assert_baseline_matches_codebase()` were **not modified at all** in this
task (verified: `git diff` touches only `eval/test_dev_sweep.py`). A real
`eval/dev_sweep.py` invocation (outside of tests) still calls the real,
live `resolve_ollama_model_identity()` and still fails closed exactly as
before — unreachable Ollama, wrong tag, or digest mismatch all still block
a sweep with `BaselineMismatchError`, precisely as Task 04 established and
Task 04.1 leaves untouched.

## Confirmation: historical B1 result and Task 04 validity metadata untouched

`eval/configs/b1_frozen.json` was not opened for writing in this task
(not in the `git diff`, not in `git status`). `_source_top1_accuracy`
(`"54/130"`), the source CSV, timestamp, all hashes, the recorded
`implementation_fingerprint.composite_sha256`, and the Task 04
`baseline_validity` block (`status`, `b2_sweep_permitted`, `reason`,
`permitted_use`, `re_freeze_requires`) are all exactly as Task 04 left
them.

## Test commands and results

```
pytest eval/test_dev_sweep.py eval/test_pre_run_check.py -q
→ 128 passed in 83.46s

pytest eval/test_run_eval_b2.py -q
→ 21 passed in 22.06s

pytest backend/tests eval/ -q
→ 1 failed, 1834 passed, 1 deselected, 1 warning in 296.70s (0:04:56)
  FAILED backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity
```

Test count is unchanged from post-Task-04 (1834) — this task changed one
existing test's fixture usage and docstring, adding/removing no test.
Only the known pre-existing, unrelated mock-signature failure remains.

## Confirmation: Task 04.1 itself made no network, inference, benchmark, or dataset operation

No `eval/run_eval.py` or `eval/ablation_runner.py` invocation was made; no
classifier was constructed; no Qdrant call was made; no Ollama call was
made anywhere in this task's own work (the fix's entire purpose was
removing the one Ollama call that previously existed in this test — after
this change, running `eval/test_dev_sweep.py` makes zero network calls of
any kind). No dataset file (`eval/test_set_full130.csv`, WISCO records,
`eval/local_benchmarks/`, `eval/local_runs/`) was read or written. No
manuscript, benchmark output, or reviewer-evidence file was touched.

## Changed files

- `eval/test_dev_sweep.py` — one test function updated (added
  `fake_ollama_identity` fixture parameter, rewrote docstring). No other
  file changed.

## Protected branch SHA check

| Branch | SHA |
|---|---|
| `master` | `5e0ff5d88c6c973f636b48cacc25e5885c11d41c` (unchanged) |
| `conference1-b2-evaluation` | `675121bbf656dad3e611f3280d95606b5b123e92` (unchanged) |
| `reviewer2-wip-snapshot-20260807` | `64e6ec4d1d1dac940d85242b791748ca1468050d` (unchanged) |
| `reviewer2-enhancement` | fetched read-only only, to read the task-handoff file itself |

No `git reset`, `git clean`, `git stash`, `git rebase`, `git pull`, `git
merge`, or force-push was run at any point in this task.
