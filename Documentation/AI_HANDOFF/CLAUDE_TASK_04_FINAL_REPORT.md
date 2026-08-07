# Task 04 Final Report — B1 Baseline Quarantine

Produced in response to
`Documentation/AI_HANDOFF/PERPLEXITY_TO_CLAUDE_04_B1_BASELINE_QUARANTINE.md`
(task ID `04-b1-baseline-quarantine`), executed on
`reviewer2-b2-integration-20260807` after explicit user instruction in the
current session.

## Starting and final commit SHAs

| | SHA |
|---|---|
| Starting `HEAD` (verified against task's expected value) | `1561f6742b04aaa965dda0f982b0c6330c1a3bae` |
| Implementation commit | `97fbec003729197153837d3a9f5f31dd7a64482f` |
| This report's commit | (recorded after commit below) |

Start-state check passed exactly: working tree clean, branch
`reviewer2-b2-integration-20260807`, `HEAD` equal to the expected
`1561f6742b04aaa965dda0f982b0c6330c1a3bae` before any edit was made.

## Changed files

- `eval/configs/b1_frozen.json` — added `baseline_validity` object (one new
  top-level key; every pre-existing field byte-for-byte unchanged, verified
  via `git diff` showing zero removed content lines).
- `eval/dev_sweep.py` — added `BASELINE_VALIDITY_STATUSES`,
  `_validate_baseline_validity_shape()` (called from
  `validate_baseline_shape()`), `check_baseline_validity_permits_sweep()`
  (added to `BASELINE_CODEBASE_CHECKS`), and a docstring update to
  `assert_baseline_matches_codebase()`. No existing function's pass/fail
  logic was changed.
- `eval/test_dev_sweep.py` — `valid_baseline()` fixture now includes a
  permitting `baseline_validity`; 6 new shape/rejection tests; the
  misleading `test_the_actual_shipped_b1_frozen_json_passes_shape_and_codebase_checks`
  replaced with `test_the_actual_shipped_b1_frozen_json_is_correctly_quarantined`.
- `eval/PRE_RUN_B2_CHECKLIST.md` — added a "Currently blocked" notice.
- `Documentation/AI_HANDOFF/CLAUDE_B1_BASELINE_STATUS_REPORT.md` — new.
- `Documentation/AI_HANDOFF/CLAUDE_B2_INTEGRATION_REPORT.md` — corrected
  the Task 03 test-count arithmetic only (no other content changed).

## Exact historical B1 fields confirmed unchanged

Verified by direct read of the file after editing:

| Field | Value |
|---|---|
| `_source_top1_accuracy` | `54/130` |
| `_source_csv` | `eval/results/raw_runs/20260805T055741Z_full130_leafvote_beam3_llama3b_pooled.csv` |
| `_verified_on` | `2026-08-05` |
| `implementation_fingerprint.composite_sha256` | `61a57d32625eaef68eca00d94d7d1975f87f459a1242636683cc116fb49acd53` |
| `provenance.b1_result_csv_sha256` | `540b8d7d048f941a01f672bfafa3ac6ba9177b5f79f6ec2d75d32ca57df7b4fb` |

`git diff eval/configs/b1_frozen.json` contains zero `-` (removed) content
lines — the change is a pure addition of the `baseline_validity` block.

## Status and enforcement behaviour

`eval/configs/b1_frozen.json.baseline_validity` = `{"status":
"historical_stale_requires_rerun", "b2_sweep_permitted": false, ...}`.
`eval/dev_sweep.py::assert_baseline_matches_codebase()` still raises
`BaselineMismatchError` for this file — unchanged, not bypassed — now for
**two independent reasons** reported in the same exception: the new
`baseline_validity` self-report, and the pre-existing implementation-
fingerprint mismatch. `eval/pre_run_check.py` inherits the new check
automatically via its generic `BASELINE_CODEBASE_CHECKS` iteration. No
B2 K-sweep can currently proceed against the shipped baseline.

## Test commands and exact results

```
pytest eval/test_dev_sweep.py eval/test_pre_run_check.py -q
→ 128 passed in 107.10s

pytest eval/test_run_eval_b2.py -q
→ 21 passed in 23.02s

pytest backend/tests eval/ -q
→ 1 failed, 1834 passed, 1 deselected, 1 warning in 296.81s (0:04:56)
  FAILED backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity
```

The former B1 fingerprint-mismatch failure
(`test_the_actual_shipped_b1_frozen_json_passes_shape_and_codebase_checks`)
**no longer appears as a failing test** — it is now tested, and passes, as
an expected fail-closed safety state under its renamed replacement
(`test_the_actual_shipped_b1_frozen_json_is_correctly_quarantined`). Net
+8 tests vs. the pre-Task-04 full-suite count of 1826 (6 new shape/
rejection tests plus the net effect of the test rename/rewrite).

## Confirmation: no benchmark/inference/data operation occurred

No `eval/run_eval.py` or `eval/ablation_runner.py` invocation was made; no
classifier was constructed; no Qdrant call was made. The one live network
call in the entire test run is the same one that already existed before
this task — `test_the_actual_shipped_b1_frozen_json_is_correctly_quarantined`
(via `check_ollama_model_identity()`) makes a metadata-only `GET
/api/tags` request to confirm the pinned model's digest, never
`/api/generate` or `/api/chat` — no inference was performed. No dataset
file (`eval/test_set_full130.csv`, WISCO records, `eval/local_benchmarks/`,
`eval/local_runs/`) was read or written. No `--sre`/`--use-llm-reranker`
CLI path or any manuscript/generated-evidence file was touched.

## Confirmation: protected branches remain unchanged

| Branch | SHA |
|---|---|
| `master` | `5e0ff5d88c6c973f636b48cacc25e5885c11d41c` (unchanged) |
| `conference1-b2-evaluation` | `675121bbf656dad3e611f3280d95606b5b123e92` (unchanged) |
| `reviewer2-wip-snapshot-20260807` | `64e6ec4d1d1dac940d85242b791748ca1468050d` (unchanged) |
| `reviewer2-enhancement` | moved to `0c5bd5ad13441aa7085dff81e4ae3ab45af83777` — **not by this task**; fetched read-only only, to read the task-handoff file itself, exactly as instructed |

No `git reset`, `git clean`, `git stash`, `git rebase`, `git pull`, `git
merge`, or force-push was run at any point in this task.

## Any remaining known failure

`backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity`
— the pre-existing, documented, unrelated mock-signature failure. No other
failure occurred.
