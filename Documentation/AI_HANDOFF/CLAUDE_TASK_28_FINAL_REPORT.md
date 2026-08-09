# Task 28 Final Report — Integrate Bounded Qdrant Resilience Into the Official WISCO Evidence Line

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_28_INTEGRATE_QDRANT_RESILIENCE.md`.
This is a Git integration and test-verification task only. No
preflight, smoke test, WISCO evaluation, benchmark rerun, analysis,
collection build, or manuscript work was performed, and no live Qdrant
call was made.

## 1. Verified source branches and SHAs

| | Branch | SHA | Verified match |
|---|---|---|---|
| Canonical evidence base | `reviewer2-wisco-official-tier1-telemetry-rerun-20260809` | `6d0453a310d44c91dd911f746bfc25b5db54dffd` | Yes (local == origin) |
| Audited feature source | `reviewer2-qdrant-timeout-resilience-20260809` | `e0b0649d43fa01dddd3c707bc7f4d2400b230ad2` | Yes (local == origin) |

Working tree was clean before branching.

## 2. Integration branch, merge SHA, final report SHA, push

| | |
|---|---|
| New branch | `reviewer2-wisco-timeout-resilience-baseline-20260809`, created from `origin/reviewer2-wisco-official-tier1-telemetry-rerun-20260809` |
| Merge command | `git merge --no-ff origin/reviewer2-qdrant-timeout-resilience-20260809` |
| Merge commit SHA | `f2e7cb398021184d098035973f93bdf2b63de874` |
| Merge exit code | 0 — **zero conflicts** |
| Final report commit SHA | recorded after this report's commit (see push confirmation below) |

`git log --oneline --graph` after the merge:

```text
*   f2e7cb3 Task 28: integrate Task 27 bounded Qdrant timeout resilience into the official WISCO evidence line
|\
| * e0b0649 Task 27: bounded, opt-in Qdrant query retry with strict exception allowlist and stage telemetry
|/
* 6d0453a Task 26: clean full official WISCO telemetry rerun -- flat gate passes, hierarchical gate fails on second transient timeout
```

No cherry-pick, squash, or reimplementation was used — an explicit
`--no-ff` merge commit, exactly as required. Task 27's logic was not
touched during integration.

## 3. Conflict status and changed-file list (base → merge)

**Conflict status: none.** The merge completed cleanly (`git merge
--no-ff` exited 0, "Merge made by the 'ort' strategy", no conflict
markers, no `git status` output requiring resolution).

`git diff --stat 6d0453a f2e7cb3`:

```text
 Documentation/AI_HANDOFF/CLAUDE_TASK_27_FINAL_REPORT.md      | 388 +++++++++++++++++++
 backend/rag/hierarchical_store.py                            |  96 +++++
 backend/rag/hierarchy_engine.py                              | 298 ++++++++++++---
 backend/tests/test_qdrant_retry_resilience.py                | 413 +++++++++++++++++++++
 eval/run_eval.py                                              |  46 +++
 eval/test_qdrant_retry_resilience_serialization.py            | 237 ++++++++++++
 6 files changed, 1428 insertions(+), 50 deletions(-)
```

This is exactly Task 27's own changed-file set (confirmed identical to
Task 27's final report §2) — no additional file was touched by the
merge itself beyond what Task 27 already introduced.

## 4. Required preservation checks — all passed, with direct evidence

### 1. Task 24's four raw artifacts remain byte-identical

Re-verified both before and after the merge:

| Artefact | Match |
|---|---|
| Flat raw CSV | Yes |
| `flat_stdout.log` | Yes |
| `flat_integrity_gate_report.json` | Yes |
| `failing_row_detail.json` | Yes |

(All four SHA-256 checksums matched their Task 25-recorded values
exactly, both pre- and post-merge.)

### 2. Task 25 remains the diagnostic record; Task 25 telemetry present

Task 25's diagnostic output root
(`eval/local_runs/official_flat_unavailability_diagnostics_20260808T221507Z/`)
still contains all 18 recorded artefacts, both before and after the
merge. Task 25's four flat telemetry fields
(`flat_query_outcome`/`flat_query_duration_ms`/`flat_query_exception_type`/
`flat_query_exception_message`) are present in `eval/run_eval.py`'s
`CaseResult` post-merge (grep-confirmed).

### 3. Task 26 remains exactly `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no`

`Documentation/AI_HANDOFF/CLAUDE_TASK_26_FINAL_REPORT.md`'s first line
is unchanged (`OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no`), both
pre- and post-merge. Task 26's flat CSV/logs and the failed
hierarchical run's log
(`eval/local_runs/wisco_official_tier1_telemetry_rerun_20260808T223144Z/hierarchical_stdout.log`,
SHA-256 `aca8d13d2f7272d84a7c63b129e904a86b02741c0ef7935da0e3b99e899b376d`)
are untouched; the `hierarchical/` output directory remains empty (0
files) both before and after the merge, matching Task 26's own
documented "no CSV written" abort state.

### 4. Task 27 retry configuration semantics — confirmed exactly

Re-read directly from the merged `backend/rag/hierarchy_engine.py`:

- `DEFAULT_MAX_QUERY_ATTEMPTS = 1`, `DEFAULT_RETRY_BACKOFF_SECONDS = 0.0` — confirmed.
- `MAX_QUERY_ATTEMPTS_HARD_CAP = 3` — confirmed.
- `MAX_RETRY_BACKOFF_SECONDS_HARD_CAP = 2.0` — confirmed.
- `_is_retryable_exception()` retries **only** a direct
  `httpx.TimeoutException` or a `qdrant_client.http.exceptions.ResponseHandlingException`
  whose `.source` `isinstance`-checks as `httpx.TimeoutException` —
  confirmed by reading the merged function body verbatim (reproduced in
  §5 below).
- No message-substring/string-matching retry decision exists anywhere
  in the function — confirmed (the function contains exactly two
  `isinstance` checks and no string comparison).
- Unknown, non-timeout, validation, authentication, authorization,
  rate-limit, schema, and generic errors remain non-retryable and fail
  closed — confirmed both by direct code reading (the function returns
  `False` for anything not matching the two `isinstance` checks) and by
  the merged hermetic test suite (`test_non_retryable_exception_not_retried_even_with_attempts_available`,
  parametrized over `RuntimeError`, `UnexpectedResponse` (HTTP 400),
  `ResponseHandlingException(ValueError(...))`,
  `ResponseHandlingException(ConnectionResetError(...))`), which passed
  post-merge (see §6).

### 5. Task 27 flat telemetry backward-compatible; hierarchical-stage telemetry present

`eval/run_eval.py`'s `CaseResult` post-merge contains, unmodified in
name/default: `flat_query_outcome`, `flat_query_duration_ms`,
`flat_query_exception_type`, `flat_query_exception_message` (Task 25
originals) plus the additive `flat_query_attempts`,
`flat_query_attempt_durations_ms` (Task 27), and the separate
`hier_stage_query_telemetry` field — confirmed present by direct grep
(§4 of the changed-file evidence above) and exercised by the merged
serialization tests (§6).

### 6. `--require-genuine-hierarchical` unchanged and fail-closed

`check_strict_hierarchical()`'s function body was diffed directly
between the canonical base commit (`6d0453a`) and the merge commit
(`f2e7cb3`) using a regex extraction of the exact function text: **byte-
identical, confirmed programmatically** (`identical: True`). It still
rejects, unchanged: a non-`hierarchical_`-prefixed `pred_method`
(fallback/unavailable), missing/empty `stageN_candidates`, and (with
`--max-stage-latency-ms`) any `stageN_latency_ms` exceeding the bound —
confirmed by the merged tests
`test_strict_guard_rejects_flat_fallback_caused_by_retry_exhaustion`,
`test_strict_guard_rejects_missing_stage_evidence_unchanged`, and
`test_strict_guard_rejects_excessive_total_stage_latency_including_retry_time`.

### 7. `--max-stage-latency-ms` covers retry attempts and backoff

Confirmed by direct code reading: `HierarchyBeamSearchEngine.search()`'s
`_timed_query()` closure times the entire `self._query(...)` call, and
the retry loop (including any `time.sleep(backoff)`) lives entirely
inside that single call — so `stageN_latency_ms` includes every retry
attempt and any backoff by construction, with no separate accounting
needed. Confirmed by the merged test
`test_stage_latency_includes_retry_and_backoff_time`.

### 8. Official ILO catalogue/profile code and Task 23 collections unchanged

| File | Pre-merge SHA-256 | Post-merge SHA-256 | Match |
|---|---|---|---|
| `eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv` | `29b7539e...` | `29b7539e...` | Yes |
| `eval/verified_catalogue_counts.yaml` | `dad2a758...` | `dad2a758...` | Yes |
| `backend/rag/official_isco08_catalogue.py` | `10ee26ab...` | `10ee26ab...` | Yes |
| `backend/rag/build_official_isco08_collections.py` | `d6a15f2c...` | `d6a15f2c...` | Yes |

No Qdrant connection was made in this task (§7), so Task 23's built
official collections were not queried, inspected, rebuilt, or mutated
— this check is a source/artefact-hash confirmation only, not a live
collection query.

### 9. B1 quarantine and B2 safety files byte-identical to canonical base

| File | Match |
|---|---|
| `eval/configs/b1_frozen.json` | Yes |
| `eval/full130_access_guard.py` | Yes |
| `eval/configs/full130_leakage_manifest.json` | Yes |
| `eval/local_benchmarks/wisco_isco08_v2_group_split/dataset_hash.txt` | Yes |

## 5. Retry allowlist and strict-guard semantics — exact confirmation

Verbatim from the merged `backend/rag/hierarchy_engine.py`:

```python
def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, ResponseHandlingException):
        source = getattr(exc, "source", None)
        if isinstance(source, httpx.TimeoutException):
            return True
    return False
```

No other code path in the merged tree constructs or checks any other
retry-eligibility condition.

## 6. Future-configuration safety clarification (constraint, not a measured result)

Recorded as required, **not acted on or tested in this task**: with the
default `QDRANT_TIMEOUT_SECONDS=30`, a retry-enabled stage (e.g.
`max_query_attempts=2` or `3`) could take longer than a
`--max-stage-latency-ms=30000` strict-guard threshold purely from
summing multiple 30-second per-attempt timeouts plus backoff — this is
an interaction between two independently-configured bounds, not a
defect in either bound individually, and is not repaired or hidden
here. **Any later live preflight or benchmark run that enables retry
must select and validate a lower per-attempt `QDRANT_TIMEOUT_SECONDS`
together with the chosen `QDRANT_QUERY_MAX_ATTEMPTS`/
`QDRANT_QUERY_RETRY_BACKOFF_SECONDS`, such that the full worst-case
stage duration (every attempt plus every backoff) remains under
whatever `--max-stage-latency-ms` value that future run intends to
use.** No such configuration was selected, computed, or tested in this
task, and no live Qdrant call of any kind was made.

## 7. Focused and full test results

```
python -m pytest backend/tests/test_qdrant_retry_resilience.py eval/test_qdrant_retry_resilience_serialization.py backend/tests/test_flat_query_telemetry.py eval/test_flat_query_telemetry_serialization.py backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchical_store.py backend/tests/test_standard_hierarchical_store.py backend/tests/test_isco_classifier_official_profile.py backend/tests/test_official_isco08_catalogue.py backend/tests/test_official_isco08_profiles.py backend/tests/test_build_official_isco08_collections.py eval/test_require_genuine_hierarchical.py eval/test_official_isco08_profile_evaluator.py eval/test_model_free_isco_evaluation.py eval/test_run_eval_b2.py eval/test_docs_consistency.py eval/test_wisco_leakage_audit.py eval/test_analyze_wisco_tier1.py -q
→ 321 passed in 3.15s

python -m pytest backend/tests eval/ -q
→ 2129 passed, 1 deselected, 1 warning in 297.81s (0:04:57)
```

**Exact match to the expected Task 27 baseline of 2,129 passed.** No
environment-neutral difference exists to explain — the merge introduced
no new test, removed no test, and altered no test outcome. Zero
failures. No unrelated test was skipped, modified, or repaired.

## 8. Confirmation of zero live operations

No Qdrant connection, mutation, query, or health check; no
SentenceTransformer/model load; no Ollama/CrewAI/LLM/paid API call; no
WISCO file export/build/read-write operation (only read-only SHA-256
hashing of already-existing files for the preservation checks above);
no benchmark/preflight/smoke/evaluation run; no official or legacy
collection build/deletion/overwrite/mutation; no accuracy, Wilson,
McNemar, latency-comparison, cost, or manuscript analysis was performed
anywhere in this task.

## 9. Protected branches and historical artifacts — unchanged

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| `reviewer2-wip-snapshot-20260807` | not touched |
| `reviewer2-b2-integration-20260807` | not touched |
| every prior `reviewer2-*` task/integration branch (through `reviewer2-qdrant-timeout-resilience-20260809`) | not touched |

No `git rebase`, `git reset`, `git clean`, `git stash`, `git pull`, or
force-push occurred (only the one required `git merge --no-ff`). No PR
was created. No historical `eval/local_runs/` output was altered
(§4, items 1-3).

## 10. Mandatory limitation

No new benchmark evidence, accuracy/statistics, or paper claim was
produced by this task. Task 24 remains invalid, and Task 26 remains
`OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no` because no valid full
hierarchical CSV exists — this integration does not change either fact.
WISCO remains a controlled multilingual ISCO-08 benchmark, not real
Labour Force Survey validation, ISIC/ISCED/SRE evidence, or a
real-world performance claim.

Stopping here, per the task's own instruction, after pushing this
report and the merge commit. No preflight, smoke test, evaluation, or
further work was started.
