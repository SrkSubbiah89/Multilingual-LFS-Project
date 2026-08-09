OFFICIAL_TIER1_STAGE_BUDGET_RERUN_COMPLETED: no

# Task 33 Final Report — Final Full Official WISCO Run With Strict Stage Budget

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_33_FINAL_FULL_STAGE_BUDGET_RERUN.md`.
This is a raw, integrity-checked evaluation run only. The flat run
passed cleanly with a perfect result. The strict hierarchical run's
guard correctly aborted mid-run on exactly the fail-closed scenario the
task itself anticipated: a stage-budget exhaustion following a
retryable failure. No retry, repair, or analysis was performed.

## 1. Branch and base SHA

| | |
|---|---|
| Base branch | `reviewer2-stage-budget-live-preflight-20260809` |
| Required SHA | `f2ba8ccb03bab177616f3b39e596f050761fc0d4` |
| Verified `origin` SHA | `f2ba8ccb03bab177616f3b39e596f050761fc0d4` — match |
| New branch | `reviewer2-wisco-official-tier1-stage-budget-rerun-20260809` |

Working tree was clean before branching and clean throughout (no
source/test code was changed; every artefact this task produced lives
under the Git-ignored
`eval/local_runs/wisco_official_tier1_stage_budget_rerun_20260809T104006Z/`
output root). **The final pushed branch tip SHA is reported in
Claude's end-of-turn response, not inside this file.**

## 2. Historical preservation — before and after

| Check | Result |
|---|---|
| Task 24's four raw artifacts | All 4 SHA-256 match recorded values, both before and after |
| Task 25 diagnostic root (18 artefacts) | Present, unmodified |
| Task 26 report/empty hierarchical output | `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no`, unchanged; 0 files, unchanged |
| Task 29 preflight artefacts | Present, unmodified |
| Task 30 report/flat CSV/empty hierarchical output | `OFFICIAL_TIER1_RETRY_RERUN_COMPLETED: no`, unchanged; flat CSV SHA-256 re-verified byte-identical; 0 files, unchanged |
| Task 31 report/source behaviour | `STAGE_BUDGET_ENFORCEMENT_READY: yes`, unchanged; retry-config constants unchanged |
| Task 32 preflight artefacts (16 artefacts) | Present, unmodified |

No historical data was edited, reclassified, or used for accuracy or
flat-versus-hierarchical analysis anywhere in this task.

## 3. Exact runtime configuration and limitations

```text
QDRANT_TIMEOUT_SECONDS=8
QDRANT_QUERY_MAX_ATTEMPTS=3
QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5
```

Hierarchical run additionally used `--require-genuine-hierarchical
--max-stage-latency-ms 30000`.

**Limitations recorded before running, per the task's explicit
requirement** (all restated from Task 31/32, none re-derived here):

- `query_points(timeout=...)` is a Qdrant **server-side**
  operation-timeout hint, not a guaranteed client-side cancellation
  mechanism.
- The shared stage budget covers all branch queries, retries, and
  backoffs for one stage, and blocks new work from starting after the
  deadline.
- A query already in flight cannot be forcibly cancelled by the
  current code.
- **This task was required to fail closed if an in-flight overrun
  ultimately caused a strict-guard or integrity failure — §8 reports
  exactly this scenario occurring.**

## 4. Full preflight — all passed

| Gate | Result |
|---|---|
| WISCO dataset hash | `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c` — exact match |
| Package totals | 20,760 / 2,013 / 18,747 — exact |
| Leakage/duplicates/malformed codes | `leakage_found: false`; 0 malformed; `validate_benchmark_package ok: true` |
| Fresh heldout export | 18,747 rows; SHA-256 `41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c` (byte-identical to every prior independently-produced export of the same canonical source); ISCO-only; 0 non-blank `gold_isic`/`gold_isced`; all `case_id` unique |
| Official catalogue SHA-256 | `29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3` — exact match |
| Official levels | 10/43/130/436 — exact |
| Qdrant official collections (pre-run) | `major=10, submajor=43, minor=130, unit=436, flat=436` — exact match |
| Legacy collections (pre-run) | `major=10, submajor=43, minor=131, unit=441, isco_occupations=124` |
| Task 23 success manifest | `status: "success"` — confirmed present, not rebuilt |
| Task 31 telemetry/pass-through, Task 32 preflight status | Confirmed present (§2) |

All preflight gates passed; both evaluator commands were authorized to
begin.

## 5. Durable execution — one-time process launches

Both commands were launched directly as background (`nohup ... &`)
processes from their first invocation — no foreground execution, no
preliminary probe, no alternative command, and no duplicate invocation
for either run. Each original process was monitored via `ps -p <pid>`
polling until it exited on its own; neither was interrupted or killed
by this task.

| Run | PID | Actual process start (from log's own first timestamp) | Actual completion (output/log file mtime) | Exit | Log |
|---|---:|---|---|---|---|
| Flat | 907 | 2026-08-09T10:42:25Z | 2026-08-09T10:53:13Z (≈10.8 min) | 0 (normal; "Wrote 18747 row(s)" printed; internal timer: 626.5s) | `flat_stdout.log` |
| Hierarchical | 653 | 2026-08-09T16:18:37Z | 2026-08-09T16:56:07Z (≈37.5 min) | non-zero (`sys.exit(1)`, strict-guard abort; no CSV written) | `hierarchical_stdout.log` |

**Note on monitoring-loop detection delay, disclosed transparently**:
this session's own `ps -p <pid>` polling loops detected each process's
exit later than the process's own actual completion time (flat:
detected at 16:15:00Z vs. actual completion 10:53:13Z, a ≈5.4-hour
gap; hierarchical: detected at 17:01:09Z vs. actual completion
16:56:07Z, a ≈5-minute gap) — both consistent with this unattended
session's environment being intermittently suspended between polling
checks. This is a delay in **this task's own detection of an
already-completed, already-written, unaltered output**, not a delay,
interruption, or defect in either evaluator process itself: each
process's own internal completion message and its output file's
timestamp are authoritative and were used for all timing figures
above. Neither process was killed, restarted, or replaced due to this
detection lag; each ran to its own single natural termination and was
observed exactly once.

Output root: `eval/local_runs/wisco_official_tier1_stage_budget_rerun_20260809T104006Z/`
(confirmed Git-ignored). No raw CSV or log was edited after being
written.

## 6. Exact commands

**Flat:**

```bash
QDRANT_TIMEOUT_SECONDS=8 QDRANT_QUERY_MAX_ATTEMPTS=3 QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5 \
python eval/run_eval.py \
  --test-set eval/local_runs/wisco_official_tier1_stage_budget_rerun_20260809T104006Z/heldout_export_fresh.csv \
  --system flat --use-llm-reranker off --isco-catalogue-profile official_ilo2021_v1 \
  --config wisco_official_tier1_stage_budget_rerun_flat \
  --run-id wisco-official-tier1-stage-budget-rerun-flat \
  --output-dir eval/local_runs/wisco_official_tier1_stage_budget_rerun_20260809T104006Z/flat
```

**Hierarchical:**

```bash
QDRANT_TIMEOUT_SECONDS=8 QDRANT_QUERY_MAX_ATTEMPTS=3 QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5 \
python eval/run_eval.py \
  --test-set eval/local_runs/wisco_official_tier1_stage_budget_rerun_20260809T104006Z/heldout_export_fresh.csv \
  --system hierarchical --use-llm-reranker off --isco-catalogue-profile official_ilo2021_v1 \
  --require-genuine-hierarchical --max-stage-latency-ms 30000 \
  --config wisco_official_tier1_stage_budget_rerun_hierarchical \
  --run-id wisco-official-tier1-stage-budget-rerun-hierarchical \
  --output-dir eval/local_runs/wisco_official_tier1_stage_budget_rerun_20260809T104006Z/hierarchical
```

## 7. Full flat run — passed

Output: `.../flat/20260809T104240Z_wisco_official_tier1_stage_budget_rerun_flat.csv`
(18,747 rows).

| # | Condition | Result |
|---|---|---|
| 1 | Exit 0, 18,747 unique rows | Pass |
| 2 | No row error, valid 4-digit code, method exactly `flat_isco08_official_ilo2021_v1` | Pass — 0 errors, 0 bad codes, 1 distinct method value |
| 3 | No LLM/reranker/API/cost/token/ISIC/ISCED/SRE activity | Pass — 0 reranker-fired, 0 nonzero-cost, ISIC/ISCED blank, SRE `not_applicable` on all |
| 4 | Every flat telemetry record parseable and consistent | Pass — 0 malformed duration/attempts/attempt-list rows |
| 5 | `success_after_retry` transparent and genuine; failures fail the gate | **0 of 18,747 rows required a retry** — `flat_query_outcome` distribution: `{"success": 18747}` |
| 6 | No final exception/retry exhaustion/unavailable/coarse-missing code/malformed telemetry/output-cardinality issue/row error | Pass — none observed |
| 7 | Qdrant official/legacy counts unchanged | Pass — identical before/after (`diff` empty) |

Zero `WARNING backend.rag.hierarchy_engine` lines anywhere in
`flat_stdout.log`. Max observed `flat_query_duration_ms`: **130.094**;
mean: **8.62**. This is a **perfect result** — every one of 18,747
queries succeeded on its first attempt, well within the 8-second
per-request timeout, with zero retries anywhere.

**Flat gate: PASSED.** Proceeding to the hierarchical run was authorized.

## 8. Full strict hierarchical run — FAILED

Reached case 3,918 of 18,747 (progress lines printed) before the
strict guard fired on case 3,919's evaluation, at 2026-08-09T16:56:07Z
(≈37.5 minutes of actual processing). **Exit non-zero**
(`sys.exit(1)`). **No output CSV was written** (`hierarchical/` output
directory confirmed empty, 0 files).

**Exact failing evidence, verbatim from `hierarchical_stdout.log`:**

```text
WARNING backend.rag.hierarchy_engine: HierarchyBeamSearchEngine: stage budget exhausted after a retryable failure on 'isco08_major_groups_ilo2021_v1' (attempt 1/3) -- not retrying.

STRICT GUARD FAILURE (--require-genuine-hierarchical): case_id=WISCO-2423002100018-ur: pred_method='flat_isco08_official_ilo2021_v1' is not a genuine hierarchical method -- the flat fallback fired for this case
Aborting immediately -- no result CSV written. This run cannot be used as genuine hierarchical benchmark evidence.
```

| Field | Value |
|---|---|
| Failing `case_id` | `WISCO-2423002100018-ur` |
| Case index | 3,919 of 18,747 (case 3,918, `WISCO-2423002100018-ar`, was the last one printed before the abort) |
| Failing stage/collection | Stage 1, `isco08_major_groups_ilo2021_v1` |
| Failure mechanism (from the warning text, verbatim) | A retryable failure occurred on attempt 1 of 3; by the time a retry was considered, the shared stage-1 budget was already exhausted, so — per Task 31's explicit design — **no retry was attempted and no additional query was started** |
| Resulting `pred_method` | `flat_isco08_official_ilo2021_v1` (the existing, unmodified flat-fallback path, exactly as designed) |
| `WARNING` line count in the entire log | exactly 1 |
| `STRICT GUARD FAILURE` line count | exactly 1 (same case) |
| Rows written to output CSV | **0** |

**This is exactly the fail-closed scenario Task 33 explicitly
anticipated and required** ("This task must fail closed if an
in-flight overrun ultimately causes a strict-guard or integrity
failure"). The mechanism worked as designed at every layer: (1) a
retryable exception occurred; (2) Task 31's stage-budget check
correctly determined no budget remained for a retry and refused to
start one (rather than sleeping or querying uselessly); (3) the
existing, unmodified flat-fallback path fired, exactly as it does for
any other hierarchical-unavailability cause; (4) the existing,
unmodified strict guard correctly rejected the resulting
`flat_isco08_official_ilo2021_v1` method label; (5) zero rows were
written and zero Qdrant mutation occurred.

**What is not recoverable without violating the "no rerun" rule**:
because the strict guard aborts before any CSV row is written for the
violating case, this specific case's full structured
`stage1_query_telemetry` (per-query outcome/attempts/duration list)
does not exist in any output file and cannot be retrieved — only the
single log warning line above is available as evidence. This is stated
explicitly, not inferred or fabricated.

**Qdrant counts, before/after the aborted hierarchical run**: identical
in every collection — confirmed by `diff`. The abort caused zero
mutation.

**Hierarchical gate: FAILED at condition 5** (stage-budget exhaustion
following a retryable failure). Per the task's explicit instruction,
this run was **not retried**, no data was altered, no code was
repaired, and no collection was rebuilt. The original process's log
was preserved exactly as produced.

## 9. OFFICIAL_TIER1_STAGE_BUDGET_RERUN_COMPLETED

```text
OFFICIAL_TIER1_STAGE_BUDGET_RERUN_COMPLETED: no
```

The flat run passed every condition of its integrity gate with a
perfect, zero-retry result. The hierarchical run failed condition 5 of
its integrity gate: a stage-1 retryable failure exhausted the shared
30,000 ms stage budget before a retry could be attempted, correctly
triggering the existing flat-fallback path, which
`--require-genuine-hierarchical` correctly rejected — zero rows
written. Both full raw-output gates are required for `yes`; since the
hierarchical gate did not pass, the overall result is `no`.

## 10. Focused and full test results

```
python -m pytest backend/tests/test_stage_budget_enforcement.py eval/test_stage_budget_enforcement_serialization.py backend/tests/test_qdrant_retry_resilience.py eval/test_qdrant_retry_resilience_serialization.py backend/tests/test_flat_query_telemetry.py eval/test_flat_query_telemetry_serialization.py backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchical_store.py backend/tests/test_isco_classifier_official_profile.py backend/tests/test_official_isco08_catalogue.py backend/tests/test_official_isco08_profiles.py eval/test_require_genuine_hierarchical.py eval/test_official_isco08_profile_evaluator.py eval/test_model_free_isco_evaluation.py eval/test_wisco_leakage_audit.py eval/test_analyze_wisco_tier1.py -q
→ 267 passed in 2.97s

python -m pytest backend/tests eval/ -q
→ 2156 passed, 1 deselected, 1 warning in 353.44s (0:05:53)
```

**Exact match to the expected 2,156-passed baseline — zero deviation.**
No source or test code was changed in this task.

## 11. Confirmations

- No official or legacy Qdrant collection was created, deleted,
  overwritten, or otherwise mutated — confirmed by identical
  before/after point counts after both the flat run and the aborted
  hierarchical run (§7, §8).
- No protected branch was touched. No `git merge`, `git rebase`,
  `git reset`, `git clean`, `git stash`, `git pull`, or force-push
  occurred. No PR was created.
- No raw result row was edited after being written; the flat CSV and
  both process logs are preserved exactly as produced. The
  hierarchical run's empty output directory (0 files) is itself the
  correct, preserved evidence of its abort.
- No whole-run retry, case retry, selection rerun, or repeated
  evaluator invocation occurred for either command — each was launched
  and observed exactly once (§5).
- No accuracy, confidence interval, McNemar or other significance
  test, latency/cost comparison, result manifest, or paper/reviewer
  document edit was produced anywhere in this task.
- No `eval/local_runs/` artefact was committed — only this report is
  staged for commit.

## 12. Mandatory limitation

This task reports raw controlled WISCO ISCO-08 pipeline evidence only
— one passed flat integrity gate (perfect, zero-retry result) and one
failed hierarchical integrity gate (with full failure disclosure). It
is not a statistics or accuracy result, not real Labour Force Survey
validation, and supports no ISIC, ISCED, SRE, reranking, cost, or
real-world performance claim. WISCO remains a controlled multilingual
ISCO-08 benchmark. A separate, independently authorized analysis task
would be required to further investigate the stage-budget-exhaustion
event reported in §8, and — separately — to analyze the flat run's raw
output once a corresponding valid hierarchical run exists; neither was
performed here.

Stopping here, per the task's own instruction, after pushing this
report.
