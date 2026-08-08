OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no

# Task 26 Final Report — Clean Full Official WISCO ISCO-08 Rerun With Exception Telemetry

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_26_CLEAN_FULL_OFFICIAL_WISCO_TELEMETRY_RERUN.md`.
This is a raw, integrity-checked evaluation run only. The flat run fully
passed; the hierarchical run's strict guard correctly aborted mid-run
on a second, independent transient Qdrant timeout. No retry, repair,
or analysis was performed.

## 1. Branch, base SHA, final SHA, push, clean-tree status

| | |
|---|---|
| Base branch | `reviewer2-official-flat-unavailability-diagnostics-20260809` |
| Required SHA | `0590781c4a7562bb29e83da32bfb16b3fb01ca7d` |
| Verified `origin` SHA | `0590781c4a7562bb29e83da32bfb16b3fb01ca7d` — match |
| New branch | `reviewer2-wisco-official-tier1-telemetry-rerun-20260809` |
| Final commit SHA | recorded after this report's commit (see push confirmation below) |

Working tree was clean immediately before branching, clean throughout
(every artefact this task produced lives under the Git-ignored
`eval/local_runs/wisco_official_tier1_telemetry_rerun_20260808T223144Z/`
output root — confirmed via `git check-ignore -v`), and clean
immediately before this report's own commit (`git status --short` was
empty; no source or test code was changed by this task).

## 2. Exact commands run

**Preflight — Task 25 telemetry / focused tests:**

```bash
python -m pytest backend/tests/test_flat_query_telemetry.py eval/test_flat_query_telemetry_serialization.py backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchical_store.py backend/tests/test_isco_classifier_official_profile.py backend/tests/test_official_isco08_catalogue.py backend/tests/test_official_isco08_profiles.py eval/test_require_genuine_hierarchical.py eval/test_official_isco08_profile_evaluator.py eval/test_model_free_isco_evaluation.py eval/test_wisco_leakage_audit.py eval/test_analyze_wisco_tier1.py -q
```
→ 185 passed.

**Heldout export:**

```bash
python eval/export_benchmark_to_run_eval_csv.py \
  --records eval/local_benchmarks/wisco_isco08_v2_group_split/records.json \
  --split heldout \
  --out eval/local_runs/wisco_official_tier1_telemetry_rerun_20260808T223144Z/heldout_export_fresh.csv
```

**Flat run:**

```bash
QDRANT_TIMEOUT_SECONDS=30 python eval/run_eval.py \
  --test-set eval/local_runs/wisco_official_tier1_telemetry_rerun_20260808T223144Z/heldout_export_fresh.csv \
  --system flat \
  --use-llm-reranker off \
  --isco-catalogue-profile official_ilo2021_v1 \
  --config wisco_official_tier1_telemetry_rerun_flat \
  --run-id wisco-official-tier1-telemetry-rerun-flat \
  --output-dir eval/local_runs/wisco_official_tier1_telemetry_rerun_20260808T223144Z/flat
```

**Hierarchical run:**

```bash
QDRANT_TIMEOUT_SECONDS=30 python eval/run_eval.py \
  --test-set eval/local_runs/wisco_official_tier1_telemetry_rerun_20260808T223144Z/heldout_export_fresh.csv \
  --system hierarchical \
  --use-llm-reranker off \
  --isco-catalogue-profile official_ilo2021_v1 \
  --require-genuine-hierarchical \
  --max-stage-latency-ms 30000 \
  --config wisco_official_tier1_telemetry_rerun_hierarchical \
  --run-id wisco-official-tier1-telemetry-rerun-hierarchical \
  --output-dir eval/local_runs/wisco_official_tier1_telemetry_rerun_20260808T223144Z/hierarchical
```

`python eval/run_eval.py --help` was inspected first; every flag above
is a documented, existing flag. No `--limit`, `--reranker-model`, or
unsupported flag was used in either command. Each command was run
**exactly once**; no retry occurred for either command.

## 3. Preflight results and hashes/counts

| Gate | Result |
|---|---|
| 1. Task 25 telemetry present in code | `flat_query_outcome`/`flat_query_duration_ms`/`flat_query_exception_type`/`flat_query_exception_message` confirmed present in `eval/run_eval.py`'s `CaseResult` and `run_one_case()` (grep-verified); 185 relevant focused tests passed |
| 2. WISCO package hash | Recomputed `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c` — matches exactly |
| 2. Package totals | `20760` total; dev `2013`, heldout `18747` — exact |
| 2. Leakage/duplicates/malformed codes | `leakage_found: false`; `0` malformed heldout codes; `validate_benchmark_package` returned `ok: true` |
| 2. Fresh heldout export | `18747` rows; SHA-256 `41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c` — byte-identical to Task 17's and Task 24's independently-produced exports of the same canonical source; ISCO-only (zero `industry_text`/`education_text` columns, zero non-blank `gold_isic`/`gold_isced`) |
| 3. Official catalogue SHA-256 | `29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3` — matches exactly |
| 3. Official levels | `10/43/130/436` — exact, from `eval/verified_catalogue_counts.yaml` |
| 4. Qdrant official collections (pre-run) | `major=10, submajor=43, minor=130, unit=436, flat=436` — all present, non-empty, exact match to required table |
| 5. Legacy collections (pre-run) | `major=10, submajor=43, minor=131, unit=441, isco_occupations=124` |
| 6. Task 23 success manifest | `status: "success"`, `profile: "official_ilo2021_v1"` — confirmed present, not rebuilt |
| 7. Task 24/25 artefacts untouched (pre-run check) | All 4 Task 24 raw-artefact SHA-256 checksums matched their Task 25-recorded values; Task 25's diagnostic output root confirmed still present |

All preflight gates passed; both evaluation commands were authorized
to begin.

## 4. Wall times and output paths

| Run | Wall-clock | Exit behaviour | Output path |
|---|---|---|---|
| Flat | 715.0s (0.04s/case), per the run's own printed timer | Exit 0; CSV written | `eval/local_runs/wisco_official_tier1_telemetry_rerun_20260808T223144Z/flat/20260808T223406Z_wisco_official_tier1_telemetry_rerun_flat.csv` |
| Hierarchical | ~1483s (≈24.7 min) elapsed before abort, from process start (`02:47:47.85`) to the finalized log's last write (`03:12:30.36`) — reached case 8983/18,747 before the strict guard aborted | Strict-guard abort (`sys.exit(1)`); **no CSV written** (`hierarchical/` output directory confirmed empty) | n/a — no output CSV exists |

No raw CSV or process log was edited after being written. The
hierarchical run's own log (`hierarchical_stdout.log`) is preserved
exactly as produced.

## 5. Flat run — full results

| Check | Result |
|---|---|
| Exit / row count | Exit 0; exactly `18747` rows |
| Row-level errors | `0` |
| `pred_method` distribution | `{"flat_isco08_official_ilo2021_v1": 18747}` — single value, 100% correct |
| Valid 4-digit code count | `18747`/`18747` |
| Reranker / cost / token evidence | `reranker_fired` true on 0 rows; `reranker_model` blank on all; cost/token fields zero on all |
| ISIC/ISCED/SRE | Not constructed (`pred_isic_section`/`pred_isced_level` blank on all; `sre_status="not_applicable"` on all) |
| `flat_query_outcome` distribution | `{"success": 18747}` — 100% success, 0 exceptions |
| `flat_query_duration_ms` | min `6.541`, mean `8.809`, max `195.572` — all non-negative, all far below the 30,000 ms bound |
| `flat_query_exception_type` / `flat_query_exception_message` | blank on all 18,747 rows |
| Unique `case_id` | Yes, 18,747/18,747 |
| Post-run Qdrant counts vs pre-run | Identical (official and legacy) — `diff` produced no output |

**Flat gate: PASSED — all 8 conditions met.**

Log evidence: zero `WARNING backend.rag.hierarchy_engine` lines
anywhere in `flat_stdout.log` (grep count `0`) — this run experienced
no transient Qdrant exception at all, unlike Task 24's flat run.

## 6. Hierarchical run — failure evidence

**Hierarchical gate: FAILED at condition 3** (no fallback/flat method
label permitted).

Exact failing evidence, verbatim from `hierarchical_stdout.log`:

```text
WARNING backend.rag.hierarchy_engine: HierarchyBeamSearchEngine: query on 'isco08_major_groups_ilo2021_v1' failed: timed out

STRICT GUARD FAILURE (--require-genuine-hierarchical): case_id=WISCO-3521002100018-ur: pred_method='flat_isco08_official_ilo2021_v1' is not a genuine hierarchical method -- the flat fallback fired for this case
Aborting immediately -- no result CSV written. This run cannot be used as genuine hierarchical benchmark evidence.
```

| Field | Value |
|---|---|
| Failing `case_id` | `WISCO-3521002100018-ur` |
| Case index | 8983 of 18,747 |
| Failing collection (per the warning) | `isco08_major_groups_ilo2021_v1` (stage 1 of the hierarchical pipeline) — a **different** collection than Task 24's flat-run failure, which was `isco08_unit_groups_flat_ilo2021_v1` |
| Resulting `pred_method` | `flat_isco08_official_ilo2021_v1` (the hierarchical pipeline's own existing flat-fallback behaviour fired after the stage-1 exception, per `backend/rag/hierarchical_store.py`'s documented fallback design) |
| `WARNING` line count in log | exactly `1` |
| `STRICT GUARD FAILURE` line count in log | exactly `1` (same case) |
| Rows processed before abort | 8982 (printed); the 8983rd case triggered the abort before its own progress line was printed |
| Rows written to output CSV | `0` — the strict guard's documented behaviour is to write nothing on the first violation |

Per the task's own instruction, this run was **not retried**, no data
was altered, no code was repaired, and no collection was rebuilt.

**Qdrant counts after the aborted hierarchical run:**

| Collection | Pre-run | Post-run |
|---|---:|---:|
| `isco08_major_groups_ilo2021_v1` | 10 | 10 |
| `isco08_submajor_groups_ilo2021_v1` | 43 | 43 |
| `isco08_minor_groups_ilo2021_v1` | 130 | 130 |
| `isco08_unit_groups_ilo2021_v1` | 436 | 436 |
| `isco08_unit_groups_flat_ilo2021_v1` | 436 | 436 |
| `isco08_major_groups` (legacy) | 10 | 10 |
| `isco08_submajor_groups` (legacy) | 43 | 43 |
| `isco08_minor_groups` (legacy) | 131 | 131 |
| `isco08_unit_groups` (legacy) | 441 | 441 |
| `isco_occupations` (legacy) | 124 | 124 |

Identical in every collection (condition 6 of the hierarchical gate is
satisfied even though the run itself failed on condition 3 — the abort
caused zero Qdrant mutation).

**Note on interpretation, offered as observation only, not a
diagnosis performed in this task**: this is a second, independent
instance of the same class of event Task 25 diagnosed and classified
`confirmed_exception` for Task 24's flat run — a single transient
Qdrant client-side timeout, this time on a different collection
(`isco08_major_groups_ilo2021_v1`, stage 1) partway through a second
independent ~18,700-query sustained run. This task performed no new
root-cause diagnosis, added no retry/resilience logic, and draws no
conclusion about frequency, cause, or whether a further rerun would
succeed — that would be new diagnostic or remediation work outside
this task's scope.

## 7. Task 24 / Task 25 artefact integrity — confirmed unchanged

Re-verified after both runs completed: all four Task 24 raw-artefact
SHA-256 checksums (flat CSV, `flat_stdout.log`,
`flat_integrity_gate_report.json`, `failing_row_detail.json`) matched
their previously-recorded values exactly. Task 25's diagnostic output
root remains present and was not opened for writing. Task 24's and
Task 25's git-ignored output roots were not touched by any command in
this task.

## 8. Focused and full test results

```
python -m pytest backend/tests/test_flat_query_telemetry.py eval/test_flat_query_telemetry_serialization.py backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchical_store.py backend/tests/test_isco_classifier_official_profile.py backend/tests/test_official_isco08_catalogue.py backend/tests/test_official_isco08_profiles.py backend/tests/test_build_official_isco08_collections.py eval/test_require_genuine_hierarchical.py eval/test_official_isco08_profile_evaluator.py eval/test_model_free_isco_evaluation.py eval/test_run_eval_b2.py eval/test_docs_consistency.py eval/test_wisco_leakage_audit.py eval/test_analyze_wisco_tier1.py -q
→ 239 passed in 2.20s

python -m pytest backend/tests eval/ -q
→ 2074 passed, 1 deselected, 1 warning in 299.95s (0:04:59)
```

**Zero failures.** Exact match to Task 25's baseline (`2074`) — this
task made no source or test code change (execution-only), so an
unchanged total confirms zero regressions and confirms no test was
added, removed, repaired, or skipped.

## 9. OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED

```text
OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no
```

The flat run passed every condition of its integrity gate. The
hierarchical run failed condition 3 of its integrity gate (a
transient Qdrant timeout on `isco08_major_groups_ilo2021_v1` caused
the hierarchical pipeline's existing flat-fallback to fire once, at
case `WISCO-3521002100018-ur`, which `--require-genuine-hierarchical`
correctly caught and aborted on before any row was written). Per this
task's own instructions, both completing gates are required for a
`yes`; since the hierarchical gate did not pass, the overall result is
`no`. No retry was attempted.

## 10. Mandatory limitation

WISCO is a controlled multilingual ISCO-08 benchmark using a public
occupation-title dataset. It is not real Labour Force Survey
validation. No accuracy, precision, recall, confidence interval,
significance test, latency comparison, cost comparison, or manuscript
result is reported anywhere in this task — this report is limited to
raw process/output integrity evidence. The flat run's raw CSV (18,747
rows, gate-passed) is preserved for a future, separately authorized
analysis task to consume; it was not itself analyzed here.

## 11. Confirmations

- No protected branch (`master`, `conference1-b2-evaluation`,
  `reviewer2-wip-snapshot-20260807`, `reviewer2-b2-integration-20260807`,
  or any prior `reviewer2-*` task/integration branch through
  `reviewer2-official-flat-unavailability-diagnostics-20260809`) was
  touched.
- No PR was created.
- No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
  `git pull`, or force-push occurred.
- No Qdrant collection was rebuilt, overwritten, deleted, or recreated
  — both runs were read-only against Qdrant, confirmed by identical
  before/after point counts (§5, §6).
- No result row was edited after being written; the flat CSV and both
  process logs are preserved exactly as produced.
- No automatic retry of any full run or any failed case occurred at
  any point in this task.
- No `eval/local_runs/` raw output was committed — only this report is
  staged for commit.

Stopping here, per the task's own instruction, after pushing this
report.
