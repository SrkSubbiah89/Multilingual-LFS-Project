PRECISE_DEADLINE_PREFLIGHT_READY: yes

# Task 35 Final Report — Precise Client-Deadline Live Preflight

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_35_PRECISE_DEADLINE_LIVE_PREFLIGHT.md`.
This is a small, live, read-only Qdrant preflight over a deterministic
528-case selection, run once each for the official flat and strict
hierarchical systems, deciding operational eligibility only. It is not
a benchmark, accuracy analysis, or manuscript-update task.

## 1. Branch and base SHA

| | |
|---|---|
| Base branch | `reviewer2-qdrant-client-deadline-precision-20260809` |
| Required/verified base SHA | `656bbe144f9ab635766fd77ecbc549b498c9431e` (confirmed against both the local branch and `origin` before branching, re-confirmed identical at close) |
| New branch | `reviewer2-precise-deadline-live-preflight-20260809` |
| Final commit SHA | reported in Claude's end-of-turn response, not inside this file |

Working tree was clean before branching. No source, test, dependency,
configuration, or manuscript file was modified anywhere in this task.
The only committed file is this report.

## 2. PRECISE_DEADLINE_PREFLIGHT_READY

```text
PRECISE_DEADLINE_PREFLIGHT_READY: yes
```

Both the official-flat and strict-hierarchical preflight runs passed
every required gate (Sections 6/7) with zero row-level errors, zero
budget exhaustion, and no historical-evidence or collection-count
drift (Sections 4/8).

## 3. Deterministic selection

**Base**: `eval/local_benchmarks/wisco_isco08_v2_group_split/reranking_subset_500.json`
(read directly, unmodified — the established 500-record stratified
seed-42 subset, `actual_size=500`, `seed=42`).

**Known-risk IDs**: the 24 fixed IDs from
`eval/analyze_wisco_tier1.py::KNOWN_RISK_IDS` (Task 12).

**Exceptional cases added**: `WISCO-3521002100018-ur` (Task 26
stage-1 timeout case), `WISCO-7123000400018-ur` and
`WISCO-7212080000000-hi` (both named Task 30 anomaly cases),
`WISCO-2423002100018-ur` (Task 33 failure case).

| | |
|---|---:|
| Base count | 500 |
| Known-risk IDs | 24 |
| Task 26 case | 1 |
| Task 30 cases | 2 |
| Task 33 case | 1 |
| Overlap (base ∩ any exceptional ID) | 0 |
| **Final selection size (deduplicated union)** | **528** |
| Selection ID-list hash (SHA-256 of ordered final ID list) | `5932a98be0d4ea77d31182df65c1d72430cb78727f7d9049ebc1071e716b3cdc` |

No count was asserted in advance; 528 is the genuine union size with
zero overlap between the base 500 and the 28 added exceptional IDs.

**Language distribution**: `{"en": 112, "ar": 111, "hi": 103, "ur": 101, "tl": 101}`

**ISCO major-group distribution**: `{"0": 5, "1": 30, "2": 117, "3": 95, "4": 26, "5": 40, "6": 43, "7": 95, "8": 45, "9": 32}`

**Verification performed**: every one of the 528 selected `case_id`
values confirmed present in `heldout_run_eval_format.csv` (18,747
rows, the frozen heldout split only — the dev split was never
consulted); all 528 `case_id` values unique; all 528
`gold_isco_4digit` values match `^\d{4}$` (read with `dtype=str` to
preserve leading zeros, e.g. major group "0" codes like `0110`); the
exported CSV schema is exactly `case_id, input_text, input_language,
gold_isco_4digit, gold_isic, gold_isced` with no
`industry_text`/`education_text` columns and 0/528 non-blank
`gold_isic`/`gold_isced`. All four exceptional IDs confirmed present
in the final export.

Selection and evaluator outputs were written only beneath
`eval/local_runs/precise_deadline_live_preflight_20260809T120000Z/`
(Git-ignored, not committed).

## 4. Historical preservation — before and after

**Pre-run snapshot** (before any live read): SHA-256 hashes of every
file under the Task 24, 25, 26, 29, 30, 32, and 33 local-run output
roots (110 files total, recursively), plus the official ISCO catalogue
CSV, `verified_catalogue_counts.yaml`, B1 frozen config,
`full130_access_guard.py` and its leakage manifest, the WISCO dataset
hash/heldout export/split manifest/500-record subset,
`eval/analyze_wisco_tier1.py` (source of `KNOWN_RISK_IDS`), the
official-catalogue/collection-builder source files, and all seven
prior tasks' own final reports.

**Post-run re-verification**: all 110 files re-hashed after both
preflight runs completed — **zero mismatches, zero missing files**.

**Qdrant point counts** — five official `ilo2021_v1` collections,
read-only (`get_collection()` only; no create/delete/rebuild/populate/
compact/alias operation of any kind):

| Collection | Pre-run | Post-run |
|---|---:|---:|
| `isco08_major_groups_ilo2021_v1` | 10 | 10 |
| `isco08_submajor_groups_ilo2021_v1` | 43 | 43 |
| `isco08_minor_groups_ilo2021_v1` | 130 | 130 |
| `isco08_unit_groups_ilo2021_v1` | 436 | 436 |
| `isco08_unit_groups_flat_ilo2021_v1` | 436 | 436 |

Exact match, before and after both runs.

## 5. Exact runtime configuration and worst-case bound

Environment variables set for both evaluator processes:

```text
QDRANT_TIMEOUT_SECONDS=8
QDRANT_QUERY_MAX_ATTEMPTS=3
QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5
```

**Worst-case single-query bound**, calculated in Python before running:

```python
worst_case_single_query_seconds = (
    QDRANT_QUERY_MAX_ATTEMPTS * QDRANT_TIMEOUT_SECONDS
    + (QDRANT_QUERY_MAX_ATTEMPTS - 1) * QDRANT_QUERY_RETRY_BACKOFF_SECONDS
)
# = 3 * 8 + 2 * 0.5 = 25.0 seconds (25,000 ms)
```

Recorded before running: Task 34.1's minimum practical client-side
deadline is **1.0 second**
(`hierarchy_engine.MIN_PRACTICAL_CLIENT_DEADLINE_SECONDS`); the
hierarchical run's stage budget is **30,000 ms**
(`--max-stage-latency-ms 30000`). The worst-case single-query bound
(25,000 ms) is below the stage budget (30,000 ms), consistent with
Task 32's same calculation and precedent.

## 6. Exact command lines, timestamps, exit status, output locations

Both evaluator processes were launched directly as durable background
jobs (`nohup ... &`) from their first invocation, with stdout/stderr
captured to a log file and monitored via the process's own PID and the
log/output file's own timestamps — never treated as retryable on a
controller/foreground timeout.

| Run | PID | Command | Launched (UTC) | Output CSV written (UTC, from filename/mtime) | Exit | Log |
|---|---:|---|---|---|---|---|
| Flat | 398 | `python eval/run_eval.py --test-set .../preflight_selected.csv --system flat --use-llm-reranker off --isco-catalogue-profile official_ilo2021_v1 --config precise_deadline_preflight_flat --run-id precise-deadline-preflight-flat --output-dir .../flat` | 2026-08-09T18:52:57Z | 2026-08-09T18:53:38Z | 0 (normal completion; "Completed 528 case(s) in 15.1s"; "Wrote 528 row(s)"; no traceback in log) | `flat_stdout.log` |
| Hierarchical | 1061 | `python eval/run_eval.py --test-set .../preflight_selected.csv --system hierarchical --use-llm-reranker off --isco-catalogue-profile official_ilo2021_v1 --require-genuine-hierarchical --max-stage-latency-ms 30000 --config precise_deadline_preflight_hierarchical --run-id precise-deadline-preflight-hierarchical --output-dir .../hierarchical` | 2026-08-09T18:53:50Z | 2026-08-09T18:55:19Z | 0 (normal completion; "Completed 528 case(s) in 62.4s"; "Wrote 528 row(s)"; no traceback in log) | `hierarchical_stdout.log` |

Output root:
`eval/local_runs/precise_deadline_live_preflight_20260809T120000Z/`

Flat CSV:
`.../flat/20260809T185318Z_precise_deadline_preflight_flat.csv` (528 rows).

Hierarchical CSV:
`.../hierarchical/20260809T185411Z_precise_deadline_preflight_hierarchical.csv`
(528 rows).

Both CSVs record `git_commit=656bbe1` (matches the base branch's SHA
short form) and a `config_hash`, confirming they ran against the
exact intended checked-out code.

## 7. Flat gate — passed

| Gate | Result |
|---|---|
| Exit code 0 | Pass |
| Exact selected-row count | 528/528 |
| Zero row-level errors | 0/528 non-blank `error` |
| Every row `pred_method=flat_isco08_official_ilo2021_v1` | 528/528 |
| Every predicted code a valid official 4-digit code | 528/528 match `^\d{4}$`, 0 invalid |
| No unavailable/fallback/empty prediction | Confirmed by method-label uniformity above |
| Telemetry present for every row | 0/528 blank `flat_query_duration_ms` |
| Query outcome `success`/`success_after_retry`, disclosed | `flat_query_outcome` distribution: `{"success": 528}` — 0 retries needed |
| Retryable exceptions disclosed | 0/528 non-blank `flat_query_exception_type`/`_message`; `flat_query_attempts` distribution: `{"1": 528}` |
| No Qdrant mutation | Confirmed — read-only searches only (Section 4) |
| No forbidden model/LLM/reranker activity | `reranker_fired`: `{"False": 528}`; `prompt_tokens`/`completion_tokens`/`estimated_cost_usd` all `0`/`0.0` for all 528 rows |

Max `flat_query_duration_ms` observed: 110.69 ms (far below the 8,000 ms
configured timeout and the 25,000 ms worst-case bound).

## 8. Strict hierarchical gate — passed

| Gate | Result |
|---|---|
| Exit code 0 | Pass |
| Exact selected-row count | 528/528 |
| Zero row-level errors | 0/528 non-blank `error` |
| Genuine hierarchical method label, never flat/fallback/unavailable/synthetic | `pred_method` distribution: `{"hierarchical_isco08_official_ilo2021_v1": 528}` |
| Complete valid stage-1..4 evidence for every row | 0/528 blank/null `stage1_candidates`..`stage4_candidates` |
| No stage-budget exhaustion | `stage_budget_exhausted=true` count across all stages/rows: **0** |
| No missing stage telemetry | `hier_stage_query_telemetry` present and JSON-parseable for all 528/528 rows |
| Each stage's observed latency ≤ 30,000 ms | Max `stageN_latency_ms` observed: 168.69 ms (stage4); max single-attempt query duration across all 6,381 queries: 70.697 ms |
| Per-stage deadline established and propagated | `initial_stage_budget_ms=30000.0` and `configured_query_timeout_seconds=8.0` on every one of the 6,381 stage-query telemetry entries, with `remaining_stage_budget_ms_at_entry` correctly decreasing across successive branch queries within a stage |
| All exceptional IDs present and clean | All 4 confirmed present with `pred_method=hierarchical_isco08_official_ilo2021_v1` and blank `error` |
| Retries policy-compliant, final latency within cap | `any_retry=true` count: **0** (no retry was needed anywhere in this run); `any_exception=true` count: **0**; `max_attempts_used` observed: 1 (uniformly) |
| No Qdrant mutation | Confirmed — read-only searches only (Section 4) |
| No forbidden model/LLM/reranker activity | `reranker_fired`: `{"False": 528}` |

Total stage-level queries executed: 6,381 (528 cases × up to 4 stages,
1–8 branch queries per stage depending on beam width), every one
`outcome="success"` on the first attempt.

`stage1_source` distribution: `{"semantic_retrieval": 405,
"keyword_map": 123}` — both are documented genuine-hierarchical
routes (see `hierarchical_store.py`'s Task 13 revision docstring);
`keyword_anchor_retry_used`: `{"False": 528}`.

## 9. Confirmations

- **Live operations were read-only**: the only live Qdrant operations
  performed were `get_collection()` point-count reads (Section 4) and
  the two evaluator runs' own read-only vector searches. No
  create/delete/rebuild/populate/compact/alias/mutation of any
  collection occurred, confirmed by the exact pre/post point-count
  match (Section 4).
- **No forbidden activity**: no Ollama call, paid API call, LLM call,
  reranker invocation, or `SentenceTransformer` load outside the
  existing evaluator path occurred — confirmed by `reranker_fired=False`
  on all 1,056 rows across both runs and zero prompt/completion tokens
  in the flat run.
- **No benchmark, analysis, or manuscript activity**: no accuracy,
  latency, cost, or coverage statistic was computed or claimed beyond
  the gate-condition checks in Sections 7/8; no manuscript file was
  touched.

## 10. Exact test commands and outputs

Full suite (run before any live operation):
```
python -m pytest backend/tests eval/ -q
```
Result: `2174 passed, 1 deselected, 1 warning in 327.21s` — matches
the expected baseline exactly.

Smallest relevant preflight/evaluator validation tests:
```
python -m pytest eval/test_pre_run_check.py eval/test_official_isco08_profile_evaluator.py eval/test_full130_access_guard.py eval/test_require_genuine_hierarchical.py eval/test_validate_controlled_benchmark.py -q
```
Result: `97 passed in 84.31s`.

## 11. Protected-branch preservation

No merge, rebase, reset, clean, stash, pull, or force-push was
performed. No protected or prior-task branch was touched. Only
`reviewer2-precise-deadline-live-preflight-20260809` was created, and
only that branch is pushed.

## 12. Conservative evidence statement

- Passing this task is an **operational preflight only** — it confirms
  the repaired official flat and strict hierarchical paths are
  eligible to attempt one future full 18,747-case WISCO rerun; it is
  not that rerun.
- This is **not** an accuracy, latency, cost, coverage, real-LFS,
  ISIC, ISCED, SRE, or manuscript claim of any kind. No such measure
  was computed here.
- WISCO remains a controlled, externally-sourced multilingual ISCO-08
  benchmark — never real Labour Force Survey respondent data, and this
  task does not change that.
- B1 remains stale/quarantined; nothing in this task touches or
  revalidates it.
- Per the task's explicit instruction, the full 18,747-case rerun was
  **not** started. That requires a separate task and approval.
