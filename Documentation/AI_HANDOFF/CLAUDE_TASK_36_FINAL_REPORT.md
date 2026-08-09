OFFICIAL_TIER1_PRECISE_DEADLINE_FULL_RERUN_COMPLETED: yes

# Task 36 Final Report — Official Tier-1 Precise-Deadline Full Rerun

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_36_OFFICIAL_TIER1_PRECISE_DEADLINE_FULL_RERUN.md`.
This is a raw, integrity-checked evaluation run only. Both the full
official-flat and full strict-hierarchical evaluator processes
completed cleanly on all 18,747 heldout cases, passing every required
gate. No accuracy, significance, or manuscript claim is made here —
see Section 11.

## 1. Status, branch, SHA, push, working tree

```text
OFFICIAL_TIER1_PRECISE_DEADLINE_FULL_RERUN_COMPLETED: yes
```

| | |
|---|---|
| Base branch | `reviewer2-precise-deadline-live-preflight-20260809` |
| Required/verified base SHA | `280ad76c7cafa3d6ff8ffc55199fbe7efd447306` (confirmed against both the local branch and `origin` before branching, re-confirmed identical at close) |
| New branch | `reviewer2-wisco-official-tier1-precise-deadline-full-results-20260809` |
| Final commit SHA | reported in Claude's end-of-turn response, not inside this file |

Working tree was clean before branching. All generated artefacts live
under the Git-ignored
`eval/local_runs/wisco_official_tier1_precise_deadline_full_20260809T190821Z/`
output root. The only committed file is this report.

## 2. Exact commands, environment, PIDs, timestamps, exit status

Environment variables set for both evaluator processes:

```text
QDRANT_TIMEOUT_SECONDS=8
QDRANT_QUERY_MAX_ATTEMPTS=3
QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5
```

**Worst-case single-query bound**, calculated in Python before running:
```text
3 * 8 + 2 * 0.5 = 25.0 seconds = 25,000 ms
```
Recorded before running: the strict stage budget is 30,000 ms; Task
34.1 enforces a minimum practical client-side deadline of 1.0 second;
a bounded client-side deadline bounds the caller's wait but does not
claim to cancel server-side work already in flight (Task 31/34
limitation, unchanged).

**Heldout export** (fresh, deterministic reformatting, no
classifier/network call):
```bash
python eval/export_benchmark_to_run_eval_csv.py \
  --records eval/local_benchmarks/wisco_isco08_v2_group_split/records.json \
  --split heldout \
  --out eval/local_runs/wisco_official_tier1_precise_deadline_full_20260809T190821Z/heldout_export_fresh.csv
```
Wrote 18,747 rows.

**Full official-flat run**:
```bash
python eval/run_eval.py \
  --test-set .../heldout_export_fresh.csv \
  --system flat \
  --use-llm-reranker off \
  --isco-catalogue-profile official_ilo2021_v1 \
  --config wisco_official_tier1_precise_deadline_full_flat \
  --run-id wisco-official-tier1-precise-deadline-full-flat \
  --output-dir .../flat
```

| | |
|---|---|
| PID | 288 |
| Launched (UTC) | 2026-08-09T19:11:11Z |
| Output CSV written (UTC, from log) | 2026-08-09T19:11:33Z |
| Duration (from log) | 587.6s (0.03s/case) |
| Exit | 0 — normal completion; "Completed 18747 case(s) in 587.6s"; "Wrote 18747 row(s)"; zero `Traceback` occurrences in the log |
| Output CSV | `.../flat/20260809T191133Z_wisco_official_tier1_precise_deadline_full_flat.csv` |
| Output CSV SHA-256 | `d0692b4a87db11945dbd046ead79a32dcc36d715fbaa66fcc4e4853102be5f02` |
| Log SHA-256 | `f3350f2f05a118af3f5486b07b3f5fca88d0704f659159ae4924b0c31bd4cbef` |

**Full official strict-hierarchical run** (only launched after the
flat gate passed in full — Section 6):
```bash
python eval/run_eval.py \
  --test-set .../heldout_export_fresh.csv \
  --system hierarchical \
  --use-llm-reranker off \
  --isco-catalogue-profile official_ilo2021_v1 \
  --require-genuine-hierarchical \
  --max-stage-latency-ms 30000 \
  --config wisco_official_tier1_precise_deadline_full_hierarchical \
  --run-id wisco-official-tier1-precise-deadline-full-hierarchical \
  --output-dir .../hierarchical
```

| | |
|---|---|
| PID | 400 |
| Launched (UTC) | 2026-08-09T19:22:35Z |
| Output CSV written (UTC, from log) | 2026-08-09T19:22:56Z |
| Duration (from log) | 2,518.9s (0.13s/case, ≈42 minutes) |
| Exit | 0 — normal completion; "Completed 18747 case(s) in 2518.9s"; "Wrote 18747 row(s)"; zero `Traceback` occurrences in the log |
| Output CSV | `.../hierarchical/20260809T192256Z_wisco_official_tier1_precise_deadline_full_hierarchical.csv` |
| Output CSV SHA-256 | `b72193c8411b076df827abf2fa8bc6c2c72ac60f86799e435b94ff5eb237a8a4` |
| Log SHA-256 | `341caed36104ca35ede025651626fda9564e52d8e007629ac38300e9148f9d8a` |

Both processes were started directly as durable background jobs
(`nohup ... &`) from their first invocation, with stdout/stderr
captured to a separate log file per run and monitored via each
process's own PID plus its log/output file's own timestamps — never
treated as retryable on a controller/foreground tool timeout.

Both output CSVs record `git_commit=280ad76` (matches the base
branch's SHA short form): flat `config_hash=91887124f2b5`,
hierarchical `config_hash=d15543a4fbb8`.

Heldout export SHA-256:
`41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c`.

## 3. Pre-run asset/package/catalogue/hash/point-count checks

**WISCO frozen package** (`eval.audit_wisco_benchmark_leakage.recompute_dataset_hash`/`run_full_audit`, live against `records.json`, not a cached summary):

| Check | Result |
|---|---|
| Total records | 20,760 |
| Dev records | 2,013 |
| Heldout records | 18,747 |
| Dataset hash (recomputed vs. `dataset_hash.txt`) | `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c` — match |
| `source_family_split_check.ok` | `true` (0 leaking source keys of 4,232) |
| `text_duplicate_check.ok` | `true` (0 cross-split exact duplicate groups) |
| `integrity_check.ok` | `true`; `n_malformed_isco_codes: 0` |
| `leakage_found` | `false` |
| `overall_ok_no_leakage` | `true` |
| `evaluation_readiness` | `ready_for_evaluation` |
| Fresh heldout export row count | 18,747; unique `case_id`: 18,747 |
| Heldout export schema | exactly `case_id, input_text, input_language, gold_isco_4digit, gold_isic, gold_isced`; no `industry_text`/`education_text` columns |
| Heldout `gold_isic`/`gold_isced` blank | 0/18,747 non-blank (both) |
| Heldout malformed `gold_isco_4digit` | 0/18,747 |

**Official ILO ISCO-08 catalogue**:

| Check | Result |
|---|---|
| Normalized catalogue SHA-256 | `29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3` — matches `eval/verified_catalogue_counts.yaml` |
| Verified counts | major 10 / submajor 43 / minor 130 / unit 436 |

**Qdrant point counts** — five official `ilo2021_v1` collections, read-only:

| Collection | Pre-run | Post-run (after both evaluator runs) |
|---|---:|---:|
| `isco08_major_groups_ilo2021_v1` | 10 | 10 |
| `isco08_submajor_groups_ilo2021_v1` | 43 | 43 |
| `isco08_minor_groups_ilo2021_v1` | 130 | 130 |
| `isco08_unit_groups_ilo2021_v1` | 436 | 436 |
| `isco08_unit_groups_flat_ilo2021_v1` | 436 | 436 |

Exact match, before and after. Also re-checked (and confirmed
unchanged) immediately after the flat run, before launching the
hierarchical run.

**Historical evidence hashes**: 120 files hashed before any live
operation — every file under the Task 24, 25, 26, 29, 30, 32, 33, and
35 local-run output roots (recursively), plus the official catalogue
CSV, `verified_catalogue_counts.yaml`, B1 frozen config,
`full130_access_guard.py`/leakage manifest, the WISCO dataset
hash/heldout export/split manifest/records.json/500-record subset,
`eval/analyze_wisco_tier1.py`, the official-catalogue/collection-builder
source files, and the final reports for Tasks 24, 25, 26, 29, 30, 32,
33, 34, 34.1, and 35. Re-hashed at task completion: **zero mismatches,
zero missing files, all 120/120 byte-identical.**

## 4. Full flat gate — passed

| Gate | Result |
|---|---|
| Exit code 0 | Pass |
| Exactly 18,747 rows, unique, matching heldout exactly | Pass — `set(case_id) == heldout_ids` |
| Zero row-level error | 0/18,747 non-blank `error` |
| Every `pred_method` exactly `flat_isco08_official_ilo2021_v1` | 18,747/18,747 |
| Every predicted code a valid official 4-digit code | 18,747/18,747 match `^\d{4}$`, 0 invalid |
| No unavailable/fallback/empty/method-drift | Confirmed by method-label uniformity above |
| Flat telemetry present for every row | 0/18,747 blank `flat_query_duration_ms` |
| `flat_query_outcome` success or transparent `success_after_retry` | `{"success": 18747}` — **0 retries occurred anywhere in this run** |
| Retry/exception fully represented in telemetry | 0/18,747 non-blank `flat_query_exception_type`/`_message`; `flat_query_attempts`: `{"1": 18747}` |
| No forbidden model/LLM/reranker/cost/token/ISIC/ISCED/SRE activity | `reranker_fired`: `{"False": 18747}`; `prompt_tokens`/`completion_tokens`: `{"0": 18747}`; `estimated_cost_usd`: `{"0.0": 18747}`; `pred_isic_section`/`pred_isced_level` non-blank: 0; `sre_status`: `{"not_applicable": 18747}` |
| Qdrant point counts and historical hashes unchanged | Confirmed (Section 3) |

`flat_query_duration_ms`: mean 8.37 ms, p95 10.57 ms, max 119.21 ms —
all far below the 8,000 ms configured timeout and 25,000 ms worst-case
bound.

No exceptional `success_after_retry` cases occurred — the run was
100% first-attempt success, so no per-case retry disclosure table is
needed.

## 5. Full strict-hierarchical gate — passed

| Gate | Result |
|---|---|
| Exit code 0 | Pass |
| Exactly 18,747 rows, unique, matching heldout exactly | Pass — `set(case_id) == heldout_ids` |
| Zero row-level error | 0/18,747 non-blank `error` |
| Every `pred_method` exactly `hierarchical_isco08_official_ilo2021_v1` | 18,747/18,747 — no flat, fallback, unavailable, synthetic, missing, or mixed label |
| Complete, valid, distinct stage-1..4 evidence for every row | 0/18,747 blank/null `stage1_candidates`..`stage4_candidates` |
| Stage telemetry JSON-parseable and present on every row | 18,747/18,747 parsed successfully, 0 parse errors |
| Every stage query has deadline/budget telemetry | `initial_stage_budget_ms=30000.0` and `configured_query_timeout_seconds=8.0` present on all 228,232 stage-query telemetry entries |
| Zero stage-budget exhaustion | `stage_budget_exhausted=true` count across all stages/rows: **0** |
| Every row's stage latency ≤ 30,000 ms | `stageN_latency_ms` max across all rows: stage1 208.01 ms, stage2 86.73 ms, stage3 2,581.40 ms, stage4 202.29 ms — 0 rows over 30,000 ms at any stage |
| All Task 12 known-risk IDs + Task 26/30/33 exceptional IDs present and pass | All 28 confirmed present in both flat and hierarchical output, all with blank `error`, correct method label, and zero retry/exception/budget-exhaustion flags in per-case stage telemetry |
| Retries policy-compliant, final latency under cap | `any_retry=true` count: **0** across all 228,232 stage queries — no retry occurred anywhere in this run |
| Zero exhausted retries, non-timeout exceptions, hidden failure | `any_exception=true` count: **0**; `max_attempts_used` observed: 1 (uniformly) |
| No forbidden model/LLM/reranker/cost/token/ISIC/ISCED/SRE activity | `reranker_fired`: `{"False": 18747}`; `prompt_tokens`: `{"0": 18747}`; `estimated_cost_usd`: `{"0.0": 18747}`; `pred_isic_section`/`pred_isced_level` non-blank: 0; `sre_status`: `{"not_applicable": 18747}` |
| Qdrant point counts and historical hashes unchanged | Confirmed (Section 3) |

Total stage-level queries executed: 228,232 (18,747 cases × up to 4
stages, 1–8 branch queries per stage), every one `outcome="success"`
on the first attempt.

Per-stage latency summary (ms):

| Stage | Mean | p95 | Max |
|---|---:|---:|---:|
| stage1 | 6.71 | 9.97 | 208.01 |
| stage2 | 15.32 | 27.73 | 86.73 |
| stage3 | 31.26 | 49.09 | 2,581.40 |
| stage4 | 54.30 | 83.26 | 202.29 |

`stage1_source` distribution: `{"semantic_retrieval": 14647,
"keyword_map": 4100}` — both documented genuine-hierarchical routes
(Task 13 revision). `keyword_anchor_retry_used`: `{"False": 18747}`;
`timed_out_flag`: `{"False": 18747}`; `degraded`: `{"False": 18747}`.

## 6. Order of execution

The hierarchical process was launched only after every flat gate
condition in Section 4 was independently confirmed passing — consistent
with the task's explicit "do not begin the hierarchical process until
all flat gate checks pass" requirement.

## 7. Confirmations

- **Live operations were read-only / no mutation**: the only live
  Qdrant operations were `get_collection()` point-count reads (before,
  between, and after both runs) and the two evaluator runs' own
  read-only vector searches. No create/delete/rebuild/populate/
  compact/alias/mutation of any collection occurred at any point,
  confirmed by the exact pre/post point-count match (Section 3).
- **No forbidden activity**: no Ollama call, paid API call, LLM call,
  or reranker invocation occurred anywhere in either run — confirmed
  by `reranker_fired=False` on all 37,494 rows across both runs and
  zero prompt/completion tokens and zero cost throughout. No ISIC,
  ISCED, or SRE classification was computed (`pred_isic_section`/
  `pred_isced_level` blank, `sre_status="not_applicable"` on every
  row).
- **No analysis or manuscript activity**: no accuracy, Wilson-interval,
  McNemar, paired-comparison, or other analysis script was invoked;
  no manuscript, figure, README, or document file was touched. This
  run produced validated raw data only.

## 8. Exact full-suite result

```bash
python -m pytest backend/tests eval/ -q
```
Result: `2174 passed, 1 deselected, 1 warning in 331.95s` — matches
the expected baseline exactly. Run before any live evaluation.

## 9. Protected-branch preservation

No merge, rebase, reset, clean, stash, pull, or force-push was
performed. No protected or prior-task branch was touched. Only
`reviewer2-wisco-official-tier1-precise-deadline-full-results-20260809`
was created, and only that branch is pushed.

## 10. Conservative evidence boundary

- This report validates **raw evaluator output only** — row counts,
  method-label distributions, code validity, telemetry completeness,
  retry/exception/budget-exhaustion counts, and latency distributions.
  It is **not yet analyzed** and is **not manuscript-ready**: no
  accuracy figure, confidence interval, significance test, or
  flat-vs-hierarchical comparison was computed here, per the task's
  explicit prohibition.
- WISCO remains a controlled, externally-sourced multilingual ISCO-08
  benchmark — never real Labour Force Survey respondent data, and this
  task does not change that.
- This run cannot support any ISIC, ISCED, SRE, real-field, cost, or
  coverage claim (none of those fields were populated — see Sections
  4/5).
- B1 remains stale/quarantined; nothing in this task touches or
  revalidates it.
- Per the task's explicit instruction, no follow-on analysis or
  manuscript task was started. A separate task and approval are
  required before this raw output can be analyzed or cited.
