OFFICIAL_TIER1_RETRY_RERUN_COMPLETED: no

# Task 30 Final Report — Full Official WISCO ISCO-08 Rerun With Bounded Retry Telemetry

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_30_FULL_OFFICIAL_WISCO_RETRY_RERUN.md`.
This is a raw, integrity-checked evaluation run only. The flat run
passed cleanly, including one genuine, correctly-recorded
`success_after_retry` case. The strict hierarchical run's guard
correctly aborted mid-run on a severe stage-4 latency violation. No
retry, repair, or analysis was performed.

## 1. Branch and push confirmation

| | |
|---|---|
| Base branch | `reviewer2-wisco-retry-config-preflight-20260809` |
| Required SHA | `10c5f1a8755a375e373ca38ab49cbe269d9426d1` |
| Verified `origin` SHA | `10c5f1a8755a375e373ca38ab49cbe269d9426d1` — match |
| New branch | `reviewer2-wisco-official-tier1-retry-rerun-20260809` |

Working tree was clean before branching and clean immediately before
this report's own commit (no source/test code was changed; every
artefact this task produced lives under the Git-ignored
`eval/local_runs/wisco_official_tier1_retry_rerun_20260809T065122Z/`
output root). **The final pushed branch tip SHA is reported in
Claude's end-of-turn response, not inside this file**, per the task's
own instruction.

## 2. Historical evidence preservation — before and after

| Check | Before | After |
|---|---|---|
| Task 24's four raw artifacts | All 4 SHA-256 match recorded values | All 4 SHA-256 match recorded values (unchanged) |
| Task 25 diagnostic evidence | Present, unmodified | Present, unmodified |
| Task 26 report first line | `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no` | Unchanged |
| Task 26 failed hierarchical output directory | 0 files | 0 files (unchanged) |
| Task 29 report first line | `RETRY_CONFIGURATION_PREFLIGHT_READY: yes` | Unchanged |
| Task 29 output root (13 artefacts) | Present, unmodified | Present, unmodified |
| Task 27 retry semantics (`DEFAULT_MAX_QUERY_ATTEMPTS=1`, `MAX_QUERY_ATTEMPTS_HARD_CAP=3`, `DEFAULT_RETRY_BACKOFF_SECONDS=0.0`, `MAX_RETRY_BACKOFF_SECONDS_HARD_CAP=2.0`) | Present | Unchanged |

`git status --short` was empty at task end — no historical CSV, log,
JSON, WISCO package file, or catalogue file was edited.

## 3. Runtime configuration and Python calculation

Environment variables set for both evaluator processes:

```text
QDRANT_TIMEOUT_SECONDS=8
QDRANT_QUERY_MAX_ATTEMPTS=3
QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5
```

```python
attempts = 3
per_attempt_timeout_s = 8
backoffs = attempts - 1          # = 2
backoff_s = 0.5
worst_case_ms = (attempts * per_attempt_timeout_s + backoffs * backoff_s) * 1000
# = 25000.0
strict_cap_ms = 30000
below_strict_cap = worst_case_ms < strict_cap_ms   # True
```

`worst_case_ms = 25000.0 < 30000` — this is a **per-query configuration
bound**, not a guarantee of actual observed behaviour. §7 below reports
an observed value that substantially exceeded this bound, which the
strict runtime guard (the empirical enforcement mechanism) correctly
caught.

## 4. Full dataset and collection preflight — all passed

| Gate | Result |
|---|---|
| WISCO dataset hash | Recomputed `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c` — exact match |
| Package totals | 20,760 total; dev 2,013, heldout 18,747 — exact |
| Leakage/duplicates/malformed codes | `leakage_found: false`; 0 malformed codes; `validate_benchmark_package` `ok: true` |
| Fresh heldout export | 18,747 rows; SHA-256 `41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c` (byte-identical to every prior independently-produced export of the same canonical source — Tasks 17/24/26); ISCO-only; 0 non-blank `gold_isic`/`gold_isced`; all `case_id` unique |
| Official catalogue SHA-256 | `29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3` — exact match |
| Official levels | 10/43/130/436 — exact, from `eval/verified_catalogue_counts.yaml` |
| Qdrant official collections (pre-run) | `major=10, submajor=43, minor=130, unit=436, flat=436` — exact match to required table |
| Legacy collections (pre-run) | `major=10, submajor=43, minor=131, unit=441, isco_occupations=124` |
| Task 23 success manifest | `status: "success"` — confirmed present, not rebuilt |
| Task 29 artefacts unchanged before beginning | Confirmed (§2) |

All preflight gates passed; both evaluator commands were authorized to
begin.

## 5. Durable execution — process IDs, timing, logs

Both commands were launched directly as background (`nohup ... &`)
processes from their first invocation — no foreground command that
could be killed by a tool/session timeout was used, and no
preliminary probe, duplicate command, or alternate evaluator command
was launched for either run. Each original process was monitored via
`ps -p <pid>` polling until it exited on its own.

| Run | PID | Start (UTC) | End (UTC) | Exit behaviour | Log |
|---|---:|---|---|---|---|
| Flat | 1710 | 2026-08-09T06:54:44Z | 2026-08-09T07:10:02Z | Completed normally; "Wrote 18747 row(s)" printed | `flat_stdout.log` |
| Hierarchical | 332 | 2026-08-09T07:13:43Z | 2026-08-09T08:06:00Z | Strict-guard abort (`sys.exit(1)`); no CSV written | `hierarchical_stdout.log` |

Neither process was interrupted, killed, or replaced by this task; each
ran to its own natural termination (normal completion for flat,
strict-guard-triggered abort for hierarchical) and was observed exactly
once.

## 6. Full flat run — passed

**Exact command:**

```bash
QDRANT_TIMEOUT_SECONDS=8 QDRANT_QUERY_MAX_ATTEMPTS=3 QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5 \
python eval/run_eval.py \
  --test-set eval/local_runs/wisco_official_tier1_retry_rerun_20260809T065122Z/heldout_export_fresh.csv \
  --system flat \
  --use-llm-reranker off \
  --isco-catalogue-profile official_ilo2021_v1 \
  --config wisco_official_tier1_retry_rerun_flat \
  --run-id wisco-official-tier1-retry-rerun-flat \
  --output-dir eval/local_runs/wisco_official_tier1_retry_rerun_20260809T065122Z/flat
```

Wall time (run's own timer): **822.4s** (0.04s/case). Output:
`.../flat/20260809T065524Z_wisco_official_tier1_retry_rerun_flat.csv`
(18,747 rows).

### Flat integrity gate — all 7 conditions met

| # | Condition | Result |
|---|---|---|
| 1 | Exit 0, exactly 18,747 unique rows | Pass |
| 2 | No row-level error, valid 4-digit code, method exactly `flat_isco08_official_ilo2021_v1` | Pass — 0 errors, 0 bad codes, 1 distinct method value |
| 3 | No reranker/LLM/API/ISIC/ISCED/SRE activity | Pass — 0 reranker-fired rows, 0 nonzero-cost rows, ISIC/ISCED blank, SRE `not_applicable` on all |
| 4 | Well-formed flat telemetry on every row | Pass — 0 malformed duration/attempts/attempt-list rows |
| 5 | `success_after_retry` permitted, reported transparently | **1 row**: `WISCO-7123000400018-ur` — see detail below |
| 6 | No exception/retry_exhausted/unavailable/invalid/duplicate/missing/extra row | Pass — none observed |
| 7 | Qdrant official/legacy counts unchanged after the run | Pass — identical before/after |

**Telemetry distribution**: `flat_query_outcome` = `{"success": 18746, "success_after_retry": 1}`;
`flat_query_attempts` = `{"1": 18746, "2": 1}`. Max
`flat_query_duration_ms` across all 18,747 rows: **30,501.28** (the
retried row itself); mean: **11.58 ms**.

**The one `success_after_retry` row, in full:**

```text
case_id: WISCO-7123000400018-ur
pred_method: flat_isco08_official_ilo2021_v1
pred_isco_4digit: 7123   (a valid, final, genuine code -- not fabricated)
pred_confidence: 0.8631
flat_query_outcome: success_after_retry
flat_query_attempts: 2
flat_query_duration_ms: 30501.28   (total across both attempts)
flat_query_attempt_durations_ms: [29540.302, 11.381]
flat_query_exception_type: ""   (blank -- populated only on a final exception/retry_exhausted outcome, not on eventual success)
```

Corresponding log line: `WARNING backend.rag.hierarchy_engine:
HierarchyBeamSearchEngine: query on 'isco08_unit_groups_flat_ilo2021_v1'
failed on attempt 1/3 (retryable): timed out -- retrying.` This retry
was not hidden or reclassified as a zero-hit result — the row's final
code and confidence are genuine, and the retry is fully disclosed in
both the CSV telemetry and this report.

**Notable, directly observed, unexplained fact** (recorded per the
task's evidence-preservation discipline, not diagnosed further): the
first attempt's own duration was **29,540.302 ms** — over 3.5× the
configured 8-second `QDRANT_TIMEOUT_SECONDS`. This means the
per-attempt client-side timeout did not appear to bound this
particular attempt's wall-clock duration to anywhere near 8 seconds.
No root cause is claimed or investigated here (this task's scope
excludes diagnosis/repair); it is reported as a fact relevant to
interpreting §7's larger anomaly.

**Qdrant counts, before/after flat run**: identical in every collection
— confirmed by `diff`.

**Flat gate: PASSED.** Proceeding to the hierarchical run was authorized.

## 7. Full strict hierarchical run — FAILED

**Exact command:**

```bash
QDRANT_TIMEOUT_SECONDS=8 QDRANT_QUERY_MAX_ATTEMPTS=3 QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5 \
python eval/run_eval.py \
  --test-set eval/local_runs/wisco_official_tier1_retry_rerun_20260809T065122Z/heldout_export_fresh.csv \
  --system hierarchical \
  --use-llm-reranker off \
  --isco-catalogue-profile official_ilo2021_v1 \
  --require-genuine-hierarchical \
  --max-stage-latency-ms 30000 \
  --config wisco_official_tier1_retry_rerun_hierarchical \
  --run-id wisco-official-tier1-retry-rerun-hierarchical \
  --output-dir eval/local_runs/wisco_official_tier1_retry_rerun_20260809T065122Z/hierarchical
```

Ran for **3,137s** (≈52.3 minutes) before aborting, reaching case
13,173 of 18,747 (progress lines printed) before the strict guard fired
on case 13,174's evaluation. **Exit non-zero** (`sys.exit(1)`, the
documented strict-guard abort path). **No output CSV was written**
(`hierarchical/` output directory confirmed empty, 0 files).

**Exact failing evidence, verbatim from `hierarchical_stdout.log`:**

```text
WARNING backend.rag.hierarchy_engine: HierarchyBeamSearchEngine: query on 'isco08_unit_groups_ilo2021_v1' failed on attempt 1/3 (retryable): timed out -- retrying.

STRICT GUARD FAILURE (--require-genuine-hierarchical): case_id=WISCO-7212080000000-hi: stage4_latency_ms=1155613.62 exceeds --max-stage-latency-ms=30000.0
Aborting immediately -- no result CSV written. This run cannot be used as genuine hierarchical benchmark evidence.
```

| Field | Value |
|---|---|
| Failing `case_id` | `WISCO-7212080000000-hi` |
| Failing stage | stage 4 (`isco08_unit_groups_ilo2021_v1`, the unit-group collection) |
| Recorded `stage4_latency_ms` | **1,155,613.62 ms** (≈19 minutes 16 seconds) |
| Strict cap | 30,000 ms |
| Exceeded cap by | ≈38.5× |
| `WARNING` line count in the entire log | exactly 1 |
| `STRICT GUARD FAILURE` line count | exactly 1 (same case) |
| Rows processed before abort | 13,173 (printed); case 13,174 triggered the abort before its own progress line printed |
| Rows written to output CSV | **0** |

**Retry disclosure, to the fullest extent the evidence allows**: the
one logged `WARNING` shows a retryable timeout on
`isco08_unit_groups_ilo2021_v1` (stage 4's collection) during this
case's evaluation, attempt 1 of 3, before the strict guard aborted.
Because the strict guard aborts **before** any CSV row is written for
the violating case, this case's own `hier_stage_query_telemetry`
(which would show per-branch `queries`/`any_retry`/`any_exception`/
`max_attempts_used`/`exception_types` for every stage) **does not
exist and cannot be recovered without violating the "no rerun" rule**
— this is stated explicitly rather than inferred or fabricated. What
is known: `stage4_latency_ms` is an **accumulated** total across every
beam-branch query issued at stage 4 for this case (confirmed by direct
code reading of `HierarchyBeamSearchEngine.search()`'s `_timed_query()`,
which sums `elapsed_ms` across every branch's stage-4 query rather than
recording a single query's time) — so the 1,155,613.62 ms figure most
plausibly reflects the sum of several stage-4 branch queries' durations
(each potentially itself retried), not necessarily one single query
stalling for 19 minutes. This is offered as a structurally-grounded
observation from the code's own accumulation design, **not a diagnosed
root cause** of why those queries were slow — no server/network/load
explanation is claimed or investigated here, consistent with this
task's explicit scope boundary and the discipline established in
Tasks 25/26/28.

**Maximum stage latency observed and compared to caps**: the single
recorded value, **1,155,613.62 ms**, is dramatically above both the
25,000 ms configuration bound (§3) and the 30,000 ms strict cap —
confirming §3's own explicit caveat that the configuration bound is
not a guarantee of observed behaviour and that the strict runtime
guard is the actual empirical enforcement mechanism. This is the same
class of anomaly flagged (in miniature — 29.5s vs. an expected ~8s) in
the one retried flat-run row (§6); here its scale is far larger.

**Qdrant counts, before/after the aborted hierarchical run**: identical
in every collection — confirmed by `diff`. The abort caused zero
mutation.

**Hierarchical gate: FAILED at condition 3** (excessive stage latency,
30,000 ms cap). Per the task's explicit instruction, this run was
**not retried**, no data was altered, no code was repaired, and no
collection was rebuilt. The original process's log was preserved
exactly as produced.

## 8. OFFICIAL_TIER1_RETRY_RERUN_COMPLETED

```text
OFFICIAL_TIER1_RETRY_RERUN_COMPLETED: no
```

The flat run passed every condition of its integrity gate, including
correctly disclosing its one genuine `success_after_retry` case. The
hierarchical run failed condition 3 of its integrity gate: a single
stage-4 latency value of 1,155,613.62 ms vastly exceeded the 30,000 ms
strict cap for case `WISCO-7212080000000-hi`, correctly triggering the
`--require-genuine-hierarchical` guard's abort with zero rows written.
Both full raw-output gates are required for `yes`; since the
hierarchical gate did not pass, the overall result is `no`.

## 9. Focused and full test results

```
python -m pytest backend/tests/test_qdrant_retry_resilience.py eval/test_qdrant_retry_resilience_serialization.py backend/tests/test_flat_query_telemetry.py eval/test_flat_query_telemetry_serialization.py backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchical_store.py backend/tests/test_isco_classifier_official_profile.py backend/tests/test_official_isco08_catalogue.py backend/tests/test_official_isco08_profiles.py eval/test_require_genuine_hierarchical.py eval/test_official_isco08_profile_evaluator.py eval/test_model_free_isco_evaluation.py eval/test_wisco_leakage_audit.py eval/test_analyze_wisco_tier1.py eval/test_docs_consistency.py -q
→ 247 passed in 2.58s

python -m pytest backend/tests eval/ -q
→ 2129 passed, 1 deselected, 1 warning in 302.58s (0:05:02)
```

**Exact match to the expected 2,129-passed baseline — zero deviation.**
No source or test code was changed in this task, so this exact match
confirms zero regressions.

## 10. Confirmations

- No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
  `git pull`, or force-push occurred. No PR was created. No protected
  branch was touched.
- No official/legacy Qdrant collection was created, deleted,
  overwritten, or otherwise mutated — confirmed by identical
  before/after point counts after both the flat run and the aborted
  hierarchical run (§6, §7).
- No raw result row was edited after being written; the flat CSV and
  both process logs are preserved exactly as produced. The
  hierarchical run's empty output directory (0 files) is itself the
  correct, preserved evidence of its abort.
- No whole-run retry, case retry, or repeated evaluator invocation
  occurred for either command — each was launched and observed exactly
  once (§5).
- No accuracy, Wilson interval, McNemar test, latency comparison
  table, cost figure, result manifest, or manuscript statement was
  produced anywhere in this task.
- No `eval/local_runs/` artefact was committed — only this report is
  staged for commit.

## 11. Mandatory limitation

This task reports raw controlled WISCO ISCO-08 pipeline evidence only
— specifically, one passed flat integrity gate (with full retry
disclosure) and one failed hierarchical integrity gate (with full
failure disclosure). It is not an accuracy or statistical result, not
real Labour Force Survey validation, and supports no ISIC, ISCED, SRE,
reranking, cost, or real-world performance claim. WISCO remains a
controlled multilingual ISCO-08 benchmark. A separate, independently
authorized task would be required to further investigate the stage-4
latency anomaly reported in §7, and — separately — to analyze the
flat run's raw output once a corresponding valid hierarchical run
exists; neither was performed here.

Stopping here, per the task's own instruction, after pushing this
report.
