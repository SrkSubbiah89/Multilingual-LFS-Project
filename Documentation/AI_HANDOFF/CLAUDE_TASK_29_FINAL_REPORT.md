RETRY_CONFIGURATION_PREFLIGHT_READY: yes

# Task 29 Final Report — Live Strict Preflight for Bounded Qdrant Retry Configuration

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_29_LIVE_STRICT_RETRY_CONFIGURATION_PREFLIGHT.md`.
This is a small, live, read-only Qdrant preflight over a deterministic
525-case subset only — not a full 18,747-case benchmark, not a rerun of
Task 24 or Task 26, not an accuracy experiment, and not manuscript
evidence.

## 1. Branch and push confirmation

| | |
|---|---|
| Base branch | `reviewer2-wisco-timeout-resilience-baseline-20260809` |
| Required SHA | `1aad2d4c9b0f497eaa75b3c54bb769ee63a049b7` |
| Verified `origin` SHA | `1aad2d4c9b0f497eaa75b3c54bb769ee63a049b7` — match |
| New branch | `reviewer2-wisco-retry-config-preflight-20260809` |

Working tree was clean before branching and clean immediately before
this report's own commit (no source/test code was changed; the only
committed file is this report). **The final pushed branch tip SHA is
reported in Claude's end-of-turn response, not inside this file**, per
the task's own instruction.

## 2. Historical-evidence preservation — before and after all live commands

| Check | Before | After |
|---|---|---|
| Task 24's four raw artifacts (flat CSV, `flat_stdout.log`, `flat_integrity_gate_report.json`, `failing_row_detail.json`) | All 4 SHA-256 match recorded values | All 4 SHA-256 match recorded values (unchanged) |
| Task 25 diagnostic root (18 artefacts) | Present, unmodified | Present, unmodified |
| Task 26 report first line | `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no` | Unchanged |
| Task 26 failed hierarchical log SHA-256 | `aca8d13d2f7272d84a7c63b129e904a86b02741c0ef7935da0e3b99e899b376d` | Unchanged (re-hashed identical) |
| Task 26 hierarchical output directory | 0 files (no CSV, matching its documented abort) | 0 files (unchanged) |
| Task 27/28 retry code constants | `DEFAULT_MAX_QUERY_ATTEMPTS=1`, `MAX_QUERY_ATTEMPTS_HARD_CAP=3`, `DEFAULT_RETRY_BACKOFF_SECONDS=0.0`, `MAX_RETRY_BACKOFF_SECONDS_HARD_CAP=2.0` | Unchanged |
| No historical CSV/log/JSON/WISCO package/catalogue file edited | `git status --short` empty throughout | `git status --short` empty at task end |

## 3. Runtime configuration validated

Environment variables set for every evaluator process in this task:

```text
QDRANT_TIMEOUT_SECONDS=8
QDRANT_QUERY_MAX_ATTEMPTS=3
QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5
```

**Python-calculated theoretical upper bound** (not mental arithmetic):

```python
attempts = 3
per_attempt_timeout_s = 8
backoffs = attempts - 1          # = 2
backoff_s = 0.5
worst_case_s = attempts * per_attempt_timeout_s + backoffs * backoff_s
# worst_case_s = 3*8 + 2*0.5 = 25.0
worst_case_ms = worst_case_s * 1000   # = 25000.0
strict_cap_ms = 30000
below_strict_cap = worst_case_ms < strict_cap_ms   # True
```

Result: `worst_case_ms = 25000.0 < 30000` — below the strict
`--max-stage-latency-ms 30000` cap. **This is a configuration bound
only, not proof that any external call obeys it exactly** — it assumes
every attempt uses its full 8-second allowance, which is a worst case,
not a guarantee about actual network/server behaviour. The strict guard
(`--require-genuine-hierarchical --max-stage-latency-ms 30000`) remains
the empirical enforcement mechanism; this preflight's actual observed
maximum stage latency (§6) is reported separately and was far below
both this bound and the cap.

No code-level default or hard cap was changed. No timeout above 8s, no
more than 3 attempts, no more than 0.5s backoff (well under the 2.0s
hard cap), no unbounded retry, no whole-case retry, and no automatic
evaluation rerun was used.

## 4. Deterministic selection

**Base**: `eval/local_benchmarks/wisco_isco08_v2_group_split/reranking_subset_500.json`
(the repository's existing, established selection tool's output —
`eval/select_wisco_reranking_subset.py`, stratified by (language,
ISCO-08 major group), deterministic `sha256(seed:benchmark_id)`
ranking, `seed=42`, `target_size=500`, `actual_size=500`) — read
directly, not regenerated, so the canonical WISCO package was never
touched.

**Known-risk IDs**: the 24 fixed IDs from `eval/analyze_wisco_tier1.py::KNOWN_RISK_IDS`
(Task 12).

**Task 26 case**: `WISCO-3521002100018-ur` (the stage-1 timeout case
that aborted Task 26's hierarchical run).

**Combination and overlap**:

| | |
|---|---:|
| Base selection size | 500 |
| Known-risk IDs | 24 |
| Overlap (base ∩ known-risk) | 0 |
| Task 26 case already present in base/known-risk | No |
| **Final selection size** | **525** |
| Selection hash (SHA-256 of ordered final ID list) | `fcbd336775a9293fa86d97775dd7c763d0fee84ef67e839f14512c54a949da1e` |

No fabrication was needed — the union genuinely totals exactly 525 with
zero overlap.

**Language distribution (final 525)**: `{"ur": 99, "ar": 111, "hi": 102, "en": 112, "tl": 101}`

**ISCO major-group distribution (final 525)**: `{"0": 5, "1": 30, "2": 116, "3": 95, "4": 26, "5": 40, "6": 43, "7": 93, "8": 45, "9": 32}`

**Export schema check**: the materialized CSV
(`preflight_selected_525.csv`, 525 rows) has exactly the columns
`case_id, input_text, input_language, gold_isco_4digit, gold_isic,
gold_isced`; zero `industry_text`/`education_text` columns; 100% blank
`gold_isic`/`gold_isced` (0 non-blank of 525); 0 malformed
`gold_isco_4digit` values; all 525 `case_id` values unique; confirmed
`WISCO-3521002100018-ur` present.

No accuracy, precision, recall, language performance, or model
comparison statistic was calculated from this selection or its
results — it is a diagnostic input only.

## 5. Live Qdrant preflight gates

| Collection | Required | Observed (pre-run) |
|---|---:|---:|
| `isco08_major_groups_ilo2021_v1` | 10 | 10 |
| `isco08_submajor_groups_ilo2021_v1` | 43 | 43 |
| `isco08_minor_groups_ilo2021_v1` | 130 | 130 |
| `isco08_unit_groups_ilo2021_v1` | 436 | 436 |
| `isco08_unit_groups_flat_ilo2021_v1` | 436 | 436 |

Legacy collections (recorded, not required to match a specific value):
`isco08_major_groups=10, isco08_submajor_groups=43,
isco08_minor_groups=131, isco08_unit_groups=441,
isco_occupations=124`.

Task 23's success manifest confirmed present:
`eval/local_runs/official_isco08_collection_build_20260808T211235Z/build_manifest.json`
— `status: "success"`. No collection was built, deleted, overwritten,
optimized, or otherwise mutated by this task (read-only `get_collections()`/`count()` calls only).

## 6. Flat preflight run

Ignored output root:
`eval/local_runs/wisco_retry_config_preflight_20260809T062006Z/`

**Note on execution**: the first invocation of the flat command was
terminated after 3 minutes by this session's own foreground-command
tooling timeout, **before any output was written** (no CSV, no exit
code, no evaluator-level result of any kind existed — confirmed by an
empty output directory immediately after). This was a tooling/
orchestration artifact on the operator's side, not an evaluator
failure, gate outcome, or retry of a completed evaluation — nothing
evaluative had occurred yet to retry. The log from that interrupted
attempt was preserved, renamed
`flat_attempt1_interrupted_by_harness_timeout.log`, and is not counted
as a run. The command was then launched once, in the background, and
run to completion — this is **the one authorized flat preflight run**
this report evaluates.

**Exact command:**

```bash
QDRANT_TIMEOUT_SECONDS=8 QDRANT_QUERY_MAX_ATTEMPTS=3 QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5 \
python eval/run_eval.py \
  --test-set eval/local_runs/wisco_retry_config_preflight_20260809T062006Z/preflight_selected_525.csv \
  --system flat \
  --use-llm-reranker off \
  --isco-catalogue-profile official_ilo2021_v1 \
  --config wisco_retry_preflight_flat \
  --run-id wisco-retry-preflight-flat \
  --output-dir eval/local_runs/wisco_retry_config_preflight_20260809T062006Z/flat
```

Exit **0**. Wall time: **23.4s** (0.04s/case) — the run's own printed
timer. Output:
`.../flat/20260809T062611Z_wisco_retry_preflight_flat.csv` (525 rows).

### Flat pass gate — all 7 conditions met

| # | Condition | Result |
|---|---|---|
| 1 | Exit 0, exactly the selected number of unique cases | Pass — 525/525, all unique |
| 2 | No row-level error, valid 4-digit code, method exactly `flat_isco08_official_ilo2021_v1` | Pass — 0 errors, 0 bad codes, 1 distinct method value across all 525 rows |
| 3 | No LLM/reranker, ISIC, ISCED, SRE activity | Pass — `reranker_fired` true on 0 rows; cost 0 on all; `pred_isic_section`/`pred_isced_level` blank on all; `sre_status="not_applicable"` on all |
| 4 | `flat_query_outcome` is `success`/`success_after_retry`, non-negative total duration, valid positive attempt count, serializable attempt-duration list whose length equals the attempt count | Pass — all 525 rows well-formed; 0 malformed-duration/attempts/attempt-list rows |
| 5 | A `success_after_retry` outcome is permitted and reported transparently | N/A this run — **0 of 525 rows required a retry**; `flat_query_outcome` distribution: `{"success": 525}` |
| 6 | No `exception`/`retry_exhausted`/unavailable/missing/coarse-code/malformed-telemetry/duplicate/missing/extra row/row error | Pass — none observed |
| 7 | Qdrant official and legacy counts unchanged | Pass — identical before/after (`diff` produced no output) |

**Telemetry distribution**: `flat_query_outcome` = `{"success": 525}`;
`flat_query_attempts` = `{"1": 525}` (every row succeeded on the first
attempt — the 8-second per-attempt timeout was never approached: max
observed `flat_query_duration_ms` = **172.803**, mean = **10.67**).
The Task 26 timeout case (`WISCO-3521002100018-ur`) succeeded cleanly
here too: `pred_method=flat_isco08_official_ilo2021_v1`,
`pred_isco_4digit=3521`, `flat_query_outcome=success`,
`flat_query_attempts=1`, `flat_query_duration_ms=8.373`.

**Qdrant counts, before/after flat run**: identical in every collection
(both official and legacy) — confirmed by `diff`.

**Flat gate: PASSED.** Proceeding to the hierarchical run was authorized.

## 7. Strict hierarchical preflight run

**Exact command:**

```bash
QDRANT_TIMEOUT_SECONDS=8 QDRANT_QUERY_MAX_ATTEMPTS=3 QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5 \
python eval/run_eval.py \
  --test-set eval/local_runs/wisco_retry_config_preflight_20260809T062006Z/preflight_selected_525.csv \
  --system hierarchical \
  --use-llm-reranker off \
  --isco-catalogue-profile official_ilo2021_v1 \
  --require-genuine-hierarchical \
  --max-stage-latency-ms 30000 \
  --config wisco_retry_preflight_hierarchical \
  --run-id wisco-retry-preflight-hierarchical \
  --output-dir eval/local_runs/wisco_retry_config_preflight_20260809T062006Z/hierarchical
```

Run once, in the background, to completion — no interruption occurred
for this command. Exit **0**. Wall time: **81.3s** (0.15s/case).
Output: `.../hierarchical/20260809T062832Z_wisco_retry_preflight_hierarchical.csv`
(525 rows).

### Hierarchical pass gate — all 8 conditions met

| # | Condition | Result |
|---|---|---|
| 1 | Exit 0, exactly the selected number of unique cases | Pass — 525/525, all unique |
| 2 | No error, valid 4-digit code, method exactly `hierarchical_isco08_official_ilo2021_v1` | Pass — 0 errors, 0 bad codes, 1 distinct method value across all 525 rows |
| 3 | Genuine valid 4-stage evidence; no fallback/flat/unavailable/missing evidence/latency ≥30,000ms | Pass — 0 rows with missing/empty stage evidence, 0 rows at/over the latency cap |
| 4 | No LLM/reranker, ISIC, ISCED, SRE activity | Pass — same zero-activity confirmation as the flat run |
| 5 | `hier_stage_query_telemetry` parseable JSON, retry/exception summarized exactly | Pass — 0 malformed-JSON rows; every row's per-stage summary structurally valid |
| 6 | A successfully-retried stage may pass only with genuine final evidence and full latency < 30,000ms, disclosed | N/A this run — **0 stages, across all 525 cases and all 4 stages, required any retry** |
| 7 | `flat_query_*` fields blank on hierarchical rows | Pass — 0 rows with non-blank `flat_query_outcome` |
| 8 | Qdrant official and legacy counts unchanged after both runs | Pass — identical to the pre-run inventory |

**Retry disclosure**: zero rows show `any_retry: true` in any stage's
`hier_stage_query_telemetry`, and zero rows show `any_exception: true`.
The full 525-case × 4-stage run (2,100 stage-slots) experienced no
transient Qdrant exception of any kind under this configuration during
this preflight — this is an observation about this one run, not a
guarantee for any future run.

**Maximum stage latency observed across all 525 cases and all 4
stages: 468.26 ms** — far below both the 25,000 ms configuration bound
(§3) and the 30,000 ms strict cap.

**The Task 26 timeout case, in detail** (`WISCO-3521002100018-ur`):

```text
pred_method: hierarchical_isco08_official_ilo2021_v1
pred_isco_4digit: 3521
stage1_latency_ms: 8.99   stage2_latency_ms: 17.43
stage3_latency_ms: 36.33  stage4_latency_ms: 82.51
hier_stage_query_telemetry:
  stage1: {queries: 1, any_retry: false, any_exception: false, max_attempts_used: 1}
  stage2: {queries: 2, any_retry: false, any_exception: false, max_attempts_used: 1}
  stage3: {queries: 4, any_retry: false, any_exception: false, max_attempts_used: 1}
  stage4: {queries: 8, any_retry: false, any_exception: false, max_attempts_used: 1}
```

This case — the one that aborted Task 26's full hierarchical run with
a stage-1 timeout on `isco08_major_groups_ilo2021_v1` — completed
cleanly here, on the first attempt, with no retry needed. This is
consistent with Task 25's `confirmed_exception` diagnosis of a rare,
transient, non-reproducible event; it is **not** evidence that the
original Task 26 timeout would or would not recur under any other
configuration or run.

**Qdrant counts, before/after both runs**: identical in every
collection — confirmed by `diff`.

**Hierarchical gate: PASSED.**

## 8. RETRY_CONFIGURATION_PREFLIGHT_READY

```text
RETRY_CONFIGURATION_PREFLIGHT_READY: yes
```

Both the flat and strict hierarchical preflight runs passed every
condition of their respective integrity gates over the deterministic
525-case selection (including the Task 26 timeout case and all 24
Task 12 known-risk IDs), under the exact runtime configuration
specified in §3, with zero retries actually required by either run.

## 9. Focused and full test results

```
python -m pytest backend/tests/test_qdrant_retry_resilience.py eval/test_qdrant_retry_resilience_serialization.py backend/tests/test_flat_query_telemetry.py eval/test_flat_query_telemetry_serialization.py backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchical_store.py backend/tests/test_isco_classifier_official_profile.py backend/tests/test_official_isco08_catalogue.py backend/tests/test_official_isco08_profiles.py eval/test_require_genuine_hierarchical.py eval/test_official_isco08_profile_evaluator.py eval/test_model_free_isco_evaluation.py eval/test_wisco_leakage_audit.py eval/test_analyze_wisco_tier1.py -q
→ 240 passed in 4.39s

python -m pytest backend/tests eval/ -q
→ 2129 passed, 1 deselected, 1 warning in 354.50s (0:05:54)
```

**Exact match to the expected 2,129-passed baseline — no deviation to
report.** No source or test code was changed in this task, so this
exact match is expected and confirms zero regressions.

## 10. Mandatory limitation

This is a small controlled operational preflight over 525 deterministically-selected
cases, not a full benchmark and not real Labour Force Survey
validation. It reports no accuracy, statistical comparison, or paper
result. It confirms only that the Task 27/28 bounded retry mechanism
and its telemetry operate correctly, and that both the official flat
and strict hierarchical retrieval paths complete cleanly (with zero
retries actually needed) under the specified `QDRANT_TIMEOUT_SECONDS=8
/ QDRANT_QUERY_MAX_ATTEMPTS=3 / QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5`
configuration on this selection, including the specific case that
aborted Task 26. It does not establish, and this report does not
claim, WISCO accuracy, ISIC/ISCED/SRE evidence, reranking behaviour,
cost, or real-world performance. No full 18,747-case evaluation, no
WISCO rebuild/split/label/catalogue change, and no collection mutation
occurred. Stopping here, per the task's own instruction, after pushing
this report.
