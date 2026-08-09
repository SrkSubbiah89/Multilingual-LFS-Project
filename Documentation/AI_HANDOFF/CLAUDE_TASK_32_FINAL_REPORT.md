STAGE_BUDGET_PREFLIGHT_READY: yes

# Task 32 Final Report — Live Strict Preflight of Stage-Budget Enforcement

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_32_LIVE_STAGE_BUDGET_PREFLIGHT.md`.
This is a small, live, read-only Qdrant preflight over a deterministic
527-case selection only — not a full 18,747-case benchmark, not a
rerun of Task 30, not analysis, and not manuscript evidence.

## 1. Branch and base SHA

| | |
|---|---|
| Base branch | `reviewer2-qdrant-stage-budget-enforcement-20260809` |
| Required SHA | `bfa02c508054fedd7886a1a43e85db816298ed14` |
| Verified `origin` SHA | `bfa02c508054fedd7886a1a43e85db816298ed14` — match |
| New branch | `reviewer2-stage-budget-live-preflight-20260809` |

Working tree was clean before branching and clean throughout (no
source/test code was changed; `git status --short` was empty
immediately before this report's own commit). **The final pushed
branch tip SHA is reported in Claude's end-of-turn response, not
inside this file.**

## 2. Historical preservation — before and after

| Check | Result |
|---|---|
| Task 24's four raw artifacts | All 4 SHA-256 match recorded values, both before and after |
| Task 25 diagnostic root (18 artefacts) | Present, unmodified |
| Task 26 report first line | `OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no`, unchanged; failed hierarchical output directory: 0 files, unchanged |
| Task 29 preflight output root (13 artefacts) | Present, unmodified |
| Task 30 report first line | `OFFICIAL_TIER1_RETRY_RERUN_COMPLETED: no`, unchanged; flat CSV SHA-256 re-verified byte-identical; failed hierarchical output directory: 0 files, unchanged |
| Task 31 report first line | `STAGE_BUDGET_ENFORCEMENT_READY: yes`, unchanged |
| Task 31 source behaviour | `stage_budget_exhausted` telemetry field confirmed present (6 occurrences across `hierarchy_engine.py`/`hierarchical_store.py`/`run_eval.py`); retry-config constants unchanged (`DEFAULT_MAX_QUERY_ATTEMPTS=1`, `MAX_QUERY_ATTEMPTS_HARD_CAP=3`, `DEFAULT_RETRY_BACKOFF_SECONDS=0.0`, `MAX_RETRY_BACKOFF_SECONDS_HARD_CAP=2.0`) |

No historical output was modified, reinterpreted, or used for any
flat-versus-hierarchical accuracy/statistical analysis.

## 3. Exact runtime configuration and explicit limitations

Environment variables set for both evaluator processes:

```text
QDRANT_TIMEOUT_SECONDS=8
QDRANT_QUERY_MAX_ATTEMPTS=3
QDRANT_QUERY_RETRY_BACKOFF_SECONDS=0.5
```

Hierarchical run additionally used `--require-genuine-hierarchical
--max-stage-latency-ms 30000`. Confirmed from source (Task 31,
`eval/run_eval.py::main()`): `effective_stage_budget_ms =
args.max_stage_latency_ms` only when both `args.system == "hierarchical"`
and `args.require_genuine_hierarchical` are true, threaded into
`clf.classify(max_stage_latency_ms=...)` → `HierarchicalISCOStore.search()`
→ `HierarchyBeamSearchEngine.search(max_stage_latency_ms=30000.0)`,
which establishes one shared monotonic deadline per stage. This was
confirmed **operationally**, not merely by code reading: every
successful row's `hier_stage_query_telemetry` (§7) shows
`"initial_stage_budget_ms": 30000.0` for every stage.

**Limitations recorded before running, per the task's explicit
requirement**:
- `query_points(timeout=...)` is a Qdrant **server-side**
  operation-timeout hint (a `?timeout=N` query-string parameter),
  audited and documented in Task 31's final report — not a client-side
  socket timeout.
- The client-side HTTP stack (httpx, via qdrant-client) is **not**
  claimed to forcibly cancel an already-started request; no
  thread/process-level cancellation exists in this codebase.
- The shared stage budget prevents a **subsequent** branch
  query/retry/backoff from starting once the deadline has passed, but a
  single already-in-flight query can still overrun before control
  returns to the budget check. This preflight's clean result (§6/§7)
  does not contradict or resolve this limitation — it simply reports
  that no query in this run exhibited such an overrun.

## 4. Deterministic risk-enriched selection

**Base**: `eval/local_benchmarks/wisco_isco08_v2_group_split/reranking_subset_500.json`
(read directly, unmodified — the established 500-record stratified
seed-42 subset, `actual_size=500`).

**Known-risk IDs**: the 24 fixed IDs from
`eval/analyze_wisco_tier1.py::KNOWN_RISK_IDS` (Task 12).

**Three exact additional cases**: `WISCO-3521002100018-ur` (Task 26
stage-1 timeout case), `WISCO-7123000400018-ur` (Task 30 flat
`success_after_retry` case), `WISCO-7212080000000-hi` (Task 30 stage-4
budget/latency anomaly case).

| | |
|---|---:|
| Base count | 500 |
| Known-risk IDs | 24 |
| Overlap (base ∩ known-risk) | 0 |
| Extra-case overlap (already present in base/known-risk) | 0 |
| Added-risk count | 27 |
| **Final selection size** | **527** |
| Selection hash (SHA-256 of ordered final ID list) | `f85ce29c4ff831a1fa630a0973fcb25a18961ce0418b5996fe93037e0a5cafa4` |

No count was asserted in advance; 527 is the genuine union size with
zero overlap.

**Language distribution**: `{"ur": 100, "ar": 111, "hi": 103, "en": 112, "tl": 101}`

**ISCO major-group distribution**: `{"0": 5, "1": 30, "2": 116, "3": 95, "4": 26, "5": 40, "6": 43, "7": 95, "8": 45, "9": 32}`

**Export schema check**: `preflight_selected.csv` (527 rows) has
exactly `case_id, input_text, input_language, gold_isco_4digit,
gold_isic, gold_isced`; zero `industry_text`/`education_text` columns;
0/527 non-blank `gold_isic`/`gold_isced`; 0 malformed
`gold_isco_4digit`; all 527 `case_id` unique; all three required exact
cases confirmed present.

No accuracy, language-performance, group-performance, or
classification-quality measure was calculated from this selection or
its results.

## 5. Preflight checks — all passed

| Gate | Result |
|---|---|
| WISCO dataset hash | `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c` — exact match |
| Package totals | 20,760 / 2,013 / 18,747 — exact |
| Leakage/duplicates/malformed codes | `leakage_found: false`; 0 malformed; `validate_benchmark_package ok: true` |
| Official catalogue SHA-256 | `29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3` — exact match |
| Official levels | 10/43/130/436 — exact |
| Qdrant official collections (pre-run) | `major=10, submajor=43, minor=130, unit=436, flat=436` — exact match |
| Legacy collections (pre-run) | `major=10, submajor=43, minor=131, unit=441, isco_occupations=124` |
| Task 23 success manifest | `status: "success"` — confirmed present, not rebuilt |
| Task 31 telemetry/pass-through present in source | Confirmed (§2) |

No collection was built, deleted, overwritten, optimized, or mutated —
every check above was a read-only `get_collections()`/`count()` call.

## 6. Durable execution — one-time process launches

Both commands were launched directly as background (`nohup ... &`)
processes from their first invocation — no foreground execution, no
preliminary probe, no alternative command, and no duplicate invocation
for either run. Each original process was monitored via `ps -p <pid>`
polling until it exited on its own; neither was interrupted or killed.

| Run | PID | Command | Start (UTC) | End (UTC) | Exit | Log |
|---|---:|---|---|---|---|---|
| Flat | 1523 | `python eval/run_eval.py --test-set .../preflight_selected.csv --system flat --use-llm-reranker off --isco-catalogue-profile official_ilo2021_v1 --config stage_budget_preflight_flat --run-id stage-budget-preflight-flat --output-dir .../flat` | 2026-08-09T09:19:07Z | 2026-08-09T09:20:44Z | 0 (normal completion; "Wrote 527 row(s)" printed) | `flat_stdout.log` |
| Hierarchical | 828 | `python eval/run_eval.py --test-set .../preflight_selected.csv --system hierarchical --use-llm-reranker off --isco-catalogue-profile official_ilo2021_v1 --require-genuine-hierarchical --max-stage-latency-ms 30000 --config stage_budget_preflight_hierarchical --run-id stage-budget-preflight-hierarchical --output-dir .../hierarchical` | 2026-08-09T09:22:03Z | 2026-08-09T09:24:14Z | 0 (normal completion; "Wrote 527 row(s)" printed) | `hierarchical_stdout.log` |

Output root: `eval/local_runs/stage_budget_live_preflight_20260809T091704Z/`
(confirmed Git-ignored via `git check-ignore -v`). No raw CSV or log
was edited after being written.

## 7. Flat run — passed

Wall time (run's own timer): **22.0s** (0.04s/case). Output:
`.../flat/20260809T091932Z_stage_budget_preflight_flat.csv` (527 rows).

| # | Condition | Result |
|---|---|---|
| 1 | Exit 0, exact 527 unique rows | Pass |
| 2 | No row error, valid 4-digit code, method exactly `flat_isco08_official_ilo2021_v1` | Pass — 0 errors, 0 bad codes, 1 distinct method value |
| 3 | No LLM/reranker/API/ISIC/ISCED/SRE activity | Pass — 0 reranker-fired, 0 nonzero-cost, ISIC/ISCED blank, SRE `not_applicable` on all |
| 4 | Every flat telemetry record parseable and internally consistent | Pass — 0 malformed duration/attempts/attempt-list rows |
| 5 | Successful retries transparent and genuine; failures fail the gate | **0 of 527 rows required a retry** — `flat_query_outcome` distribution: `{"success": 527}` |
| 6 | Official/legacy collection counts unchanged | Pass — identical before/after (`diff` empty) |

Zero `WARNING backend.rag.hierarchy_engine` lines anywhere in
`flat_stdout.log` (grep count 0) — every query succeeded on its first
attempt. Max observed `flat_query_duration_ms`: **326.116**; mean:
**11.32**. The Task 30 `success_after_retry` case
(`WISCO-7123000400018-ur`) succeeded cleanly on the first attempt here
(`flat_query_attempts=1`) — consistent with the transient,
non-reproducible nature already established for these anomalies. The
Task 26 timeout case (`WISCO-3521002100018-ur`) also succeeded cleanly
(`flat_query_duration_ms=8.176`).

**Flat gate: PASSED.** Proceeding to the hierarchical run was authorized.

## 8. Strict hierarchical run — passed

Wall time (run's own timer): **78.3s** (0.15s/case). Output:
`.../hierarchical/20260809T092227Z_stage_budget_preflight_hierarchical.csv`
(527 rows).

| # | Condition | Result |
|---|---|---|
| 1 | Exit 0, exact 527 unique rows | Pass |
| 2 | No row error, valid 4-digit code, method exactly `hierarchical_isco08_official_ilo2021_v1` | Pass — 0 errors, 0 bad codes, 1 distinct method value |
| 3 | Genuine complete stage evidence; no fallback/flat/unavailable/missing evidence/latency ≥30,000ms | Pass — 0 rows with bad stage evidence, 0 rows at/over the cap |
| 4 | Every `stage{i}_query_telemetry` parseable | Pass — 0 malformed-JSON rows |
| 5 | `stage_budget_exhausted` false on every successful row; no budget/retry exhaustion, final exception, or guard abort | Pass — 0 rows with any stage exception; 0 rows with any stage retry; strict guard never fired (log confirms zero `STRICT GUARD FAILURE` lines) |
| 6 | `flat_query_*` blank on all hierarchical rows | Pass — 0 rows with non-blank `flat_query_outcome` |
| 7 | No LLM/reranker/API/ISIC/ISCED/SRE activity | Pass — same zero-activity confirmation as the flat run |
| 8 | Qdrant counts unchanged after both runs | Pass — identical to the pre-run inventory |

**Stage query counts and retry/exhaustion summary** (aggregated across
all 527 cases × 4 stages = up to 2,108 stage-slots, up to 8 branch
queries per case at stage 4): **zero** rows show `any_retry: true` in
any stage's telemetry, and **zero** rows show `any_exception: true` or
`stage_budget_exhausted: true` anywhere. **Maximum stage latency
observed across all 527 cases and all 4 stages: 201.83 ms** — far below
both the 30,000 ms strict cap and the theoretical 25,000 ms per-query
configuration bound from Task 29/30.

**The two Task 30 anomaly cases, in detail:**

- `WISCO-3521002100018-ur` (Task 26 stage-1 timeout case): completed
  cleanly, `stage1_latency_ms=8.07` .. `stage4_latency_ms=96.85`; every
  stage's telemetry shows `"stage_budget_exhausted": false`,
  `"configured_query_timeout_seconds": 8.0`,
  `"initial_stage_budget_ms": 30000.0`, and per-query
  `"remaining_stage_budget_ms_at_entry"` values around 29,900-30,000 ms
  — confirming the budget was established, tracked, and never
  approached.
- `WISCO-7212080000000-hi` (Task 30 stage-4 budget/latency anomaly
  case, `stage4_latency_ms` was `1,155,613.62` in Task 30): completed
  cleanly here, `pred_method=hierarchical_isco08_official_ilo2021_v1`,
  `pred_isco_4digit=7212`, with all four stages' telemetry showing
  `"stage_budget_exhausted": false` and no retries or exceptions.

This is consistent with — and does not contradict — Task 25's
`confirmed_exception` diagnosis of rare, transient, non-reproducible
events; it is not evidence that either historical anomaly would or
would not recur under a different run, and no such claim is made.

**Qdrant counts, before/after both runs**: identical in every
collection — confirmed by `diff`.

**Hierarchical gate: PASSED.**

## 9. STAGE_BUDGET_PREFLIGHT_READY

```text
STAGE_BUDGET_PREFLIGHT_READY: yes
```

Both the flat and strict hierarchical preflight runs passed every
condition of their respective integrity gates over the deterministic
527-case risk-enriched selection (including all 24 Task 12 known-risk
IDs and all three named anomaly cases from Tasks 26/30), under the
exact runtime configuration specified in §3, with zero retries and zero
budget exhaustions actually required by either run, and with rich,
structurally-consistent Task 31 telemetry confirming the 30,000 ms
value was genuinely established and tracked as the shared per-stage
monotonic deadline.

## 10. Focused and full test results

```
python -m pytest backend/tests/test_stage_budget_enforcement.py eval/test_stage_budget_enforcement_serialization.py backend/tests/test_qdrant_retry_resilience.py eval/test_qdrant_retry_resilience_serialization.py backend/tests/test_flat_query_telemetry.py eval/test_flat_query_telemetry_serialization.py backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchical_store.py backend/tests/test_isco_classifier_official_profile.py backend/tests/test_official_isco08_catalogue.py backend/tests/test_official_isco08_profiles.py eval/test_require_genuine_hierarchical.py eval/test_official_isco08_profile_evaluator.py eval/test_model_free_isco_evaluation.py eval/test_wisco_leakage_audit.py eval/test_analyze_wisco_tier1.py -q
→ 267 passed in 2.41s

python -m pytest backend/tests eval/ -q
→ 2156 passed, 1 deselected, 1 warning in 326.01s (0:05:26)
```

**Exact match to the expected 2,156-passed baseline — zero deviation.**
No source or test code was changed in this task.

## 11. Confirmations

- No official or legacy Qdrant collection was built, deleted,
  overwritten, or otherwise mutated — confirmed by identical
  before/after point counts after both runs (§5, §7, §8).
- No full 18,747-case evaluation was run.
- No code or test file was modified.
- No model download, LLM/API/Ollama/CrewAI call occurred.
- No accuracy, Wilson interval, McNemar test, latency-comparison
  table, cost table, manifest, or manuscript/reviewer-document edit
  was produced.
- No `eval/local_runs/` artefact was committed — only this report is
  staged for commit.
- No protected branch (`master`, `conference1-b2-evaluation`,
  `reviewer2-wip-snapshot-20260807`, `reviewer2-b2-integration-20260807`,
  or any prior `reviewer2-*` branch through
  `reviewer2-qdrant-stage-budget-enforcement-20260809`) was touched. No
  `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
  `git pull`, or force-push occurred. No PR was created.

## 12. Mandatory limitation

A passing result here is a small operational preflight only — it
confirms that Task 31's per-request timeout propagation and
stage-budget mechanism operate correctly and transparently on a
527-case risk-enriched selection under the specified configuration,
with zero retries/exhaustions actually observed. It is **not** full
controlled-benchmark evidence, not real Labour Force Survey validation,
and supports no ISIC, ISCED, SRE, reranking, accuracy, or real-world
performance claim. It does not prove the shared stage budget can
forcibly cancel an in-flight query (§3's limitation remains
unaddressed and unclaimed). A separate, independently authorized full
rerun would be required before any such evidence could exist, and this
task does not perform one.

Stopping here, per the task's own instruction, after pushing this
report. No preflight repair, rerun, or full benchmark was started.
