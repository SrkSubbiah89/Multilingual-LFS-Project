# Final Results Package

**Purpose:** every number below is read directly from a real, committed
artifact (CSV/JSON/doc), not retyped from memory of an earlier chat
report. Each section states the exact file path and the commit that
last touched it, so this document is itself traceable back to a real
source. Compiled 2026-08-16.

**Test suite re-run fresh for this compilation** (not cited from an
earlier report): `pytest backend/tests eval/ -q` → **2,363 passed, 0
failed, 1 deselected** (297.8s). Matches the last-known count from Step
8's own fresh verification exactly — no drift found.

> **Update, 2026-08-21**: this 2,363 figure is a historical, point-in-time
> record and is left unedited above. It has since changed to **2,282
> passed, 0 failed** after a post-Module-J cleanup pass removed
> `backend/agents/survey_orchestrator.py` (confirmed dead code, never
> imported by the live API) along with its dedicated 65-test file and the
> orchestrator-dependent tests in `test_hitl_and_e2e_extended.py` — a
> real, intentional test-count drop, not drift. Current, live counts are
> maintained in `README.md`'s own header banner.

---

## Table 6.1b — WISCO External Validation (Step 3)

### Canonical result: 18,747-case official-profile heldout, per-language top-1/top-3/κ/95% CI

**Source:** `Documentation/Conference_I_Reviewer_2/generated/wisco_tier1_topk_kappa.json`
(commit `cabcc75`). Offline analysis of Task 36's raw evaluator output,
verified SHA-256-identical to the values in
`OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`.

#### Flat retrieval

| Language | n | Top-1 | Top-1 95% CI | Top-3 | Top-3 95% CI | Cohen's κ (major group) | κ 95% CI (bootstrap) |
|---|---:|---:|---|---:|---|---:|---|
| ar | 3,762 | 14.62% | [13.53%, 15.79%] | 25.70% | [24.33%, 27.13%] | 0.297 | [0.279, 0.315] |
| en | 3,818 | 37.98% | [36.45%, 39.53%] | 54.16% | [52.58%, 55.74%] | 0.560 | [0.542, 0.577] |
| hi | 3,793 | 23.07% | [21.76%, 24.44%] | 35.64% | [34.14%, 37.18%] | 0.393 | [0.375, 0.410] |
| tl | 3,766 | 14.47% | [13.38%, 15.63%] | 23.74% | [22.41%, 25.12%] | 0.289 | [0.270, 0.306] |
| ur | 3,608 | 15.33% | [14.19%, 16.54%] | 25.42% | [24.02%, 26.86%] | 0.299 | [0.280, 0.317] |
| **Aggregate** | **18,747** | **21.19%** | **[20.61%, 21.78%]** | **33.06%** | **[32.39%, 33.74%]** | **0.368** | **[0.360, 0.377]** |

#### Strict hierarchical retrieval (4-stage)

| Language | n | Top-1 | Top-1 95% CI | Top-3 | Top-3 95% CI | Cohen's κ (major group) | κ 95% CI (bootstrap) |
|---|---:|---:|---|---:|---|---:|---|
| ar | 3,762 | 9.78% | [8.87%, 10.77%] | 15.92% | [14.79%, 17.13%] | 0.357 | [0.341, 0.374] |
| en | 3,818 | 22.52% | [21.23%, 23.88%] | 30.62% | [29.18%, 32.10%] | 0.502 | [0.484, 0.520] |
| hi | 3,793 | 8.09% | [7.27%, 9.00%] | 12.02% | [11.03%, 13.10%] | 0.207 | [0.192, 0.223] |
| tl | 3,766 | 5.31% | [4.64%, 6.07%] | 8.68% | [7.83%, 9.62%] | 0.166 | [0.151, 0.180] |
| ur | 3,608 | 5.71% | [5.00%, 6.51%] | 9.37% | [8.46%, 10.36%] | 0.180 | [0.164, 0.197] |
| **Aggregate** | **18,747** | **10.35%** | **[9.93%, 10.80%]** | **15.41%** | **[14.90%, 15.93%]** | **0.282** | **[0.274, 0.290]** |

Hierarchical is 10.84 percentage points worse than flat in aggregate
(McNemar p ≈ 1.86×10⁻³⁰¹, per the canonical result doc) — this remains
the authoritative, disclosed negative finding. Never describe
hierarchical as outperforming flat.

### Separate result: legacy-profile 500-case subsample, 3-system comparison

**⚠️ This is a DIFFERENT dataset/profile from the canonical result above
— do not merge the two.** Legacy (non-official-ILO-2021) catalogue
profile, WISCO's pre-existing deterministic 500-case subset, first run
in Step 3.

**Source:** `backend/evaluation/results_wisco.csv` (commit `cabcc75`), read verbatim:

| System | Top-1 | Top-3 | Cohen's κ | HITL rate | Mean latency | P95 latency | P99 latency | n | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 1.6% | 2.8% | 0.0347 | 87.8% | 0.08 ms | 0.19 ms | 0.27 ms | 500 | 0 |
| Flat | 8.6% | 13.8% | 0.307 | 0.0% | 30.39 ms | 47.59 ms | 72.85 ms | 500 | 0 |
| Hierarchical | 11.0% | 15.6% | 0.264 | 0.0% | 756.96 ms | 990.07 ms | 1,352.70 ms | 500 | 0 |

Note: these accuracy percentages are on the **legacy catalogue profile**
and a different 500-case subsample — they are numerically much lower
than a naive comparison to the canonical 18,747-case numbers above would
suggest is consistent, precisely because they are not the same
measurement. Reported as-is, not reconciled.

---

## Table 6.3 — SRE Validation (Steps 5 / 5.5)

### Aggregate by severity, 61-case set

**Source:** `backend/evaluation/sre_expanded_validation.json` (commit `f44cd79`).
`n_mismatches: 0`, `all_predictions_matched: true` — every case's severity was
predicted from the engine's real crosswalk tables *before* running, then
confirmed against actual engine output.

| Severity | n | Mean coherence score |
|---|---:|---:|
| COHERENT | 20 | 1.000 |
| LOW | 12 | 0.550 |
| MODERATE | 14 | 0.754 |
| HIGH | 15 | 0.510 |

(Mean score is not monotonic across severity bands — MODERATE's mean
exceeds LOW's — because severity is driven by the *worse* of the
independent ISCO↔ISIC and ISCO↔ISCED checks, while the coherence score
is a weighted sum of both; reported exactly as computed, not smoothed.)

### Real HITL enforcement verification (Step 5.5)

**Source:** `backend/evaluation/sre_hitl_enforcement_endpoint_verification.json`
(commit `0eba4d4`). Drives the real live `/survey/sessions/{id}/message`
endpoint (not just the engine directly) for all 61 cases, 3 runs.

| Run | n HIGH | HIGH escalation rate | n non-HIGH | non-HIGH escalation rate | All correct | Non-200 responses |
|---|---:|---:|---:|---:|---|---:|
| 1 | 15 | 100.0% | 46 | 0.0% | true | 0 |
| 2 | 15 | 100.0% | 46 | 0.0% | true | 0 |
| 3 | 15 | 100.0% | 46 | 0.0% | true | 0 |

`stable_across_runs: true` — byte-identical across all 3 runs.

---

## Module I Efficiency Table (Step 6)

**Sources:** `Documentation/Phase_2/Week_2/week2_brief.md` §4 and
`Documentation/Phase_2/Week_2/module_i_computational_efficiency_report.md`,
plus `backend/evaluation/qdrant_collection_memory_audit.json` and
`backend/evaluation/embedding_timing_benchmark.json` (all commit `b73dbca`).
Real-vs-estimated labels preserved exactly as Step 6 left them.

| Component | Value | Status |
|---|---|---|
| Hierarchical RAG stage, end-to-end latency (mean / median) | 133.9 ms / 137.4 ms | **Real** — measured across the real 18,747-case Task 36 run |
| Flat RAG stage, end-to-end latency (mean) | 31.1 ms | **Real** — same run |
| Embedding compute per call (batch=1, mean / P95) | 19.1 ms / 25.6 ms | **Real** — 120 calls, this sandbox's CPU, exact production call pattern |
| Qdrant on-disk size, 4 official-profile hierarchical collections (total) | ~4.3 MB | **Real** — `du` inside the Qdrant container |
| Qdrant whole-process RSS (all 10 ISCO collections) | 94.9 MB | **Real** — Qdrant `/metrics` |
| Vector dimension | 384 | **Real** — live collection config |
| LLM re-ranking trigger rate | 30–70% | **Estimate, still unmeasured** — Task 36 excluded the LLM tier; Task 43's attempt was an invalidated silent-failure artifact |
| Claude 3.5 Sonnet cost per re-ranking call | ~$0.006 | **Estimate, still unmeasured** — no valid real Claude-reranked run exists anywhere in this repo |
| Claude 3.5 Sonnet latency per call | ~1.5–2.5s | **Estimate, still unmeasured** — same reason |
| "$84 cost-per-interview target" | — | **Not found** — searched the full repo, no such figure exists anywhere; not extended |
| Load test, 10 concurrent users (3-run range) | 100% success, P95 2,412–2,458 ms | **Real** |
| Load test breaking point | Between 42 users (100% success, P95 2,715ms) and **50 users** (66.0% success, P95 5,382ms) | **Real**, single-instance only — multi-instance horizontal scaling explicitly not tested |
| Per-turn ~2.1s latency floor (all concurrency levels) | Root-caused to `_OLLAMA_HEALTH_TIMEOUT=2s` firing because Ollama is unreachable in this sandbox | **Real finding, sandbox-specific** — not a RAG-pipeline cost, would not exist with Ollama reachable |

---

## Corrected System Facts

Re-verified directly against current source for this compilation (not
carried over from any earlier report):

| Fact | Value | Source |
|---|---|---|
| ISCO-08 structure | 10 major / 43 sub-major / 130 minor / 436 unit groups | `backend/rag/load_full_isco.py` (`_MAJOR`, `_SUBMAJOR`, `_MINOR`, `_UNIT`), imported and counted directly |
| ISIC Rev.4 coverage | 134 unique classes / 419 official | `backend/agents/isic_classifier.py` (`_ISIC_DATA`), deduplicated by `class_code` and counted directly |
| ISCED-F 2013 coverage | 63 detailed fields / ~80 official | `backend/agents/isced_classifier.py` (`_ISCED_FIELDS`), counted directly |
| Embedding model | `intfloat/multilingual-e5-small`, 384-dim | `backend/rag/vector_store.py` line 54-55 (`MODEL_NAME`, `VECTOR_DIM`) |

---

## Module J — LLM Task-Routing Ablation (Step 7)

**Source**: `Documentation/Phase_2/Week_2/module_j_llm_task_routing_ablation_status.md`
and its cited raw artifacts (all committed):
`backend/evaluation/ner_{llama,qwen,gemma}_run{1,2,3}.json`,
`backend/evaluation/conversation_manager_warmed_comparison.json`,
`backend/evaluation/emotional_intelligence_{llama,qwen}_warmed.json`.

**Methodology note**: every comparison below explicitly warms each model
with one throwaway call before timing anything, reported separately as
`cold_start_s` — real measured cold start was 78.0s vs. 4.7-6.1s warm
(`conversation_manager.py`'s `_CORRECTION_TIMEOUT` comment). Two
environmental artifacts were found and excluded from these results: a
laptop idle-sleep timeout that inflated two `EmotionalIntelligence` calls
to 9.9h/62min wall-clock (root-caused, sleep disabled, run discarded and
redone) — those two data points appear nowhere below.

### LanguageProcessor (NER), 9 runs — llama3.2 vs. qwen2.5:3b vs. gemma3:4b

| Model | Run 1 F1 | Run 2 F1 | Run 3 F1 | Range | Mean |
|---|---:|---:|---:|---|---:|
| llama3.2 | 0.397 | 0.483 | 0.384 | [0.384, 0.483] | 0.421 |
| qwen2.5:3b | 0.519 | 0.533 | 0.474 | [0.474, 0.533] | **0.509** |
| gemma3:4b | 0.394 | 0.465 | 0.432 | [0.394, 0.465] | 0.430 |

qwen2.5:3b wins clearly, driven by real Arabic-script strength (F1
0.67-0.82 vs. llama3.2's 0.29-0.33). English is stable across every
model/run (F1 0.76-0.81, all 9 runs).

### ConversationManager, warmed, FSM-sensitivity checked

FSM-sensitivity check (3 cases, warm): **PASS for both models** (3/3
each).

| Model | Cold start | Run 1 | Run 2 | Run 3 |
|---|---:|---|---|---|
| llama3.2 | 103.14s | 10/10, mean 54.87s | 10/10, mean 13.00s | 10/10, mean 10.23s |
| qwen2.5:3b | 72.58s | 10/10, mean 49.01s | 10/10, mean 6.49s | 10/10, mean 9.61s |

Perfect accuracy for both models (30/30 each) — no accuracy
differentiator. Steady-state latency (Runs 2-3 average): llama3.2
≈11.6s, qwen2.5:3b ≈8.05s.

### EmotionalIntelligence, warmed

| Model | Cold start | Run 1 | Run 2 | Run 3 |
|---|---:|---|---|---|
| llama3.2 | 76.65s | 9/10, mean 113.92s | 8/10, mean 111.13s | 9/10, mean 124.15s |
| qwen2.5:3b | 51.60s | 9/10, mean 103.06s | 9/10, mean 109.52s | 9/10, mean 117.28s |

qwen2.5:3b slightly more consistent (9/10 every run) at comparable or
lower latency.

### Recommendation (input for a human decision — `llm_client.py`'s default routing was not changed)

qwen2.5:3b matched or outperformed llama3.2 on every real comparison in
this module, most clearly on LanguageProcessor's Arabic handling.
gemma3:4b was only measured for LanguageProcessor, not the other two
agents, so it is not fully evaluated against qwen2.5:3b.

**Module J status: COMPLETE** — all 3 agents have real, warmed, 3-run
comparisons with no outstanding confounds.

---

## Gaps and Discrepancies Found

- No gaps: every artifact referenced by this task's instructions exists
  and contains real, usable data.
- No cross-source numeric discrepancies found — `results_wisco.csv`
  (legacy 500-case subsample) and `wisco_tier1_topk_kappa.json`
  (canonical 18,747-case) describe genuinely different measurements by
  design (different catalogue profile, different sample), not
  conflicting measurements of the same thing, so they are reported side
  by side above rather than reconciled.
- Test suite: current fresh run (2,363 passed, 0 failed) matches Step
  8's most recent number exactly — no drift found in this pass.
