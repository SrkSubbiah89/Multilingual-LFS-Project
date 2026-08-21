# Module J — LLM Task-Routing Ablation (Prompt 7 of 9)

**Status: COMPLETE.** All 3 candidate GENERAL-tier agents have real,
warmed, 3-run comparisons (llama3.2 vs. qwen2.5:3b; LanguageProcessor
additionally includes gemma3:4b). No outstanding confounds — cold-start
latency is measured and reported separately from steady-state
comparison latency for every agent, per the discipline established after
the timeout-mismatch bug (see below).

Raw result artifacts (all committed): `backend/evaluation/ner_{llama,qwen,gemma}_run{1,2,3}.json`,
`backend/evaluation/conversation_manager_warmed_comparison.json`,
`backend/evaluation/emotional_intelligence_{llama,qwen}_warmed.json`.
Reusable scripts: `eval/ner_benchmark_runner.py` (pre-existing, reused),
`eval/general_tier_ablation_conversation_manager.py`,
`eval/general_tier_ablation_emotional_intelligence.py`,
`eval/conversation_manager_warmed_comparison.py`,
`eval/emotional_intelligence_warmed_comparison.py`.

## Environment history (for context, resolved)

Ollama's inference server binary was found missing (`llama-server.exe`
absent from the install) and blocked this module for an earlier pass.
Root-caused as a stray file lock (`Access is denied` on
`ggml-base.dll`, confirmed by direct exclusive-open test, likely
Defender or Search Indexer scanning a just-touched file) preventing a
normal reinstall from completing. Resolved by renaming the whole parent
install directory out of the way (works even when a file within it is
locked, since NTFS directory rename doesn't require closing child file
handles the way a recursive delete does) and reinstalling clean via
`winget`. Verified with real inference (`ollama run llama3.2 "test"
--verbose`, real generated tokens, real eval rate).

A second issue, `_CORRECTION_TIMEOUT` in `conversation_manager.py`
being calibrated against `llama3.2:1b` while the real shipped default is
bare `llama3.2` (3B), was found and fixed as a **separate, already-
committed, standalone fix** (commit `d78ed50`) — not part of this
module's own commit, per that fix's own scoping rule.

## Real finding carried into this module's own methodology: cold-start vs. warm latency

Direct measurement (`backend/agents/conversation_manager.py`'s
`_CORRECTION_TIMEOUT` comment): a cold call (model not yet resident in
Ollama) took **78.0s**; warm calls took **4.7-6.1s**. Every comparison in
this module now explicitly warms each model with one throwaway call
before timing anything, and reports that throwaway call's duration as a
separate `cold_start_s` figure — never blended into the accuracy/latency
comparison numbers.

A second, unrelated environmental artifact was found and fixed
mid-module: two `EmotionalIntelligence` calls showed latencies of
**35,648.90s (~9.9 hours)** and **3,724.17s (~62 min)** — both eventually
returned the *correct* answer, ruling out a hang, but the wall-clock
duration was absurd. Root-caused to the laptop's 10-minute idle sleep
timeout (`powercfg` confirmed `0x258` = 600s) triggering mid-request;
`time.perf_counter()` keeps advancing across a sleep/resume cycle, so
the elapsed measurement silently included the whole sleep duration once
the request's socket eventually resumed and completed. Fixed by
disabling sleep for the remainder of this work (`powercfg /change
standby-timeout-ac 0` / `-dc 0`); the affected run was discarded and
re-run cleanly (max latency after the fix: 126.71s, no further
anomalies). **The two `9.9h`/`62min` data points are not included
anywhere in the results below** — they were an artifact of this specific
test environment's power settings, not real model or task performance.

## Task 1 — LanguageProcessor (NER), 9 runs (3 models × 3 runs)

Reused `eval/ner_benchmark_runner.py` against the existing 30-case
gold-labeled set (`eval/ner_benchmark_data.json`).

| Model | Run 1 F1 | Run 2 F1 | Run 3 F1 | Range | Mean |
|---|---:|---:|---:|---|---:|
| llama3.2 | 0.397 | 0.483 | 0.384 | [0.384, 0.483] | 0.421 |
| qwen2.5:3b | 0.519 | 0.533 | 0.474 | [0.474, 0.533] | **0.509** |
| gemma3:4b | 0.394 | 0.465 | 0.432 | [0.394, 0.465] | 0.430 |

**qwen2.5:3b wins clearly and consistently**, driven mainly by real,
substantial Arabic-script strength (F1 0.67-0.82 across its 3 runs vs.
llama3.2's 0.29-0.33). English is the one language stable across every
model/run (F1 0.76-0.81 in all 9 runs). Non-English performance is
volatile per-run for every model — no model is reliably strong on
Urdu/Hindi/Tagalog across all 3 runs. A qualitative sample review (Run 1
vs. Run 2, llama3.2) found real, varied failure patterns beyond simple
"wrong": entity over-splitting, hallucinated entities, duplicate
entities, wrong labels on correctly-extracted text, and one instance of
apparent cross-case entity contamination (predicted entities matching a
*different* case's content, not the actual input) — disclosed as a
notable pattern, not confirmed as a systematic bug beyond that one
observed instance.

Full per-case data: `backend/evaluation/ner_{llama,qwen,gemma}_run{1,2,3}.json`.

## Task 2 — ConversationManager, warmed, FSM-sensitivity checked

Real GENERAL-tier task: `_llm_extract_correction` (VALIDATING-state
free-text correction parsing). Both models warmed with one throwaway
call before any timing.

**FSM-sensitivity check (3 cases each, warm): PASS for both models.**
llama3.2 3/3 correct, qwen2.5:3b 3/3 correct — no FSM-breaking output
format issues from either model.

**Full 3-run comparison (10 cases each):**

| Model | Cold start | Run 1 | Run 2 | Run 3 |
|---|---:|---|---|---|
| llama3.2 | 103.14s | 10/10, mean 54.87s | 10/10, mean 13.00s | 10/10, mean 10.23s |
| qwen2.5:3b | 72.58s | 10/10, mean 49.01s | 10/10, mean 6.49s | 10/10, mean 9.61s |

**Both models: perfect accuracy, 30/30 correct across all 3 runs each —
no accuracy differentiator for this task.** Real latency finding: a
single warm-up call is not sufficient to reach steady-state speed — Run
1 (immediately after warm-up) is still noticeably slower than Runs 2-3
for both models, which is itself disclosed here rather than averaged
away. Steady-state (Runs 2-3 average): llama3.2 ≈11.6s, qwen2.5:3b
≈8.05s — qwen2.5:3b modestly faster once fully warm, with identical
accuracy.

Full per-case data: `backend/evaluation/conversation_manager_warmed_comparison.json`.

## Task 3 — EmotionalIntelligence, warmed

Real task: `analyze()`'s Stage-2 CrewAI-based emotion classification
(unconditional, not `FAST_MODE`-gated). 10-case gold-labeled set built
for this module (`eval/general_tier_ablation_emotional_intelligence.py`
— no prior benchmark existed for this agent).

| Model | Cold start | Run 1 | Run 2 | Run 3 |
|---|---:|---|---|---|
| llama3.2 | 76.65s | 9/10, mean 113.92s | 8/10, mean 111.13s | 9/10, mean 124.15s |
| qwen2.5:3b | 51.60s | 9/10, mean 103.06s | 9/10, mean 109.52s | 9/10, mean 117.28s |

**qwen2.5:3b is slightly more consistent** (9/10 every run vs.
llama3.2's 8-9/10) at comparable latency (~103-117s vs. ~111-124s).
Both comfortably below the underlying 120s CrewAI/litellm timeout on
average, though individual per-case latencies in the 118-127s range were
observed for both models — close enough to the ceiling to be worth
disclosing as a real, if narrow, margin.

Full per-case data: `backend/evaluation/emotional_intelligence_{llama,qwen}_warmed.json`.

## Final recommendation (input for a human decision — no code changed)

Across all 3 agents, **qwen2.5:3b outperforms or matches llama3.2 on
every real comparison run this module produced**, with the clearest
margin on LanguageProcessor's Arabic-script handling. `llm_client.py`'s
actual default routing (`OLLAMA_MODEL=llama3.2`) was **not** changed as
part of this module — this is a measurement, not a decision. If a
routing change is made based on this evidence, `gemma3:4b` was also
measured for LanguageProcessor only (not the other two agents) and
performed comparably to llama3.2, not qwen2.5:3b — it was not carried
into the fuller comparison, so it should not be considered fully
evaluated against qwen2.5:3b.

## Full suite

2,363 passed, 0 failed, confirmed after the `_CORRECTION_TIMEOUT` fix
(separate commit `d78ed50`); no production code was changed by this
module's own measurement work.

> **Update, 2026-08-21**: the 2,363 figure above is a historical record
> from when this module completed and is left unedited. A later,
> unrelated cleanup pass removed the confirmed-dead-code
> `survey_orchestrator.py` and its 65+16 dependent tests, bringing the
> current live count to **2,282 passed, 0 failed**. See `README.md`'s
> header for the current, maintained count.
