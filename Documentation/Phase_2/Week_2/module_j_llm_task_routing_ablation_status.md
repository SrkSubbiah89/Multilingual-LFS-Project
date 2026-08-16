# Module J — LLM Task-Routing Ablation (Prompt 7 of 9)

**Status: PARTIAL — blocked on local environment, not complete, not
abandoned.**

## What is done

### Phase 0 — verified, real, complete

All three of Prompt 7's "already real here" claims confirmed exactly as
stated, directly against the running repo:

1. `backend/llm/llm_client.py`'s routing confirmed: `TaskType.GENERAL` →
   Ollama (`OLLAMA_MODEL` env var, default `llama3.2`) with Claude 3.5
   Sonnet fallback if Ollama is unreachable; `TaskType.CRITICAL` → Claude
   3.5 Sonnet always.
2. The three real GENERAL-tier callers confirmed by direct grep at the
   exact claimed line numbers: `language_processor.py:374`,
   `conversation_manager.py:636`, `emotional_intelligence.py:374`.
3. `eval/ner_benchmark_runner.py` confirmed to exist (one commit,
   `9ac324d`). **No result files exist anywhere in this repository or on
   this machine's filesystem** — not `ner_results_aya_fire.json`, not
   under any other name. The earlier Llama-vs-Qwen-vs-Tiny-Aya comparison
   numbers exist only in this session's own prior conversation history,
   never persisted as a file artifact. This means Phase 1, once
   unblocked, needs fresh runs — there is nothing on disk to reuse.

### Real, additional code tracing done beyond what Prompt 7 assumed

Tracing `conversation_manager.py` directly (not assumed) found that its
"FSM-driven field extraction" characterization needs a correction:
`_extract_fields` — the FSM's primary field extraction — is **fully
regex/rule-based and never calls the LLM at all.** The only genuine,
LLM-touching, checkable code path in `ConversationManager` is
`_llm_extract_correction` (VALIDATING-state free-text correction
parsing, e.g. "actually I'm from India not Pakistan"), which:

- Is called via a **direct Ollama REST API call** (bypasses CrewAI
  overhead entirely), not the CrewAI `Agent`/`Crew` path.
- Has a hard **45-second** timeout (not 15s — see stale-comment fix
  below).
- Is **not** bypassed by `LFS_FAST_MODE` (unlike `process_message`'s
  main CrewAI response-generation path, which the `.env` file's real
  `LFS_FAST_MODE=true` setting means never actually runs the LLM in this
  project's real default configuration). This makes
  `_llm_extract_correction` the only representative, always-active
  GENERAL-tier task in this agent — and the one Phase 1 will test once
  unblocked. A test harness for it already exists, written and verified
  syntactically correct (not yet run to completion — see blocker below):
  `eval/general_tier_ablation_conversation_manager.py`, 10 gold-labeled
  correction cases + `--subset N` isolation-check support per Prompt 7's
  FSM-sensitivity requirement.

`emotional_intelligence.py`'s `analyze()` was also traced: its Stage-2
LLM call is unconditional (not `FAST_MODE`-gated), so it is a genuinely
representative task once Ollama is available again.

### Two stale-comment fixes — done, comment-only, verified

1. **The one Prompt 7 named**: `emotional_intelligence.py:374` said
   `# GPT-4o-mini, temp 0.3` next to a call that actually routes to
   Llama 3.2 via Ollama. Fixed.
2. **Two more found in the same file while tracing it** (not the exact
   line Prompt 7 named, but the identical stale-drift pattern): the
   module docstring's "LLM routing" section and the class docstring both
   also said "GPT-4o-mini" — fixed both, consistent with this project's
   standing discipline of not leaving known-wrong comments uncorrected
   once found, even mid-task.
3. **A separate, unrelated stale-comment bug found while investigating
   `_llm_extract_correction`** (not something Prompt 7 asked about, but
   directly adjacent to the code this task required reading closely):
   its own docstring said "a hard 15-second timeout" while the real,
   in-force constant is `_CORRECTION_TIMEOUT = 45` — a comment that had
   drifted after a real fix changed the timeout value (the surrounding
   code comment explains why: 15s was measured to always time out
   against the real ~35s worst-case call, so it was raised to 45s, but
   the docstring two lines up was never updated to match). Fixed.

`git diff` for both files confirmed comment/docstring-only — zero
production logic touched.

## What is NOT done — the actual blocker

**Phase 1 (the real model-swap ablation) could not be run.** Ollama's
own inference server binary is missing on this machine:

```
$ curl -X POST http://localhost:11434/api/chat ...
{"error":"error starting llama-server: llama-server binary not found
(checked: ...\\lib\\ollama\\llama-server.exe [7 candidate paths, all
checked, none found])..."}
```

Confirmed directly (not assumed): `lib/ollama/` contains only
`ggml-base.dll` — no `llama-server.exe` anywhere. This is **not**
model-specific — `llama3.2`, `llama3.2:latest`, and `llama3.2:1b` all
fail with the identical error, confirming the inference *server* itself
is broken, not any one model. `ollama list` still shows all 7 previously
pulled models present (`llama3.2:latest`, `qwen2.5:3b`,
`hf.co/CohereLabs/tiny-aya-*-GGUF`, `aya:latest`, `gemma3:4b`,
`llama3.2:1b`) — only the server executable is missing, so no re-pulling
is needed once the binary is restored.

Checked again after a reported reinstall attempt — **identical error,
unchanged.** Per this task's own ground rules ("don't keep attempting
Ollama fixes yourself beyond the health check" — repairing a local
Windows install is not something this session can do), no further
repair attempts were made from here.

**None of the following happened, and none of these numbers exist:**
per-agent/per-model accuracy or latency deltas, 3-run ranges, the
`ConversationManager` FSM-sensitivity check's actual result, or any
recommendation about which model to route. Nothing here is a
placeholder — these sections are simply absent because they were never
run.

## Status for tracking purposes

**Module J: PARTIAL — blocked on local environment.** Distinct from
"complete" (Phase 1 has produced zero real numbers) and from
"not started" (Phase 0 is genuinely done, two real code-comment bugs are
fixed, and a ready-to-run test harness exists for one of the three
agents). Resume from Phase 1 once Ollama's `llama-server.exe` is
restored — the health check to run first is exactly the one at the top
of this document.
