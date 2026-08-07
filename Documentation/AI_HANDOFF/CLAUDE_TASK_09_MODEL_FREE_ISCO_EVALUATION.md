# Task 09 — Make Retrieval-Only ISCO Evaluation Truly Model-Free

## Purpose

Correct a preflight inconsistency before any WISCO benchmark is run. `eval/run_eval.py --use-llm-reranker off` currently still requires `--reranker-model` for `hierarchical`/`flat`, constructs `ISCOClassifier` in a way that initializes an LLM, and unconditionally constructs `ISICClassifier`, `ISCEDClassifier`, and `SemanticRelationEngine` even when every CSV row has blank `industry_text` and `education_text`.

This task must make the planned WISCO ISCO-only, no-reranker evaluation genuinely retrieval-only. It is a code-and-test task only. Do not run any benchmark or evaluation.

## Starting point

1. Fetch and verify the exact remote tip before making any change:
   - Base branch: `reviewer2-pre-evaluation-baseline-20260808`
   - Required base SHA: `c3d39d89f1b164f4fc5595c319e90546a2c17716`
2. Require a clean working tree after checkout.
3. Create a new branch from that exact base:
   - `reviewer2-model-free-isco-evaluation-20260808`
4. Do not merge, rebase, reset, clean, stash, pull, force-push, create a PR, or change any protected or prior-task branch.

## Scope

### A. Add an opt-in, production-preserving ISCO LLM initialization control

In `backend/agents/isco_classifier.py`:

1. Add a final, additive constructor parameter `enable_llm: bool = True` to `ISCOClassifier.__init__`.
2. Preserve every existing default caller exactly: `enable_llm=True` must retain the current strict pinned-model behavior, existing graceful fallback behavior, agent initialization, and output behavior.
3. When `enable_llm=False`:
   - Initialize the requested retrieval store(s) exactly as before.
   - Do not call `get_llm_strict`, `get_llm`, `_build_reranker_agent`, or construct an LLM/agent.
   - Leave `_agent_available=False`.
   - Set `reranker_model_resolved` to a clear non-empty, non-model disclosure such as `"none (reranking disabled)"`.
   - Return through the existing semantic retrieval path when `classify(..., use_llm=False)` is used. Do not add a second retrieval algorithm, fabricate a reranking trace, or silently change prediction logic.
4. If no vector store is available, retain the existing early-return behavior and never initialize an LLM.

### B. Wire the evaluator correctly

In `eval/run_eval.py`:

1. Change `build_system(...)` to receive and respect whether LLM reranking is enabled.
   - For `hierarchical` / `flat` with reranking on: retain the current pinned-model requirement and strict model validation.
   - For `hierarchical` / `flat` with `--use-llm-reranker off`: construct `ISCOClassifier(..., enable_llm=False, reranker_model=None)`.
   - For `bm25`: preserve existing behavior.
2. Change the CLI validation so `--reranker-model` is required only when `--system` is `hierarchical` or `flat` **and** `--use-llm-reranker on`, outside `--dry-run`. It must be valid to omit `--reranker-model` when reranking is off.
3. Update the CLI help text and module documentation so they explicitly distinguish reranking-on requirements from retrieval-only runs.
4. Ensure the configuration hash records the true behavior. A retrieval-only run must not look as though it used an LLM model. Preserve hash behavior for default reranking-on runs.
5. Before constructing ISIC, ISCED, or SRE components, inspect the loaded rows. Construct all three only when at least one selected row has both non-blank `industry_text` and `education_text`; otherwise set them to `None` and do not initialize them.
6. Make the smallest safe type/guard adjustment to `run_one_case(...)` so the existing behavior remains unchanged for rows that do contain both texts, including the Task 05.1 guarantee: `--sre off` disables only coherence analysis, not ISIC/ISCED classification. For a WISCO-style input with no paired industry/education text, retain the existing `sre_status="not_applicable"` semantics and leave ISIC/ISCED prediction fields blank.
7. Do not modify the documented semantics of `--dry-run`.

### C. Tests

Add or update focused hermetic tests. Do not weaken existing tests or use skips/xfail.

At minimum, prove:

1. `ISCOClassifier(enable_llm=False)` does not call `get_llm_strict`, `get_llm`, or `_build_reranker_agent`, but still makes a semantic-only classification when supplied with a fake ready retrieval store.
2. Existing/default `ISCOClassifier()` behavior remains unchanged, including the pinned strict model path when `enable_llm=True`.
3. `build_system("hierarchical"/"flat", use_llm_reranker=False)` passes `enable_llm=False` and no reranker model; reranking-on retains the model pin.
4. CLI parsing accepts `--system hierarchical --use-llm-reranker off` and `--system flat --use-llm-reranker off` without `--reranker-model`; it still rejects reranking-on hierarchical/flat runs without a pin.
5. With WISCO-style rows lacking paired `industry_text` and `education_text`, `ISICClassifier`, `ISCEDClassifier`, and `SemanticRelationEngine` are not constructed at all. The test must use mocks that would fail if constructed.
6. With paired industry/education text, all existing ISIC/ISCED/SRE behavior remains available; with `--sre off`, ISIC/ISCED still run and only SRE is skipped.
7. A no-reranker run produces explicit retrieval-only metadata, never a fake model identity or an LLM-reranker trace.

All tests must use fakes/mocks. No Qdrant, SentenceTransformer download, Ollama, paid LLM/API call, benchmark run, WISCO dataset build, or real dataset access is permitted.

### D. Documentation

Update only documentation directly affected by this correction, including the WISCO run plan if its exact command or statement about no-LLM execution is now inaccurate. State that a retrieval-only run is not a reranking comparison and that it supports ISCO-08 only.

Create:

- `Documentation/AI_HANDOFF/CLAUDE_TASK_09_FINAL_REPORT.md`

The report must include: starting and final SHAs; branch name; exact changed files; a before/after explanation of the preflight defect; exact test commands and outputs; confirmation that no live Qdrant/model/LLM/benchmark/dataset operation occurred; protected branch SHAs; working-tree status; and remaining manuscript limits.

## Required verification

Run the smallest meaningful focused suites first, then:

```bash
pytest backend/tests/test_isco_classifier.py backend/tests/test_isco_classifier_extended.py eval/test_run_eval_b2.py eval/test_sre_isic_isced_coupling_fix.py -q
pytest backend/tests eval/ -q
```

If a named file does not exist on this branch, use the closest existing focused test file and explain the substitution. Do not claim zero regressions unless the full suite is actually green.

## Git boundary

Commit only the scoped code, tests, documentation, and final report to the new branch. Push only `reviewer2-model-free-isco-evaluation-20260808`. Do not create a PR and do not merge into the canonical branch.

## Stop condition

Stop after the pushed branch, final report, and clean-tree verification. Do not start the WISCO benchmark. The next task will independently review and integrate this correction before any measured evaluation occurs.
