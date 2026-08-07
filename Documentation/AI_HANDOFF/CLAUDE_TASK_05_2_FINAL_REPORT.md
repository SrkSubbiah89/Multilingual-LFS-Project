# Task 05.2 Final Report — Embedding-Model Initialization Resilience

Produced in response to
`Documentation/AI_HANDOFF/PERPLEXITY_TO_CLAUDE_05_2_MODEL_INITIALIZATION_RESILIENCE.md`
(task ID `05.2-model-initialization-resilience`), executed on a new branch
per the task's explicit instruction. Continuation of Task 05.1, not an
integration or evaluation task.

## Branch / SHA

| | |
|---|---|
| Base branch | `reviewer2-isic-iscedf-hierarchical-resilience-20260808` |
| Base SHA (verified) | `cd991faef753bd210047d3f621288e7f7fc5ac37` |
| Working branch | `reviewer2-isic-iscedf-hierarchical-model-init-resilience-20260808` (new, created by this task) |
| Final SHA | commit created and pushed as part of this task's own closing step (see push confirmation below; this report is committed in the same commit) |

Start-state check passed exactly: `git fetch origin` confirmed
`origin/reviewer2-isic-iscedf-hierarchical-resilience-20260808` at
`cd991faef753bd210047d3f621288e7f7fc5ac37`; the local branch of that name
was already checked out at that exact SHA with a clean working tree
(`git status --short` produced no output) before
`git switch -c reviewer2-isic-iscedf-hierarchical-model-init-resilience-20260808`
was run.

## The gap this task fixes

Task 05.1 protected the Qdrant readiness check, the query-embedding call,
and the engine-search call. It missed one thing: `__init__()` still
constructed `SentenceTransformer(MODEL_NAME)` directly (unconditionally,
whenever no `embedder=` was injected and the store was `ready`) — **outside**
any of those protected try/except boundaries. If model initialization
itself failed (missing/corrupt local cache, out-of-memory, etc.), that
exception would propagate straight out of `__init__()`, through
`get_isic_hierarchical_store()` / `get_iscedf_hierarchical_store()`, and
into the classifier's `classify(..., method=...)` call — crashing the
request instead of reaching the already-tested fallback path.

## What was implemented

All changes confined to `backend/rag/standard_hierarchical_store.py`. No
classifier file needed changes.

1. **Lazy model construction**: `__init__()` no longer constructs
   `SentenceTransformer(MODEL_NAME)` at all. `self._embedder = embedder`
   (the injected value, or `None` if not supplied) is now the entire body
   of that step — down from a 5-line ready/injected/unavailable branch.
2. **Construction moved inside the protected boundary**: `_embed_query()`
   now does `if self._embedder is None: self._embedder =
   SentenceTransformer(MODEL_NAME)` as its first line, before building the
   query vector. Since `_embed_query()` is called inside the
   `try/except Exception` block Task 05.1 already added in `search()`, a
   construction failure is now caught by that exact same handler and
   reported as `"{standard} query embedding failed: {exc}"` — indistinguishable,
   from the caller's perspective, from a query-encoding failure. No new
   try/except was added; the existing one now covers a wider boundary.
3. **No model load when already unavailable**: unchanged in effect,
   simplified in mechanism — a store that is not `ready` returns from
   `search()` before ever calling `_embed_query()`, so lazy construction is
   never attempted (verified by a dedicated test that counts constructor
   calls).
4. **Dependency injection preserved**: an injected `embedder=` is stored
   as-is in `__init__()` and `_embed_query()`'s `if self._embedder is
   None` guard means it is never replaced by a real model — verified by a
   test that makes the real `SentenceTransformer` always raise and
   confirms the injected-fake positive path still succeeds untouched.
5. No new blanket catch was added anywhere; the protected scope is
   unchanged (readiness check, embedding — now inclusive of lazy model
   construction — and engine search).

## Changed files

**Modified (3 files, 192 insertions / 24 deletions total per `git diff --stat`):**

- `backend/rag/standard_hierarchical_store.py` — lazy model construction
  as described above; module docstring gained a new "Model initialization
  resilience (Task 05.2)" paragraph.
- `backend/tests/test_standard_hierarchical_store.py` — 5 new hermetic
  tests (see below); all pre-existing tests in this file (including all of
  Task 05.1's resilience tests) unchanged and still passing.
- `Documentation/Conference_I_Reviewer_2/ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md`
  — "Explicit fallback semantics" section updated to note that the
  embedding-failure fallback cause now explicitly includes model
  initialization failure; "Hermetic test coverage" section gained item 10
  describing these tests.

**New:**

- `Documentation/AI_HANDOFF/CLAUDE_TASK_05_2_FINAL_REPORT.md` (this file).

No other file was touched — `git status --short` before staging showed
exactly these 3 modified files plus this new report.

## Proof that model-construction failures now fall back

- `test_model_construction_never_attempted_in_init_no_embedder_injected` —
  swaps in a call-counting stand-in for `SentenceTransformer`; asserts the
  count is `0` immediately after `__init__()` and becomes `1` only after
  the first `search()` call.
- `test_model_construction_failure_yields_explicit_embedding_unavailable_result`
  — swaps in `_RaisingSentenceTransformer` (constructor always raises);
  `search()` on a `ready` store returns `ready=True`, `code=""`, a
  non-empty `unavailable_reason` containing `"embed"`, and confirms
  `client.calls == []` (no Qdrant query was ever reached).
- `test_isic_classifier_falls_back_on_model_construction_failure` — same
  failure injected through the real `ISICClassifier.classify(method=
  isic_hierarchical_retrieval)` entry point; asserts the fallback label
  (`isic_hierarchical_fallback_keyword`/`_llm`), `fallback_used=True`, a
  non-empty `fallback_reason`, and that the legacy pipeline genuinely ran
  (`section != ""`).
- `test_isced_classifier_falls_back_on_model_construction_failure_and_keeps_level`
  — same through `ISCEDClassifier.classify(method=
  iscedf_hierarchical_retrieval)`; additionally asserts `result.level == 6`,
  proving the independent `_score_level()` scorer is unaffected.
- `test_injected_embedder_bypasses_model_construction_even_if_it_would_fail`
  — with the real `SentenceTransformer` monkeypatched to always raise, the
  full positive ISIC parent-filtered path (`A -> 01 -> 011 -> 0111`) still
  succeeds end-to-end using only the injected `FakeEmbedder`, proving
  construction is never attempted when an embedder is supplied.

## Test commands and exact outputs

```
pytest backend/tests/test_standard_hierarchical_store.py backend/tests/test_isic_classifier.py backend/tests/test_isced_classifier.py backend/tests/test_hierarchy_nodes.py -q
→ 86 passed in 0.52s

pytest backend/tests eval/ -q
→ 1 failed, 1884 passed, 1 deselected, 1 warning in 299.17s (0:04:59)
  FAILED backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity
```

## Remaining failures

Exactly one, and it is **pre-existing**:
`backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity`
— the same CrewAI LLM mock-signature mismatch first identified in Task
04.1 and reconfirmed in Tasks 05 and 05.1's final reports, unrelated to
ISIC/ISCED-F and unmodified by this task. Test count went from 1879 passed
(Task 05.1's final state) to 1884 passed (+5, exactly the new tests added
this task), same single known failure, same 1 deselected — no regression,
no new failure. Not repaired, per the task's instruction.

## Confirmation: no live network, Qdrant, LLM, model download, dataset, or evaluation operation occurred

- Every new test uses `FakeQdrantClient`, `FakeEmbedder`,
  `_RaisingSentenceTransformer`, or a call-counting stand-in — no real
  `SentenceTransformer` model was ever loaded and no real `QdrantClient`
  was ever connected to a live host during this task's test runs.
- `SentenceTransformer` was monkeypatched at the
  `backend.rag.standard_hierarchical_store` module-attribute level in
  every test that exercises the lazy-construction path — the real
  `sentence-transformers` package's model-loading code was never invoked.
- No `python -m backend.rag.build_standard_hierarchical_collections
  --execute` was run. No Qdrant collection was created or populated.
- No Ollama/LLM inference call was made. The one test that exercises the
  real `ISICClassifier` patches `get_llm` with a `MagicMock`, matching the
  existing fixture pattern already used throughout `test_isic_classifier.py`.
- No `eval/run_eval.py`, `ablation_runner.py`, or benchmark script was
  invoked. No dataset file was read or written.

## Confirmation: protected branches unchanged

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| `reviewer2-wip-snapshot-20260807` | not touched |
| `reviewer2-b2-integration-20260807` | not touched |
| `reviewer2-isic-iscedf-hierarchical-rag-20260808` | not touched |
| `reviewer2-isic-iscedf-hierarchical-resilience-20260808` | not touched (this task branched from it, did not merge back into it) |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push was run at any point in this task.

## Confirmation: working tree

Clean before this task started (verified) and clean again immediately
before this report's commit — `git status --short` showed exactly the 3
modified files plus this new report file, matching the scope described
above with nothing else pending.

## What remains blocked / still unsafe to claim in the paper

Unchanged from Tasks 05 and 05.1 — this task only hardens error handling
one boundary further, it does not build, populate, or evaluate anything
new:

- No live Qdrant collection has been built or populated for ISIC/ISCED-F.
- No accuracy, latency, cost, or improvement measurement exists for either
  standard's hierarchical retrieval.
- No official-catalogue coverage claim is supported.
- It is now additionally true (and safe to state) that the hierarchical
  retrieval code degrades gracefully — never crashes a classification
  request — if Qdrant is unreachable, the embedding model fails to
  initialize, the query-encoding step fails, or the search itself fails;
  this remains an operational-robustness property only, not an accuracy or
  coverage claim, and should not be conflated with either in manuscript
  text.
