# Task 05.1 Final Report — Operational Resilience for ISIC/ISCED-F Hierarchical Retrieval

Produced in response to
`Documentation/AI_HANDOFF/PERPLEXITY_TO_CLAUDE_05_1_HIERARCHICAL_RETRIEVAL_RESILIENCE.md`
(task ID `05.1-hierarchical-retrieval-resilience`), executed on a new
branch per the task's explicit instruction.

## Branch / SHA

| | |
|---|---|
| Base branch | `reviewer2-isic-iscedf-hierarchical-rag-20260808` |
| Base SHA (verified) | `b03a60f6160824643335884f497bd32e53655c64` |
| Working branch | `reviewer2-isic-iscedf-hierarchical-resilience-20260808` (new, created by this task) |
| Final SHA | commit created and pushed as part of this task's own closing step (see the push confirmation below; this report is committed in the same commit) |

Start-state check passed exactly: `git fetch origin` confirmed
`origin/reviewer2-isic-iscedf-hierarchical-rag-20260808` at
`b03a60f6160824643335884f497bd32e53655c64`; the local branch of that name
was already checked out at that exact SHA with a clean working tree
(`git status --short` produced no output) before
`git switch -c reviewer2-isic-iscedf-hierarchical-resilience-20260808` was
run.

## The gap this task fixes

`StandardHierarchicalStore.__init__()` called
`self._client.get_collections()` with no exception handling. If Qdrant is
unreachable, times out, or errors, that call raises out of `__init__()`,
out of `get_isic_hierarchical_store()` / `get_iscedf_hierarchical_store()`,
and out of `ISICClassifier._classify_hierarchical()` /
`ISCEDClassifier._classify_hierarchical()` — before either classifier's
explicit fallback path ever runs. A `classify(..., method=
isic_hierarchical_retrieval)` call could crash the whole classification
request instead of degrading to the existing, already-tested legacy
fallback.

## What was implemented

All changes confined to `backend/rag/standard_hierarchical_store.py`. No
classifier file needed changes — `ISICClassifier`/`ISCEDClassifier`
already branch correctly on `ready`/`unavailable_reason`; the fix was to
make sure that state is always reached instead of an exception escaping
first.

1. **Readiness-check resilience** (`__init__`): the
   `self._client.get_collections()` call and the collection-membership
   computation are now wrapped in a `try/except Exception`. On success,
   behaviour is unchanged (`ready`/`_missing_collections`/reason text
   identical to before). On failure, `self.ready = False` and a new
   `self._unavailable_reason` is set to an operational-failure message
   ("Qdrant readiness check failed for {standard}: {exc}. ..."),
   logged via `_logger.warning`, mirroring the existing missing-collections
   log line's shape.
2. **Reason storage refactor**: `_unavailable_reason` is now computed once
   at construction time (for both the missing-collections case and the new
   operational-failure case) and `search()`'s `not self.ready` branch
   simply returns it, instead of re-deriving the missing-collections
   message inline. Behaviourally identical for the missing-collections
   case; new for the operational-failure case.
3. **No embedding-model load when already unavailable**: the
   `SentenceTransformer(MODEL_NAME)` construction (previously unconditional
   whenever no `embedder=` was injected) now only happens when
   `self.ready is True`. If readiness failed and no `embedder=` was
   injected, `self._embedder` is `None` — never constructed. Dependency
   injection is unaffected: an explicitly passed `embedder=` is always
   stored regardless of readiness (verified by a dedicated test), so
   hermetic tests keep working exactly as before.
4. **Embedding-call resilience** (`search()`): `self._embed_query(text)` is
   now wrapped in `try/except Exception`. On failure, an explicit
   `StandardHierarchyResult(code="", ..., ready=True,
   unavailable_reason="{standard} query embedding failed: {exc}")` is
   returned instead of letting the exception propagate.
5. **Engine-search-call resilience** (`search()`): `self._engine.search(...)`
   is now wrapped in `try/except Exception` with the same pattern
   (`"{standard} hierarchical engine search failed: {exc}"`). Note:
   `HierarchyBeamSearchEngine._query()` already catches per-query Qdrant
   exceptions internally and returns `[]` (verified by reading
   `backend/rag/hierarchy_engine.py`, unmodified by this task), so most
   Qdrant-level failures during an active search already degrade to
   `engine_result is None` (the pre-existing "no candidates" fallback).
   This new try/except is defense-in-depth for any other exception at
   that call boundary, per the task's explicit requirement; it does not
   change `hierarchy_engine.py` at all.
6. No score, candidate, or hierarchy path is fabricated on any of these
   paths — every new failure branch returns the same empty-field
   `StandardHierarchyResult` shape the pre-existing missing-collections/
   zero-hit/empty-text branches already used.
7. Exception handling is scoped narrowly to exactly these three external
   call boundaries (readiness check, embedding, engine search) — no other
   classifier or store logic is wrapped in a blanket catch.

## Changed files

**Modified (3 files, 298 insertions / 23 deletions total per `git diff --stat`):**

- `backend/rag/standard_hierarchical_store.py` — the resilience changes
  above, plus an expanded module docstring "Operational resilience"
  paragraph documenting the new state model.
- `backend/tests/test_standard_hierarchical_store.py` — 8 new hermetic
  tests (see below); all pre-existing tests in this file unchanged and
  still passing.
- `Documentation/Conference_I_Reviewer_2/ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md`
  — "Explicit fallback semantics" section updated to list the new
  operational failure causes (Qdrant unreachable, embedding failure,
  engine-search failure) alongside the pre-existing "collections missing"
  / "no candidates" causes; "Hermetic test coverage" section gained a new
  item (9) describing the resilience tests.

**New:**

- `Documentation/AI_HANDOFF/CLAUDE_TASK_05_1_FINAL_REPORT.md` (this file).

No other file was touched — `git status --short` before staging showed
exactly these 3 modified files plus this new report.

## Exact failure modes now handled

| Failure point | Before | After |
|---|---|---|
| `QdrantClient.get_collections()` raises (Qdrant unreachable/erroring) at store construction | Exception propagates out of `__init__`, out of the classifier's `classify()` call | `ready=False`, explicit `unavailable_reason`, classifier falls back with an explicit label |
| Embedding model load (`SentenceTransformer(MODEL_NAME)`) when store is already known unavailable | Always attempted regardless of readiness | Skipped entirely when `ready is False` and no `embedder=` injected |
| `self._embed_query(text)` raises during `search()` | Exception propagates | Explicit no-code `StandardHierarchyResult` with `unavailable_reason` naming the embedding failure |
| `self._engine.search(...)` raises during `search()` (any cause not already swallowed inside `hierarchy_engine.py`) | Exception propagates | Explicit no-code `StandardHierarchyResult` with `unavailable_reason` naming the search failure |

## Required hermetic tests added (`backend/tests/test_standard_hierarchical_store.py`)

1. `test_readiness_check_failure_yields_explicit_unavailable_result_and_never_queries`
   — a `_RaisingGetCollectionsClient` whose `get_collections()` raises
   yields `ready=False`, a non-empty reason, and `query_points()` is
   proven never called (it raises `AssertionError` if invoked).
2. `test_readiness_check_failure_never_loads_embedding_model` — confirms
   `store._embedder is None` when readiness failed and no embedder was
   injected.
3. `test_readiness_check_failure_still_honours_injected_embedder` —
   confirms DI still works on the failure path and the injected embedder
   is never called (short-circuited before `_embed_query`).
4. `test_isic_classifier_hierarchical_mode_falls_back_on_readiness_failure`
   — through the real `ISICClassifier.classify(method=
   isic_hierarchical_retrieval)` entry point (store injected via
   `monkeypatch` on `get_isic_hierarchical_store`), asserts the fallback
   label, `fallback_used=True`, non-empty `fallback_reason`, and that the
   legacy pipeline genuinely ran (`section != ""`).
5. `test_isced_classifier_hierarchical_mode_falls_back_on_readiness_failure_and_keeps_level`
   — same through `ISCEDClassifier.classify(method=
   iscedf_hierarchical_retrieval)`; additionally asserts
   `result.level == 6`, proving the independent `_score_level()` scorer is
   unaffected by the store failure.
6. `test_embedder_failure_does_not_fabricate_result` — collections ready,
   but `_RaisingEmbedder.encode()` raises; asserts an explicit,
   empty-code result and that no Qdrant query was ever issued
   (`client.calls == []`).
7. `test_engine_search_failure_does_not_fabricate_result` — collections
   and embedder both fine, but `store._engine.search` replaced with a
   function that raises; asserts an explicit, empty-code result.
8. `test_positive_path_parent_filtering_unaffected_by_resilience_changes`
   — the full ISIC `A -> 01 -> 011 -> 0111` positive path (pre-existing
   scenario) re-asserted end-to-end to prove none of the above changes
   altered the successful path.

Category "default classifier calls remain unchanged" is covered by the
pre-existing, unmodified `test_method_none_default_path_unchanged` tests
in `test_isic_classifier.py` / `test_isced_classifier.py`, which still
pass unchanged (see full-suite results below) — no new test was needed
for that guarantee since this task did not touch either classifier file.

## Test commands and exact outputs

```
pytest backend/tests/test_standard_hierarchical_store.py backend/tests/test_isic_classifier.py backend/tests/test_isced_classifier.py backend/tests/test_hierarchy_nodes.py -q
→ 81 passed in 0.37s

pytest backend/tests eval/ -q
→ 1 failed, 1879 passed, 1 deselected, 1 warning in 308.81s (0:05:08)
  FAILED backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity
```

## Remaining failures

Exactly one failure in the full suite:
`backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity`.

This is the **same pre-existing, unrelated failure** first identified in
Task 04.1's final report and reconfirmed in Task 05's final report (a
CrewAI LLM mock-signature mismatch — `clf.<locals>.<lambda>() got an
unexpected keyword argument 'temperature'` — in an ISCO-08 test, unrelated
to ISIC/ISCED-F, unrelated to this task's changes). Per the task's
instruction, it was reported, not repaired. Test count went from 1871
passed (Task 05's final state) to 1879 passed (+8, exactly the new
resilience tests added this task) with the same single known failure and
the same 1 deselected — no regression, no new failure.

## Confirmation: no live network, Qdrant, LLM, model download, dataset, or evaluation operation occurred

- Every new and existing test in `test_standard_hierarchical_store.py`
  uses `FakeQdrantClient`/`FakeEmbedder`/`_RaisingGetCollectionsClient`/
  `_RaisingEmbedder` — no real `QdrantClient` was ever connected to a live
  host, and no real `SentenceTransformer` model was ever loaded, during
  this task's test runs.
- No `python -m backend.rag.build_standard_hierarchical_collections
  --execute` was run. No Qdrant collection was created or populated.
- No Ollama/LLM inference call was made. The one test that exercises the
  real `ISICClassifier` (`test_isic_classifier_hierarchical_mode_falls_back_on_readiness_failure`)
  patches `get_llm` with a `MagicMock`, exactly matching the existing
  `clf` fixture pattern already used throughout `test_isic_classifier.py`.
- No `eval/run_eval.py`, `ablation_runner.py`, or benchmark script was
  invoked. No dataset file was read or written.
- `hierarchy_engine.py` was read (to confirm its existing internal
  exception handling) but not modified.

## Confirmation: protected branches unchanged

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| `reviewer2-wip-snapshot-20260807` | not touched |
| `reviewer2-b2-integration-20260807` | not touched |
| `reviewer2-isic-iscedf-hierarchical-rag-20260808` | not touched (this task branched from it, did not merge back into it) |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push was run at any point in this task.

## Confirmation: working tree

Clean before this task started (verified) and clean again immediately
before this report's commit — `git status --short` showed exactly the 3
modified files plus this new report file, matching the scope described
above with nothing else pending.

## What remains blocked / still unsafe to claim in the paper

Unchanged from Task 05's final report — this task only hardens error
handling, it does not build, populate, or evaluate anything new:

- No live Qdrant collection has been built or populated for ISIC/ISCED-F.
- No accuracy, latency, cost, or improvement measurement exists for either
  standard's hierarchical retrieval.
- No official-catalogue coverage claim is supported.
- It is now additionally true (and safe to state) that the hierarchical
  retrieval code degrades gracefully — never crashes a classification
  request — if Qdrant is unreachable, the embedding step fails, or the
  search itself fails; this is an operational-robustness property, not an
  accuracy or coverage claim, and should not be conflated with either in
  manuscript text.
