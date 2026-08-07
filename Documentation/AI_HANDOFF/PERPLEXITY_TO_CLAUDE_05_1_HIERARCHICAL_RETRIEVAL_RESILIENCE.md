# Claude Task 05.1: Make ISIC / ISCED-F hierarchical retrieval operationally fail-closed

## Purpose

Fix one reliability gap found in the independent Task 05 audit before any integration: `StandardHierarchicalStore.__init__()` calls `QdrantClient.get_collections()` without handling a connection/readiness failure. If Qdrant is not running or cannot be reached, `ISICClassifier.classify(..., method=isic_hierarchical_retrieval)` or `ISCEDClassifier.classify(..., method=iscedf_hierarchical_retrieval)` can raise before its explicit fallback path runs.

This task must make that condition return the existing explicitly labelled fallback result. It must not weaken normal errors elsewhere or silently claim a hierarchical result.

## Start state and branch discipline

1. Fetch origin and verify that `reviewer2-isic-iscedf-hierarchical-rag-20260808` is at `b03a60f6160824643335884f497bd32e53655c64`.
2. Verify it is clean.
3. Create a new branch from exactly that SHA:
   `reviewer2-isic-iscedf-hierarchical-resilience-20260808`
4. Do not merge, rebase, reset, clean, stash, pull, or force-push.
5. Do not touch the protected branches: `master`, `conference1-b2-evaluation`, `reviewer2-wip-snapshot-20260807`, or `reviewer2-b2-integration-20260807`.

## Required implementation

### Store availability handling

In `backend/rag/standard_hierarchical_store.py`:

- Make the one-time Qdrant collection-readiness check resilient to an unreachable or failing Qdrant service.
- Restrict exception handling to the external readiness, embedding, and search boundary. Do not wrap unrelated classifier logic in a blanket catch.
- Preserve structured, explicit state. A Qdrant readiness failure must lead `search()` to return `StandardHierarchyResult` with no code, a non-empty `unavailable_reason` that identifies the operational unavailability, and a state that makes the caller choose the existing explicit fallback path.
- Avoid creating/loading `SentenceTransformer` when the store is already known to be unavailable because Qdrant readiness failed or required collections are missing. Keep dependency injection with fake clients/embedders working.
- Ensure an embedding-load or engine-search operational failure also produces a no-code result with a non-empty explicit reason rather than crashing the classification request. Do not fabricate scores, candidates, or hierarchy paths.
- Keep the current meanings of `ready`, `unavailable_reason`, missing-collection results, empty input, and zero-hit results clear and documented. Update the implementation documentation if the state model needs a precise wording change.

### Classifier behavior

- With `method=isic_hierarchical_retrieval`, unavailable Qdrant, missing collections, embedding failure, or no usable hierarchical result must return the existing legacy result with `isic_hierarchical_fallback_keyword` or `isic_hierarchical_fallback_llm`, `fallback_used=True`, and a non-empty reason.
- With `method=iscedf_hierarchical_retrieval`, the same conditions must return `iscedf_hierarchical_fallback_keyword`, `fallback_used=True`, and a non-empty reason. ISCED 2011 attainment level must still be independently classified.
- Default `classify(text)` behavior must remain unchanged.

## Required hermetic tests

Use fake clients and fake embedders only. No real Qdrant, model download, Ollama, LLM, network, evaluation, benchmark, or dataset operation.

Add focused tests that prove:

1. A fake client whose `get_collections()` raises yields an explicit unavailable result and never runs a hierarchy query.
2. ISIC hierarchical mode falls back with the correct explicit label and metadata when the readiness check fails.
3. ISCED-F hierarchical mode falls back with the correct explicit label and metadata when the readiness check fails, while retaining independently classified ISCED 2011 level output.
4. An embedder failure and an engine/search failure do not fabricate results and are explicitly surfaced to the fallback path.
5. Existing positive-path parent filtering still passes unchanged.
6. Default classifier calls remain unchanged.

Run at minimum:

```bash
pytest backend/tests/test_standard_hierarchical_store.py backend/tests/test_isic_classifier.py backend/tests/test_isced_classifier.py backend/tests/test_hierarchy_nodes.py -q
pytest backend/tests eval/ -q
```

Report the known historical `test_llm_used_for_low_similarity` mock-signature failure separately if it remains the only failure. Do not repair that older test in this task.

## Documentation and report

- Update only documentation that becomes inaccurate due to the resilience state model.
- Add `Documentation/AI_HANDOFF/CLAUDE_TASK_05_1_FINAL_REPORT.md` stating exact branch/SHA, changed files, test results, the exact failure modes now handled, and confirmation that no live service, model, dataset, or evaluation was used.
- Commit and push only the new feature branch.
- Stop after reporting. Do not integrate this branch.

## Prohibited

No official catalogue download or import. No Qdrant collection build. No benchmark run. No synthetic-data evaluation. No changes to paper claims. No commit to protected branches.
