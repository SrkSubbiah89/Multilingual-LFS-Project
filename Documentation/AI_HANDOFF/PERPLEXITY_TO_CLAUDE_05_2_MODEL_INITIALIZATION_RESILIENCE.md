# Claude Task 05.2: Make embedding-model initialization fail closed

## Purpose

An independent audit of Task 05.1 confirmed the Qdrant readiness, query-embedding, and engine-search fallback paths. One final operational gap remains: `StandardHierarchicalStore.__init__()` constructs `SentenceTransformer(MODEL_NAME)` outside the protected embedding boundary. If model initialization fails, hierarchical classification can still raise before returning the existing explicit fallback result.

Fix only that gap. This is a continuation of Task 05.1, not an integration or evaluation task.

## Start state and branch discipline

1. Fetch origin and verify `reviewer2-isic-iscedf-hierarchical-resilience-20260808` is exactly `cd991faef753bd210047d3f621288e7f7fc5ac37` with a clean tree.
2. Create a new branch from that SHA:
   `reviewer2-isic-iscedf-hierarchical-model-init-resilience-20260808`
3. Do not merge, rebase, reset, clean, stash, pull, force-push, or touch protected branches.
4. Protected branches include `master`, `conference1-b2-evaluation`, `reviewer2-wip-snapshot-20260807`, `reviewer2-b2-integration-20260807`, and all Task 05 / 05.1 source branches.

## Required implementation

In `backend/rag/standard_hierarchical_store.py`:

- Make production `SentenceTransformer(MODEL_NAME)` construction lazy, or otherwise move it entirely inside the existing protected embedding boundary used by `search()`.
- If model construction or query encoding fails, `search()` must return an explicit no-code `StandardHierarchyResult` with a non-empty embedding-related `unavailable_reason`. It must not raise to either classifier.
- Preserve dependency injection: a supplied fake embedder must still be used directly and must not construct a real model.
- Do not construct or load an embedding model when Qdrant readiness failed or collections are missing.
- Preserve existing positive-path behavior, parent-filtered engine behavior, fallback labels, and default `classify(text)` behavior.
- Do not add a broad catch around unrelated application logic. The protected scope is only the external model construction/encoding boundary.

## Required hermetic tests

Use monkeypatch and fakes only. No network, model download, Qdrant, Ollama, LLM, dataset, evaluation, benchmark, or collection build.

Add tests that prove:

1. When `SentenceTransformer` construction raises, a ready-store `search()` returns an explicit no-code embedding-unavailable result rather than raising.
2. `ISICClassifier.classify(..., method=isic_hierarchical_retrieval)` receives that condition as the correct explicitly labelled ISIC fallback result.
3. `ISCEDClassifier.classify(..., method=iscedf_hierarchical_retrieval)` receives that condition as the correct explicitly labelled ISCED-F fallback result and retains independent ISCED 2011 level classification.
4. A fake injected embedder bypasses model construction and the existing positive parent-filtered tests continue to pass.
5. The Task 05.1 Qdrant-readiness and engine-search resilience tests remain green.

Run:

```bash
pytest backend/tests/test_standard_hierarchical_store.py backend/tests/test_isic_classifier.py backend/tests/test_isced_classifier.py backend/tests/test_hierarchy_nodes.py -q
pytest backend/tests eval/ -q
```

Report the historical ISCO mock-signature failure separately if it remains the only failure. Do not repair that unrelated legacy test in this task.

## Documentation and report

- Update only the resilience documentation wording affected by lazy model initialization.
- Add `Documentation/AI_HANDOFF/CLAUDE_TASK_05_2_FINAL_REPORT.md` with branch/SHA, changed files, exact test output, explicit proof that model construction failures now fall back, and confirmation of zero live operations.
- Commit and push only this new branch, then stop. Do not integrate.

## Prohibited

No code beyond this narrow resilience correction. No benchmark, evaluation, or dataset work. No live model or Qdrant operation. No official catalogue work. No manuscript claim changes.
