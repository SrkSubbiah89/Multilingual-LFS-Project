# Claude Task 07: Repair the verified legacy ISCO LLM mock-signature test defect

## Purpose

The integrated branch has one remaining full-suite failure:
`backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity`.

Independent review located the exact cause in that test file's shared `clf` fixture:

```python
monkeypatch.setattr("backend.agents.isco_classifier.get_llm", lambda t: MagicMock())
```

Production calls `get_llm` with a `temperature=` keyword, so the fixture mock raises `TypeError` and changes the test path instead of testing intended low-confidence LLM reranking. Correct the test double signature only. This is a test-fixture repair, not a classifier behavior change.

## Start state and branch discipline

1. Fetch origin and verify `reviewer2-isic-iscedf-integration-20260808` is exactly `53c09e40a2383b102bfa8e432e60426e6c26bda7`, clean.
2. Create a new branch from this exact SHA:
   `reviewer2-isic-iscedf-test-green-20260808`
3. Do not merge, rebase, reset, clean, stash, pull, force-push, or alter any existing branch.
4. Do not modify the Task 05 source/resilience branches or protected historical branches.

## Required change

- In `backend/tests/test_isco_classifier_extended.py`, modify only the `get_llm` test double so it accepts the production call signature, including positional and keyword arguments, while still returning the same `MagicMock()`.
- Do not alter `backend/agents/isco_classifier.py`, production LLM configuration, thresholds, retrieval logic, classifier outputs, or unrelated tests.
- Do not use a skip, xfail, exception swallowing, weakened assertion, or hard-coded result. The existing `test_llm_used_for_low_similarity` assertion must execute the intended LLM reranking path and pass for the right reason.
- Add a narrowly scoped assertion only if needed to prove the mock received the expected temperature keyword. Do not expand scope.

## Required verification

Run:

```bash
pytest backend/tests/test_isco_classifier_extended.py -q
pytest backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchy_nodes.py backend/tests/test_standard_hierarchical_store.py backend/tests/test_isic_classifier.py backend/tests/test_isced_classifier.py backend/tests/test_method_registry.py -q
pytest eval/test_run_eval_b2.py eval/test_dev_sweep.py eval/test_pre_run_check.py eval/test_validate_evaluation_discipline.py eval/test_docs_consistency.py -q
pytest backend/tests eval/ -q
```

The expected outcome is zero failed tests in the full suite. If any test fails, stop and report it as a blocker. Do not work around it.

## Prohibited operations

No production-code change. No Qdrant, model download, Ollama/LLM network call, collection build, benchmark, evaluation, dataset operation, official-catalogue work, B1/B2 configuration change, or paper claim change.

## Final report and push

- Add `Documentation/AI_HANDOFF/CLAUDE_TASK_07_FINAL_REPORT.md`.
- Include: exact root cause, exact fixture change, changed files, targeted/full test commands and output, explicit proof the full suite is green, no-live-operation confirmation, protected-branch confirmation, and clean-tree confirmation.
- Commit and push only `reviewer2-isic-iscedf-test-green-20260808`.
- Stop. Do not integrate or create a PR.
