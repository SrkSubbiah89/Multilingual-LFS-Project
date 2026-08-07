# Claude Task 08: Create the canonical pre-evaluation baseline

## Purpose

Create one clear, canonical branch containing the completed B2 integration, audited ISIC/ISCED-F hierarchical retrieval feature chain, all resilience work, and the now-green test-suite fixture repair. This establishes the single branch from which controlled benchmark/data work can later begin.

This is source-control consolidation only. It is not a benchmark, data, Qdrant, model, evaluation, or paper-editing task.

## Required sources

- Base integration branch: `reviewer2-isic-iscedf-integration-20260808` at `53c09e40a2383b102bfa8e432e60426e6c26bda7`.
- Green-test feature branch: `reviewer2-isic-iscedf-test-green-20260808` at `52e7f34b6643e05e00edf9c7a8ffb08a05cfefa4`.

Fetch origin. Verify both exact SHAs and a clean starting worktree before proceeding.

## Branch discipline

1. Create a new branch from the base integration SHA only:
   `reviewer2-pre-evaluation-baseline-20260808`
2. Merge the green-test source into it using `git merge --no-ff` so the branch records an explicit two-parent consolidation merge.
3. Do not alter the source branches or any other existing branch.
4. Do not merge into `master`, `conference1-b2-evaluation`, `reviewer2-wip-snapshot-20260807`, or `reviewer2-b2-integration-20260807`.
5. Do not reset, clean, stash, rebase, pull, force-push, or rewrite Git history.

## Required preservation checks

After merging, prove:

- The final branch includes Task 05, 05.1, and 05.2 ISIC/ISCED-F hierarchical retrieval and resilience files.
- `backend/tests/test_isco_classifier_extended.py` includes the repaired `get_llm` mock that accepts keyword arguments and retains the intended low-confidence LLM-path assertion.
- `eval/configs/b1_frozen.json` and `eval/dev_sweep.py` remain byte-identical to the integration base. Historical B1 stays quarantined and B2 sweep remains blocked until a valid re-freeze.
- No source-code edits beyond an unavoidable merge conflict resolution are permitted. If a conflict occurs, stop and report it rather than silently redesigning behavior.

## Required tests

Run exactly:

```bash
pytest backend/tests/test_isco_classifier_extended.py -q
pytest backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchy_nodes.py backend/tests/test_standard_hierarchical_store.py backend/tests/test_isic_classifier.py backend/tests/test_isced_classifier.py backend/tests/test_method_registry.py -q
pytest eval/test_run_eval_b2.py eval/test_dev_sweep.py eval/test_pre_run_check.py eval/test_validate_evaluation_discipline.py eval/test_docs_consistency.py -q
pytest backend/tests eval/ -q
```

Expected full-suite result: `1885 passed, 1 deselected, 1 warning`, with zero failures. If it differs or any test fails, stop and report it as a blocker.

## Prohibited operations

No live Qdrant, embedding-model download/load, Ollama/LLM invocation, collection build, dataset operation, WISCO/full130 run, evaluation, synthetic generation, official catalogue import, B1 re-freeze, B2 sweep, manuscript change, or PR.

## Final report and push

- Add `Documentation/AI_HANDOFF/CLAUDE_TASK_08_FINAL_REPORT.md`.
- Include source SHAs, merge SHA, final SHA, conflict status, preservation-check evidence, exact test output, confirmation of no live operations, protected-branch status, and clean working tree.
- Commit the report and push only `reviewer2-pre-evaluation-baseline-20260808`.
- Stop after reporting. Do not begin benchmark or evaluation work.
