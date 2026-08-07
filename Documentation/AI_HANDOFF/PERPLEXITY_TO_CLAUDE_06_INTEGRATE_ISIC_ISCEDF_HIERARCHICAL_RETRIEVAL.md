# Claude Task 06: Integrate the audited ISIC / ISCED-F hierarchical retrieval feature chain

## Purpose

Integrate the completed and independently audited Task 05, Task 05.1, and Task 05.2 feature chain into the existing B2 integration line. This is a controlled source-code integration only. It is not a Qdrant build, an evaluation, a benchmark, a dataset operation, or a paper-editing task.

## Required sources

- Integration base: `reviewer2-b2-integration-20260807` at `2ababd56b1b5e50349e8f7556de4639c4b0bc3fa`.
- Feature source: `reviewer2-isic-iscedf-hierarchical-model-init-resilience-20260808` at `f86a11b8ff032a862d2cbbb0755532a717e239af`.

Before changing anything, fetch origin and verify both exact SHAs and clean worktrees.

## Branch discipline

1. Create a new branch from the integration base only:
   `reviewer2-isic-iscedf-integration-20260808`
2. Merge the feature source into this new branch using an explicit merge commit (`git merge --no-ff`). Do not fast-forward the branch.
3. Do not merge into `master`, `conference1-b2-evaluation`, `reviewer2-wip-snapshot-20260807`, `reviewer2-b2-integration-20260807`, `reviewer2-isic-iscedf-hierarchical-rag-20260808`, `reviewer2-isic-iscedf-hierarchical-resilience-20260808`, or `reviewer2-isic-iscedf-hierarchical-model-init-resilience-20260808`.
4. Do not reset, clean, stash, rebase, pull, force-push, or alter Git history.

## Integration requirements

- Preserve the entire final feature chain: real parent-filtered ISIC Rev.4 and ISCED-F 2013 hierarchical retrieval, explicit legacy fallback labels, Qdrant-readiness resilience, embedding/search resilience, lazy model initialization, and all hermetic tests.
- Preserve B2 safety work: historical B1 baseline remains quarantined and B2 sweep remains blocked until a valid re-freeze. Do not alter `eval/configs/b1_frozen.json` or bypass the sweep gate.
- Resolve conflicts only if they are real merge conflicts. If a conflict occurs, preserve both feature-chain behavior and B2 safety behavior. Do not make unrelated refactors.
- Do not edit classifier behavior, data, evaluation configuration, generated evidence, or manuscript wording beyond changes mechanically necessary to resolve a merge conflict.
- Confirm that `method=None` remains the legacy default for ISIC and ISCED, and that hierarchical retrieval occurs only with its explicit method constants.

## Required verification

Run all of the following after the merge:

```bash
pytest backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchy_nodes.py backend/tests/test_standard_hierarchical_store.py backend/tests/test_isic_classifier.py backend/tests/test_isced_classifier.py backend/tests/test_method_registry.py -q
pytest eval/test_run_eval_b2.py eval/test_dev_sweep.py eval/test_pre_run_check.py eval/test_validate_evaluation_discipline.py eval/test_docs_consistency.py -q
pytest backend/tests eval/ -q
```

- Report exact outputs.
- The known legacy `backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity` failure may remain. Report it exactly and do not suppress, skip, xfail, modify, or repair it in this integration task.
- Any additional failure is a blocker. Stop and report it rather than working around it.

## Prohibited operations

No live Qdrant connection, collection build, model download, Ollama/LLM invocation, network benchmark, dataset read/write, WISCO/full130 run, evaluation manifest, or synthetic-data evaluation. No official catalogue import. No paper claim changes.

## Final report and push

1. Add `Documentation/AI_HANDOFF/CLAUDE_TASK_06_FINAL_REPORT.md`.
2. The report must include: source SHAs, merge commit SHA, final SHA, conflict status, exact changed files, exact test commands/results, the status of the known legacy failure, confirmation of no live operations, confirmation that protected branches are unchanged, and confirmation the working tree is clean.
3. Commit the report to the new integration branch and push only that branch.
4. Stop. Do not create a PR and do not merge this branch into any protected branch.
