# Task 10 — Integrate the Model-Free Evaluation Fix into the WISCO Measurement Baseline

## Purpose

Create one clean, auditable branch that combines the current canonical pre-evaluation baseline with the verified Task 09 retrieval-only evaluator correction. This task is Git integration plus test verification only. It must not build collections, prepare a dataset, call a model, or run a WISCO benchmark.

## Verified source commits

| Role | Branch | Required SHA |
| --- | --- | --- |
| Canonical base | `reviewer2-pre-evaluation-baseline-20260808` | `c3d39d89f1b164f4fc5595c319e90546a2c17716` |
| Task 09 feature | `reviewer2-model-free-isco-evaluation-20260808` | `2348566e54828546cf7ac821eb17cc96213c8ebf` |

The Task 09 feature is exactly one commit ahead of the canonical base. Its parent is the required canonical SHA and its scope is limited to six files: `backend/agents/isco_classifier.py`, `eval/run_eval.py`, their focused tests, the WISCO run plan, and the Task 09 report.

## Starting point and branch boundary

1. Fetch remote refs and verify both source branch tips match the exact SHAs above.
2. Start with a clean working tree.
3. Create a new branch from the canonical base only:
   - `reviewer2-wisco-measurement-baseline-20260808`
4. Merge the Task 09 feature using an explicit merge commit:

```bash
git merge --no-ff reviewer2-model-free-isco-evaluation-20260808
```

5. If Git reports a conflict, stop immediately. Do not hand-resolve it. Record the conflict paths and leave all protected/source branches untouched.

## Required preservation checks

After a clean merge, prove all of the following before running tests:

1. The Task 05 / 05.1 / 05.2 ISIC and ISCED-F hierarchy files remain present:
   - `backend/rag/hierarchy_nodes.py`
   - `backend/rag/standard_hierarchical_store.py`
   - `backend/rag/build_standard_hierarchical_collections.py`
2. The Task 07 repaired `get_llm` mock remains in `backend/tests/test_isco_classifier_extended.py`, including the assertion that passes `temperature`.
3. The Task 04 B1 quarantine remains intact and byte-identical to the canonical base:

```bash
git diff c3d39d89f1b164f4fc5595c319e90546a2c17716..HEAD -- eval/configs/b1_frozen.json eval/dev_sweep.py
```

Expected: no output. Do not alter, re-freeze, or bypass B1/B2 safety.

4. Confirm the Task 09 behavior is present:
   - `ISCOClassifier(enable_llm=False)` skips LLM/agent construction but leaves default production behavior unchanged.
   - `--use-llm-reranker off` does not require `--reranker-model`.
   - ISCO-only inputs with no paired industry and education text do not construct ISIC, ISCED, or SRE components.
   - The configuration hash records the true LLM-reranker state.
5. Confirm the WISCO documentation still limits it to a controlled multilingual ISCO-08 benchmark, not real Labour Force Survey validation and not ISIC/ISCED/SRE evidence.

## Tests

Run focused tests first:

```bash
pytest backend/tests/test_isco_classifier.py backend/tests/test_isco_classifier_extended.py eval/test_model_free_isco_evaluation.py eval/test_run_eval_b2.py eval/test_sre_isic_isced_coupling_fix.py -q
pytest backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchy_nodes.py backend/tests/test_standard_hierarchical_store.py backend/tests/test_isic_classifier.py backend/tests/test_isced_classifier.py backend/tests/test_method_registry.py -q
```

Then run the full suite:

```bash
pytest backend/tests eval/ -q
```

Report exact results only. Do not claim a green suite unless the full command is green.

## Strict non-goals

Do not perform any of the following:

- build or populate Qdrant collections;
- download or load SentenceTransformer models;
- invoke Ollama, any paid LLM, or any API model;
- build, export, modify, or evaluate WISCO data;
- run `eval/run_eval.py`, `eval/ablation_runner.py`, `eval/build_wisco_isco_benchmark*.py`, or any benchmark command;
- re-freeze B1 or start a B2 sweep;
- change a dataset, benchmark split, fixture, production classifier behavior beyond the clean merge, configuration, requirements, or manuscript claims;
- merge/rebase/reset/clean/stash/pull/force-push/create a PR.

## Documentation and Git action

Create only:

- `Documentation/AI_HANDOFF/CLAUDE_TASK_10_FINAL_REPORT.md`

The final report must state: verified source SHAs; merge SHA and final SHA; conflict status; exact changed files base-to-merge; preservation-check outputs; focused and full test commands/results; confirmation of zero live model/Qdrant/dataset/benchmark operations; protected branch status; working-tree status; and remaining manuscript limits.

Commit the merge and the final report only to `reviewer2-wisco-measurement-baseline-20260808`, then push only that branch. Do not create a PR and do not merge into the canonical or protected branches.

## Stop condition

Stop after the branch is pushed, the final report is committed, and the working tree is clean. Do not begin any WISCO preparation or measured evaluation. The next task will inspect this integrated baseline and present the long-running benchmark for explicit approval.
