# Task 14 — Integrate Hierarchical Integrity Hardening into the Strict WISCO Benchmark Baseline

## Purpose

Create one clean, auditable branch that combines the stopped Task 12 benchmark lineage with the independently verified Task 13 integrity hardening. This task is Git integration and hermetic test verification only. It must not read WISCO data, connect to Qdrant, load an embedding model, call an LLM, or execute any evaluation.

## Verified source commits

| Role | Branch | Required SHA |
| --- | --- | --- |
| Failed-run/report base | `reviewer2-wisco-tier1-results-20260808` | `6710192ea255c2cfb00599db9e55f65f9a53ca53` |
| Integrity-hardening feature | `reviewer2-hierarchical-integrity-hardening-20260808` | `68cfcd452b880b7cd62db416c66e5db96681f306` |

Task 13 is exactly one commit ahead of the Task 12 results branch. It fixes the verified keyword-anchor silent fallback, adds a bounded configurable Qdrant timeout, and adds strict evaluation-only guards. It does not change the Task 12 raw artifacts.

## Starting point and Git boundary

1. Fetch remote refs and verify both source branch tips match the exact SHAs above.
2. Start with a clean working tree.
3. Create a new branch from the Task 12 results base only:
   - `reviewer2-wisco-strict-benchmark-baseline-20260808`
4. Merge the Task 13 feature using an explicit merge commit:

```bash
git merge --no-ff reviewer2-hierarchical-integrity-hardening-20260808
```

5. If Git reports a conflict, stop immediately. Do not hand-resolve it. Record conflict paths and leave all source/protected branches untouched.

## Required preservation checks

After a clean merge, prove all of the following before tests:

1. The Task 12 raw output CSVs remain local, Git-ignored, and byte-identical to the Task 13 recorded checksums. Do not open them for writing.
2. The Task 12 final report remains unchanged and still states `TIER1_COMPLETED: no`.
3. The Task 05 / 05.1 / 05.2 ISIC and ISCED-F hierarchy files remain present:
   - `backend/rag/hierarchy_nodes.py`
   - `backend/rag/standard_hierarchical_store.py`
   - `backend/rag/build_standard_hierarchical_collections.py`
4. The Task 07 repaired `get_llm` mock and its `temperature` assertion remain present.
5. The B1 quarantine remains byte-identical to the Task 12 base:

```bash
git diff 6710192ea255c2cfb00599db9e55f65f9a53ca53..HEAD -- eval/configs/b1_frozen.json eval/dev_sweep.py
```

Expected: no output. Do not re-freeze or bypass B1/B2 safety.

6. Confirm Task 13 behavior exists after the merge:
   - seeded keyword-anchor failure triggers exactly one unseeded hierarchical retry;
   - only the winning retry trace becomes final stage evidence;
   - remaining flat fallback stays explicitly labelled;
   - Qdrant timeout has a finite default and safe environment override;
   - `--require-genuine-hierarchical` and `--max-stage-latency-ms` are opt-in and fail a contaminated evaluation before CSV output.
7. Confirm Task 09 model-free behavior remains present: no LLM initialization when reranking is off, and no ISIC/ISCED/SRE construction for WISCO-style rows.

## Tests

Run focused tests first:

```bash
pytest backend/tests/test_hierarchical_store.py eval/test_require_genuine_hierarchical.py backend/tests/test_hierarchy_engine.py backend/tests/test_isco_classifier.py backend/tests/test_isco_classifier_extended.py eval/test_model_free_isco_evaluation.py eval/test_run_eval_b2.py eval/test_sre_isic_isced_coupling_fix.py -q
```

Then run:

```bash
pytest backend/tests eval/ -q
```

Report exact outputs. Do not claim a green suite unless the full command has zero failures.

## Strict non-goals

Do not perform any live operation, including:

- reading, rebuilding, exporting, or evaluating WISCO;
- connecting to Qdrant, listing Qdrant collections, or mutating Qdrant;
- loading SentenceTransformer or any embedding model;
- calling Ollama, any LLM, paid API, or model API;
- running `eval/run_eval.py`, `eval/analyze.py`, `eval/ablation_runner.py`, any benchmark script, B1 re-freeze, or B2 sweep.

Do not change production code, tests, configurations, requirements, datasets, benchmark split, manuscript wording, or Task 12 artifacts beyond the clean merge and the final report.

## Documentation and Git action

Create only:

- `Documentation/AI_HANDOFF/CLAUDE_TASK_14_FINAL_REPORT.md`

The report must state: verified source SHAs; merge/final SHAs; conflict status; exact base-to-merge file scope; preservation-check outputs; exact tests/results; confirmation of zero live WISCO/Qdrant/model/LLM/benchmark activity; protected-branch status; working-tree status; and remaining manuscript limits.

Commit the merge and final report only to `reviewer2-wisco-strict-benchmark-baseline-20260808`, then push only that branch. Do not create a PR or merge into any protected branch.

## Stop condition

Stop after pushing the branch, committing the report, and verifying the working tree is clean. Do not run a WISCO preflight or rerun. The next task will independently inspect this integrated baseline and authorize a larger strict preflight before any new full benchmark run.
