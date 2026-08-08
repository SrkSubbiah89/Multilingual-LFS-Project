# Task 19 — Integrate Analysis Gate and Audit Full Flat Baseline

## Purpose

This task has two tightly limited purposes:

1. merge the verified Task 18 fail-closed WISCO analysis utility into the Task 17 evidence line; and
2. perform a read-only source/provenance audit for a future, fair **full-unit-group flat ISCO comparator**.

The Task 18 analysis gate exposed a real property of the legacy `isco_occupations` collection: it is intentionally curated and mixed-granularity, so it is not eligible for a four-digit unit-group accuracy comparison. Do not weaken the gate and do not report a metric from the Task 17 legacy-flat CSV.

Do not implement, build, populate, query, evaluate, or benchmark a replacement flat comparator in this task. This is integration plus audit/specification only.

## Verified sources

Fetch `origin` and verify before changing anything:

| Role | Branch | Required SHA |
|---|---|---|
| Task 17 evidence base | `reviewer2-wisco-tier1-strict-full-results-20260808` | `f7c40ab3675ab8a7ee239a490fbef261c2e5bc44` |
| Task 18 analysis-gate source | `reviewer2-wisco-tier1-analysis-20260808` | `1b15b19fe8f60cf217caeb076bf3a134059793c4` |

Confirm:

1. Task 18 is exactly one commit ahead of Task 17.
2. Task 18’s base-to-tip changed files are exactly:

```text
eval/analyze_wisco_tier1.py
eval/test_analyze_wisco_tier1.py
Documentation/AI_HANDOFF/CLAUDE_TASK_18_FINAL_REPORT.md
```

3. Working tree is clean.

If any verification fails, do not merge. Create a final report describing the mismatch and stop.

## Branch and Git rules

1. Create and work only on:

```text
reviewer2-flat-baseline-coverage-audit-20260808
```

2. Make an explicit no-fast-forward merge:

```bash
git merge --no-ff reviewer2-wisco-tier1-analysis-20260808
```

3. If a merge conflict occurs, stop and report it. Do not improvise a conflict-resolution change.
4. After the merge, add only:

```text
Documentation/Conference_I_Reviewer_2/FLAT_BASELINE_COVERAGE_AUDIT.md
Documentation/AI_HANDOFF/CLAUDE_TASK_19_FINAL_REPORT.md
```

5. Commit the audit document and final report separately after the merge.
6. Push only the new Task 19 branch. Do not create a PR.
7. Do not merge, rebase, reset, clean, stash, pull, force-push, or modify protected/prior branches.

## Required preservation checks

Before and after the merge, use read-only Git/source inspection to confirm:

1. Task 17 raw outputs remain ignored/local and Task 17 report remains unchanged.
2. Task 17’s `TIER1_STRICT_COMPLETED: yes` remains a raw-output-integrity status only, not an accuracy result.
3. Task 18’s `analyze_wisco_tier1.py` and its test file are present unchanged by the merge.
4. Task 18’s real-data gate failure remains documented:
   - flat invalid/non-four-digit predictions: 4,754 of 18,747;
   - 1,372 one-digit codes;
   - 3,382 two-digit codes;
   - no Task 18 metrics output.
5. Task 13 strict hierarchical retry/timeout/strict-guard behavior remains intact.
6. Task 09’s model-free path remains intact.
7. B1 quarantine and B2 sweep gate remain byte-identical to the Task 17 base:

```text
eval/configs/b1_frozen.json
eval/dev_sweep.py
```

No project code, test, configuration, data, local raw artifact, Qdrant collection, or benchmark definition may be modified in this task other than the permitted merge content and two audit/report documents.

## Audit scope

Create:

```text
Documentation/Conference_I_Reviewer_2/FLAT_BASELINE_COVERAGE_AUDIT.md
```

The audit must be evidence-led and cite exact repository file paths, line ranges, counts, and existing source URLs/provenance fields where available. Do not invent an official count, a source licence, or a data lineage.

### Legacy flat baseline audit

Inspect `backend/rag/vector_store.py`, the `ISCOClassifier` call path, and relevant tests. Establish:

1. the legacy collection name and its intended role;
2. the exact source of its records;
3. record counts by code length and declared hierarchy level;
4. whether it contains 1-digit, 2-digit, 3-digit, and 4-digit records;
5. search behavior and why it can return a coarse code;
6. whether returned results are filtered to 4-digit codes;
7. the precise relationship to Task 18’s 4,754 invalid four-digit predictions;
8. why this is a baseline-coverage limitation, not data corruption or a Task 17 execution failure.

State clearly:

```text
The legacy `isco_occupations` collection is a curated mixed-granularity integration resource. It must not be called a four-digit flat ISCO-08 accuracy baseline, and its Task 17 raw output must not be used for a four-digit flat-versus-hierarchical accuracy calculation.
```

### Hierarchical unit-group source audit

Search the repository for every source, builder, static data structure, test fixture, and documentation path that contributes to the existing `isco08_unit_groups` hierarchy collection. Determine, with evidence:

1. the source file(s) and transformation path that defines its records;
2. its expected code format and whether every node is exactly four digits;
3. declared versus observed/count-tested unit-group totals;
4. its payload schema, including code, labels, parent fields, any source/provenance fields, and embedding text;
5. whether it is read directly by the hierarchical stage-4 retrieval;
6. whether a direct unfiltered query over only those 4-digit nodes is technically feasible;
7. whether any WISCO source file, WISCO title, WISCO code, or WISCO dataset-derived mapping is imported, read, or used to create the unit-group nodes.

Do not infer non-leakage merely because a file name does not include “WISCO.” Search imports, data-loading paths, builder inputs, and documentation. If provenance cannot be proven from repository evidence, say so and mark the comparator blocked.

### Count discrepancy audit

Resolve or precisely describe the discrepancy among:

- the official/provenance documentation that previously cited 436 ISCO-08 unit groups;
- hierarchy architecture/code documentation that may say approximately 430 or 436;
- Task 11/17 local Qdrant point counts that recorded 441 points in `isco08_unit_groups`.

For every count, identify:

1. what it counts;
2. where it comes from;
3. whether it is a source-catalogue count, static-record count, Qdrant-point count, unique-code count, or another measure;
4. whether duplicates, aliases, additional codes, metadata points, or a documented revision explain the difference.

Do not “resolve” the discrepancy by picking a convenient number. If the source code does not prove the explanation, write:

```text
UNRESOLVED: a full unit-group flat comparator must not be implemented until this count and catalogue identity are reconciled.
```

### Future comparator specification

Only if source provenance, no-WISCO-leakage evidence, and catalogue identity are adequately demonstrated, write a **proposed**, not implemented, specification for a future comparator. It must include:

1. a separately named collection, distinct from legacy `isco_occupations`, such as `isco08_unit_groups_flat_v1`;
2. one record per accepted four-digit ISCO unit-group code only;
3. an immutable catalogue/source hash and exact expected unique-code count checked before any collection build;
4. payload fields and embedding text construction;
5. direct unfiltered nearest-neighbor retrieval over unit-group leaves only, with no parent-beam or hierarchical traversal;
6. a hard runtime assertion that every returned candidate code is exactly four digits;
7. a distinct method label, never `flat_semantic`, so papers/results cannot conflate it with the legacy curated flat path;
8. an explicit collection-building command requiring a separate approval before execution;
9. hermetic tests for source counts, all-code format, payload schema, WISCO non-dependence, runtime four-digit filtering, and collection identity;
10. a future evaluation protocol that runs only this new comparator against the already validated Task 17 hierarchical raw output, with no changes to WISCO labels/split/order.

If any source/provenance/count gate is unresolved, provide a blocker list and do not write an implementation-ready specification.

### Paper/reviewer implications

Conclude the audit with a simple evidence-status table:

| Claim area | Current status | What is safe now | What remains blocked |
|---|---|---|---|

At minimum distinguish:

- architecture/implementation;
- hierarchical strict integrity;
- WISCO controlled four-digit hierarchical output integrity;
- fair flat-versus-hierarchical accuracy comparison;
- controlled multilingual ISCO-08 accuracy;
- latency/computational evidence;
- ISIC/ISCED/SRE evidence;
- real LFS validation.

Do not state that Reviewer #2 is fully satisfied. Do not claim any WISCO accuracy result, because Task 18 produced no metric output.

## Operations prohibited in this task

Do not run:

- `eval/run_eval.py`, `eval/analyze_wisco_tier1.py` on real data, `eval/analyze.py`, or any benchmark/evaluation/analysis command;
- Qdrant query, collection build, mutation, count, inspect, populate, rebuild, or deletion;
- SentenceTransformer/model loading, Ollama, CrewAI, LLM, paid API, or network inference;
- WISCO package read/write beyond source-path/provenance inspection needed for no-leakage audit;
- B1 re-freeze, B2 sweep, ISIC/ISCED/SRE evaluation;
- a new unit-group flat comparator implementation;
- collection-builder implementation or execution.

This task may run only static/read-only source inspection, Git inspection, and automated tests.

## Tests

Run:

```bash
pytest eval/test_analyze_wisco_tier1.py eval/test_analyze.py eval/test_run_eval_b2.py eval/test_require_genuine_hierarchical.py eval/test_docs_consistency.py -q
pytest backend/tests eval/ -q
```

Report exact results. Do not fix, weaken, skip, xfail, or change unrelated code/tests if a test fails.

## Final report

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_19_FINAL_REPORT.md
```

Include:

1. source SHAs; branch; merge commit and parent SHAs; report commit and remote push confirmation;
2. conflict status and exact changed-file lists for merge and audit commit;
3. every preservation check;
4. audit conclusion:

```text
FLAT_COMPARATOR_IMPLEMENTATION_READY: yes
```

or:

```text
FLAT_COMPARATOR_IMPLEMENTATION_READY: no
```

5. legacy baseline coverage conclusion and Task 18 invalid-code breakdown;
6. hierarchy source/provenance/no-WISCO-leakage conclusion;
7. complete count-discrepancy reconciliation or blocker;
8. implementation specification readiness or precise blockers;
9. reviewer/paper evidence-status summary;
10. focused and full test results;
11. confirmation that no live Qdrant/model/LLM/evaluation/data operation occurred;
12. protected-branch and clean-tree confirmation.

After pushing the final report, stop. Do not implement the comparator or run another evaluation.
