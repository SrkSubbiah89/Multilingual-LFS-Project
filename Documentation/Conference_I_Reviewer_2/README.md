# Conference I — Reviewer #2 Response: Evidence & Documentation

This directory contains the infrastructure, audits, schemas, tests, and
documentation built to give the authors defensible, non-fabricated evidence
when responding to Conference I Reviewer #2's feedback and revising the
manuscript. It does **not** contain manuscript text — it contains the code,
data, and reports the manuscript's claims can now cite.

**Hard rule that governs every file here**: nothing in this directory
fabricates a benchmark result, coverage percentage, latency/memory/cost
figure, or citation. Where real evidence doesn't exist yet, the tooling
reports that honestly (`null` + a reason, or a literal `"not yet run"` /
`"not yet measured"` / `"Awaiting data"`), never a placeholder number.

## Reviewer #2's 8 comments, and where each is addressed

See **[REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md](REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md)**
for the authoritative, comment-by-comment mapping to code/docs/tests/evidence
and current closure status.

## Documents in this directory

| Document | What it's for |
|---|---|
| [REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md](REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md) | Comment-by-comment status (5 allowed labels only) — **the authoritative index of every Step's outcome, read this first** |
| [FINAL_QA_BASELINE.md](FINAL_QA_BASELINE.md) | Historical point-in-time QA baseline (Step 1) — test counts there are superseded by whatever `pytest backend/tests eval/ -q` reports now; see the matrix for current status |
| [CLASSIFIER_METHOD_REGISTRY.md](CLASSIFIER_METHOD_REGISTRY.md) | How to (re)generate and read the classifier/agent method registry |
| [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) | Manifest schema, dev/held-out split discipline, relationship to the pre-existing thesis evaluation framework |
| [EVALUATION_READINESS_AND_DRY_RUN.md](EVALUATION_READINESS_AND_DRY_RUN.md) | Step 4: `--dry-run` mode for `run_eval.py`/`ablation_runner.py`, verified with zero live model/DB calls |
| [REAL_LFS_VALIDATION_DATASET_CARD_TEMPLATE.md](REAL_LFS_VALIDATION_DATASET_CARD_TEMPLATE.md) | Template + governance rules for any future real LFS dataset |
| [REAL_LFS_DATA_INTAKE_CHECKLIST.md](REAL_LFS_DATA_INTAKE_CHECKLIST.md) | Step-by-step checklist for a data custodian moving from "we have real data" to a citable `approved_real_lfs_validation` result |
| [ANNOTATION_AND_ADJUDICATION_GUIDE.md](ANNOTATION_AND_ADJUDICATION_GUIDE.md) | Coder qualifications, double-coding, adjudication, and ambiguous/multilingual-response handling for occupation/industry/education labelling — covers both real-LFS intake and controlled-benchmark labelling |
| [COVERAGE_AUDIT_GUIDE.md](COVERAGE_AUDIT_GUIDE.md) | How the ISCO/ISIC/ISCED coverage audit works, how to import an official catalogue for a verified percentage, and the abstract-wording implications |
| [STANDARDS_SOURCE_PROVENANCE.md](STANDARDS_SOURCE_PROVENANCE.md) | Authoritative sources for ISCO-08/ISIC Rev.4/ISCED 2011/ISCED-F 2013, and the exact research trail behind every sourced-but-unverified count |
| [COMPUTATIONAL_ANALYSIS_GUIDE.md](COMPUTATIONAL_ANALYSIS_GUIDE.md) | How to produce a real computational-cost manifest, and what's honestly not measured yet |
| [FIGURE_DATA_EXPORT_GUIDE.md](FIGURE_DATA_EXPORT_GUIDE.md) | How to turn the exported CSV/JSON into vector figures for the paper |
| [SRE_COUPLING_BUGFIX.md](SRE_COUPLING_BUGFIX.md) | Step 5.1: root-cause + fix for the `--sre off` also disabling ISIC/ISCED classification bug |
| [CONTROLLED_BENCHMARK_AUDIT.md](CONTROLLED_BENCHMARK_AUDIT.md) | Step 6: audit of every candidate benchmark dataset in the repo (incl. the newly-found WISCO dataset) against a 10-item provenance checklist |
| [CONTROLLED_BENCHMARK_DATASET_CARD_TEMPLATE.md](CONTROLLED_BENCHMARK_DATASET_CARD_TEMPLATE.md) | Template for any controlled (non-real-LFS) benchmark package, e.g. WISCO |
| [WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md](WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md) | Step 7A: strict leakage audit of the WISCO benchmark split, the v2 group-aware split fix, and the prepared (not yet executed) Step 7B evaluation plan |
| [STEP_7_COMMAND.md](STEP_7_COMMAND.md) | Superseded by `WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md` — kept for Step 6's historical record only |
| `generated/` | Landing directory for every JSON/CSV/MD artifact the tooling below produces (`MEASURED_EVALUATION_EVIDENCE_SUMMARY.md` lives here) |

## Code added (by section)

- **Section A** — `backend/agents/method_registry.py`, `backend/agents/classifier_methods.py`
- **Section B (generic engine only)** — `backend/rag/hierarchy_engine.py`; `backend/rag/hierarchical_store.py` refactored to use it (behavior-preserving); `ISICClassifier`/`ISCEDClassifier` gained a `method=` stub parameter for the (not yet implemented) hierarchical-retrieval modes
- **Section C** — `eval/coverage_audit.py`, `eval/standards_reference.yaml`, `eval/catalogue_importer.py` (validates a user-supplied official catalogue file and produces `eval/verified_catalogue_counts.yaml`, the only source of a citable `coverage_percentage`)
- **Section D** — `eval/manifest.py`, `eval/analyze.py`; `eval/run_eval.py`'s `CaseResult` gained additive ISIC/ISCED-F full-depth prediction columns
- **Section E** — `eval/ablation_runner.py`; `eval/run_eval.py` gained `--use-llm-reranker` and `--sre` flags
- **Section F** — `eval/dataset_card_schema.py` (closed 3-value `dataset_label` vocabulary + full 5-group governance schema), `eval/validate_real_lfs_governance.py` (strict fail-closed gate with content-aware synthetic-marker and in-repo-path safeguards); Step 3 hardening pass also touched `eval/manifest.py` and `eval/ablation_runner.py` (label enforcement + `invalid_incomplete_governance` visibility) — see `REAL_LFS_DATA_INTAKE_CHECKLIST.md`
- **Section G** — `backend/agents/semantic_relation.py` gained violation provenance + a genuine LOW severity tier; `eval/sre_eval_format.py`
- **Section I** — `eval/figure_exports/*.py`

Every module above has a corresponding test file (`test_*.py` next to it, or
under `backend/tests/`) — see [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)
and the individual guides for how to run each tool.

### Later steps (2 through 7A) — not part of the original A-J section plan

The A-J sections above were the original single-pass plan. Work continued
in numbered "Steps" after that pass shipped; each has its own doc (see the
table above) and is tracked in
[REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md](REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md)'s
per-step sections at the bottom of that file. Briefly: Step 2 (standards
provenance + verified-vs-unverified coverage counts), Step 3 (real-LFS
governance hardening), Step 4 (`--dry-run` mode), Step 5/5.1 (measured
synthetic-fixture run + a real bug found and fixed), Step 6 (controlled-
benchmark audit — found and prepared the WISCO dataset), Step 7A (WISCO
leakage audit + group-aware split fix; Step 7B, the actual measured
benchmark run, has not happened yet). New modules: `eval/manifest.py`,
`eval/dataset_card_schema.py`, `eval/validate_real_lfs_governance.py`,
`eval/validate_evaluation_discipline.py`, `eval/controlled_benchmark_schema.py`,
`eval/validate_controlled_benchmark.py`, `eval/build_wisco_isco_benchmark.py`
(+ `_v2_group_split.py`), `eval/audit_wisco_benchmark_leakage.py`,
`eval/select_wisco_reranking_subset.py`, `eval/export_benchmark_to_run_eval_csv.py`,
plus the `eval/local_runs/` and `eval/local_benchmarks/` gitignored output
conventions.

## What is explicitly out of scope for this pass

Full ISIC Rev.4 and ISCED-F 2013 **hierarchical retrieval** (new Qdrant
collections, a loader, live multi-stage search) was not built in this pass —
only method-label stubs that clearly report "not yet implemented"
(`ISICClassifier.classify(text, method="isic_hierarchical_retrieval")` /
`ISCEDClassifier.classify(text, method="iscedf_hierarchical_retrieval")`).
The generic `backend/rag/hierarchy_engine.py` built for ISCO is designed so
that building these later is a natural extension (new `StageConfig` list +
a loader, following `backend/rag/load_full_isco.py`'s template), not a
rewrite. Never describe ISIC/ISCED as "hierarchical RAG" in the manuscript
until that future work lands and is tested.
