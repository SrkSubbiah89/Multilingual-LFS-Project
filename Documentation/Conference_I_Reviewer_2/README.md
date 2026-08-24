# Conference I — Reviewer #2 Response: Evidence & Documentation

> **Evidence trail, not a status source.** For "what's the project's
> current state," use `CLAUDE.md` (repo root) or
> `Documentation/PROJECT_FLOW_AND_STATUS.md`. This folder is where those
> documents point *from* when a claim needs a citable source — read it
> when you need the underlying evidence for one specific reviewer comment,
> not as a way to find out what's true right now.

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
| [ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md](ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md) | Task 05: real (but unevaluated) ISIC Rev.4 / ISCED-F 2013 hierarchical retrieval — architecture, collection names, explicit fallback semantics, operator build commands, safe/unsafe manuscript wording |
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
| [WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md](WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md) | Step 7A: strict leakage audit of the WISCO benchmark split, the v2 group-aware split fix, and the Step 7B evaluation plan — non-reranked tier now completed, see below |
| [OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md](OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md) | **Tasks 36/37.1 (2026-08-10): the canonical, manuscript-ready WISCO v2 official-profile controlled result** — headline/paired/subgroup tables, exact hashes, evidence chain, and safe/unsafe interpretation. Read this first for any WISCO accuracy citation. |
| [MANUSCRIPT_SAFE_WISCO_WORDING.md](MANUSCRIPT_SAFE_WISCO_WORDING.md) | Ready-to-paste abstract/methods/results/limitations wording and reviewer-comment responses for the WISCO result above, plus an explicit "do not write" list |
| [STEP_7_COMMAND.md](STEP_7_COMMAND.md) | Superseded by `WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md` — kept for Step 6's historical record only |
| `generated/` | Landing directory for every JSON/CSV/MD artifact the tooling below produces (`MEASURED_EVALUATION_EVIDENCE_SUMMARY.md` lives here) |
| [FLAT_BASELINE_COVERAGE_AUDIT.md](FLAT_BASELINE_COVERAGE_AUDIT.md) | Task 19: why the legacy `isco_occupations` flat baseline can't support a 4-digit accuracy comparison, and the (then-unresolved) 441-vs-436 ISCO-08 unit-group count discrepancy |
| [ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md](ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md) | Task 20: primary ILO ISCO-08 catalogue import (`eval/verified_catalogue_counts.yaml`) and full code-by-code reconciliation against `backend/rag/load_full_isco.py` — closes Task 19's catalogue-identity blocker; catalogue itself not yet corrected |
| [OFFICIAL_ISCO08_RUNTIME_AND_FLAT_COMPARATOR.md](OFFICIAL_ISCO08_RUNTIME_AND_FLAT_COMPARATOR.md) | Task 21: official-source runtime loader, dry-run collection builder, versioned official retrieval profiles (hierarchical + genuine four-digit-only flat comparator), and `eval/run_eval.py --isco-catalogue-profile` wiring. Task 22: the real, guarded `--execute` collection-build path (dual acknowledgement, local-only, no-overwrite, ordered create+verify, manifest). No collection built, no evaluation run by either task |

## Code added (by section)

- **Section A** — `backend/agents/method_registry.py`, `backend/agents/classifier_methods.py`
- **Section B** — `backend/rag/hierarchy_engine.py`; `backend/rag/hierarchical_store.py` refactored to use it (behavior-preserving). Extended by Task 05: `backend/rag/hierarchy_nodes.py`, `backend/rag/standard_hierarchical_store.py`, `backend/rag/build_standard_hierarchical_collections.py` — real, tested (but unevaluated, and not yet built against a live Qdrant instance) ISIC Rev.4 / ISCED-F 2013 hierarchical retrieval, reusing the same generic engine; see [ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md](ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md)
- **Section C** — `eval/coverage_audit.py`, `eval/standards_reference.yaml`, `eval/catalogue_importer.py` (validates a user-supplied official catalogue file and produces `eval/verified_catalogue_counts.yaml`, the only source of a citable `coverage_percentage`). Task 20 supplied the first real catalogue: `eval/normalize_ilo_isco08_catalogue.py` parses the official ILO ISCO-08 structure workbook into `catalogue_importer.py`'s input shape; `eval/verified_catalogue_counts.yaml` now has a real ISCO-08 entry (`{major: 10, submajor: 43, minor: 130, unit: 436}`) — see [ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md](ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md).
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
leakage audit + group-aware split fix). Step 7B, the actual measured
benchmark run, is tracked as Tasks 23-37.1 below (non-reranked tier
completed 2026-08-10; reranking tier still not run). New modules: `eval/manifest.py`,
`eval/dataset_card_schema.py`, `eval/validate_real_lfs_governance.py`,
`eval/validate_evaluation_discipline.py`, `eval/controlled_benchmark_schema.py`,
`eval/validate_controlled_benchmark.py`, `eval/build_wisco_isco_benchmark.py`
(+ `_v2_group_split.py`), `eval/audit_wisco_benchmark_leakage.py`,
`eval/select_wisco_reranking_subset.py`, `eval/export_benchmark_to_run_eval_csv.py`,
plus the `eval/local_runs/` and `eval/local_benchmarks/` gitignored output
conventions.

### Tasks 09-22 — WISCO Tier 1 evidence line, catalogue reconciliation, and official runtime/build

Tracked individually under `Documentation/AI_HANDOFF/CLAUDE_TASK_09_*`
through `CLAUDE_TASK_22_*` (numbered task-handoff reports, not part of
either the A-J or Step-2-through-7A schemes above). Briefly, in order:
model-free ISCO evaluation mode; WISCO v2 group-aware measurement
baseline; Tier-1 preflight and a full Tier-1 run that found 23 silent
hierarchical fallbacks and stopped (Task 12); a retry/timeout fix (Task
13) verified by a 524-case strict preflight (Task 15) and then a full
18,747-case strict run with zero fallbacks (Task 17); a reproducible,
fail-closed WISCO analysis utility (`eval/analyze_wisco_tier1.py`, Task
18) whose first real run correctly refused to score the legacy flat
baseline (§ below); a read-only audit of that refusal
(`FLAT_BASELINE_COVERAGE_AUDIT.md`, Task 19); a primary-source ILO
ISCO-08 catalogue reconciliation finding 20 non-standard codes, 14
missing codes, and 84 title mismatches
(`ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md`, Task 20); an
official-source runtime path plus a genuine, versioned, four-digit-only
flat comparator built against that verified catalogue (Task 21); and
the real, guarded local Qdrant collection-build execution for those
same five official collections (dual acknowledgement, no-overwrite,
ordered create+verify, manifest — `OFFICIAL_ISCO08_RUNTIME_AND_FLAT_COMPARATOR.md`,
Task 22). As of Task 22, no WISCO accuracy number had yet been
produced by any of these tasks — Task 18 produced zero metric output (a
gate failure, not a bug), Task 20 explicitly did not correct or
re-evaluate anything, and Tasks 21-22 built runtime/build infrastructure
only (zero Qdrant collections built, zero evaluations run, zero live
Qdrant connections at any point). `eval/local_runs/` and
`eval/local_catalogues/` (both gitignored) hold every raw artifact
these tasks produced.

### Tasks 23-37.1 — client-side deadline hardening, and the completed official-profile WISCO Tier-1 run

Tracked individually under `Documentation/AI_HANDOFF/CLAUDE_TASK_23_*`
through `CLAUDE_TASK_37_1_*`. Tasks 23-34.1 audited and hardened the
Qdrant retrieval transport's client-side deadline enforcement (a
reliability prerequisite, not a WISCO-specific change); Tasks 32/33/35
validated it under live, increasingly realistic preflights. **Task 36
then executed the official ILO 2021 ISCO-08 profile's Tier-1 run to
completion**: both flat and strict hierarchical retrieval, full
18,747-case WISCO v2 heldout split, no LLM reranking, zero errors/
retries/exceptions/fallbacks. **Task 37** produced the first offline
accuracy analysis and disclosed one read-only, out-of-scope Qdrant call
made outside the analyzer itself (never hidden — see the canonical
document below). **Task 37.1** independently reproduced that analysis
with zero Qdrant connection anywhere in its own execution and is the
clean-reproduction record. **A real WISCO accuracy number now exists**:
flat retrieval 21.19% exact 4-digit accuracy vs. strict hierarchical
retrieval's 10.35% (McNemar p ≈ 1.86e-301) — see
[OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md](OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md)
for the full, exact, citable record. This is a **controlled WISCO v2
benchmark result, not real Labour Force Survey validation**, and covers
only the non-reranked ISCO-08 tier — the reranking tier and any ISIC/
ISCED/SRE measurement on WISCO remain not run.

## What is explicitly out of scope, still

Task 05 implemented real ISIC Rev.4 / ISCED-F 2013 **hierarchical
retrieval** code (`ISICClassifier.classify(text,
method="isic_hierarchical_retrieval")` / `ISCEDClassifier.classify(text,
method="iscedf_hierarchical_retrieval")`), reusing the same generic
`backend/rag/hierarchy_engine.py` built for ISCO — see
[ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md](ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md).
Still out of scope:

- **No live Qdrant collections have been built or populated.** The code
  path exists and is hermetically tested, but until an operator runs
  `python -m backend.rag.build_standard_hierarchical_collections --standard
  {isic,iscedf} --execute`, every real call falls back to the existing
  keyword/rule pipeline under an explicit `*_hierarchical_fallback_*` label.
- **No accuracy measurement.** `evaluated=False` in the method registry for
  both methods; no manuscript claim about ISIC/ISCED-F hierarchical-RAG
  accuracy, latency, or improvement is supported yet.
- **No official-catalogue coverage claim.** The indexed node counts
  (21/68/118/134 for ISIC; 11/25/63 for ISCED-F) reflect this repository's
  currently embedded classifier tables only, not verified official-standard
  completeness — see `COVERAGE_AUDIT_GUIDE.md`.

Never describe ISIC/ISCED-F hierarchical retrieval as "validated",
"evaluated", or "run on real LFS data" in the manuscript until a future
build + evaluation pass produces a real, citable run manifest.
