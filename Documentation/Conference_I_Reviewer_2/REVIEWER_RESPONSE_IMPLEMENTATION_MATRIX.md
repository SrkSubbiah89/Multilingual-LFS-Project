# Reviewer #2 Response — Implementation Matrix

Status column uses **only** these 5 labels, per the governing instructions
for this work: `Implemented but not evaluated`, `Partially evidenced`,
`Awaiting data`, `Awaiting measurement`, `Ready for paper update`.

No row below claims more than what has actually been built and verified in
this codebase as of this pass. Where a status depends on a real number
(coverage counts, latency, accuracy), that number is cited from a real
generated report under `generated/`, never invented for this table.

| # | Reviewer #2 comment | Code / docs / tests added | Evidence output | Status | Remaining action |
|---|---|---|---|---|---|
| 1 | Springer formatting was not followed. | — (not a code task) | — | **Ready for paper update** | Authors reformat the manuscript to the Springer template directly; no evidence-gathering blocks this. |
| 2 | Novelty was unclear (framework mainly combines existing techniques). | `backend/agents/method_registry.py` (Section A) documents every classifier/agent's actual method, decoding config, and HITL-escalation wiring — several of which (e.g. `HITLQualityManager`'s independence from `ISCOClassifier.HITL_THRESHOLD`, `SemanticRelationEngine`'s deterministic 3-way crosswalk with per-violation provenance) are genuine, non-obvious engineering contributions, not just "combining existing techniques." `eval/ablation_runner.py` (Section E) can now empirically isolate the contribution of each design choice (hierarchical vs. flat retrieval, reranking on/off, SRE on/off) — and, per Step 4, its `--dry-run` mode has been verified end-to-end for all 5 configs against synthetic data (see `EVALUATION_READINESS_AND_DRY_RUN.md`), so the pipeline itself is confirmed ready. | `generated/classifier_method_registry.{json,md}` (real, generated from code); `eval/local_runs/dry_run_*/` (Step 4, synthetic dry-run only — infrastructure verification, not a measurement) | **Partially evidenced** | Run the 5 ablation configs on a held-out **real or operationally-realistic** set via `eval/ablation_runner.py` (no `--dry-run`) and cite the resulting accuracy/latency deltas as the empirical novelty argument — no non-dry run has been executed yet. |
| 3 | Manuscript lacks validation on real Labour Force Survey data. | `eval/dataset_card_schema.py` (closed 3-value `dataset_label` vocabulary: `synthetic_or_operationally_realistic` / `approved_real_lfs_validation` / `invalid_incomplete_governance`; full 5-group `DatasetCard` schema — identity, governance, classification-label provenance, split discipline, privacy), `eval/validate_real_lfs_governance.py` (fail-closed validator with content-aware safeguards: rejects any card containing a synthetic/placeholder marker regardless of completeness, and any path-shaped field that resolves inside this Git repository), `REAL_LFS_VALIDATION_DATASET_CARD_TEMPLATE.md`, `REAL_LFS_DATA_INTAKE_CHECKLIST.md`, `ANNOTATION_AND_ADJUDICATION_GUIDE.md` (coder qualifications, mandatory double-coding + adjudication, ambiguous/multilingual-response handling), `eval/validate_evaluation_discipline.py` + `eval/split_manifest_schema.py` (Step 4 — rejects a missing/leaking split manifest, an absent dataset hash, or a "measured" run with no actual metrics). | `eval/fixtures/synthetic_lfs_intake_package/` — a fully-filled-out SYNTHETIC intake package proven to still fail the approved-real gate (`eval/test_validate_real_lfs_governance.py`); `eval/fixtures/complete_approved_real_lfs_dataset_card_example.json` — a schema-complete TEST FIXTURE proving the accept path works. No real respondent data was added (by design). | **Awaiting data** | Obtain a real, permissioned LFS dataset; follow `REAL_LFS_DATA_INTAKE_CHECKLIST.md` end-to-end; run `eval/ablation_runner.py run --dataset-label approved_real_lfs_validation --dataset-card <your card>` (see `EVALUATION_READINESS_AND_DRY_RUN.md` §2d for the exact future command). |
| 4 | Computational analysis is missing (training/inference cost, memory, latency, scalability). | `eval/manifest.py` — full experiment-run manifest (hardware, OS, dependency versions, latency mean/p50/p95, throughput, cost, HITL escalation rate, retrieval candidate pool size, reranker invocation rate, SRE status); `eval/ablation_runner.py` wires it into every ablation run. Step 4 confirmed every one of these fields is either populated on a real run or `None` with an explicit reason — see `EVALUATION_READINESS_AND_DRY_RUN.md` §4's field-by-field inventory. | `Documentation/Conference_I_Reviewer_2/generated/figure_data/latency_scalability.{json,csv}` — currently `no_manifests_found: true` (honest — no run has been executed against a real test set yet); `eval/local_runs/dry_run_*/` (Step 4 synthetic dry-run artifacts, infrastructure verification only) | **Awaiting measurement** | Run `eval/ablation_runner.py run ...` against `test_set_full130.csv` (`--split heldout`, no `--dry-run`) to produce a real, citable manifest. |
| 5 | Abstract implies complete ISIC coverage rather than disclosing the real limitation. | `eval/coverage_audit.py`, `eval/standards_reference.yaml`, `eval/catalogue_importer.py` — computes real IMPLEMENTED unique code counts per level directly from the embedded data tables, and now distinguishes a sourced-but-**unverified** official count (transcribed from a cited ILO/UNSD/UNESCO source, see `STANDARDS_SOURCE_PROVENANCE.md`) from a **verified** official count (only ever produced by importing and validating a real catalogue file). | `generated/coverage_audit_isic_rev4_*.{json,csv,md}` — 134 unique 4-digit ISIC classes implemented (real, computed) vs. `official_count_unverified=419` (sourced via secondary corroboration of UNSD's ISIC Rev.4 publication, NOT primary-document-confirmed — see `STANDARDS_SOURCE_PROVENANCE.md`) vs. `official_count_verified=null` (no catalogue imported yet) → `coverage_percentage=null` | **Partially evidenced** | Obtain the real ISIC Rev.4 catalogue, run `eval/catalogue_importer.py --standard isic_rev4 --catalogue ...`, re-run the audit, and use the resulting **verified** `coverage_percentage` (never `official_count_unverified`) to write the abstract's coverage-disclosure sentence. |
| 6 | Multiple LLM roles are unclear — the specific task handled by each model must be documented and measurable. | `backend/agents/method_registry.py` + `backend/agents/classifier_methods.py` — one row per (component, method), each with `model_name`, `decoding_config`, `fallback_behaviour`, `evaluated`, `affects_hitl_escalation`, all hand-verified against the actual code (not assumed). | `generated/classifier_method_registry.{json,md}` | **Ready for paper update** | Cite the registry table (or an adapted version of it) directly in the manuscript's methods section — the underlying facts are complete and code-verified. |
| 7 | Figures need better readability and resolution. | `eval/figure_exports/*.py` (classifier hierarchy, agent-role diagram, evaluation results, latency/scalability, coverage charts) export clean CSV/JSON — never raster screenshots. `FIGURE_DATA_EXPORT_GUIDE.md` documents building vector PDF/SVG/TikZ figures from them. | `generated/figure_data/*.{json,csv}` — hierarchy/agent-role/coverage exports have real data now; evaluation-results/latency exports report `no_manifests_found: true` until a real run exists | **Partially evidenced** | Build the actual vector figures from the exported data (see the guide); re-export evaluation-results/latency once a real ablation run exists. |
| 8 | References and technical claims need traceable evidence. | This matrix + every artifact under `generated/` + the classifier method registry + coverage audit + (once run) evaluation manifests together form a traceable evidence chain from manuscript claim → generated report → source code. | This document | **Partially evidenced** | Authors cross-check each remaining manuscript claim against the corresponding `generated/` artifact (or this matrix's "Remaining action" column) before submission. |

## Step 4 — evaluation-readiness dry run

Confirmed (via `EVALUATION_READINESS_AND_DRY_RUN.md`) that the evaluation,
ablation, instrumentation, and evidence-export pipeline is technically ready
for a future approved dataset: all 5 named ablation configs run end-to-end
in `--dry-run` mode with zero live Qdrant/Ollama/LLM/API/network calls,
producing correctly-shaped, honestly-labelled (`dataset_label=
synthetic_or_operationally_realistic`, `evaluation_status=
dry_run_not_measured`) manifests, null-metric reports, a coverage-audit
linkage, an SRE evaluation-schema check, and figure-export validation
output. This is **infrastructure verification only** — it does not move any
row above out of "Awaiting data" / "Awaiting measurement," and no number
from this pass may be cited in the manuscript.

## Step 5 / 5.1 — measured synthetic-fixture integration run + SRE coupling bugfix

Step 5 executed a real (non-dry-run) measured evaluation and 5-config
ablation study against the small `eval/fixtures/synthetic_lfs_intake_package/`
fixture (n=5), per the user's explicit "synthetic fixture integration run
only" instruction. Every artifact carries `dataset_label=
synthetic_or_operationally_realistic`, `evaluation_status=
measured_synthetic_fixture_only`, and `manuscript_eligible=false` (a new,
automatically-computed manifest field — see `eval/manifest.py`). Full
findings: `Documentation/Conference_I_Reviewer_2/generated/
MEASURED_EVALUATION_EVIDENCE_SUMMARY.md`. This is **infrastructure and
instrumentation verification only** — like Step 4, it does not move any row
above out of its current status, and no number from it may be cited in the
manuscript.

Step 5's `no_sre` config surfaced a real defect: `--sre off` also disabled
ISIC/ISCED classification (`eval/run_eval.py:611`, a shared conditional).
Step 5.1 fixed this (see `SRE_COUPLING_BUGFIX.md`), added 17 regression
tests (`eval/test_sre_isic_isced_coupling_fix.py`) plus a corrected existing
test, and re-ran all 5 configs post-fix
(`eval/local_runs/step5_1_synthetic_post_sre_fix_20260807T173123Z/`,
verified: `no_sre` now produces zero blank ISIC/ISCED predictions). The
original Step 5 run's `no_sre` vs. `with_sre` ISIC/ISCED comparison is
superseded by the post-fix run for that specific comparison; the original
run otherwise remains valid pipeline-integration evidence (see that
directory's `SUPERSEDED_FOR_SRE_COMPARISON.md`). Confirmed via code review
that this defect never affected production/API paths
(`backend/agents/survey_orchestrator.py`'s `_classify_isic()`/
`_classify_isced()` have no SRE-enabled gate of any kind) — evaluation-harness-only.

## Step 6 — controlled benchmark audit and preparation

Full audit: `Documentation/Conference_I_Reviewer_2/CONTROLLED_BENCHMARK_AUDIT.md`.

**The existing `eval/test_set_full130.csv` (130-case ISCO set) does not
qualify as a defensible controlled benchmark** — no documented label
source, no coder identity, no double-coding/adjudication evidence, no
independent split. It is preserved unchanged as a development/engineering
fixture only (status: `not_eligible_unknown_provenance`); no manuscript
claim should cite it as validated/benchmark evidence.

**A materially stronger, previously-dormant candidate was found**: WISCO
(`backend/evaluation/wisco/`), a real, externally-published, CC-BY-4.0,
DOI-anchored (Zenodo `10.5281/zenodo.8262593`) multilingual occupation-
title-to-ISCO-08-code dataset, downloaded and parsed in an earlier project
phase (`Documentation/Phase_2/Week_1/`) for a different purpose (knowledge-
base coverage comparison) and never used as an evaluation benchmark.
Status for ISCO-08: `conditionally_eligible_needs_documentation` (label
source, provenance, and independence are satisfied; double-coding/
adjudication evidence and a formal split/dataset card were missing and have
now been supplied — see below). Status for ISIC/ISCED: `not_eligible_
missing_independent_labels` (no direct gold codes for those standards
exist in this source).

**New in this pass** (Phase D/F): `eval/controlled_benchmark_schema.py`
(`BenchmarkRecord` schema), `eval/validate_controlled_benchmark.py`
(fail-closed eligibility validator — rejects absent provenance, self-
generated labels, split overlap, missing hashes, and any
`approved_real_lfs_validation` claim on a benchmark unconditionally),
`eval/build_wisco_isco_benchmark.py` (converts WISCO into a 20,760-record
`BenchmarkRecord` package, 2,005 dev / 18,755 heldout, deterministic
per-occupation split, `dataset_hash` computed, validated `ok=True`),
`eval/export_benchmark_to_run_eval_csv.py` (reformats to `eval/run_eval.py`'s
CSV schema). Package output lives at the Git-ignored
`eval/local_benchmarks/wisco_isco08_v1/` (regenerable from tracked source +
script; nothing new committed to Git beyond the tooling itself).

**Not done in this pass** (restriction 5 / Phase G): no substantive
ablation was run against the WISCO benchmark. `STEP_7_COMMAND.md` prepares
the exact command for a future Step 7, including an explicit scale warning
(18,755 heldout rows would take days at Step 5/5.1's observed per-case
reranked latency) and a reminder that dev-split parameter selection must
happen before any heldout run.

All Step 4/5/5.1 five-row synthetic-fixture outputs remain exactly as
produced — **integration-only, manuscript-ineligible** — and were not
altered, deleted, or relabelled by this Step 6 pass.

## Step 7A — WISCO leakage audit and group-aware split

Full detail: `Documentation/Conference_I_Reviewer_2/
WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`. A strict, deterministic (no
classifier/LLM/network call) leakage audit of the Step 6 WISCO benchmark
(`eval/audit_wisco_benchmark_leakage.py`) found that v1
(`eval/local_benchmarks/wisco_isco08_v1/`) — while free of raw source-key
leakage — had **4 cross-split exact-duplicate-text groups** (different
WISCO occupation keys sharing byte-identical titles and gold codes, split
apart by coincidence). A corrected, group-aware **v2**
(`eval/local_benchmarks/wisco_isco08_v2_group_split/`, fixed seed 42,
union-find duplicate-text grouping, 20,760 records / 2,013 dev / 18,747
heldout, dataset hash
`a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c`) is now
the valid split — re-audited clean (0 leaking groups, 0 cross-split
duplicates). v1 is retained unchanged as the pre-fix audit record.

A tiered evaluation plan was prepared (not executed): a full-heldout,
no-LLM tier (~75 min/config, both configs feasible) and a fixed,
deterministically-selected, stratified 500-record reranking subset
(~2.3–2.6 hours/config; the full heldout set with reranking would take
~90–98 hours/config and was explicitly NOT selected). The no-SRE/with-SRE
ablation axis is marked `not_evaluable_on_wisco_isco_only` (no gold
ISIC/ISCED exists in WISCO). **No benchmark measurement was run in Step
7A** — Step 7B execution requires separate explicit user approval,
especially for the reranking tier.

## Notes on methodology choices flagged as provisional

- **Section G's LOW severity tier** (`backend/agents/semantic_relation.py`,
  `_check_isco_isced`): a genuine 3-tier severity band was added
  (`gap<=1 -> LOW`, `gap==2 -> MODERATE`, `gap>=3 -> HIGH`, where `gap` is
  the ISCED level's distance outside the ISCO major group's expected
  range). This is deterministic and tested, but the exact boundary values
  are an engineering judgement call, not sourced from an external
  publication — review before citing the specific thresholds in the
  manuscript. See `EVALUATION_PROTOCOL.md`.
