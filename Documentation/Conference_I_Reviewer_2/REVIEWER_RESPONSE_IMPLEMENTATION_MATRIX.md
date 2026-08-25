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
| 2 | Novelty was unclear (framework mainly combines existing techniques). | `backend/agents/method_registry.py` (Section A) documents every classifier/agent's actual method, decoding config, and HITL-escalation wiring — several of which (e.g. `HITLQualityManager`'s independence from `ISCOClassifier.HITL_THRESHOLD`, `SemanticRelationEngine`'s deterministic 3-way crosswalk with per-violation provenance) are genuine, non-obvious engineering contributions, not just "combining existing techniques." `eval/ablation_runner.py` (Section E) can now empirically isolate the contribution of each design choice (hierarchical vs. flat retrieval, reranking on/off, SRE on/off) — and, per Step 4, its `--dry-run` mode has been verified end-to-end for all 5 configs against synthetic data (see `EVALUATION_READINESS_AND_DRY_RUN.md`), so the pipeline itself is confirmed ready. **Tasks 23-37.1 (2026-08-10) added a real, non-dry-run controlled comparison**: on the WISCO v2 benchmark's full 18,747-case heldout split (official ILO 2021 ISCO-08 profile, no LLM reranking), flat retrieval scored 21.19% exact-match accuracy vs. strict hierarchical retrieval's 10.35% (McNemar p ≈ 1.86e-301) — see `OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`. **This result does not support a hierarchy-accuracy superiority claim; the empirical novelty argument cannot rest on a hierarchical-retrieval accuracy advantage.** | `generated/classifier_method_registry.{json,md}` (real, generated from code); `eval/local_runs/dry_run_*/` (Step 4, synthetic dry-run only — infrastructure verification, not a measurement); `OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md` (Tasks 23-37.1, real non-dry-run measurement, non-reranked only) | **Partially evidenced** | A novelty claim now requires either (a) a reranked/ablated WISCO comparison isolating the reranker/SRE contribution (not yet run), or (b) grounding novelty in the documented engineering contributions (method registry) rather than a hierarchy-accuracy claim, since the measured hierarchy-vs-flat result is unfavorable to hierarchical retrieval. |
| 3 | Manuscript lacks validation on real Labour Force Survey data. | `eval/dataset_card_schema.py` (closed 3-value `dataset_label` vocabulary: `synthetic_or_operationally_realistic` / `approved_real_lfs_validation` / `invalid_incomplete_governance`; full 5-group `DatasetCard` schema — identity, governance, classification-label provenance, split discipline, privacy), `eval/validate_real_lfs_governance.py` (fail-closed validator with content-aware safeguards: rejects any card containing a synthetic/placeholder marker regardless of completeness, and any path-shaped field that resolves inside this Git repository), `REAL_LFS_VALIDATION_DATASET_CARD_TEMPLATE.md`, `REAL_LFS_DATA_INTAKE_CHECKLIST.md`, `ANNOTATION_AND_ADJUDICATION_GUIDE.md` (coder qualifications, mandatory double-coding + adjudication, ambiguous/multilingual-response handling), `eval/validate_evaluation_discipline.py` + `eval/split_manifest_schema.py` (Step 4 — rejects a missing/leaking split manifest, an absent dataset hash, or a "measured" run with no actual metrics). **Tasks 23-37.1's WISCO v2 controlled result (see row 2) uses externally sourced occupation-title reference data, not real LFS respondent data, and does not close this request.** | `eval/fixtures/synthetic_lfs_intake_package/` — a fully-filled-out SYNTHETIC intake package proven to still fail the approved-real gate (`eval/test_validate_real_lfs_governance.py`); `eval/fixtures/complete_approved_real_lfs_dataset_card_example.json` — a schema-complete TEST FIXTURE proving the accept path works. No real respondent data was added (by design). | **Awaiting data** | Obtain a real, permissioned LFS dataset; follow `REAL_LFS_DATA_INTAKE_CHECKLIST.md` end-to-end; run `eval/ablation_runner.py run --dataset-label approved_real_lfs_validation --dataset-card <your card>` (see `EVALUATION_READINESS_AND_DRY_RUN.md` §2d for the exact future command). |
| 4 | Computational analysis is missing (training/inference cost, memory, latency, scalability). | `eval/manifest.py` — full experiment-run manifest (hardware, OS, dependency versions, latency mean/p50/p95, throughput, cost, HITL escalation rate, retrieval candidate pool size, reranker invocation rate, SRE status); `eval/ablation_runner.py` wires it into every ablation run. Step 4 confirmed every one of these fields is either populated on a real run or `None` with an explicit reason — see `EVALUATION_READINESS_AND_DRY_RUN.md` §4's field-by-field inventory. **Tasks 23-37.1 (2026-08-10)**: real, controlled, non-reranked query/stage-latency measurements now exist from the WISCO v2 full-heldout run — see `OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`'s operational table. | `Documentation/Conference_I_Reviewer_2/generated/figure_data/latency_scalability.{json,csv}` — currently `no_manifests_found: true` (honest — no `ablation_runner.py` manifest run has been executed yet); `eval/local_runs/dry_run_*/` (Step 4 synthetic dry-run artifacts, infrastructure verification only); `OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md` — real local single-run flat-query and per-stage hierarchical latency distributions (mean/median/p95/p99/max), 228,232 stage queries, zero retries/exceptions/fallbacks/budget-exhaustion | **Partially evidenced** | Controlled local retrieval-latency measurements now exist (non-reranked only); still missing: memory (`peak_process_memory_mb` remains unpopulated), throughput/scalability under concurrency, reranked-configuration latency, production/deployment measurement, and any real-LFS-data measurement. |
| 5 | Abstract implies complete ISIC coverage rather than disclosing the real limitation. | `eval/coverage_audit.py`, `eval/standards_reference.yaml`, `eval/catalogue_importer.py` — computes real IMPLEMENTED unique code counts per level directly from the embedded data tables, and now distinguishes a sourced-but-**unverified** official count (transcribed from a cited ILO/UNSD/UNESCO source, see `STANDARDS_SOURCE_PROVENANCE.md`) from a **verified** official count (only ever produced by importing and validating a real catalogue file). **Task 20 (ISCO-08 only, not ISIC):** `eval/normalize_ilo_isco08_catalogue.py` + a primary ILO workbook import produced the project's first-ever **verified** entry, `eval/verified_catalogue_counts.yaml` (`{major: 10, submajor: 43, minor: 130, unit: 436}`, zero import issues) — see `ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md`. **Task 21 (ISCO-08 only):** built the official-source runtime path and a distinct, versioned, four-digit-only flat comparator (`backend/rag/official_isco08_catalogue.py`, `backend/rag/build_official_isco08_collections.py`, `HierarchicalISCOStore(profile=...)`, `ISCOClassifier(isco_catalogue_profile=...)`, `eval/run_eval.py --isco-catalogue-profile`) — see `OFFICIAL_ISCO08_RUNTIME_AND_FLAT_COMPARATOR.md`. **Task 22 (ISCO-08 only):** implemented the real, guarded `--execute` collection-build path (dual acknowledgement, local-only Qdrant, no-overwrite preflight, ordered create+verify, success/failure manifest), 26 hermetic tests against fakes only. No collection built, no evaluation run. | `generated/coverage_audit_isic_rev4_*.{json,csv,md}` — 134 unique 4-digit ISIC classes implemented (real, computed) vs. `official_count_unverified=419` (sourced via secondary corroboration of UNSD's ISIC Rev.4 publication, NOT primary-document-confirmed — see `STANDARDS_SOURCE_PROVENANCE.md`) vs. `official_count_verified=null` (no catalogue imported yet) → `coverage_percentage=null`. **ISCO-08's own catalogue (`backend/rag/load_full_isco.py`) is now known, code-for-code, to contain 20 non-standard codes and to be missing 14 real ones (`ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md` §4) — a real, quantified limitation, not yet disclosed anywhere in the manuscript.** | **Partially evidenced** | Obtain the real ISIC Rev.4 catalogue (same path ISCO-08 just used) and re-run the ISIC audit — ISIC's `official_count_verified` remains `null`, unchanged by anything below. For ISCO-08 specifically: Tasks 23-37.1 (2026-08-10) built the official ILO 2021 ISCO-08 catalogue profile's Qdrant collections and ran a real, non-reranked controlled evaluation against them on the WISCO v2 benchmark (`OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`) — this closes the "no fresh ISCO-08 evaluation run" gap for the official profile specifically, but is a **controlled WISCO benchmark result, not real-LFS validation**, is **ISCO-08 only** (no ISIC coverage implication), and does not by itself support a complete/exact ISCO-08 coverage claim in the abstract. |
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
ISIC/ISCED exists in WISCO). No benchmark measurement was run in Step
7A itself — Step 7A was audit and planning only. **The non-reranked
tier of this plan (Tier 1a/1b) has since been executed and completed —
see the dated section immediately below; the reranking tier (Tier 2)
remains not executed.**

## 2026-08-10 — Tasks 23-37.1: official-profile WISCO Tier-1 run completed (non-reranked tier)

Full detail: `OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md` (canonical) and
`MANUSCRIPT_SAFE_WISCO_WORDING.md` (safe phrasing). This section
summarizes; the canonical document is authoritative for exact numbers.

Tasks 23-34.1 built and hardened a genuine client-side Qdrant deadline
mechanism (superseding the server-side-only timeout Task 31 had
believed was client-side); Tasks 32/33/35 ran live preflights proving it
under increasingly realistic load; **Task 36** then ran the Phase D.1
Tier-1 plan above to completion — official ILO 2021 ISCO-08 profile,
full 18,747-case WISCO v2 heldout split, both flat and strict
hierarchical, no LLM reranking — with zero row-level errors, zero
retries, zero exceptions, and zero fallbacks in either arm. **Task 37**
produced the first offline accuracy analysis (disclosing one read-only,
out-of-scope Qdrant call made outside the analyzer itself — see the
canonical document's disclosure section). **Task 37.1** independently
reproduced that analysis with zero Qdrant connection anywhere in its own
execution and is the clean-reproduction record of citable numbers.

**Headline result**: flat retrieval 21.1927% exact 4-digit accuracy
(3,973/18,747); strict hierarchical retrieval 10.3537% (1,941/18,747);
McNemar exact two-sided p ≈ 1.8573559951149046e-301. **Flat outperformed
strict hierarchical retrieval in this controlled, non-reranked,
official-profile configuration — this is a negative finding for
hierarchical retrieval relative to flat retrieval here, not a system
superiority claim of any kind.**

**Limitations** (restated from the canonical document — read it in
full before citing): WISCO v2 is a controlled multilingual benchmark,
not real LFS data; this does not resolve reviewer comment 3; no ISIC,
ISCED, SRE, reranking, cost, memory, throughput, or production-latency
claim is supported; the operational latency figures are local single-run
descriptions only; B1 remains stale/quarantined and unrelated to this
result. The reranking tier (Phase D.2 above) has still not been run.

## Notes on methodology choices flagged as provisional

- **Section G's LOW severity tier** (`backend/agents/semantic_relation.py`,
  `_check_isco_isced`): a genuine 3-tier severity band was added
  (`gap<=1 -> LOW`, `gap==2 -> MODERATE`, `gap>=3 -> HIGH`, where `gap` is
  the ISCED level's distance outside the ISCO major group's expected
  range). This is deterministic and tested, but the exact boundary values
  are an engineering judgement call, not sourced from an external
  publication — review before citing the specific thresholds in the
  manuscript. See `EVALUATION_PROTOCOL.md`.

## 2026-08-24 update — real findings since this matrix's last pass (2026-08-10)

This matrix predates the work below; rows above are left unedited (per
this document's own discipline of not rewriting history), but the
following materially affects rows 2, 4, and 5's evidence base. Full
detail: `Documentation/PROJECT_FLOW_AND_STATUS.md` (the maintained,
current status document — read that first for anything not covered
here).

**Row 2 (novelty)** — a much stronger, now-repeated empirical finding
exists beyond the flat-vs-hierarchical result this row already cites:
across every reranker/retrieval-quality experiment run (reranker model
swap, corrective retry, and reranking on a stronger retrieval base),
**LLM reranking never changed the accuracy outcome — proven three
independent times, on two different retrieval bases and two different
cloud providers** (case-for-case identical predictions each time). The
one intervention that *did* move the number: a larger embedding model
(`intfloat/multilingual-e5-large`, 1024-dim vs. the default `-small`,
384-dim) — **+8.6pp (20.60%→29.20%), McNemar p≈1.77×10⁻⁶**, on a
500-case independent sample from the WISCO dev split; a full 18,747-case
heldout confirmation run (same split/config Tasks 23-37.1 used) was
started 2026-08-24 and its result should be folded in here once
complete. This is a materially stronger novelty grounding than what
existed at this matrix's last pass: not just "flat beats hierarchical"
but "we identified and confirmed *why* accuracy plateaus, and what
actually moves it."

**Row 4 (computational analysis)** — e5-large's real latency cost is now
measured: ~145ms/case vs. e5-small's ~28ms/case (both local, no
network), a real ~5.2x cost for the +8.6pp accuracy gain.

**Update, 2026-08-25**: two of the four gaps this row still listed were
already closed by data already sitting in the repo — never extracted
and reported back to this row. `peak_memory_mb`/`end_to_end_latency_ms`
turn out to be populated on every real `run_eval.py` run (`eval/
run_eval.py:1443`'s `_peak_rss_mb()`, a real psutil-based sample, not
fabricated) — just never pulled out and documented here.
- **Memory: real, not missing.** Retrieval-only (enriched catalogue +
  e5-large, the current best config, 18,747 cases): peak RSS 1822.9MB
  max, 1243.3MB mean. Reranked config (enriched catalogue + e5-small +
  Groq, 642 cases — a different embedding model, not a matched pair with
  the figure above): peak RSS 856.8MB max, 625.1MB mean.
- **Reranked-configuration latency: real, not missing.** Same reranked
  run: end-to-end latency mean 536.1ms, p50 556.0ms, p95 769.1ms, max
  2152.7ms. For contrast, the retrieval-only e5-large run: mean 208.1ms,
  p50 194.6ms, p95 291.1ms, p99 386.8ms.
- **Throughput/scalability under concurrency: also already closed**, by
  Module I's real load test (`PROJECT_FLOW_AND_STATUS.md` §12.3) —
  100% success through 42 concurrent users, fails at 50. This matrix's
  original row 4 text predates that finding and was never updated to
  point at it.

**Genuinely still missing, not resolved by anything above**: production/
deployment-environment measurement (everything above is a local dev
machine, not a deployed instance) and any real-LFS-data measurement
(gated on Module E, the pilot, unrelated to any of this).

**Row 5 (ISIC coverage)** — infrastructure progress only, coverage
numbers unchanged: ISIC/ISCED-F hierarchical retrieval Qdrant
collections were built and live-verified 2026-08-23 (previously
implemented but never populated); both classifiers gained the same
LLM-reranker parity ISCO-08 already had, plus a real, previously-
undiscovered bug was found and fixed in both (a keyword-confidence score
that was mathematically always 1.0, silently making the LLM-rerank path
unreachable since these classifiers were first written). **None of this
changes `official_count_verified=null` or the 134/419 and 63/~80
coverage figures** — no official ISIC/ISCED catalogue has been imported,
unchanged from this row's original "Awaiting" status. Accuracy against a
labelled ISIC/ISCED test set still does not exist — WISCO has no
independent gold labels for either standard (confirmed in
`CONTROLLED_BENCHMARK_AUDIT.md`), so closing this gap needs either an
external labelled source or a disclosed, small, hand-curated set — not
yet decided.

**New since this matrix's last pass, not covered by any existing row**:
`get_llm(TaskType.GENERAL)` now has a real, tested, local-first
automatic fallback chain (Ollama → Claude → Gemini → Groq → OpenRouter),
addressing operational resilience to any single provider's
unavailability — relevant context for row 6's LLM-role documentation if
cited.

**Module H (CrewAI architecture evaluation), built 2026-08-24 — relevant
to row 2 (novelty)**. This was the one item in the project's internal
backlog with zero prior work and no external blocker. Its original
"delegation correctness" framing doesn't apply to this codebase — as row
2 already establishes, this system never uses CrewAI's hierarchical
delegation (no `Process.hierarchical`, no `manager_agent`; every
`backend/agents/*.py` module is invoked by ordinary Python calling code
in `backend/api/survey_routes.py`, not by an LLM manager deciding who to
call). The real evaluable target — does the calling code invoke the
right agent at the right time, in the right order — is a correctness
property, not a statistical one, so it's implemented as
`backend/tests/test_orchestration_correctness.py` (8 tests, all passing)
rather than an accuracy metric with a confidence interval:
- **Full per-turn order** (`test_stage_order_matches_documented_pipeline`):
  asserts the real recorded call sequence
  (ConversationManager → ISCOClassifier → ISICClassifier →
  ISCEDClassifier → NationalityClassifier → SemanticRelationEngine →
  ValidationAgent → ContextMemory → AuditLogger) exactly matches
  `survey_routes.py`'s own documented Stage 1/3/4/4b-g comments.
- **6 conditional-gating tests**: each agent whose trigger condition
  isn't met this turn (no `industry` collected → ISICClassifier must not
  fire; wrong FSM state → ValidationAgent must not fire; etc.) is
  asserted absent from the call list, not just present-in-the-right-order
  when it does fire.
- **1 cross-function ordering test**
  (`test_ensure_isco_classification_runs_before_quality_review`) guards
  the one invariant `survey_routes.py` explicitly documents in a comment
  (line 1224) — that `_ensure_isco_classification` must run before
  `_trigger_quality_review` so HITL scoring sees the freshly-assigned
  ISCO code. This is exactly the kind of bug an output-only test
  (asserting a `QualityReview` row exists) would never catch.

**A real finding surfaced while building this**: the first draft of the
full-order test asserted `EmotionalIntelligence` fires between
`SemanticRelationEngine` and `ValidationAgent`, per
`survey_routes.py`'s own Stage 4f comment — it failed. `EmotionalIntelligence`
turns out to share the exact same `not skip_ner` gate as `LanguageProcessor`
(both silently skipped whenever `LFS_FAST_MODE=true`, not just on short
acknowledgement tokens as Stage 4f's comment alone would suggest). Not a
bug — correct, intentional behavior — but a real case of documentation
(a code comment) not fully describing a shared dependency, caught by
writing an order-sensitive test rather than trusting the comment. Added
as its own explicit test
(`test_emotional_intelligence_and_language_processor_share_skip_ner_gate`)
rather than left as an implicit side-effect of the main test passing.
No production code was changed — this is evaluation-only,
zero-risk-to-ship. Full test suite re-run after: 2,384 passed, 0 failed.

**Fourth reranking check (2026-08-24, later the same day) — relevant to
row 2 (novelty)'s reranker-contribution question.** The three prior
reranking checks (63-case and 500-case, two providers, two retrieval
bases — see `PROJECT_FLOW_AND_STATUS.md` §12 rows 5–11) all found
reranking made zero difference, byte-identical predictions every time.
This check used the real validation split (642 cases, built 2026-08-23,
first genuine use of it) on the production-default e5-small profile — a
combination the prior three hadn't covered — and found the first
non-zero effect: 18.22%→18.69% (+0.47pp), McNemar exact p=0.25 (not
significant), 638/642 predictions identical. Investigated the 4
differences: all 4 are the same occupation ("Air force captain," gold
code `0110`), the already-disclosed Armed Forces catalogue quirk (see
row 5's evidence column). 3 of 4 flipped wrong→right. **This does not
change row 2's evidence base or its "Partially evidenced" status** — a
non-significant, single-occupation effect isn't grounds for a reranker-
contribution novelty claim; if anything it strengthens confidence in the
prior null result by showing the method is sensitive enough to detect a
real (if narrow) effect when one exists, rather than being uniformly
insensitive. Full detail and numbers: `PROJECT_FLOW_AND_STATUS.md` §12
row 13; real artifacts:
`eval/results/dev_selection/validation_reranking_check/*.csv`.
