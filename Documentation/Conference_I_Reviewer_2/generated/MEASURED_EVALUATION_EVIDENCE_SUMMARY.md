# Measured Evaluation Evidence Summary

Conference I Reviewer #2 response, Step 5: "execute a controlled evaluation
and ablation study using a user-supplied, locally stored, independently
labelled dataset." The user directed this run to be a **synthetic fixture
integration run only** (`eval/fixtures/synthetic_lfs_intake_package/`, 5
rows), not a substantive evaluation. Every table, figure, and manifest this
run produced carries `dataset_label=synthetic_or_operationally_realistic`,
`evaluation_status=measured_synthetic_fixture_only`, and
`manuscript_eligible=false`.

> **Results are from a controlled synthetic or operationally realistic
> evaluation and do not constitute real Labour Force Survey validation.**

Raw artifacts: `eval/local_runs/step5_synthetic_integration_20260807/`
(Git-ignored, not referenced by path content below beyond naming which
phase produced what — see the final report for exact paths).

> **Step 5.1 update (2026-08-07):** the `no_sre` config's ISIC/ISCED
> figures below (§2) reflect a since-fixed bug (`--sre off` incorrectly
> also disabled ISIC/ISCED classification — see
> `Documentation/Conference_I_Reviewer_2/SRE_COUPLING_BUGFIX.md`) and are
> **superseded** for any SRE-isolation comparison. See §8 for the post-fix
> results. The other 4 configs' figures are unaffected by the bug and
> remain as originally reported.

## 1. Implemented architecture facts

- ISCO-08 uses a genuine 4-stage hierarchical Qdrant retrieval pipeline
  (`backend/rag/hierarchical_store.py` / `hierarchy_engine.py`), optionally
  reranked by an LLM (`ISCOClassifier`). A `flat` single-stage retrieval
  mode and a `bm25` mode also exist for the ablation baseline.
- ISIC Rev.4 uses keyword lookup over a flat code table
  (`backend/agents/isic_classifier.py`) with optional LLM reranking — **not**
  hierarchical RAG (no ISIC Qdrant collections exist; confirmed via Phase B's
  Qdrant collection listing: ISCO-only).
- ISCED 2011 (attainment level) and ISCED-F 2013 (field of education) use a
  deterministic rule/keyword method (`backend/agents/isced_classifier.py`) —
  **not** hierarchical RAG.
- The Semantic Relation Engine (SRE) checks cross-standard coherence between
  ISCO/ISIC/ISCED outputs and can flag `NONE`/`LOW`/`MODERATE`/`HIGH`
  severity, optionally triggering escalation.
- The evaluation harness (`eval/run_eval.py`, `eval/ablation_runner.py`,
  `eval/manifest.py`, `eval/analyze.py`) captures per-case predictions,
  per-run manifests, accuracy with Wilson CIs, and (this pass) real
  operational metrics (latency, throughput, cost, HITL escalation rate).

## 2. Measured results (synthetic fixture, n=5, `heldout` split)

All 5 named ablation configs completed (`flat_baseline`,
`hierarchical_no_rerank`, `hierarchical_with_rerank`, `no_sre`, `with_sre`),
identical 5 test cases, embedding model `intfloat/multilingual-e5-small`,
reranker `ollama/llama3.2:1b` (local, zero-cost).

| Config | ISCO top-1 4-digit | 95% CI | ISIC section | ISCED level | Latency mean (ms) | HITL esc. rate | Cost (USD) |
|---|---|---|---|---|---|---|---|
| flat_baseline | 0.4 (2/5) | [0.118, 0.769] | 0.8 | 0.8 | 16269.97 | 0.2 | 0.0 |
| hierarchical_no_rerank | 0.4 (2/5) | [0.118, 0.769] | 0.8 | 1.0 | 232.72 | 0.2 | 0.0 |
| hierarchical_with_rerank | 0.4 (2/5) | [0.118, 0.769] | 0.8 | 1.0 | 17528.48 | 0.2 | 0.0 |
| no_sre | 0.4 (2/5) | [0.118, 0.769] | **not measured** (see §7) | **not measured** (see §7) | 17347.74 | 0.2 | 0.0 |
| with_sre | 0.4 (2/5) | [0.118, 0.769] | 0.8 | 1.0 | 19269.67 | 0.2 | 0.0 |

`n=5` — every confidence interval spans roughly [0.12, 0.77]; **these
numbers carry no statistical meaning and must never be read as a system
accuracy figure.** Cost is a real `$0.0` (local Ollama inference has no
per-token charge), not an unmeasured placeholder.

McNemar comparisons (paired, same 5 cases): `hierarchical_no_rerank` vs.
`hierarchical_with_rerank` — 0 discordant pairs, not measured (undefined at
b=c=0). `flat_baseline` vs. `hierarchical_with_rerank` — b=1, c=1, p=1.0.
`no_sre` vs. `with_sre` — 0 discordant pairs, not measured. All three:
`n=5` is far too small for any p-value here to mean anything; reported only
to prove the comparison pipeline itself works.

## 3. Dataset type and governance status

`dataset_label=synthetic_or_operationally_realistic`. The governance
validator (`eval/validate_real_lfs_governance.py::validate()`) was run and
returned `ok=True` with a warning that governance checks are scoped to
`approved_real_lfs_validation` only and were not applied here — correct,
since this dataset was never claimed as approved-real. The
evaluation-discipline validator (`eval/validate_evaluation_discipline.py`)
passed on the split manifest (distinct `dev`/`heldout` `split_id`s, heldout
frozen). Two Phase-A checklist items were **honestly recorded as failing**
for this fixture and do not gate a synthetic-fixture-only run: (a) reference
labels are fixture labels invented for test coverage, not independently
annotated; (b) the dataset file is version-controlled inside this Git repo
(a fixture, not real respondent data — appropriate for that reason alone).

## 4. Findings supported by evidence

- The evaluation, ablation, manifest, accuracy-analysis, and figure-export
  pipeline runs end-to-end against a real classifier stack (Qdrant +
  Ollama), producing correctly-shaped, honestly-labelled manifests with real
  (not null-by-default) latency/throughput/cost/HITL/SRE-severity figures.
- `hierarchical_no_rerank` is ~70-75x faster than any reranked configuration
  on this environment (232.72ms vs. 16,270-19,270ms mean latency) — expected
  and consistent with local CPU-bound LLM reranking via `llama3.2:1b` being
  the dominant cost, not the retrieval step itself. This is a real,
  measured, reproducible latency figure **on this specific 5-case fixture
  and this specific hardware/model** — not a claim about production-scale
  latency.
- A real, reproducible implementation bug was found: `--sre off` also
  disables ISIC/ISCED classification (not just the SRE coherence check) due
  to a shared conditional in `eval/run_eval.py` line 611. See
  `eval/local_runs/step5_synthetic_integration_20260807/FINDING_sre_isic_isced_coupling_bug.md`
  for the full root-cause analysis. This is exactly the kind of defect a
  controlled ablation study is meant to surface.

## 5. Findings NOT supported by evidence

- **No accuracy, latency, cost, or scalability claim about real Labour
  Force Survey data.** Nothing here says anything about how the system
  performs on real respondent text.
- **No claim that hierarchical retrieval beats flat retrieval**, that LLM
  reranking improves accuracy, or that the SRE changes escalation
  outcomes — the McNemar comparisons above are explicitly `not_measured` or
  statistically meaningless at `n=5`.
- **No claim about ISIC/ISCED accuracy** beyond section/level (no gold
  labels exist in this fixture for ISIC division/group/class or ISCED-F
  broad/narrow/detailed — correctly reported `not_measured` with a reason by
  `eval/analyze.py`, never fabricated).
- **No claim that the `no_sre` config's ISIC/ISCED accuracy is 0%** — it is
  *unmeasured* (blank predictions caused by the bug in §4), not a measured
  zero. The evidence-safe correction is already applied in §2's table.

## 6. Reviewer #2 comments now partially or fully addressed

Per `REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md`'s numbering:
- **#2 (novelty)**: the 5-config ablation pipeline is now proven to run
  end-to-end and produce real comparative manifests — still
  **Partially evidenced**; a substantive (non-fixture) run is required
  before citing any delta as evidence of novelty.
- **#4 (computational analysis)**: real latency/throughput/cost/HITL/SRE
  figures now exist for the first time in this codebase — but only against
  a 5-row synthetic fixture. Matrix status stays **Awaiting measurement**
  for the manuscript; this run demonstrates the instrumentation itself is
  correct and complete.
- **#3 (real LFS validation)**: unchanged, **Awaiting data** — this run
  used no real data and does not move this forward.

## 7. Remaining limitations

- `n=5` — not a statistically meaningful sample under any circumstance;
  every accuracy/CI/p-value figure above exists only to prove the pipeline
  computes them correctly, never as an accuracy claim.
- Reference labels were invented alongside the fixture, not independently
  annotated — even at real scale, this dataset could never satisfy the
  "independent reference labels" requirement without a genuine annotation
  process.
- ~~`--sre off` currently also disables ISIC/ISCED classification~~ —
  **fixed in Step 5.1**; see §8.
- `peak_process_memory_mb` remains unmeasured on every config (pre-existing
  gap: `CaseResult.peak_memory_mb` is never populated by `eval/run_eval.py`
  — unchanged since Step 4).
- ISIC sub-section (division/group/class) and ISCED-F (broad/narrow/
  detailed) accuracy remain unmeasured — no gold labels exist for them in
  any fixture or test set in this repo yet.
- `n=5` remains the limitation for every run in this document, pre- and
  post-fix alike — nothing here is a manuscript-eligible measurement.

## 8. Step 5.1 — SRE coupling bugfix and post-fix re-run

**Root cause** (`eval/run_eval.py:611`, pre-fix): ISIC/ISCED classification
was gated on the same `if sre_enabled and ...` conditional as the SRE
coherence check, so `--sre off` silently skipped ISIC/ISCED classification
too, not just the coherence check. Confirmed **evaluation-harness-only** —
`backend/agents/survey_orchestrator.py`'s production ISIC/ISCED calls have
no SRE-enabled gate at all. Full analysis:
`Documentation/Conference_I_Reviewer_2/SRE_COUPLING_BUGFIX.md`.

**Fix**: ISIC/ISCED classification now runs unconditionally whenever
`industry_text`/`education_text` are present; only `sre.analyse()` is gated
on `sre_enabled`. Two new `CaseResult` fields (`sre_status`,
`sre_status_reason`) make "disabled by configuration" explicit and
distinguishable from "evaluated, no violation found" (`sre_severity=
"NONE"`) — never a bare blank/zero standing in for either.

**Post-fix re-run**: all 5 configs re-run against the identical fixture,
dataset card, split manifest, embedding model, and reranker, at
`eval/local_runs/step5_1_synthetic_post_sre_fix_20260807T173123Z/`
(`post_fix_run_eval_py_sha256=2bcce91bb737daae1bf9e4c5e6afe4a8d374320896a28388916fb8fe11df79fc`).

| Config | ISCO top-1 4-digit | ISIC section | ISCED level | sre_status (all rows) | Latency mean (ms) |
|---|---|---|---|---|---|
| flat_baseline | 0.4 (2/5) | 0.8 | 0.8 | evaluated | 16525.55 |
| hierarchical_no_rerank | 0.4 (2/5) | 0.8 | 0.8 | evaluated | 253.86 |
| hierarchical_with_rerank | 0.4 (2/5) | 0.8 | 0.8 | evaluated | 18795.10 |
| **no_sre** | 0.4 (2/5) | **0.8** | **1.0** | **disabled_by_configuration** | 23260.55 |
| with_sre | 0.4 (2/5) | 0.8 | 0.8 | evaluated | 21725.20 |

`no_sre` now shows **zero blank ISIC/ISCED predictions** (5/5 rows
populated, vs. 0/5 pre-fix) and every row's `sre_status` is explicitly
`disabled_by_configuration` — never blank, never `"NONE"`. Still `n=5`,
still not manuscript-eligible; this table exists only to prove the fix
works, not as an accuracy claim. The original pre-fix run remains valid for
pipeline-integration evidence on the other 4 configs; only its `no_sre` vs.
`with_sre` ISIC/ISCED comparison is superseded — see that directory's
`SUPERSEDED_FOR_SRE_COMPARISON.md` and
`pre_fix_vs_post_fix_sre_comparison.json` for the full before/after record.

---

## 9. Task 36/37.1 — controlled WISCO v2 official-profile evidence (2026-08-10)

**This section is a distinct, separately-scaled evidence record. It must
never be merged, averaged, or otherwise combined with the `n=5`
synthetic-fixture numbers in §1-8 above** — those remain a
manuscript-ineligible pipeline-integration proof at `n=5`; this section
is a real, `n=18,747` controlled-benchmark measurement. The canonical,
authoritative version of everything in this section is
`Documentation/Conference_I_Reviewer_2/OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`
— read that document in full before citing any number below.

**Dataset and scope**: WISCO v2 controlled multilingual ISCO-08
benchmark (externally sourced, CC-BY-4.0, Zenodo DOI
`10.5281/zenodo.8262593`; not real Labour Force Survey respondent data),
full 18,747-case frozen heldout split, official ILO 2021 ISCO-08
catalogue profile (10/43/130/436 verified counts). Zero reranker, zero
LLM, zero ISIC, zero ISCED, and zero SRE activity in either comparison
arm — `reranker_fired=False`, zero tokens/cost, blank ISIC/ISCED
predictions, and `sre_status=not_applicable` on every one of the 37,494
rows across both systems.

**Headline** (exact 4-digit ISCO-08 match only):

| System | n | Correct | Accuracy | 95% Wilson CI |
|---|---:|---:|---:|---|
| Flat | 18,747 | 3,973 | 21.1927% | [20.6136%, 21.7836%] |
| Strict hierarchical | 18,747 | 1,941 | 10.3537% | [9.9256%, 10.7979%] |

**Paired comparison**: both correct 1,341; flat-only correct 2,632;
hierarchical-only correct 600; both incorrect 14,174; hierarchical minus
flat = -10.8391 percentage points; McNemar exact two-sided
p = 1.8573559951149046e-301.

**Raw-file identities**: flat CSV SHA-256
`d0692b4a87db11945dbd046ead79a32dcc36d715fbaa66fcc4e4853102be5f02`;
hierarchical CSV SHA-256
`b72193c8411b076df827abf2fa8bc6c2c72ac60f86799e435b94ff5eb237a8a4`;
heldout export SHA-256
`41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c`.
Evidence chain: Task 36 (raw run) →  Task 37 (first analysis; disclosed
one out-of-scope read-only Qdrant call, never hidden) → Task 37.1
(clean offline reproduction with zero Qdrant connection — the citable
clean-reproduction record).

**Strict limitation**: this is a controlled WISCO benchmark result, not
real Labour Force Survey validation, and does not resolve Reviewer #2
comment 3. It supports no ISIC, ISCED, SRE, cost, coverage,
generalization, or production-performance claim. The result is a
negative finding for hierarchical retrieval relative to flat retrieval
in this one non-reranked configuration, not a system superiority claim.
