# Official WISCO v2 Tier-1 Controlled Results (ISCO-08, Official ILO 2021 Profile)

**Canonical, durable, manuscript-ready evidence note.** This is the single
authoritative source for the WISCO v2 official-profile exact-code
comparison result. Every other document in this repository that cites this
result must match the numbers here character-for-character.

## Scope label

**Controlled exact-code evaluation** of ISCO-08 occupation classification
on the **WISCO v2 controlled multilingual ISCO-08 benchmark**, using the
**official ILO 2021 ISCO-08 catalogue profile**. This is not a field
validation, not an accuracy claim on real respondent data, and not a
production performance measurement.

## Evidence chain

| Task | Branch | Final commit SHA | Role |
|---|---|---|---|
| Task 36 | `reviewer2-wisco-official-tier1-precise-deadline-full-results-20260809` | `514bc31f78cd5d66b018714c9f99ba591df0fdfe` | Produced the raw, validated flat and strict-hierarchical evaluator output on the full 18,747-case heldout split |
| Task 37 | `reviewer2-wisco-official-tier1-precise-deadline-analysis-20260810` | `85fcda8e7907b5dbd56780766d4642be9578e872` | First offline analysis of the Task 36 raw output. **Disclosed one read-only Qdrant metadata call made outside its own declared scope** (an operator-side preservation check, not part of the analyzer itself; caused no mutation). This disclosure remains permanent — see the dedicated section below. |
| Task 37.1 | `reviewer2-wisco-official-tier1-analysis-clean-reproduction-20260810` | `6b797f77291046f7a89268a33ab116e868ef7503` | **The clean, strictly offline reproduction of record.** Re-ran Task 37's unmodified analyzer against the identical Task 36 raw inputs, plus an independent from-scratch computation, with zero Qdrant connection or other live operation anywhere in the task. Every number in this document is the value both Task 37 and Task 37.1 agree on exactly. |

Source reports (read these for full methodology, not reproduced in full
here):
- `Documentation/AI_HANDOFF/CLAUDE_TASK_36_FINAL_REPORT.md`
- `Documentation/AI_HANDOFF/CLAUDE_TASK_37_FINAL_REPORT.md`
- `Documentation/AI_HANDOFF/CLAUDE_TASK_37_1_FINAL_REPORT.md`

## Task 37's disclosed scope breach — permanent, not repeated

Task 37's final report permanently records that it made one read-only
`get_collection()` Qdrant point-count call outside its own declared
no-live-operation scope, during an operator-side preservation
sanity-check (not by the analyzer script `eval/analyze_official_tier1.py`
itself). It caused no mutation. **This disclosure is not deleted,
revised, hidden, minimized, or reinterpreted anywhere in this
documentation.** Because of it, **Task 37.1 — not Task 37 — is the citable
clean-reproduction record**: Task 37.1 independently re-derived every
number below from the same local raw files with zero Qdrant connection
anywhere in its own execution, and its results matched Task 37's exactly.
Any citation of this evidence should reference Task 37.1 as the clean
reproduction, and may mention Task 37 only with this qualification.

## Raw-file identities (SHA-256)

| File | SHA-256 |
|---|---|
| Flat result CSV | `d0692b4a87db11945dbd046ead79a32dcc36d715fbaa66fcc4e4853102be5f02` |
| Hierarchical result CSV | `b72193c8411b076df827abf2fa8bc6c2c72ac60f86799e435b94ff5eb237a8a4` |
| Heldout export | `41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c` |

Raw and derived artifacts are Git-ignored and not committed to this
repository. Their locations, for an operator with local access to the
evaluation environment:
- Raw evaluator output root (Task 36):
  `eval/local_runs/wisco_official_tier1_precise_deadline_full_20260809T190821Z/`
  (`flat/20260809T191133Z_wisco_official_tier1_precise_deadline_full_flat.csv`,
  `hierarchical/20260809T192256Z_wisco_official_tier1_precise_deadline_full_hierarchical.csv`,
  `heldout_export_fresh.csv`)
- Task 37's derived analysis: `.../analysis_task37/` (`official_tier1_analysis.json`, `official_tier1_analysis.md`, `analysis_manifest.json`)
- Task 37.1's clean-reproduction derived analysis: `.../analysis_task37_1_clean_reproduction/` (same three file names)

## WISCO v2 split/provenance

WISCO v2 (`eval/local_benchmarks/wisco_isco08_v2_group_split/`) is an
**externally sourced, CC-BY-4.0, DOI-anchored (Zenodo `10.5281/zenodo.8262593`)
multilingual occupation-title-to-ISCO-08-code reference dataset** — not
collected via this project's survey flow and not real Labour Force Survey
respondent data. It is a group-aware, leakage-audited split (fixed seed
42, union-find duplicate-text grouping): 20,760 records, **2,013 dev /
18,747 heldout**, dataset hash
`a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c`. This
result uses the **frozen 18,747-case heldout split only** — the dev split
was never consulted for this evaluation. Full leakage-audit history:
`WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`.

## Official ILO 2021 ISCO-08 catalogue profile

The **official ILO 2021 ISCO-08 catalogue profile** (`official_ilo2021_v1`)
is a separate, versioned catalogue and retrieval profile distinct from
this project's original, legacy ISCO-08 implementation tables. Verified
official counts: **10 major groups, 43 sub-major groups, 130 minor
groups, 436 unit groups** — normalized catalogue SHA-256
`29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3`
(`eval/verified_catalogue_counts.yaml`). This is distinct from — and must
never be conflated with — the historical legacy implementation's
10/43/**131**/**441** counts (see `FLAT_BASELINE_COVERAGE_AUDIT.md` and
the Phase 1 Summary addendum). This result was produced entirely against
the official 10/43/130/436 profile; the legacy 131/441 profile was not
evaluated here.

## Exact correctness definition

```text
correct = predicted_isco_4digit == gold_isco_4digit
```

Exact four-digit ISCO-08 code match only. No prefix, major-group,
partial-code, or semantic-similarity matching of any kind.

## Eligibility and integrity criteria

Every value below was computed only after a 10-condition fail-closed
eligibility gate passed in full (Task 37, re-run independently in Task
37.1): input SHA-256 identity match; exactly 18,747 rows with unique case
IDs matching the heldout export exactly, for both systems; uniform
official flat (`flat_isco08_official_ilo2021_v1`) and hierarchical
(`hierarchical_isco08_official_ilo2021_v1`) method labels on every row;
every predicted and gold code a valid four-digit code present in the
official catalogue; zero non-blank row-level error; complete, parseable
hierarchical stage 1–4 evidence and telemetry with zero exception, zero
budget exhaustion, and zero stage latency above the configured 30,000 ms
cap; reranking off, zero tokens/cost, blank ISIC/ISCED, and
`sre_status=not_applicable` throughout both arms; and unchanged
historical evidence. Full per-condition results: Task 37 and Task 37.1
final reports.

## Headline results

| Metric | Official flat | Strict hierarchical |
|---|---:|---:|
| Heldout cases | 18,747 | 18,747 |
| Exact 4-digit ISCO-08 correct | 3,973 | 1,941 |
| Accuracy | 21.1927% | 10.3537% |
| 95% Wilson interval | [20.6136%, 21.7836%] | [9.9256%, 10.7979%] |

## Paired contingency and McNemar's test

| Both correct | Flat-only correct | Hierarchical-only correct | Both incorrect |
|---:|---:|---:|---:|
| 1,341 | 2,632 | 600 | 14,174 |

- n pairs: 18,747
- Accuracy difference (hierarchical − flat): **-10.8391 percentage points**
- McNemar statistic (min(b, c)): 600.0
- McNemar exact two-sided p-value: **1.8573559951149046e-301**
- Method: exact two-sided binomial test against p=0.5 on the discordant
  pairs only (b = flat-correct/hierarchical-wrong, c =
  flat-wrong/hierarchical-correct); statistic = min(b, c); p-value = sum
  of Binomial(n=b+c, p=0.5) pmf over every outcome at least as extreme as
  min(b, c) on either tail. No SciPy dependency; no normal/chi-square
  approximation. (`eval/analyze.py::mcnemar_test`, verified against an
  independent from-scratch log-space implementation in Task 37.1.)

## Descriptive subgroup tables (Task 37.1 derived result — copied verbatim)

**By input language:**

| Language | Flat n | Flat correct | Flat accuracy | Hierarchical n | Hierarchical correct | Hierarchical accuracy |
|---|---:|---:|---:|---:|---:|---:|
| ar | 3,762 | 550 | 14.62% | 3,762 | 368 | 9.78% |
| en | 3,818 | 1,450 | 37.98% | 3,818 | 860 | 22.52% |
| hi | 3,793 | 875 | 23.07% | 3,793 | 307 | 8.09% |
| tl | 3,766 | 545 | 14.47% | 3,766 | 200 | 5.31% |
| ur | 3,608 | 553 | 15.33% | 3,608 | 206 | 5.71% |

**By ISCO-08 major group (first digit of the gold 4-digit code):**

| Major group | Flat n | Flat correct | Flat accuracy | Hierarchical n | Hierarchical correct | Hierarchical accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 143 | 27 | 18.88% | 143 | 42 | 29.37% |
| 1 | 1,129 | 330 | 29.23% | 1,129 | 202 | 17.89% |
| 2 | 4,290 | 1,129 | 26.32% | 4,290 | 360 | 8.39% |
| 3 | 3,566 | 843 | 23.64% | 3,566 | 314 | 8.81% |
| 4 | 924 | 216 | 23.38% | 924 | 169 | 18.29% |
| 5 | 1,561 | 304 | 19.47% | 1,561 | 114 | 7.30% |
| 6 | 846 | 113 | 13.36% | 846 | 90 | 10.64% |
| 7 | 3,519 | 508 | 14.44% | 3,519 | 248 | 7.05% |
| 8 | 1,671 | 329 | 19.69% | 1,671 | 308 | 18.43% |
| 9 | 1,098 | 174 | 15.85% | 1,098 | 94 | 8.56% |

Subgroup values are **descriptive only** — no subgroup significance is
claimed, and none should be inferred without a separate, explicitly
authorized multiplicity-aware analysis. Full precision (per-subgroup
Wilson CIs, paired discordant counts) is in the Git-ignored
`official_tier1_analysis.json` at the paths listed above.

## Local-only operational observations

**These are local, single-run descriptions of this one evaluation run on
this one machine — not a performance benchmark, throughput figure, cost
figure, memory figure, scalability claim, or production SLA of any
kind.**

| Metric | Mean | Median | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|
| Flat query duration (ms) | 8.37 | 7.76 | 10.58 | 26.50 | 119.21 |
| Hierarchical stage 1 latency (ms) | 6.71 | 5.55 | 9.97 | 17.87 | 208.01 |
| Hierarchical stage 2 latency (ms) | 15.32 | 13.90 | 27.73 | 38.86 | 86.73 |
| Hierarchical stage 3 latency (ms) | 31.26 | 28.62 | 49.09 | 70.53 | 2,581.40 |
| Hierarchical stage 4 latency (ms) | 54.30 | 49.68 | 83.26 | 116.30 | 202.29 |

- Total hierarchical stage-level queries executed: **228,232**
- Zero retries, zero exceptions, zero fallbacks, zero unavailable
  outcomes, and zero stage-budget exhaustion anywhere in the Task 36 full
  run.
- No reranker, LLM, ISIC, ISCED, or SRE activity occurred in either
  comparison arm (reranking off throughout; zero tokens/cost; blank
  ISIC/ISCED predictions; `sre_status=not_applicable` on every row).

## Interpretation

**In this specific controlled configuration — the official ILO 2021
ISCO-08 profile, exact four-digit-code matching, no LLM reranking, full
18,747-case WISCO v2 heldout split — the flat retrieval comparator was
substantially more accurate than the strict hierarchical retrieval
method** (21.19% vs. 10.35% exact-match accuracy; McNemar
p ≈ 1.86 × 10⁻³⁰¹, an extremely strong statistical signal against the null
hypothesis of equal accuracy, given the very large discordant-pair count
at this sample size). This is a **negative finding for hierarchical
retrieval relative to flat retrieval in this one configuration** — it is
not evidence that the system as a whole underperforms, and it is not a
claim of flat-retrieval superiority in any other configuration
(reranked, different dataset, different catalogue profile, or real LFS
data).

## What this does not establish

- **Not** real Labour Force Survey validation and **does not** resolve
  Reviewer #2 comment 3's request for real-LFS-data evidence — WISCO is
  reference data, not survey respondent data.
- **Not** an ISIC, ISCED, or Semantic Relation Engine evaluation or
  improvement claim — none of those components were exercised.
- **Not** an LLM reranking comparison or conclusion — reranking was off
  in both arms.
- **Not** a hierarchy-accuracy improvement, novelty gain, or superiority
  claim for the system's hierarchical design — the measured result is
  the opposite (flat outperformed hierarchical here).
- **Not** an official coverage percentage or generalization claim beyond
  this exact benchmark and split.
- **Not** a cost, memory, scalability, throughput, production-latency,
  or SLA claim — the operational table above is a local single-run
  observation only.
- **Not** a claim that WISCO is an official ILO dataset, or that WISCO
  is the same as ISCO-08 — WISCO is an externally published reference
  benchmark that uses ISCO-08 codes as its label space.
- **Not** a claim that the historical legacy 441/131 unit/minor-group
  counts equal the official 436/130 catalogue profile's counts — they
  are different catalogues.
- **Not** a claim that Task 37's disclosed scope breach did not occur —
  it did, and remains disclosed (see above).

## Reviewer-response-ready wording

See `MANUSCRIPT_SAFE_WISCO_WORDING.md` for ready-to-paste safe and
explicitly unsafe phrasing for the abstract, methods, results,
limitations, and responses to Reviewer #2 comments 2, 3, and 4.
