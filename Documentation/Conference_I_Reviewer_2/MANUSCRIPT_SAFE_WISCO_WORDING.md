# Manuscript-Safe WISCO Wording

Concise, ready-to-paste, evidence-safe language for citing the Task
36/37.1 WISCO v2 official-profile controlled result. Every number here
must match `OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md` character-for-
character — that document is the source of truth; this one is phrasing
only.

**Terminology to use consistently:**
- **WISCO v2 controlled multilingual ISCO-08 benchmark** for the dataset
  (never "our dataset," "the LFS dataset," or "field data").
- **official ILO 2021 ISCO-08 catalogue profile** for the runtime/profile
  used (never just "ISCO-08" without the profile qualifier when citing
  this specific result).
- **controlled exact-code evaluation** for the measurement (never "field
  validation," "real-world evaluation," or "deployment test").
- **local single-run operational observation** for any latency number
  (never "performance benchmark," "throughput," or "SLA").

## Do not write

- "validated on real Labour Force Survey data" / "real-world validation"
- "field-tested" / "deployed and evaluated"
- any ISIC, ISCED, or Semantic Relation Engine accuracy/improvement claim
  tied to this result
- "with LLM reranking" or any reranking-comparison conclusion tied to
  this result
- **"hierarchical retrieval improved accuracy"** or any phrasing implying
  hierarchical retrieval outperformed flat retrieval — the measured
  result is the opposite
- "state-of-the-art," "novel architecture proven effective," or similar
  superiority language anchored to this result
- an official ISCO-08 coverage percentage or generalization claim beyond
  this benchmark and split
- a cost, memory, scalability, throughput, or production-latency/SLA
  claim
- "WISCO, an official ILO dataset" (WISCO is an externally published
  reference benchmark that uses ISCO-08 codes, not an ILO publication)
- "WISCO is ISCO-08" (WISCO is occupation-title reference data labelled
  with ISCO-08 codes, not the standard itself)
- the legacy 441 unit-group / 131 minor-group counts presented as if
  they were the official 436/130 catalogue profile's counts
- any statement implying Reviewer #2 comment 3 (real LFS data) is now
  resolved
- any statement omitting or minimizing Task 37's disclosed Qdrant
  scope-breach when Task 37 is cited (cite Task 37.1 as the clean
  reproduction instead; Task 37 may be mentioned only with the
  qualification that it is not the clean-offline record)

## Abstract / summary

Numeric evidence must never appear in abstract wording unless it opens
with the controlled-benchmark qualification. Safe pattern:

> On a controlled, externally sourced multilingual occupation-title
> benchmark (WISCO v2, 18,747 held-out cases, official ILO 2021 ISCO-08
> catalogue), exact four-digit ISCO-08 classification without LLM
> reranking achieved 21.19% accuracy with flat retrieval versus 10.35%
> with the strict hierarchical retrieval method — a controlled-benchmark
> observation, not a real Labour Force Survey validation result.

Unsafe: "Our system achieves 21% ISCO-08 accuracy" (drops the
controlled-benchmark qualifier and the flat-vs-hierarchical context).

## Methods

> We additionally evaluated exact four-digit ISCO-08 classification
> against the WISCO v2 controlled multilingual occupation-title
> benchmark (Zenodo DOI 10.5281/zenodo.8262593; 20,760 records, 2,013
> dev / 18,747 held-out, group-aware leakage-audited split), using the
> official ILO 2021 ISCO-08 catalogue profile (10 major, 43 sub-major,
> 130 minor, 436 unit groups). Two retrieval configurations were
> compared on the full held-out split with LLM reranking disabled in
> both: flat single-stage retrieval and the system's strict 4-stage
> hierarchical retrieval. Correctness was defined as exact match between
> the predicted and gold four-digit ISCO-08 code.

## Results

> Flat retrieval reached 21.19% exact-match accuracy (3,973/18,747; 95%
> Wilson CI [20.61%, 21.78%]); strict hierarchical retrieval reached
> 10.35% (1,941/18,747; 95% CI [9.93%, 10.80%]). The paired difference
> (hierarchical − flat) was -10.84 percentage points (McNemar exact
> two-sided p ≈ 1.86 × 10⁻³⁰¹ on 3,232 discordant pairs), indicating flat
> retrieval was substantially more accurate than strict hierarchical
> retrieval in this controlled, non-reranked, official-catalogue
> configuration.

Unsafe: any sentence that reports only the hierarchical number, or only
the flat number, without the paired comparison and the "in this
controlled, non-reranked configuration" qualifier.

## Limitations

> This result is drawn from an externally sourced controlled benchmark
> (WISCO), not real Labour Force Survey respondent data, and does not
> constitute field validation. It covers ISCO-08 exact-code
> classification only, with LLM reranking disabled; no ISIC, ISCED,
> Semantic Relation Engine, cost, memory, throughput, or production-
> latency claim is supported by this measurement. The observed
> hierarchical-retrieval underperformance in this configuration should
> not be read as a general claim about hierarchical retrieval; it is
> specific to this benchmark, this catalogue profile, and reranking-off
> operation.

## Response to Reviewer #2 comment 2 (novelty)

> We have now run a controlled, non-reranked comparison of flat versus
> strict hierarchical retrieval on the WISCO v2 benchmark's full 18,747-
> case held-out split (official ILO 2021 ISCO-08 catalogue). The result
> does not support a hierarchy-accuracy superiority claim: flat retrieval
> was more accurate in this configuration (21.19% vs. 10.35%, McNemar
> p ≈ 1.86 × 10⁻³⁰¹). A novelty claim for the paper's architecture must
> therefore rest on grounds other than a hierarchical-retrieval accuracy
> advantage — e.g. the documented engineering contributions in the
> classifier method registry, or a future reranked/ablated comparison
> that has not yet been run. We report this as a genuine, unfavorable
> finding rather than omit it.

## Response to Reviewer #2 comment 3 (real LFS data)

> We have not obtained real, permissioned Labour Force Survey data. The
> WISCO v2 controlled benchmark result reported above is a controlled
> exact-code evaluation on externally sourced occupation-title reference
> data, not real survey respondent data, and does not resolve this
> request. Obtaining a real LFS dataset and following the existing
> intake/governance checklist (`REAL_LFS_DATA_INTAKE_CHECKLIST.md`)
> remains required future work.

## Response to Reviewer #2 comment 4 (computational analysis)

> We report local, single-run operational observations from the WISCO
> v2 controlled run: flat query duration (mean 8.37 ms, p95 10.58 ms,
> p99 26.50 ms, max 119.21 ms) and hierarchical per-stage latency
> (stage 1–4 means 6.71 / 15.32 / 31.26 / 54.30 ms; 228,232 total
> hierarchical stage-level queries; zero retries, exceptions, fallbacks,
> or budget-exhaustion events). These are descriptive, single-machine,
> single-run measurements, not a throughput, scalability, memory, cost,
> or production-latency/SLA benchmark, which remain unmeasured.
