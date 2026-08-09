# Task 37: Official Tier-1 Fail-Closed Analysis

## Purpose

Analyze the two validated raw outputs from Task 36, only after independently confirming that they remain eligible for analysis. Produce reproducible controlled-benchmark statistics for the official ILO 2021 ISCO-08 flat versus strict-hierarchical comparison.

This is an offline analysis task. It must not run an evaluator, contact Qdrant, load an embedding model, call an LLM or reranker, modify a dataset, or update manuscript-facing documents.

## Evidence basis and source

- Required base branch: `reviewer2-wisco-official-tier1-precise-deadline-full-results-20260809`
- Required base SHA: `514bc31f78cd5d66b018714c9f99ba591df0fdfe`
- New branch: `reviewer2-wisco-official-tier1-precise-deadline-analysis-20260810`
- Fetch `origin` and verify both local base and `origin/reviewer2-wisco-official-tier1-precise-deadline-full-results-20260809` equal the required SHA before branching.
- Task 36 raw-output identities to verify exactly before analysis:
  - Flat CSV SHA-256: `d0692b4a87db11945dbd046ead79a32dcc36d715fbaa66fcc4e4853102be5f02`
  - Hierarchical CSV SHA-256: `b72193c8411b076df827abf2fa8bc6c2c72ac60f86799e435b94ff5eb237a8a4`
  - Heldout export SHA-256: `41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c`
- Task 36 final report: `Documentation/AI_HANDOFF/CLAUDE_TASK_36_FINAL_REPORT.md`

Do not merge, rebase, reset, clean, stash, pull, force-push, or open a PR. Do not touch protected or prior-task branches.

## Strict scope

Allowed:

1. Read the Task 36 raw flat CSV, hierarchical CSV, heldout export, official verified catalogue, and Task 36 report.
2. Run deterministic, offline Python analysis using the repository's existing `eval/analyze_wisco_tier1.py` functionality and its existing Wilson/McNemar helpers where applicable.
3. Make the minimum additive, tested code change only if the existing analyzer cannot correctly analyze the Task 36 official-profile outputs.
4. Write derived analysis artifacts only under the Task 36 Git-ignored local-run output root.
5. Commit a final report and, only if required, the smallest analyzer/test changes needed for correct reproducible analysis.

Forbidden:

- any Qdrant connection or collection operation;
- any evaluator, benchmark, model, SentenceTransformer, Ollama, LLM, reranker, external API, or network call;
- any rerun, retry, data rewrite, filtering, relabelling, resampling, tuning, or cherry-picking of cases;
- any change to raw CSVs, WISCO package/splits, official catalogue, production classifier/retrieval code, B1/B2 configuration, or historical raw evidence;
- any manuscript, paper, figure, README, reviewer-response, or claim-matrix update.

## Mandatory analysis eligibility gate

Before computing any statistic, fail closed unless every condition passes:

1. The three Task 36 SHA-256 hashes listed above match exactly.
2. Each output has exactly 18,747 rows, unique `case_id` values, and a case-ID set exactly equal to the heldout export.
3. The flat CSV has exactly `flat_isco08_official_ilo2021_v1` in every `pred_method`.
4. The hierarchical CSV has exactly `hierarchical_isco08_official_ilo2021_v1` in every `pred_method`.
5. Every prediction in both files is a valid four-digit code in the verified official ILO catalogue.
6. Both files have zero non-blank row-level error.
7. The hierarchical file has complete, parseable stage-1 through stage-4 evidence and stage telemetry for every row, zero fallback/unavailable/synthetic result, zero exception, zero budget exhaustion, and zero stage latency above 30,000 ms.
8. Both files show reranking off, zero tokens/cost, blank ISIC/ISCED outputs, and `sre_status=not_applicable` throughout.
9. The heldout gold ISCO-08 codes are valid four-digit official codes, and heldout gold ISIC/ISCED fields are blank.
10. The Task 36 report remains byte-identical to the base commit, and all previous-task evidence required by Task 36 remains unchanged.

If any check fails:

- write an explicit failure manifest under the Git-ignored Task 36 output root;
- report `OFFICIAL_TIER1_ANALYSIS_COMPLETED: no`;
- commit only the final report;
- do not compute, print, or infer any accuracy, interval, comparison, or subgroup statistic.

## Analysis method

If the gate passes, calculate deterministic results using every one of the 18,747 heldout rows, with exact four-digit ISCO-08 match as the sole correctness definition:

```text
correct = predicted_isco_4digit == gold_isco_4digit
```

Do not use prefix, major-group, partial-code, semantic-similarity, or manually corrected matching.

### Required headline metrics

For flat and hierarchical systems separately, report:

- correct count / 18,747;
- exact-match accuracy;
- two-sided 95% Wilson confidence interval;
- incorrect count;
- method label;
- verified input/output SHA-256 values.

For the paired comparison, report:

- both correct;
- flat-only correct;
- hierarchical-only correct;
- both incorrect;
- hierarchical minus flat exact-match accuracy difference in percentage points;
- exact two-sided McNemar p-value calculated from the discordant pairs only;
- the exact McNemar implementation/formula used.

Use a deterministic, dependency-light implementation. If SciPy is unavailable, use a mathematically correct exact binomial calculation without silently falling back to an approximation. Test the implementation against known values.

### Required subgroup reporting

For each system, report numerator, denominator, accuracy, and two-sided 95% Wilson interval for:

1. each of the five input languages present in the heldout set;
2. each ISCO-08 major group, derived strictly from the first digit of the gold four-digit code.

For each language and major group, report the paired flat-only/hierarchical-only discordant counts. Do not claim subgroup significance unless an exact, multiplicity-aware analysis is separately authorized. These subgroup metrics are descriptive only.

### Required operational summaries

Calculate and report, as descriptive observed properties of this one local run only:

- flat query duration mean, median, p95, p99, and maximum;
- hierarchical stage 1–4 latency mean, median, p95, p99, and maximum;
- total hierarchical stage-query count;
- retry, exception, fallback, unavailable, and budget-exhaustion counts;
- retrieval path distributions already represented in the raw CSV, including `stage1_source`.

Do not generalize these runtime values to production performance, user latency, cost, or deployment capacity.

## Reproducibility outputs

If the gate passes, write these Git-ignored derived artifacts under the Task 36 output root:

1. `analysis_manifest.json` with input paths, input SHA-256s, analyzer version/commit, exact eligibility results, timestamp, and output SHA-256s;
2. `official_tier1_analysis.json` with all headline, paired, subgroup, and operational metrics;
3. `official_tier1_analysis.md` with human-readable tables and an explicit limitations block.

The JSON must retain full precision. Format the Markdown report without rounding away the underlying counts. Do not commit raw or derived local-run artifacts.

## Tests and verification

Before analysis:

```bash
python -m pytest backend/tests eval/ -q
```

Expected baseline is `2174 passed, 1 deselected, 1 warning`. If no code changes are needed, this exact result must hold before and after analysis.

If the analyzer is changed:

1. add focused hermetic tests for the new/changed behavior, including:
   - official profile method-label acceptance;
   - SHA/integrity mismatch fail-closed behavior;
   - invalid/coarse code rejection;
   - row-ID mismatch rejection;
   - known Wilson interval result;
   - known exact two-sided McNemar result;
   - paired-contingency accounting;
   - no analysis output before eligibility passes;
2. run those focused tests;
3. run the complete suite after changes;
4. report the exact pass counts and why any count differs from 2,174.

No test may use live Qdrant, network, model loading, or real raw evaluator files.

## Final report

Create:

`Documentation/AI_HANDOFF/CLAUDE_TASK_37_FINAL_REPORT.md`

Required content:

1. `OFFICIAL_TIER1_ANALYSIS_COMPLETED: yes` or `no`;
2. branch, verified base SHA, final SHA, push confirmation, and clean working tree;
3. exact scope and confirmation of zero live/network/model/evaluator activity;
4. the complete eligibility-gate table, including all input hashes;
5. if passed, headline metric table, paired contingency table, McNemar method/p-value, subgroup tables, and local operational summaries;
6. full paths and SHA-256s of derived Git-ignored output artifacts;
7. exact test commands and outputs;
8. preservation evidence for Task 36 raw outputs/report, B1/B2 safety, and protected branches;
9. a strict limitations statement:
   - this is a controlled multilingual WISCO ISCO-08 benchmark, not real Labour Force Survey validation;
   - it evaluates ISCO-08 exact-code prediction only;
   - it supports no ISIC, ISCED, SRE, cost, coverage, generalization, or real-field-performance claim;
   - the observed latency figures are local-run descriptions, not production SLAs;
   - B1 remains stale/quarantined;
   - manuscript updates remain a separate follow-on task.

Push only the new branch and stop after the report.
