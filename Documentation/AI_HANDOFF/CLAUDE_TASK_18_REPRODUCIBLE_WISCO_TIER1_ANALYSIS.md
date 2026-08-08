# Task 18 — Reproducible Controlled WISCO Tier 1 Analysis

## Purpose

Create a tested, reproducible analysis utility and use it exactly once on the two valid raw result CSVs from Task 17:

- full WISCO v2 heldout, model-free flat retrieval;
- full WISCO v2 heldout, model-free strict hierarchical retrieval.

This task is the first authorized accuracy/statistics analysis of Task 17’s raw artifacts. It may report controlled multilingual ISCO-08 benchmark results only after all input-integrity checks pass.

It must never represent WISCO results as:

- real Labour Force Survey validation;
- population-representative performance;
- ISIC, ISCED, or SRE performance;
- reranker or LLM performance;
- real-time or production performance;
- an all-language, all-country, or all-occupation guarantee.

The final report and documentation must call WISCO an **externally sourced controlled multilingual ISCO-08 occupation-title benchmark**. It is not respondent-level survey data.

## Required starting state

Fetch refs and verify before changing anything:

| Role | Branch | Required SHA |
|---|---|---|
| Task 17 evidence base | `reviewer2-wisco-tier1-strict-full-results-20260808` | `f7c40ab3675ab8a7ee239a490fbef261c2e5bc44` |

Require:

1. `origin/reviewer2-wisco-tier1-strict-full-results-20260808` exactly matches the SHA above.
2. Working tree is clean before branching.
3. Create and work only on:

```text
reviewer2-wisco-tier1-analysis-20260808
```

4. Push only this new branch.
5. Do not merge, rebase, reset, clean, stash, pull, force-push, or create a pull request.
6. Do not change protected/prior branches.

## Permitted and prohibited operations

### Permitted

1. Read Task 17’s local ignored output root:

```text
eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/
```

2. Read the canonical WISCO v2 heldout source:

```text
eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv
```

3. Write and run a deterministic offline analysis utility and hermetic tests.
4. Write analysis outputs only under a new ignored local output subdirectory, for example:

```text
eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/analysis_task18/
```

5. Add the code, tests, controlled-results document, reviewer matrix update, and final handoff report specified below.

### Prohibited

Do not run:

- `eval/run_eval.py` or any other prediction/evaluation run;
- any Qdrant query or mutation;
- SentenceTransformer/model load/download;
- Ollama, CrewAI, LLM, paid API, or external inference;
- a reranker;
- WISCO data rebuild, split regeneration, export, mutation, or deletion;
- B1 re-freeze, B2 sweep, ISIC/ISCED evaluation, SRE evaluation, or another benchmark;
- a new result-generating script outside the defined offline analysis utility.

Do not alter raw CSVs, raw manifests/logs, pre-run records, Task 17 report, code under test, evaluation configuration, or labels to make analysis pass.

If raw-input validation fails, do not calculate or report accuracy/statistics. Record the failure in the final report, commit only permitted code/test/report documentation, and stop.

## Required raw-input provenance and integrity gate

Before scoring, validate and record all of the following in the analysis output provenance JSON:

1. The canonical input CSV exists, has SHA-256 exactly:

```text
41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c
```

and has exactly 18,747 rows in canonical source order.

2. The Task 17 flat and hierarchical raw CSV paths exist exactly as identified by the Task 17 report:

```text
eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/flat/20260808T115438Z_wisco_v2_tier1_flat_model_free.csv
eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/hierarchical/20260808T153325Z_wisco_v2_tier1_hierarchical_strict_model_free.csv
```

3. Record SHA-256 for both result CSVs before analysis and do not modify them.
4. Both result files have exactly 18,747 rows, unique case IDs, and case IDs in exactly the same order as each other and the canonical heldout input.
5. Every raw output row has nonblank gold and predicted four-digit ISCO values that are valid four-digit numeric ISCO-08 code strings. Derive/check 1-, 2-, and 3-digit prefixes directly from the valid four-digit codes rather than trusting a mismatched auxiliary field.
6. Every `error` field is blank.
7. If an `evaluation_status` field exists, every row must carry the measured status expected for a real non-dry-run run. Do not silently accept dry-run or synthetic status.
8. Flat rows must be `flat_semantic`, must not have reranking fired, must have no reranker model, and must carry no nonzero LLM-cost or token field.
9. Hierarchical rows must all have method labels beginning `hierarchical_`, must have no fallback/missing-method row, no reranking, no reranker model, and no nonzero LLM-cost or token field.
10. For every hierarchical row, validate the same stage-evidence field contract used by the Task 13 strict guard in `eval/run_eval.py`: all four stage candidate fields must be non-empty valid JSON lists and all recorded stage latencies must be finite numeric values at or below 30,000 ms.
11. All 24 Task 12 known-risk IDs must appear in the hierarchical raw output and pass the same method/evidence/latency/error checks.
12. Confirm the canonical reference has no nonblank ISIC or ISCED gold label for any of the 18,747 rows, and record `sre_status` is consistently not applicable if that column exists.
13. Read Task 17’s recorded Qdrant before/after-count evidence as provenance only. Do not make any Qdrant call in this task.

Fail closed on any discrepancy. The utility must emit a human-readable error naming the violated invariant and affected case IDs where applicable. It must not produce a partially measured result bundle.

## New reproducible analysis utility

Add:

```text
eval/analyze_wisco_tier1.py
```

The utility must be deterministic and offline. It must use Python standard library plus dependencies already in the project. It must not call a network, Qdrant, model, LLM, or subprocess.

### Required CLI

Implement explicit arguments:

```text
--flat-csv <path>
--hierarchical-csv <path>
--reference-csv <path>
--out <directory>
--expected-reference-sha256 <hash>
--expected-n 18747
--expected-hierarchical-method-prefix hierarchical_
--max-stage-latency-ms 30000
```

Validate all inputs before writing final analysis outputs. The CLI must exit nonzero on a failed gate.

### Required statistics

Use transparent, documented implementations. Reuse the project’s existing Wilson and exact-McNemar definitions from `eval/analyze.py` where practical, or implement mathematically identical tested functions without changing `eval/analyze.py`.

For **each system**, calculate exact-match ISCO accuracy at:

- 1 digit;
- 2 digits;
- 3 digits;
- 4 digits.

For every accuracy, report:

- `n`;
- number correct;
- proportion;
- 95% Wilson score interval using \(z = 1.96\).

For the 4-digit metric, calculate the same fields for each observed WISCO language from the reference CSV. The language field must come from the canonical reference CSV, not inferred silently from an ID suffix. Also report the unweighted macro-average of the five language accuracies, clearly labelled as an unweighted descriptive macro-average with no pooled confidence interval.

For the paired 4-digit comparison, calculate and report:

- flat correct / hierarchical correct;
- both correct;
- flat correct and hierarchical wrong (`b`);
- flat wrong and hierarchical correct (`c`);
- both wrong;
- absolute accuracy difference \(A_{\text{hierarchical}} - A_{\text{flat}}\), in percentage points;
- exact two-sided McNemar p-value based only on `b` and `c`;
- the number of discordant pairs.

State exact McNemar’s null hypothesis: the two systems have equal marginal probability of a correct four-digit prediction on these paired WISCO cases. Do not call this a causal or real-world result.

For execution efficiency, calculate **descriptive** latency distributions from non-empty, finite per-row `end_to_end_latency_ms`:

- n;
- arithmetic mean;
- median;
- p95, using the deterministic nearest-rank definition `ceil(0.95*n)`;
- maximum.

For hierarchical only, additionally report the same descriptive fields for each available stage-latency field. Preserve units in milliseconds. Do not calculate cost, throughput, a latency significance test, or a speedup claim. Explain that these are local-machine measurements under the Task 17 environment, not a real-time or deployment guarantee.

Also report operational traceability counts:

- total `keyword_anchor_retry_used=true` rows, if the field exists;
- cross-tab of retry used versus final correct/incorrect four-digit prediction;
- observed `stage1_source` counts, if the field exists.

These are diagnostic behavior counts, not accuracy evidence for a separate method.

### Required outputs

Under `--out`, produce:

```text
provenance.json
wisco_tier1_metrics.json
wisco_tier1_metrics.md
```

The JSON must be machine-readable and contain:

- input paths/hashes;
- validation-gate outcomes;
- timestamp;
- analysis version/script hash;
- full and per-language accuracy metrics;
- paired contingency/McNemar results;
- latency descriptives;
- diagnostic traceability counts;
- complete limitations/evidence-boundary fields.

The Markdown must be a clear, concise analysis artifact with:

1. scope and evidence boundary;
2. input provenance and validation gates;
3. overall 1/2/3/4-digit ISCO accuracy table;
4. 4-digit language table;
5. paired 4-digit comparison and McNemar result;
6. latency descriptives with the non-real-time caveat;
7. diagnostic traceability;
8. limitations and prohibited inferences.

Do not write a generic “Sources” section or fabricate external citations. Do cite the local Task 17 raw artifact paths and the controlled WISCO documentation in prose only as internal provenance, not as external source claims.

## Hermetic tests

Add:

```text
eval/test_analyze_wisco_tier1.py
```

Use small temporary CSV fixtures constructed in the test. Tests must not read the 18,747-row local data or use network/model/Qdrant resources.

Cover at least:

1. correct valid-pair analysis and hand-checkable 1/2/3/4-digit results;
2. Wilson interval and exact McNemar expected values;
3. per-language grouping from reference field;
4. key/order mismatch rejection;
5. duplicate ID rejection;
6. invalid/missing four-digit code rejection;
7. nonblank row error rejection;
8. flat wrong-method/reranker/cost/token rejection;
9. hierarchical fallback/wrong-method rejection;
10. hierarchical malformed/empty stage-evidence rejection;
11. hierarchical over-cap/non-numeric stage-latency rejection;
12. known-risk missing or failed-row rejection;
13. dry-run/non-measured evaluation-status rejection when status field is present;
14. output JSON/Markdown schema presence and deterministic values;
15. input hashes are recorded and source CSVs remain unchanged after analysis.

Do not weaken existing tests or modify existing code merely to make the new analysis tool work.

## Documentation updates

Add:

```text
Documentation/Conference_I_Reviewer_2/WISCO_TIER1_CONTROLLED_RESULTS.md
```

Update:

```text
Documentation/Conference_I_Reviewer_2/REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md
Documentation/Conference_I_Reviewer_2/README.md
```

Only after the analysis utility completes successfully and its raw-input gates pass, these documents may state the measured controlled WISCO ISCO-08 results using values reproduced exactly from the generated local JSON.

Every document must contain these limitations clearly:

1. WISCO is externally sourced occupation-title reference data, not real LFS respondent data.
2. The results cover ISCO-08 only, not ISIC, ISCED, SRE, reranking, LLM performance, or system-wide survey validation.
3. Performance/latency values are descriptive local-machine measurements, not real-time or production guarantees.
4. The full 18,747-row results use a leakage-audited heldout split and model-free retrieval, but are still not evidence of national representativeness or field deployment.
5. The historical B1 baseline remains quarantined and unrelated to this controlled WISCO comparison.

Do not claim that the full project or every Reviewer #2 comment is now fully satisfied. Use an evidence-status table that distinguishes:

- addressed by controlled WISCO ISCO-08 evidence;
- improved implementation/readiness only;
- still blocked by absence of approved real LFS data or independent ISIC/ISCED/SRE labels.

## Test and reporting requirements

Run:

```bash
pytest eval/test_analyze_wisco_tier1.py eval/test_analyze.py eval/test_run_eval_b2.py eval/test_require_genuine_hierarchical.py eval/test_docs_consistency.py -q
pytest backend/tests eval/ -q
```

Report exact results. If tests fail, do not hide, skip, xfail, weaken, or repair unrelated failures. State exact attribution.

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_18_FINAL_REPORT.md
```

The final report must include:

1. source SHA, analysis branch, final commit SHA, and push confirmation;
2. exact changed tracked files;
3. raw input paths and SHA-256 values;
4. complete raw-input validation-gate outcomes;
5. exact analysis command;
6. ignored output bundle path;
7. all measured values from the generated JSON, including accuracy, Wilson intervals, language results, paired contingency counts/McNemar p-value, latency descriptives, and traceability counts;
8. focused and full test results;
9. confirmation that no evaluation, Qdrant, model, LLM, network, data mutation, or source prediction run occurred;
10. protected-branch and clean-tree confirmation;
11. a clear manuscript-safe wording example and a clear unsafe-wording example;
12. a reviewer-comment evidence-status summary;
13. a direct statement that results are controlled multilingual ISCO-08 benchmark evidence only.

Commit and push only code, tests, documentation, and the final report. Keep raw result bundles ignored. Stop after the final report is pushed. Do not start paper drafting, real-LFS work, ISIC/ISCED/SRE evaluation, or another benchmark.
