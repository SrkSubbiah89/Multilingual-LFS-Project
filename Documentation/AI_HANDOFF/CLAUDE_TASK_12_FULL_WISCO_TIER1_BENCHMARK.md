# Task 12 — Execute the Full WISCO Tier-1 Controlled ISCO-08 Benchmark

## Explicit authorization

This task is authorized to execute the first full controlled benchmark after Task 11’s readiness gate passed.

It will run **two complete, paired retrieval-only evaluations**, each over the frozen WISCO v2 heldout split of **18,747** cases, for **37,494 local classifications total**:

1. Flat ISCO retrieval, no reranking.
2. Genuine four-stage hierarchical ISCO retrieval, no reranking.

The run will use only the existing local Qdrant ISCO collections and the local embedding model. It will take several hours of local compute and write Git-ignored raw outputs, manifests, and analyses. It is expressly authorized to do that.

It is not authorized to use Ollama, any LLM, paid API, real-LFS data, or mutate Qdrant collections.

## Scientific scope and claim boundary

This is a **controlled multilingual ISCO-08 benchmark using WISCO occupation-title data**. It is not real Labour Force Survey validation.

It may measure and report flat versus hierarchical ISCO-08 retrieval outcomes on this benchmark, with the required limitation statement. It must not report or imply:

- real-LFS validation, respondent-data validation, national representativeness, or real-time deployment performance;
- ISIC, ISCED/ISCED-F, or SRE accuracy/quality;
- LLM reranking quality, because reranking is disabled;
- a B1 re-freeze, B2 sweep, or a result applicable to another split/dataset.

Every local manifest/report must use:

- `dataset_label="synthetic_or_operationally_realistic"`;
- `evaluation_status="measured"`;
- `manuscript_eligible=false` as computed by the existing manifest code;
- the exact statement: `controlled multilingual ISCO-08 benchmark; not real Labour Force Survey validation`.

Do not fabricate a `dataset_label` column in the raw `run_eval.py` CSV. That runner does not have one; the label belongs in the generated manifests and report.

## Starting point and Git boundary

1. Fetch and verify the exact remote tip:
   - Base branch: `reviewer2-wisco-tier1-preflight-20260808`
   - Required base SHA: `a1ce94213e5981dc30444fd4947ff4c0dd056888`
2. Require a clean working tree.
3. Create a new branch:
   - `reviewer2-wisco-tier1-results-20260808`
4. Do not merge, rebase, reset, clean, stash, pull, force-push, create a PR, or alter any protected/prior branch.
5. Commit only the final report. All local benchmark artifacts must remain Git-ignored.

## A. Repeat the non-negotiable pre-run gates

Before either full command, re-run and record:

1. WISCO v2 integrity and leakage validation. The exact expected values are:
   - dataset hash: `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c`;
   - 20,760 records: 2,013 dev / 18,747 heldout;
   - zero source-family leakage;
   - zero cross-split exact-text duplicate groups;
   - zero malformed ISCO codes.
2. Export the fixed heldout split if the current ignored CSV is absent or does not match the 18,747-row source order:

```bash
python eval/export_benchmark_to_run_eval_csv.py \
  --records eval/local_benchmarks/wisco_isco08_v2_group_split/records.json \
  --split heldout \
  --out eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv
```

3. Read-only Qdrant readiness checks. Require these real production collections to be present and non-empty:
   - `isco08_major_groups`
   - `isco08_submajor_groups`
   - `isco08_minor_groups`
   - `isco08_unit_groups`
   - `isco_occupations`

If any gate fails, do not start either full run. Report `TIER1_COMPLETED: no` with the exact blocker.

## B. Execute the two full paired runs

Create one timestamped ignored root:

```text
eval/local_runs/wisco_v2_tier1_full_<UTC timestamp>/
```

Use the same frozen heldout CSV, source order, dataset hash, and output root timestamp for both configurations. Do not use `--limit`, `--reranker-model`, `--use-llm-reranker on`, `ablation_runner.py`, or any non-WISCO input.

Run flat first, then hierarchical:

```bash
python eval/run_eval.py \
  --test-set eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv \
  --system flat \
  --use-llm-reranker off \
  --config wisco_v2_tier1_full_flat_norerank \
  --run-id wisco-v2-tier1-full-flat-norerank \
  --output-dir eval/local_runs/wisco_v2_tier1_full_<UTC timestamp>/flat

python eval/run_eval.py \
  --test-set eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv \
  --system hierarchical \
  --use-llm-reranker off \
  --config wisco_v2_tier1_full_hierarchical_norerank \
  --run-id wisco-v2-tier1-full-hierarchical-norerank \
  --output-dir eval/local_runs/wisco_v2_tier1_full_<UTC timestamp>/hierarchical
```

If either command exits non-zero, is interrupted, produces fewer than 18,747 rows, has non-blank row-level errors, or the hierarchical run has a fallback/flat method label or missing four-stage evidence, stop. Preserve the raw output and report the failure. Do not retry, alter data, or report an incomplete comparison as a result.

## C. Output-integrity and path checks

For each completed CSV, verify and record:

1. Exactly 18,747 data rows; exact case-ID order and gold ISCO values match across flat and hierarchical outputs and the frozen heldout export.
2. Every row has `evaluation_status="measured"`; raw CaseResult CSV does not contain a `dataset_label` field, and this absence is stated rather than filled artificially.
3. Reranker invariants: `reranker_fired=False`, disabled-model disclosure at run level, empty reranker trace, zero tokens, and zero estimated LLM cost for all rows.
4. ISIC/ISCED/SRE invariants: blank ISIC/ISCED prediction fields and `sre_status="not_applicable"` throughout. Do not compute their metrics.
5. Flat output: every `pred_method` is a flat retrieval method, not a hierarchical method.
6. Hierarchical output: every `pred_method` is genuine hierarchical retrieval, no `fallback` / `flat` label, no row-level error, and every row carries non-empty stage 1–4 evidence. If some valid non-fallback hierarchical records differ in stage count for a documented code-path reason, report the exact count/reason instead of silently accepting it.
7. Record hardware, Python/dependency version information, exact code SHA, dataset hash, split hash, and Qdrant point counts at run start and after completion. Do not expose secrets.

## D. Mechanical analysis and paired comparison

After and only after both output-integrity gates pass:

1. Run existing analysis independently for each raw CSV:

```bash
python eval/analyze.py --case-csv <flat_csv> --out <root>/analysis/flat
python eval/analyze.py --case-csv <hierarchical_csv> --out <root>/analysis/hierarchical
```

2. Use `eval.analyze`’s existing `wilson_score_interval` and `mcnemar_test` utilities, without changing project source code, to create one ignored, reproducible comparison artifact under `<root>/analysis/`. The artifact must include:
   - exact-match ISCO accuracy at 1-, 2-, 3-, and 4-digit depth for each system, with successes, n, and Wilson 95% intervals;
   - paired 4-digit outcome table: both correct, flat only correct, hierarchical only correct, both wrong;
   - exact two-sided McNemar statistic and p-value for flat versus hierarchical 4-digit accuracy;
   - aggregate end-to-end and retrieval latency mean, median/p50, p95, throughput, and zero-LLM cost confirmation;
   - per-language 4-digit accuracy and n for all five WISCO languages, marked descriptive rather than a separate generalisation claim;
   - all exclusions or unavailable metrics with reasons.
3. The comparison artifact must include the exact controlled-benchmark limitation statement and must say that `manuscript_eligible=false` is structurally computed because the data are not approved real-LFS validation.
4. Do not run any analysis that treats blank ISIC/ISCED/SRE fields as zero, and do not calculate reranking, SRE, ISIC, or ISCED metrics.

## E. Create truthful manifests

Use existing `eval.manifest.build_manifest()` and `write_manifest()` against each completed CaseResult CSV. Write manifests under `<root>/manifests/`, not Git-tracked files.

For each manifest use:

- the exact `run_id`, `classifier_method`, `split_name="heldout"`, dataset hash, code SHA, and split-manifest path;
- `dataset_label="synthetic_or_operationally_realistic"`;
- `evaluation_status="measured"`;
- retrieval/model parameters that truthfully record no reranker, the fixed seed/split, and the local embedding/Qdrant configuration;
- `manuscript_eligible` must be left to existing manifest code, never supplied or overridden.

After writing, inspect both manifests and assert:

- `dataset_label` is exactly `synthetic_or_operationally_realistic`;
- `evaluation_status` is exactly `measured`;
- `manuscript_eligible` is `false`;
- `n_cases=18747`;
- no LLM cost/reranker usage is recorded.

## F. Final report and Git action

Create only:

- `Documentation/AI_HANDOFF/CLAUDE_TASK_12_FINAL_REPORT.md`

The final report must state:

1. Starting/final SHA and branch name.
2. Whether every pre-run gate passed, including exact WISCO dataset/split hash and Qdrant collection counts.
3. Exact full-run commands, start/end time, elapsed wall time, and raw output/manifests/analysis paths.
4. Exact output-integrity results and whether any row-level errors/fallbacks occurred.
5. Exact measured metrics from the generated comparison artifact, with the controlled-benchmark limitation immediately beside the results.
6. Exact manifest statuses, especially `dataset_label`, `evaluation_status`, and `manuscript_eligible=false`.
7. Confirmation of zero Ollama/LLM/paid API calls and zero Qdrant mutations.
8. Clear findings: what this benchmark does support and what it cannot support.
9. Manuscript-safe wording and manuscript-unsafe wording.
10. Protected-branch status and clean working tree.

Do not claim the project is fully validated. No real-LFS, ISIC, ISCED, SRE, or reranking conclusion is permitted from this task.

Commit and push only the final report on `reviewer2-wisco-tier1-results-20260808`. Do not create a PR. Do not commit raw WISCO records, exports, output CSVs, manifests, analysis JSON, or local scripts.

## Stop condition

Stop after the two complete configurations have either succeeded and been analyzed, or after the first gating/run failure has been accurately reported. Do not run the 500-case reranking tier, build ISIC/ISCED-F collections, re-freeze B1, start B2, or make any further code change.
