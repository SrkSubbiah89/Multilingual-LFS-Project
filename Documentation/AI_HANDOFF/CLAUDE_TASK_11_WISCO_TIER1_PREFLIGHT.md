# Task 11 — WISCO Tier-1 Local Preflight and Path-Integrity Gate

## Purpose

Validate the exact local prerequisites for the controlled WISCO ISCO-08 Tier-1 benchmark before approving the two full held-out retrieval-only measurements. This task intentionally performs a small, non-manuscript-eligible local smoke test so that a missing Qdrant collection, missing embedding model, incorrect data hash, or silent flat fallback is found before a multi-hour run.

This is not the benchmark. It must never be reported as performance evidence.

## Starting point

1. Fetch and verify the exact remote tip:
   - Base branch: `reviewer2-wisco-measurement-baseline-20260808`
   - Required base SHA: `4e772669449703ea09b6c3a33f32f5fa409c6a3b`
2. Start with a clean working tree.
3. Create a new branch:
   - `reviewer2-wisco-tier1-preflight-20260808`
4. Do not merge, rebase, reset, clean, stash, pull, force-push, create a PR, or alter any protected/prior branch.

## Explicit authorization boundary

This task is authorized to perform only the following live local operations:

- Read the existing public WISCO source data already present locally.
- Rebuild or validate the ignored local v2 group-aware WISCO package only if it is absent or its integrity validation fails, using the existing deterministic builder. Never alter v1.
- Read the existing local Qdrant ISCO-08 collections and execute embedding/search queries needed for a two-configuration, five-case smoke test.
- Load the existing local embedding model if required by those smoke tests.
- Write all generated benchmark exports and smoke outputs under Git-ignored `eval/local_benchmarks/` and `eval/local_runs/` only.

This task is not authorized to:

- create, delete, recreate, mutate, or populate any Qdrant collection;
- call Ollama, a paid API, or any LLM. Every evaluator invocation must include `--use-llm-reranker off` and no `--reranker-model`;
- run more than five held-out WISCO cases per system;
- run `ablation_runner.py`, a B1 re-freeze, a B2 sweep, ISIC/ISCED-F collection construction, an ISIC/ISCED/SRE evaluation, or a full WISCO measurement;
- alter the benchmark split, WISCO source content, production source code, tests, requirements, configuration, or manuscript wording.

## A. Asset and split-integrity gate

1. Locate the ignored local v2 package:
   - `eval/local_benchmarks/wisco_isco08_v2_group_split/`
2. Validate it against its recorded `split_manifest.json` and rebuild it only if absent or invalid, using:

```bash
python eval/build_wisco_isco_benchmark_v2_group_split.py \
  --out-root eval/local_benchmarks/wisco_isco08_v2_group_split
```

3. Run the existing deterministic leakage/integrity audit against v2. It must confirm all of the following before any smoke retrieval:
   - dataset hash exactly `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c`;
   - 20,760 total records with 2,013 dev and 18,747 heldout;
   - zero source-family leakage;
   - zero cross-split exact-text duplicate groups;
   - zero malformed ISCO codes.
4. Export exactly the heldout split to a new ignored CSV:

```bash
python eval/export_benchmark_to_run_eval_csv.py \
  --records eval/local_benchmarks/wisco_isco08_v2_group_split/records.json \
  --split heldout \
  --out eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv
```

5. Confirm the exported CSV has exactly 18,747 data rows, required `run_eval.py` columns, blank paired `industry_text`/`education_text` fields, and no ISIC/ISCED gold labels.

## B. Non-destructive collection-readiness gate

1. Query local Qdrant read-only. Record the host/port configuration, collection names, and point counts, without exposing credentials.
2. Confirm that the ISCO hierarchical collections required by the real production retrieval path are present and non-empty:
   - `isco08_major_groups`
   - `isco08_sub_major_groups`
   - `isco08_minor_groups`
   - `isco08_unit_groups`
3. Confirm the flat ISCO collection required by `--system flat` is present and non-empty.
4. Do not infer readiness merely from a process being available. Use a read-only collection listing/count check and record exact results.
5. If any required collection is unavailable or empty, stop before the smoke test. Do not build, recreate, or populate collections. Record the blocker and do not begin the full benchmark.

## C. Five-case path-integrity smoke test

Only after A and B pass:

1. Select the first five records from the fixed, already-exported heldout CSV in source order. Do not select based on a prediction, language, code, or expected outcome.
2. Run exactly two evaluator commands, each with `--limit 5`, separate output directories, and a fixed preflight run identifier:

```bash
python eval/run_eval.py \
  --test-set eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv \
  --system flat \
  --use-llm-reranker off \
  --limit 5 \
  --config wisco_v2_preflight_flat_norerank \
  --run-id wisco-v2-tier1-preflight-flat \
  --output-dir eval/local_runs/wisco_v2_tier1_preflight/flat

python eval/run_eval.py \
  --test-set eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv \
  --system hierarchical \
  --use-llm-reranker off \
  --limit 5 \
  --config wisco_v2_preflight_hierarchical_norerank \
  --run-id wisco-v2-tier1-preflight-hierarchical \
  --output-dir eval/local_runs/wisco_v2_tier1_preflight/hierarchical
```

3. Immediately inspect both five-row CSVs. Do not calculate or report accuracy. Prove the following path integrity facts:
   - every row has `evaluation_status="measured_synthetic_or_operationally_realistic"` or the repository’s exact equivalent measured controlled-benchmark status, never a real-LFS label;
   - every row has `dataset_label="synthetic_or_operationally_realistic"` if that field is present in this runner/output version; otherwise record its absence precisely and do not fabricate it;
   - reranker is disabled, with `reranker_model`/resolved metadata equal to the Task 09 disabled disclosure and no reranker trace, call, or latency;
   - ISIC, ISCED, and SRE output fields are blank or `not_applicable` as appropriate, and no such classifier was constructed;
   - flat rows use a flat retrieval method, not a hierarchical method;
   - hierarchical rows use a genuine hierarchical method and contain non-empty stage evidence consistent with four-stage retrieval;
   - no hierarchical row has a `fallback` method label or any flat-only method label.
4. If the hierarchical smoke test falls back, has missing stage evidence, produces a row-level error, or otherwise cannot prove genuine hierarchical retrieval, stop. Do not run the full heldout benchmark. Diagnose and report the blocker only.
5. Record smoke latency and environment only as preflight diagnostics. Do not calculate accuracy, comparisons, significance, cost, or manuscript-ready metrics.

## D. Documentation, tests, and reporting

1. Make no production code/test/config changes. Do not run the full test suite again unless a local environment issue requires it; this is an operational readiness task, not a code change.
2. Create only:
   - `Documentation/AI_HANDOFF/CLAUDE_TASK_11_FINAL_REPORT.md`
3. The final report must include:
   - base/final SHAs and branch name;
   - whether the v2 package was reused or rebuilt;
   - complete asset/split hash and leakage-gate results;
   - complete read-only Qdrant collection readiness results;
   - exact smoke commands, output directories, row counts, and path-integrity checks;
   - confirmation of zero Ollama, paid API, or LLM calls;
   - confirmation that no Qdrant collection was changed;
   - confirmation that no full benchmark, performance analysis, or manuscript claim was produced;
   - a clear `READY_FOR_TIER1_FULL_RUN: yes|no` line and the precise reason;
   - protected branch status and clean working tree.
4. Do not commit any local benchmark/export/smoke artifact. They remain Git-ignored.
5. Commit and push only the final report to `reviewer2-wisco-tier1-preflight-20260808`. Do not create a PR.

## Stop condition

Stop after publishing the final report and verifying a clean tree. Whether readiness is yes or no, do not run the 18,747-case flat or hierarchical benchmark. A separately approved task will decide and execute that measurement.
