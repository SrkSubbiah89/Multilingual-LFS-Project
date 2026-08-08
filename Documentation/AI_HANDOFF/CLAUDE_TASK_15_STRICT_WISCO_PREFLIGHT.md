# Task 15 — Strict WISCO High-Coverage Preflight and Known-Risk Replay

## Purpose

Run one limited, diagnostic preflight against the existing WISCO v2 heldout package after Task 13's hierarchical-integrity hardening was integrated in Task 14. This is **not** a benchmark result, accuracy study, performance comparison, ablation, or manuscript-evidence task.

The narrow goal is to establish whether the repaired hierarchical path can process a deterministic multilingual sample, including every known-risk Task 12 case, while:

1. producing genuine hierarchical retrieval for every selected row;
2. avoiding an implicit flat fallback;
3. keeping every recorded hierarchy stage under 30 seconds; and
4. preserving the read-only, model-free evaluation conditions.

Do not claim that this preflight establishes accuracy, latency, scalability, real-LFS validation, ISIC/ISCED performance, SRE performance, or paper readiness.

## Starting point and branch discipline

1. Fetch refs and verify both source branch and SHA before changing anything:
   - Base branch: `reviewer2-wisco-strict-benchmark-baseline-20260808`
   - Required base SHA: `3716057db699527761127383f430b786fa21ff46`
2. Require a clean working tree before branching.
3. Create and work only on:
   - `reviewer2-wisco-strict-preflight-20260808`
4. Push only that new branch.
5. Do **not** modify, merge into, rebase, reset, clean, stash, pull, force-push, or switch the protected/prior-task branches.
6. Do not alter project source code, tests, dependencies, configurations, WISCO package files, Qdrant collections, benchmark definitions, or historical raw outputs in this task.
7. The only tracked repository file allowed to change is:
   - `Documentation/AI_HANDOFF/CLAUDE_TASK_15_FINAL_REPORT.md`
8. Local scripts and all generated artifacts must live under an ignored local output directory such as:
   - `eval/local_runs/wisco_v2_strict_high_coverage_preflight_<UTC_TIMESTAMP>/`
9. Before committing, verify `git status --short` contains only the final report. Commit and push the final report only.

## Explicit authorization and strict prohibitions

This task authorizes exactly one bounded live evaluation operation:

- read the existing local WISCO v2 package;
- make read-only requests to the existing local Qdrant collections;
- run one model-free, reranker-off, hierarchical evaluation command on the selected cases;
- write outputs only beneath the Git-ignored local output root.

Do **not**:

- run the full 18,747-row benchmark;
- run a flat baseline;
- run reranking, Ollama, an LLM, a paid API, B1 re-freeze, B2 sweep, ISIC/ISCED-F benchmark, SRE benchmark, or `analyze.py`;
- mutate, rebuild, populate, or delete any Qdrant collection;
- retry a failed strict run;
- weaken/disable `--require-genuine-hierarchical`;
- relax the stage limit;
- exclude a known-risk case after selection;
- change source code or tests to obtain a pass;
- create accuracy, Wilson, McNemar, aggregate latency, cost, throughput, or manuscript-result artifacts.

If any preflight gate fails, record the factual failure and stop. Do not retry or expand scope.

## Required preflight gates

Use the existing, local WISCO v2 package. Confirm and record all of the following before the live evaluation command:

1. Dataset package hash exactly equals:
   - `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c`
2. Package record total is exactly 20,760.
3. Split counts are exactly 2,013 development and 18,747 heldout records.
4. Leakage audit remains zero source-family leakage and zero cross-split exact normalized-text duplicates.
5. Heldout code validation remains clean.
6. The canonical heldout ordering contains exactly 18,747 records.
7. Existing local Qdrant has all required non-empty collections:
   - `isco08_major_groups`
   - `isco08_submajor_groups`
   - `isco08_minor_groups`
   - `isco08_unit_groups`
   - `isco_occupations`

The required collection name is `isco08_submajor_groups` with no underscore between `sub` and `major`, matching production code. Record point counts before and after the evaluation operation. They must be unchanged.

If any gate fails, do not execute the evaluation command. Create the final report with `STRICT_PREFLIGHT_READY: no`, state which gate failed, commit only the report, push the branch, and stop.

## Deterministic sample and known-risk inclusion

### Build the deterministic 500-case base selection

Use the existing selection utility. Keep the resulting JSON and its deterministic selection metadata under the ignored local output root:

```bash
python eval/select_wisco_reranking_subset.py \
  --records eval/local_benchmarks/wisco_isco08_v2_group_split/records.json \
  --split heldout \
  --target-size 500 \
  --seed 42 \
  --out <IGNORED_OUTPUT_ROOT>/stratified_500.json
```

Do not alter the selector or its algorithm. The expected selector is deterministic stratification by `(language, ISCO major group)` using SHA-256 ranking with seed 42 and largest-remainder allocation.

### Include every Task 12 known-risk ID

Use this fixed list. Do not omit, substitute, or reinterpret any ID:

```text
WISCO-2310570000000-en
WISCO-4312002200018-hi
WISCO-6113040000000-en
WISCO-6113040000000-ar
WISCO-6113990000000-ar
WISCO-6121002200018-en
WISCO-6121070000000-en
WISCO-6129000900018-en
WISCO-6210000300018-ar
WISCO-6221050000000-en
WISCO-6221050000000-ar
WISCO-6222010000000-en
WISCO-6222010000000-ar
WISCO-6222020000000-en
WISCO-6222020000000-ar
WISCO-6223000200018-en
WISCO-6223000200018-ar
WISCO-6224000500018-ar
WISCO-6224000700018-ar
WISCO-6330010000000-ar
WISCO-6340000100016-en
WISCO-9213010000000-ar
WISCO-9216010000000-en
WISCO-8160001500018-ur
```

Create a local selection-manifest JSON under the ignored output root that records:

- source WISCO v2 hash;
- split name;
- seed and base selected size;
- fixed known-risk ID list;
- overlap count between base selection and known-risk IDs;
- final selection size;
- complete final selected ID list;
- language and ISCO-major-group distribution;
- confirmation that every known-risk ID is in the final selection.

The final set must be the 500 deterministic selection plus every risk ID not already selected. It must therefore contain between 500 and 524 unique records. Export those records to a temporary CSV in their **original canonical heldout source order**, not sorted by the selection algorithm.

The CSV must preserve the required evaluation schema and meet all of the following:

- every selected `benchmark_id` appears once and only once;
- gold ISCO values correspond to the selected WISCO records;
- `industry_text` and `education_text` are blank;
- ISIC and ISCED gold columns, if present, are blank;
- no test data values are invented, translated, modified, or committed.

Use only a short local throwaway script under the ignored output directory if one is necessary. Do not add a project script.

## One authorized live preflight command

After all gates and the selection CSV pass, run exactly one evaluation command, substituting real local paths and a UTC timestamped output root:

```bash
QDRANT_TIMEOUT_SECONDS=30 python eval/run_eval.py \
  --test-set <IGNORED_OUTPUT_ROOT>/strict_high_coverage_selected.csv \
  --system hierarchical \
  --use-llm-reranker off \
  --require-genuine-hierarchical \
  --max-stage-latency-ms 30000 \
  --config wisco_v2_strict_high_coverage_preflight \
  --run-id wisco-v2-strict-high-coverage-preflight \
  --output-dir eval/local_runs/wisco_v2_strict_high_coverage_preflight_<UTC_TIMESTAMP>
```

Do not add `--limit`, a reranker model, an LLM flag, a fallback override, a retry wrapper, or any broader run.

The strict evaluation must exit with code 0. It is allowed to write a CSV only if every row passes the strict guard. If it exits non-zero or fails before output, preserve the local diagnostic information, record it accurately, set `STRICT_PREFLIGHT_READY: no`, and stop without retrying.

## Required post-run verification

Only if the one command exits 0, inspect raw row-level artifacts and verify:

1. Output row count equals the final selected case count and all selected IDs appear exactly once in original selected CSV order.
2. Every row has a method beginning with `hierarchical_`; no row is `flat_semantic`, an explicit fallback, or a missing-method error.
3. Every row has non-empty valid JSON list evidence for all stage-1 through stage-4 candidate fields.
4. Every recorded hierarchy stage latency is no more than 30,000 ms.
5. All row-level error fields are blank.
6. Reranking/LLM use is disabled throughout:
   - reranker resolved value is equivalent to `none (reranking disabled)`;
   - zero reranker trace/cost/token fields;
   - no Ollama, paid API, or external LLM call.
7. ISIC, ISCED, and SRE are not constructed or evaluated for this ISCO-only CSV. Record the actual explicit status value if one exists, such as `not_applicable`.
8. All 24 known-risk IDs are present and each satisfies the genuine-hierarchy and stage-limit checks.
9. For descriptive diagnostic traceability only, count/record whether the keyword-anchor retry was used and the final stage-1 source for the selected rows. Do not interpret this as quality, accuracy, or performance.
10. Qdrant point counts after the run exactly match the pre-run counts.

Do **not** invoke `eval/analyze.py`, calculate accuracy, confidence intervals, McNemar tests, aggregate latency, cost, throughput, or create charts/tables that could be presented as benchmark results.

## Final report and stop condition

Create only:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_15_FINAL_REPORT.md
```

The report must start with exactly one of:

```text
STRICT_PREFLIGHT_READY: yes
```

or:

```text
STRICT_PREFLIGHT_READY: no
```

It must include:

1. source base branch and verified SHA;
2. new branch name, commit SHA, and remote-push confirmation;
3. clean-tree confirmation before and after;
4. exact changed tracked files;
5. every preflight-gate result, including collection names and before/after point counts;
6. dataset hash, record/split totals, selection seed, base size, known-risk overlap, final size, and distribution summary;
7. the exact authorized evaluation command and exit status;
8. the raw ignored output-root paths;
9. strict post-run integrity checks, reported row-level and factual only;
10. explicit confirmation that no full benchmark, flat run, reranker, LLM, API, collection mutation, analysis/statistics, source/test/config edit, B1 re-freeze, or B2 sweep occurred;
11. protected-branch status;
12. a clear statement that this is only a diagnostic readiness check and contributes no manuscript performance evidence.

If the strict run or any check fails, specify the first failing condition and exact affected IDs or error text, but do not infer a root cause beyond available evidence. Do not retry.

After committing and pushing the report, stop. Do not begin a full WISCO rerun or any further task.
