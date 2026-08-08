# Task 17 — Full Strict WISCO Tier 1 Controlled Evaluation

## Purpose and evidence boundary

Run one fresh, controlled, model-free Tier 1 evaluation on the complete WISCO v2 heldout split. This task is the first evidence-producing evaluation task after Task 12 was stopped and Task 13–16 hardening/preflight work was verified.

The evaluation has only this scope:

- external WISCO v2 multilingual occupation titles;
- ISCO-08 four-digit prediction only;
- two systems evaluated under the same current source revision:
  - `flat`;
  - `hierarchical`;
- local Qdrant retrieval only;
- reranking disabled;
- no LLM, Ollama, paid API, or external inference;
- raw results and execution integrity only.

WISCO is a **controlled multilingual ISCO-08 benchmark**, not real Labour Force Survey respondent data. This task can never support claims about real-LFS validation, population representativeness, ISIC accuracy, ISCED accuracy, SRE accuracy, reranker quality, or cost of an LLM workflow.

This task does not calculate accuracy, confidence intervals, statistical tests, aggregate comparison metrics, charts, or manuscript text. A later separate task must analyze only valid raw outputs.

## Required source state

Fetch refs and verify before any operation:

| Role | Branch | Required SHA |
|---|---|---|
| Integrated Task 16 base | `reviewer2-wisco-strict-preflight-integration-20260808` | `50cba8c7f053260e2e4851f79f8f6257ed3b06f7` |

Require:

1. `origin/reviewer2-wisco-strict-preflight-integration-20260808` exactly matches the required SHA.
2. The working tree is clean before branching.
3. Create and work only on:

```text
reviewer2-wisco-tier1-strict-full-results-20260808
```

4. Push only this new branch.
5. Do not merge, rebase, reset, clean, stash, pull, force-push, create a PR, or change any protected/prior branch.
6. Do not modify project source, tests, dependencies, configuration, WISCO package files, collection-builder files, or historical artifacts.
7. The only tracked file permitted to change is:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_17_FINAL_REPORT.md
```

All raw outputs, check scripts, manifests, and logs must remain under an ignored local output root.

## Authorization and prohibited operations

This task authorizes only:

1. read existing local WISCO v2 package files;
2. read existing local Qdrant collections and point counts;
3. run exactly two full, model-free, reranker-off `eval/run_eval.py` commands described below;
4. inspect their raw row-level output for integrity only;
5. write all generated artifacts under a Git-ignored `eval/local_runs/` root.

Do not:

- run a third evaluation command, a retry, a reduced/selected subset, an ablation, a dev sweep, B1 re-freeze, B2 sweep, or any ISIC/ISCED/SRE evaluation;
- run reranking, Ollama, CrewAI, LLM, paid API, or external inference;
- run `eval/analyze.py`, compute accuracy, Wilson intervals, McNemar tests, p-values, aggregate latency/throughput/cost values, charts, rankings, or manuscript-ready comparison tables;
- mutate, build, populate, rebuild, delete, or otherwise write to Qdrant;
- alter data, test order, code, thresholds, flags, test files, or configuration to make either command pass;
- commit raw evaluation output, data, local scripts, logs, or manifests.

If any pre-run gate, raw-output integrity check, or strict hierarchical check fails, record the factual failure and stop. Do not retry a command, relax a threshold, exclude a case, substitute a method, or continue to a later analysis stage.

## Pre-run gates

Before either evaluation command, independently confirm and record:

1. WISCO v2 dataset hash exactly:

```text
a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c
```

2. Package has exactly 20,760 records:
   - development: 2,013;
   - heldout: 18,747.
3. WISCO v2 leakage audit remains zero source-family leakage and zero cross-split exact normalized-text duplicate groups.
4. Heldout codes are clean and valid.
5. The canonical heldout `run_eval` CSV contains exactly 18,747 unique rows in canonical source order, has WISCO ISCO gold values, and contains no nonblank ISIC/ISCED gold labels.
6. All required Qdrant collections are present and non-empty:

```text
isco08_major_groups
isco08_submajor_groups
isco08_minor_groups
isco08_unit_groups
isco_occupations
```

7. Record each collection’s exact point count before execution.

If any gate fails, execute neither command. Set `TIER1_STRICT_COMPLETED: no` in the final report, state the first failure, commit only the report, push the branch, and stop.

## Output root and execution identity

Create one fresh ignored root with a UTC timestamp:

```text
eval/local_runs/wisco_v2_tier1_strict_full_<UTC_TIMESTAMP>/
```

Store beneath it:

- a pre-run gate record;
- a text file identifying the canonical heldout input path and SHA-256;
- the exact two commands;
- raw CSV and manifest/log outputs from each run;
- a small local integrity-check output for each run;
- Qdrant counts before and after.

The local helper artifacts are audit records only and must not calculate benchmark metrics.

Use the current full heldout `run_eval` CSV directly. Do not create a new split, reorder the cases, add/remove rows, or apply a limit.

## Authorized command 1: flat, full heldout

Run exactly once, first:

```bash
python eval/run_eval.py \
  --test-set <CANONICAL_WISCO_V2_HELDOUT_RUN_EVAL_CSV> \
  --system flat \
  --use-llm-reranker off \
  --config wisco_v2_tier1_flat_model_free \
  --run-id wisco-v2-tier1-flat-model-free \
  --output-dir <IGNORED_OUTPUT_ROOT>/flat
```

Do not provide a reranker model. Do not add a limit. Do not call `analyze.py` after it.

Immediately inspect only raw-output integrity:

1. exit status is 0;
2. output has exactly 18,747 rows in canonical input order with unique case IDs;
3. every row is ISCO-only and has no row-level error;
4. reranking/LLM remains off;
5. ISIC/ISCED/SRE are not constructed or evaluated;
6. preserve, but do not calculate aggregate performance measures from, raw per-row timing fields.

If this command or any integrity check fails, do not run the hierarchical command. Report the failure and stop.

## Authorized command 2: hierarchical, full heldout, strict

Only if the flat command and raw integrity checks pass, run exactly once:

```bash
QDRANT_TIMEOUT_SECONDS=30 python eval/run_eval.py \
  --test-set <CANONICAL_WISCO_V2_HELDOUT_RUN_EVAL_CSV> \
  --system hierarchical \
  --use-llm-reranker off \
  --require-genuine-hierarchical \
  --max-stage-latency-ms 30000 \
  --config wisco_v2_tier1_hierarchical_strict_model_free \
  --run-id wisco-v2-tier1-hierarchical-strict-model-free \
  --output-dir <IGNORED_OUTPUT_ROOT>/hierarchical
```

Do not provide a reranker model. Do not add a limit. Do not wrap it in a retry mechanism.

The hierarchical command is valid only if it exits 0. The strict guard must remain enabled from start to finish.

Immediately inspect raw-output integrity, row-by-row:

1. output has exactly 18,747 rows in canonical input order with unique case IDs;
2. every `pred_method` begins with `hierarchical_`;
3. no row has `flat_semantic`, any explicit fallback label, a missing method, or a row-level error;
4. every stage-1 through stage-4 candidate field is a non-empty valid JSON list;
5. every stage-1 through stage-4 latency is at or below 30,000 ms;
6. all 24 fixed Task 12 known-risk IDs are present and individually meet all five preceding hierarchical requirements;
7. reranking/LLM remains off;
8. ISIC/ISCED/SRE are not constructed or evaluated;
9. preserve raw per-row timing fields but do not calculate aggregate latency/throughput/cost measures.

If the command exits nonzero or any one requirement fails, set `TIER1_STRICT_COMPLETED: no`, state the first failure and all affected IDs visible from raw output, preserve raw artifacts, do not retry, do not analyze either system, and stop.

## Qdrant integrity after execution

After successful completion of both commands, re-read the five Qdrant point counts. Each after-count must exactly equal the corresponding before-count. If any differs:

- set `TIER1_STRICT_COMPLETED: no`;
- record the exact difference;
- do not analyze either raw output;
- stop.

## Final report and stop condition

Create only:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_17_FINAL_REPORT.md
```

Start it with exactly one:

```text
TIER1_STRICT_COMPLETED: yes
```

or:

```text
TIER1_STRICT_COMPLETED: no
```

The report must include:

1. verified base source branch/SHA, new branch, final commit SHA, remote push confirmation, and clean-tree status;
2. exact changed tracked files;
3. every pre-run gate and exact WISCO/Qdrant integrity values;
4. canonical input file identity and SHA-256;
5. exact commands, execution order, exit statuses, and ignored raw-output root paths;
6. raw output row counts/order/uniqueness and all specified integrity checks;
7. factual all-row strict-hierarchy pass/fail counts and known-risk-ID pass/fail status;
8. Qdrant before/after counts;
9. explicit confirmation of no reranker, Ollama, LLM, paid API, external inference, ISIC/ISCED evaluation, SRE evaluation, Qdrant mutation, analysis/statistics, code/test/config/data change, B1 re-freeze, B2 sweep, retry, or full-run expansion beyond the two authorized commands;
10. protected-branch status;
11. evidence boundary:

```text
If completed successfully, these raw artifacts are eligible only for a later controlled multilingual ISCO-08 analysis task. They are not themselves an accuracy claim, a comparative-performance result, real-LFS validation, population-representative evidence, ISIC/ISCED/SRE evidence, or manuscript-ready result.
```

After committing and pushing the report, stop. Do not perform analysis, compute metrics, create a manuscript statement, or begin another task.
