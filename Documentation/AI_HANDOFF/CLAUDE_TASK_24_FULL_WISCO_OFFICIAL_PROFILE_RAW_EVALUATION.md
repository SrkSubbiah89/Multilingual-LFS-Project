# Task 24 — Full Controlled WISCO ISCO-08 Evaluation Using the Official ILO Profile

## Purpose and evidence boundary

On a new branch from the verified Task 23 result branch, produce the first full
controlled WISCO ISCO-08 raw outputs using the official ILO 2021 profile:

1. one full official four-digit flat-comparator run; and
2. one full strict official hierarchical-retrieval run.

This task produces **raw outputs only**. It must not calculate accuracy,
precision, recall, Wilson intervals, McNemar tests, latency comparisons, or
any comparative conclusion. It must not draft manuscript wording or treat raw
predictions as paper-ready evidence.

WISCO remains a controlled multilingual ISCO-08 benchmark, not real Labour
Force Survey validation. It cannot establish ISIC, ISCED, or SRE performance.

## Exact base and branch requirements

1. Fetch `origin` and verify this exact base before branching:

   ```text
   reviewer2-official-isco08-collection-build-smoke-20260809
   0ce9e6778c5fe982babd446a26f8445a959c6783
   ```

2. Confirm the working tree is clean.
3. Create exactly:

   ```text
   reviewer2-wisco-official-tier1-raw-results-20260809
   ```

4. Do not use `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
   `git pull`, force-push, or a PR.
5. Do not modify or merge any protected or prior-task branch.

## Absolute restrictions

- Do not mutate any Qdrant collection. This task is read-only against Qdrant.
- Do not rerun the Task 23 builder.
- Do not use a reranker, Ollama, CrewAI, any LLM, or a paid API.
- Do not construct or run ISIC, ISCED, or SRE.
- Do not modify the WISCO package, split manifest, official ILO catalogue,
  metadata, Task 17 output, or Task 23 output.
- Do not re-freeze B1, bypass the B2 gate, run B2 sweep, or do a synthetic
  evaluation.
- Do not retry a failed run, replace it, repair data, alter code, or delete
  output.
- Keep every export, raw CSV, log, and manifest under a new Git-ignored output
  root in `eval/local_runs/`. Never commit those outputs.

## Part A — Mandatory read-only pre-run gates

All gates must pass before either full run begins.

### A.1 WISCO package integrity

Verify the existing WISCO v2 group-split benchmark package, without rebuilding
it:

```text
dataset hash:
a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c

total: 20,760
dev: 2,013
heldout: 18,747
```

Confirm zero source-family leakage, zero cross-split duplicate-text groups, and
zero malformed codes using existing validation/audit tooling. Do not change the
package.

Freshly export exactly the 18,747 canonical heldout rows using the existing
approved export tool. Verify:

- the export has exactly 18,747 rows;
- required ISCO input and gold-code fields are present;
- it is ISCO-only, with no paired `industry_text` and `education_text`;
- any ISIC/ISCED gold columns are blank and are not used.

### A.2 Official-source and collection integrity

Verify, read-only:

- the official normalized catalogue SHA-256 remains:

  ```text
  29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3
  ```

- `eval/verified_catalogue_counts.yaml` remains WISCO-independent and records
  official counts 10/43/130/436;
- the Task 23 success manifest exists and identifies
  `official_ilo2021_v1`, `intfloat/multilingual-e5-small`, and dimension 384;
- local Qdrant at `localhost:6333` is reachable;
- official target collections exist with exact point counts:

  | Collection | Expected count |
  | --- | ---: |
  | `isco08_major_groups_ilo2021_v1` | 10 |
  | `isco08_submajor_groups_ilo2021_v1` | 43 |
  | `isco08_minor_groups_ilo2021_v1` | 130 |
  | `isco08_unit_groups_ilo2021_v1` | 436 |
  | `isco08_unit_groups_flat_ilo2021_v1` | 436 |

- legacy collections remain exactly at their Task 23 inventory:
  `10/43/131/441/124`.

Record the complete read-only pre-run Qdrant inventory and point counts.

### A.3 Command validation and output root

1. Inspect `python eval/run_eval.py --help` and use only supported flags.
2. Create one new ignored root:

   ```text
   eval/local_runs/wisco_official_tier1_<UTC timestamp>/
   ```

3. Validate that this output root is Git-ignored before any run.
4. Set `QDRANT_TIMEOUT_SECONDS=30` for both runs and record that value.
5. Do not pass `--reranker-model` to either command.

If any Part A check fails, report `OFFICIAL_TIER1_COMPLETED: no`, commit only
the final report, and stop. Do not begin a run.

## Part B — Exact authorised full runs

Run each command **once only**, using the freshly exported 18,747-row heldout
CSV. Both must use the official profile and disable reranking.

### B.1 Official flat comparator

Run first:

```text
--system flat
--use-llm-reranker off
--isco-catalogue-profile official_ilo2021_v1
```

Use its own output subdirectory under the Task 24 ignored root.

Immediately after completion, perform the flat output-integrity gate in Part C.
If it fails, do not run the hierarchical command.

### B.2 Strict official hierarchical retrieval

Run only after the flat gate fully passes:

```text
--system hierarchical
--use-llm-reranker off
--isco-catalogue-profile official_ilo2021_v1
--require-genuine-hierarchical
--max-stage-latency-ms 30000
```

Use its own output subdirectory under the Task 24 ignored root.

Do not add unsupported flags. Do not make a second attempt under any
circumstance.

## Part C — Fail-closed output-integrity gates

### C.1 Flat gate

The flat run passes only when all conditions are true:

1. the command exits 0;
2. exactly 18,747 rows exist in the raw output;
3. every row has no row-level error;
4. every `pred_method` is exactly
   `flat_isco08_official_ilo2021_v1`;
5. every predicted ISCO code is syntactically valid and exactly four digits;
6. reranking is disabled with zero reranker trace, cost, and token activity;
7. no ISIC, ISCED, or SRE classifier/engine is constructed or run;
8. read-only Qdrant post-run counts match the pre-run counts exactly.

If any condition fails: preserve the output, set
`OFFICIAL_TIER1_COMPLETED: no`, do not run Part B.2, do not retry, and stop.

### C.2 Hierarchical gate

The hierarchical run passes only when all conditions are true:

1. the command exits 0;
2. exactly 18,747 rows exist in the raw output;
3. every row has no row-level error;
4. every `pred_method` is exactly
   `hierarchical_isco08_official_ilo2021_v1`;
5. every row has valid non-empty stage 1, 2, 3, and 4 evidence;
6. no row is labelled a fallback or flat method;
7. no individual stage latency exceeds 30,000 ms;
8. reranking is disabled with zero reranker trace, cost, and token activity;
9. no ISIC, ISCED, or SRE classifier/engine is constructed or run;
10. read-only Qdrant post-run counts match the pre-run counts exactly.

If any condition fails: preserve the output, set
`OFFICIAL_TIER1_COMPLETED: no`, do not retry, alter data, repair code, or
delete/rebuild collections, and stop.

If both gates pass, state:

```text
OFFICIAL_TIER1_COMPLETED: yes
```

This means only that both raw output files passed operational integrity gates.
It does not permit score calculation or a manuscript claim.

## Part D — Tests and final report

After the run integrity gates, run the relevant focused official-profile and
evaluation tests, then:

```bash
pytest backend/tests eval/ -q
```

Do not change code or tests to repair any unrelated issue.

Commit and push only:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_24_FINAL_REPORT.md
```

The report must include:

1. branch, base SHA, final SHA, push confirmation, and clean-tree status;
2. all exact commands executed, including the heldout export;
3. WISCO package/split/hash checks and official catalogue hash;
4. pre-run, after-flat, and after-hierarchical Qdrant inventories;
5. raw output and manifest paths, which must be stated as Git-ignored;
6. per-run wall-clock time and exit code;
7. all flat and hierarchical integrity-gate counts, including method labels,
   valid four-digit code counts, error counts, stage-evidence counts,
   fallback counts, maximum stage latency, reranker evidence, and
   ISIC/ISCED/SRE non-construction evidence;
8. whether each authorized command was run exactly once;
9. `OFFICIAL_TIER1_COMPLETED: yes` or `no`, with every triggered stop
   condition if `no`;
10. focused and full test outputs;
11. protected branch status and confirmation that no PR was created; and
12. this exact boundary statement:

    ```text
    These are raw outputs from a controlled multilingual ISCO-08 benchmark
    using a public occupation-title dataset. They are not real Labour Force
    Survey validation and have not yet been analysed for accuracy, statistical
    comparison, or manuscript-ready conclusions. They provide no ISIC, ISCED,
    SRE, reranking, cost, or real-world performance evidence.
    ```

Stop after pushing the report. Do not run analysis, accuracy scoring,
statistical comparison, reranking, paper drafting, B1 re-freeze, or B2 sweep.
