TIER1_STRICT_COMPLETED: yes

# Task 17 Final Report — Full Strict WISCO Tier 1 Controlled Evaluation

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_17_FULL_STRICT_WISCO_TIER1_EVALUATION.md`.
This is a raw, integrity-checked evaluation run only — no accuracy,
statistics, comparison, or manuscript text was computed or written.

## 1. Base source, new branch, final commit, push, clean-tree status

| | |
|---|---|
| Base branch | `reviewer2-wisco-strict-preflight-integration-20260808` |
| Required SHA | `50cba8c7f053260e2e4851f79f8f6257ed3b06f7` |
| Verified `origin` SHA | `50cba8c7f053260e2e4851f79f8f6257ed3b06f7` — match |
| New branch | `reviewer2-wisco-tier1-strict-full-results-20260808` |
| Final commit SHA | recorded after this report's commit (see push confirmation below) |

Working tree was clean immediately before branching (`git status
--short` empty on the base branch), clean throughout both evaluation
runs (all generated artifacts live under the Git-ignored
`eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/` output
root — confirmed via `git check-ignore -v`), and clean immediately
before this report's own commit.

## 2. Exact changed tracked files

Only:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_17_FINAL_REPORT.md
```

No project source, test, dependency, configuration, WISCO package file,
collection-builder file, or historical artifact was changed.

## 3. Pre-run gates and exact WISCO/Qdrant integrity values

All gates were re-verified live (not read from a cached summary alone)
using the project's own `recompute_dataset_hash` / `run_full_audit`
(`eval/audit_wisco_benchmark_leakage.py`) and `validate_benchmark_package`
(`eval/validate_controlled_benchmark.py`), plus a direct read of the
canonical heldout CSV and a live, read-only Qdrant query.

| Gate | Result |
|---|---|
| 1. Dataset hash | Recomputed `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c` — matches `dataset_hash.txt` exactly |
| 2. Package totals | `20760` total; dev `2013`, heldout `18747` (all exact) |
| 3. Leakage audit | `leakage_found: false`; `source_family_split_check.ok: true`; `text_duplicate_check.ok: true` — zero source-family leakage, zero cross-split exact-duplicate groups |
| 4. Heldout code validation | `0` malformed ISCO-08 codes across 18,747 heldout records; full-package `validate_benchmark_package` returned `ok: true`, zero errors/warnings |
| 5. Canonical heldout CSV | `18747` rows, all `case_id` unique, in canonical source order; every row has a non-blank `gold_isco_4digit`; `0` rows have a non-blank `gold_isic` or `gold_isced` |
| 6. Qdrant collections present/non-empty | All 5 required collections present (`isco08_major_groups`, `isco08_submajor_groups`, `isco08_minor_groups`, `isco08_unit_groups`, `isco_occupations`) |
| 7. Pre-run point counts | `major=10, submajor=43, minor=131, unit=441, isco_occupations=124` |

All 7 gates passed; both evaluation commands were authorized to run.

## 4. Canonical input file identity and SHA-256

```text
path:    eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv
sha256:  41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c
rows:    18747
```

This file was used, unmodified, unreordered, unfiltered, as `--test-set`
for both commands.

## 5. Exact commands, execution order, exit statuses, output roots

Ignored output root:
`eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/`

**Command 1 (run first): flat, full heldout**

```bash
python eval/run_eval.py \
  --test-set eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv \
  --system flat \
  --use-llm-reranker off \
  --config wisco_v2_tier1_flat_model_free \
  --run-id wisco-v2-tier1-flat-model-free \
  --output-dir eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/flat
```

Exit status: **0**. Completed 18,747 cases in 628.5s (0.03s/case). Wrote
`.../flat/20260808T115438Z_wisco_v2_tier1_flat_model_free.csv`.

**Command 2 (run second, only after command 1 passed all integrity
checks): hierarchical, full heldout, strict**

```bash
QDRANT_TIMEOUT_SECONDS=30 python eval/run_eval.py \
  --test-set eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv \
  --system hierarchical \
  --use-llm-reranker off \
  --require-genuine-hierarchical \
  --max-stage-latency-ms 30000 \
  --config wisco_v2_tier1_hierarchical_strict_model_free \
  --run-id wisco-v2-tier1-hierarchical-strict-model-free \
  --output-dir eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/hierarchical
```

Exit status: **0**. Completed 18,747 cases in 2869.2s (47 min 49s;
0.15s/case). Wrote
`.../hierarchical/20260808T153325Z_wisco_v2_tier1_hierarchical_strict_model_free.csv`.
The strict guard (`--require-genuine-hierarchical`) remained enabled for
the command's entire duration — since it exits 1 with **no CSV write**
on the first violation, exit 0 with a written CSV is itself proof every
one of the 18,747 rows passed `check_strict_hierarchical()` before any
output was produced.

No third command, retry, reduced subset, ablation, dev sweep, B1
re-freeze, B2 sweep, or ISIC/ISCED/SRE evaluation was run. Neither
command was given a `--limit`, a reranker model, or a retry wrapper.

## 6/7. Raw output integrity and strict-hierarchy pass/fail counts

### Command 1 — flat (18,747 rows)

| Check | Result |
|---|---|
| Row count == 18,747, canonical order, unique IDs | Pass |
| All rows ISCO-only, zero row-level `error` | Pass (0 non-blank errors) |
| Method label | `flat_semantic` for all 18,747 rows (correct for `--system flat`) |
| Reranker/LLM off | `reranker_fired=true` on 0 rows; `reranker_model` blank on all rows; `estimated_cost_usd`/token fields zero on all rows |
| ISIC/ISCED/SRE not constructed | `pred_isic_section`/`pred_isced_level` blank on all rows; `sre_status="not_applicable"` on all rows |
| Timing fields | Preserved per-row (`stage1..4_latency_ms`, `retrieval_latency_ms`, `end_to_end_latency_ms` all present); **no aggregate was computed** |

### Command 2 — hierarchical, strict (18,747 rows)

| Check | Result |
|---|---|
| Row count == 18,747, canonical order, unique IDs | Pass |
| Every `pred_method` begins with `hierarchical_` | Pass — **18,747/18,747** rows are `hierarchical_semantic`; 0 rows are `flat_semantic`, an explicit fallback, or a missing method |
| Zero row-level `error` | Pass (0 non-blank errors) |
| Stage-1..4 candidate fields are non-empty valid JSON lists | Pass — **18,747/18,747** rows, 0 bad-evidence rows |
| Every stage-1..4 latency ≤ 30,000 ms | Pass — **0** rows over the cap; **maximum stage latency observed across all 18,747 rows and all 4 stages: 3,098.16 ms** (≈10% of the 30s cap; no case approached the limit) |
| All 24 fixed Task 12 known-risk IDs present and individually pass method/stage-evidence/latency/error checks | Pass — **24/24** present, **24/24** passing, **0** failing |
| Reranker/LLM off | `reranker_fired=true` on 0 rows; `reranker_model` blank on all rows; `estimated_cost_usd`/token fields zero on all rows |
| ISIC/ISCED/SRE not constructed | `pred_isic_section`/`pred_isced_level` blank on all rows; `sre_status="not_applicable"` on all rows |
| Timing fields | Preserved per-row; **no aggregate latency/throughput/cost was computed** |

**All-row strict pass: 18,747/18,747 (100% of rows individually
satisfy every one of the raw integrity checks above).** This is a
row-level integrity tally, not an accuracy measurement — no row's
`pred_isco_4digit` was compared against `gold_isco_4digit` anywhere in
this task.

For traceability only (not interpreted as quality/performance):
`keyword_anchor_retry_used=true` on 21/18,747 rows (all 21 are among the
24 known-risk IDs — the same 21 that required the Task 13 retry path in
the Task 15 preflight); `stage1_source` distribution:
`semantic_retrieval=14668`, `keyword_map=4079`.

## 8. Qdrant before/after point counts

| Collection | Before | After |
|---|---|---|
| `isco08_major_groups` | 10 | 10 |
| `isco08_submajor_groups` | 43 | 43 |
| `isco08_minor_groups` | 131 | 131 |
| `isco08_unit_groups` | 441 | 441 |
| `isco_occupations` | 124 | 124 |

Identical before and after both commands (`diff` of the two recorded
count files produced no output).

## 9. Explicit confirmation of scope boundaries

- No reranker, Ollama, CrewAI, LLM, paid API, or external inference call
  was made in either command (confirmed row-by-row in §6/7).
- No ISIC, ISCED, or SRE evaluation occurred — both runs' own printed
  guard confirmed *"No row has both industry_text and education_text --
  ISICClassifier/ISCEDClassifier/SemanticRelationEngine will not be
  constructed; this is an ISCO-08-only retrieval run."*
- No Qdrant mutation, build, population, rebuild, or deletion occurred —
  read-only search calls only; point counts unchanged (§8).
- No `eval/analyze.py` invocation, no accuracy/Wilson/McNemar/p-value
  computation, no aggregate latency/throughput/cost value, no chart,
  ranking, or manuscript-ready comparison table was produced.
- No source, test, configuration, or data file was changed to obtain a
  pass — both commands passed on their first and only execution; no
  retry was needed or performed.
- No B1 re-freeze or B2 sweep occurred; `git status --short` confirms no
  file outside this report changed on this branch.
- No third evaluation command, reduced/selected subset, or ablation was
  run — exactly the two authorized commands, in the required order, each
  exactly once.

## 10. Protected-branch status

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| `reviewer2-wip-snapshot-20260807` | not touched |
| `reviewer2-b2-integration-20260807` | not touched |
| `reviewer2-isic-iscedf-hierarchical-rag-20260808` | not touched |
| `reviewer2-isic-iscedf-hierarchical-resilience-20260808` | not touched |
| `reviewer2-isic-iscedf-hierarchical-model-init-resilience-20260808` | not touched |
| `reviewer2-isic-iscedf-integration-20260808` | not touched |
| `reviewer2-isic-iscedf-test-green-20260808` | not touched |
| `reviewer2-pre-evaluation-baseline-20260808` | not touched |
| `reviewer2-model-free-isco-evaluation-20260808` | not touched |
| `reviewer2-wisco-measurement-baseline-20260808` | not touched |
| `reviewer2-wisco-tier1-preflight-20260808` | not touched |
| `reviewer2-wisco-tier1-results-20260808` | not touched |
| `reviewer2-hierarchical-integrity-hardening-20260808` | not touched |
| `reviewer2-wisco-strict-benchmark-baseline-20260808` | not touched |
| `reviewer2-wisco-strict-preflight-20260808` | not touched |
| `reviewer2-wisco-strict-preflight-integration-20260808` | not touched (used only as the branch-creation base; never checked out for writing) |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push was run at any point in this task. No PR was
created.

## 11. Evidence boundary

```text
If completed successfully, these raw artifacts are eligible only for a later controlled multilingual ISCO-08 analysis task. They are not themselves an accuracy claim, a comparative-performance result, real-LFS validation, population-representative evidence, ISIC/ISCED/SRE evidence, or manuscript-ready result.
```

Both raw CSVs (18,747 rows each, flat and hierarchical-strict) remain
under the Git-ignored output root
`eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/` for a
future, separate analysis task to consume. Per this task's stop
condition, no analysis, metric computation, or manuscript statement was
produced here, and no further task was begun.
