STRICT_PREFLIGHT_READY: yes

# Task 15 Final Report — Strict WISCO High-Coverage Preflight and Known-Risk Replay

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_15_STRICT_WISCO_PREFLIGHT.md`.
This is a bounded, diagnostic readiness check only — not a benchmark
result, accuracy study, performance comparison, or ablation.

## 1. Source base branch and verified SHA

| Role | Branch | SHA |
|---|---|---|
| Base | `reviewer2-wisco-strict-benchmark-baseline-20260808` | `3716057db699527761127383f430b786fa21ff46` |

`git rev-parse HEAD` on the starting checkout, `git rev-parse
origin/reviewer2-wisco-strict-benchmark-baseline-20260808`, and the task
file's required SHA all matched exactly before any change was made.

## 2. New branch, commit SHA, remote-push confirmation

New branch: `reviewer2-wisco-strict-preflight-20260808`, created with
`git switch -c reviewer2-wisco-strict-preflight-20260808` directly from
the verified base SHA above (no other branch involved).

Commit and push confirmation are recorded after this report is committed
(see the closing steps of this task) — this branch contains exactly one
new commit on top of the base, adding this report.

## 3. Clean-tree confirmation before and after

- `git status --short` was empty immediately before branching.
- `git status --short` was empty immediately after the live evaluation
  command completed (all generated artifacts live under the Git-ignored
  `eval/local_runs/wisco_v2_strict_high_coverage_preflight_20260808T111903Z/`
  output root, confirmed covered by the existing `eval/local_runs/`
  `.gitignore` pattern — the same pattern used by Tasks 11–14).
- `git status --short` was checked again immediately before this
  report's commit and showed only this new file.

## 4. Exact changed tracked files

Only:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_15_FINAL_REPORT.md
```

No production code, test, configuration, dependency, dataset, benchmark
split, or historical raw output file was changed.

## 5. Preflight-gate results

All gates were re-verified live against the on-disk WISCO v2 package (not
read from a cached summary alone) using the project's own
`recompute_dataset_hash` / `run_full_audit` (from
`eval/audit_wisco_benchmark_leakage.py`) and `validate_benchmark_package`
(from `eval/validate_controlled_benchmark.py`), plus a direct read of the
canonical heldout CSV and a live, read-only Qdrant collections query.

| Gate | Result |
|---|---|
| 1. Dataset package hash | Recomputed `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c` — **matches** `dataset_hash.txt` exactly |
| 2. Package record total | `20760` (matches required exactly) |
| 3. Split counts | dev `2013`, heldout `18747` (matches required exactly) |
| 4. Leakage audit | `leakage_found: false`, `source_family_split_check.ok: true`, `text_duplicate_check.ok: true` — zero source-family leakage, zero cross-split exact-duplicate groups |
| 5. Heldout code validation | `0` malformed ISCO-08 codes across all 18,747 heldout records; full-package `validate_benchmark_package` also returned `ok: true` with zero errors/warnings |
| 6. Canonical heldout ordering | `heldout_run_eval_format.csv` contains exactly `18747` data rows, all `case_id` values unique |
| 7. Qdrant collections | All 5 required collections present and non-empty (point counts below); required name confirmed as `isco08_submajor_groups` (no underscore between "sub" and "major"), matching `backend/rag/hierarchical_store.py`'s `_COL_SUBMAJOR` constant |

Qdrant point counts, before the live evaluation command:

| Collection | Points |
|---|---|
| `isco08_major_groups` | 10 |
| `isco08_submajor_groups` | 43 |
| `isco08_minor_groups` | 131 |
| `isco08_unit_groups` | 441 |
| `isco_occupations` | 124 |

All 7 gates passed. Nothing was executed conditionally on a partial
gate result.

## 6. Dataset hash, record/split totals, selection seed, base size, known-risk overlap, final size, distribution summary

- Dataset hash: `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c`
- Package total: `20760` records (dev `2013` / heldout `18747`)
- Base selection: `eval/select_wisco_reranking_subset.py --records
  eval/local_benchmarks/wisco_isco08_v2_group_split/records.json --split
  heldout --target-size 500 --seed 42` — deterministic stratification by
  `(language, ISCO major group)`, SHA-256 ranking, largest-remainder
  allocation. Produced exactly `500` records (`en=101, ar=100, ur=97,
  hi=101, tl=101`).
- Known-risk fixed list: `24` IDs (verbatim from the task file).
- Overlap between the base 500 and the known-risk 24: `0` (none of the 24
  known-risk IDs happened to already be in the stratified 500).
- Final selection size: `524` (`500 + 24`, within the required 500–524
  range). All 24 known-risk IDs confirmed present in the final selection.
- Final-selection language distribution: `en=112, ar=111, ur=98, hi=102,
  tl=101`.
- Final-selection ISCO major-group distribution: `0=5, 1=30, 2=116, 3=94,
  4=26, 5=40, 6=43, 7=93, 8=45, 9=32`.
- The selected records were exported to
  `eval/local_runs/wisco_v2_strict_high_coverage_preflight_20260808T111903Z/strict_high_coverage_selected.csv`
  in **original canonical heldout source order** (filtered from
  `heldout_run_eval_format.csv` preserving its row order, not sorted by
  the selection algorithm) — `524` data rows, header
  `case_id,input_text,input_language,gold_isco_4digit,gold_isic,gold_isced`,
  every `case_id` appearing exactly once, `gold_isic`/`gold_isced`
  explicitly blank (they were already blank in the source heldout CSV;
  the export script blanks them again defensively). No test-data value
  was invented, translated, or modified. The full selection manifest
  (base selection metadata, the fixed known-risk list, overlap detail,
  final ID list in canonical order, and both distributions) is at
  `.../selection_manifest.json` under the same ignored output root.

## 7. Exact authorized evaluation command and exit status

```bash
QDRANT_TIMEOUT_SECONDS=30 python eval/run_eval.py \
  --test-set eval/local_runs/wisco_v2_strict_high_coverage_preflight_20260808T111903Z/strict_high_coverage_selected.csv \
  --system hierarchical \
  --use-llm-reranker off \
  --require-genuine-hierarchical \
  --max-stage-latency-ms 30000 \
  --config wisco_v2_strict_high_coverage_preflight \
  --run-id wisco-v2-strict-high-coverage-preflight \
  --output-dir eval/local_runs/wisco_v2_strict_high_coverage_preflight_20260808T111903Z/run_output
```

No `--limit`, reranker model, LLM flag, fallback override, retry
wrapper, or broader run was added — the command above is verbatim what
was executed, exactly once.

**Exit status: 0.** Completed all `524` cases in `63.8s` (`0.12s/case`
mean wall time) and wrote one output CSV:
`eval/local_runs/wisco_v2_strict_high_coverage_preflight_20260808T111903Z/run_output/20260808T112809Z_wisco_v2_strict_high_coverage_preflight.csv`.
The run's own header line recorded `config_hash=6e57d4759e31`,
`reranker_model=none (reranking disabled)`, `keyword_map_enabled=True`,
and: *"No row has both industry_text and education_text --
ISICClassifier/ISCEDClassifier/SemanticRelationEngine will not be
constructed; this is an ISCO-08-only retrieval run."* Every output row
recorded `git_commit=3716057` (short form of the verified base SHA) and
`seed=42`.

Because the command exited 0, `--require-genuine-hierarchical` itself
confirms every one of the 524 rows passed `check_strict_hierarchical()`
before any CSV write occurred — no partial/fallback row could have been
silently written.

## 8. Raw ignored output-root paths

All under the Git-ignored root
`eval/local_runs/wisco_v2_strict_high_coverage_preflight_20260808T111903Z/`:

- `stratified_500.json` — raw output of the base selector
- `selection_manifest.json` — full selection record (base + known-risk + final)
- `strict_high_coverage_selected.csv` — the 524-row input CSV, canonical heldout order
- `run_stdout.log` — full stdout/stderr of the evaluation command
- `qdrant_point_counts_before.txt` / `qdrant_point_counts_after.txt`
- `gate_report.json` — live preflight-gate recomputation output
- `postrun_verification_report.json` — full post-run integrity-check output (below)
- `run_output/20260808T112809Z_wisco_v2_strict_high_coverage_preflight.csv` — the run's own output CSV

## 9. Strict post-run integrity checks (row-level, factual only)

All checks below were computed directly from the output CSV's 524 rows
(script: `postrun_verification_report.json` in the output root), not
inferred from the exit code alone.

| Check | Result |
|---|---|
| 1. Row count == selected case count, all IDs present exactly once, in original selected order | `524 == 524`; IDs match the selected-CSV order exactly; all unique |
| 2. Every row's method begins with `hierarchical_` | `524/524` — all rows `pred_method=hierarchical_semantic`; zero `flat_semantic`, zero explicit fallback, zero missing-method rows |
| 3. Non-empty valid JSON list for stage1–stage4 candidates | `524/524` rows pass on all 4 stage columns; zero bad rows |
| 4. Every recorded hierarchy stage latency ≤ 30,000 ms | `524/524` pass; **maximum stage latency observed across the entire run: 278.82 ms** (far below the 30s cap — no case approached the limit) |
| 5. All row-level `error` fields blank | `524/524` blank |
| 6. Reranking/LLM disabled throughout | `reranker_fired` true for `0/524` rows; per-row `reranker_model` blank for all rows (config-level `reranker_model_resolved` = `none (reranking disabled)`, as printed in the run header); `estimated_cost_usd` and `prompt_tokens`/`completion_tokens` are `0`/blank for all 524 rows; no Ollama, paid API, or external LLM call was made |
| 7. ISIC/ISCED/SRE not constructed | Confirmed by the run's own printed guard (quoted in §7). Per-row: `pred_isic_section` and `pred_isced_level` blank for all 524 rows; explicit `sre_status = "not_applicable"` for all 524 rows |
| 8. All 24 known-risk IDs present and passing | `0` missing, `0` failing the genuine-hierarchy/stage-limit checks — all 24 pass |
| 9. Keyword-anchor retry / stage-1 source (diagnostic only) | `keyword_anchor_retry_used = true` on `21/524` rows (all 21 are known-risk IDs from the fixed list; the other 3 known-risk IDs resolved without a retry). `stage1_source` distribution: `semantic_retrieval=422`, `keyword_map=102`. These counts are recorded for traceability only and are **not** interpreted as an accuracy, quality, or performance signal |
| 10. Qdrant point counts unchanged | Before: `major=10, submajor=43, minor=131, unit=441, isco_occupations=124`. After: identical, byte-for-byte diff empty |

No accuracy, Wilson interval, McNemar test, aggregate latency statistic,
cost figure, throughput number, or chart/table was computed.
`eval/analyze.py` was never invoked.

## 10. Explicit confirmation of scope boundaries

- No full 18,747-row benchmark was run — only the 524-case strict
  selection.
- No flat baseline was run (`--system hierarchical` only).
- No reranker, Ollama, LLM, or paid API call was made (confirmed by
  §9 row 6).
- No Qdrant collection was mutated, rebuilt, or populated — read-only
  search calls only, point counts unchanged (§9 row 10).
- No B1 re-freeze or B2 sweep occurred; `eval/configs/b1_frozen.json` and
  `eval/dev_sweep.py` were not touched (`git status --short` confirms no
  file outside this report changed).
- No ISIC/ISCED-F benchmark, no SRE benchmark, no `eval/analyze.py`
  invocation.
- No source, test, or configuration file was changed to obtain a pass —
  the strict run passed on its first and only execution; no retry was
  needed or performed.
- No known-risk case was excluded from selection; all 24 were included
  and none needed a rerun.

## 11. Protected-branch status

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
| `reviewer2-wisco-strict-benchmark-baseline-20260808` | not touched (used only as the branch-creation base; never checked out for writing) |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push was run against any branch in this task. No PR
was created.

## 12. What this preflight does and does not establish

This is only a diagnostic readiness check. It confirms that, after
Task 13's integrity hardening (retry-without-seed recovery + bounded
Qdrant timeout) and Task 14's integration, the hierarchical pipeline can:

- process a deterministic, stratified, multilingual 524-case sample
  spanning all 10 ISCO major groups and all 5 languages, and
- correctly re-process every one of Task 12's 24 original silent-fallback
  case IDs without falling back and without any stage approaching the
  30-second cap (observed maximum: 278.82 ms — roughly two orders of
  magnitude under the limit),

under strict, model-free, reranker-off, read-only conditions.

It contributes **no manuscript performance evidence**. It does not
measure accuracy (no gold-vs-pred comparison was performed — the output
CSV contains `gold_isco_4digit` and `pred_isco_4digit` but this report
draws no conclusion from comparing them), latency distribution, cost,
throughput, scalability, ISIC/ISCED/SRE performance, or real-LFS
validity. It does not authorize or imply a full-benchmark rerun. Per the
task's stop condition, no further evaluation, rerun, or expansion of
scope follows this report.
