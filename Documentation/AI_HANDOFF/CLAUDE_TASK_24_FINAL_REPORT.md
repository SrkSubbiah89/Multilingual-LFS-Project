OFFICIAL_TIER1_COMPLETED: no

# Task 24 Final Report — Full Controlled WISCO ISCO-08 Evaluation Using the Official ILO Profile

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_24_FULL_WISCO_OFFICIAL_PROFILE_RAW_EVALUATION.md`.
This is a raw, integrity-checked evaluation run only. Per the task's own
fail-closed instruction, the run was stopped after the flat
output-integrity gate failed on 1 of 18,747 rows — the hierarchical
command was never started, and no retry, repair, or data alteration was
performed.

## 1. Branch, base SHA, final SHA, push, clean-tree status

| | |
|---|---|
| Base branch | `reviewer2-official-isco08-collection-build-smoke-20260809` |
| Required SHA | `0ce9e6778c5fe982babd446a26f8445a959c6783` |
| Verified `origin` SHA | `0ce9e6778c5fe982babd446a26f8445a959c6783` — match |
| New branch | `reviewer2-wisco-official-tier1-raw-results-20260809` |
| Final commit SHA | recorded after this report's commit (see push confirmation below) |

Working tree was clean immediately before branching, clean throughout
(all generated artifacts live under the Git-ignored
`eval/local_runs/wisco_official_tier1_20260808T213656Z/` output root —
confirmed via `git check-ignore -v` on every artifact path), and clean
immediately before this report's own commit.

## 2. Exact commands executed

**Heldout export** (Part A.1):

```bash
python eval/export_benchmark_to_run_eval_csv.py \
  --records eval/local_benchmarks/wisco_isco08_v2_group_split/records.json \
  --split heldout \
  --out eval/local_runs/wisco_official_tier1_20260808T213656Z/heldout_export_fresh.csv
```

Exit 0. Wrote 18,747 rows.

**Official flat comparator** (Part B.1, run once):

```bash
QDRANT_TIMEOUT_SECONDS=30 python eval/run_eval.py \
  --test-set eval/local_runs/wisco_official_tier1_20260808T213656Z/heldout_export_fresh.csv \
  --system flat \
  --use-llm-reranker off \
  --isco-catalogue-profile official_ilo2021_v1 \
  --config wisco_official_tier1_flat \
  --run-id wisco-official-tier1-flat \
  --output-dir eval/local_runs/wisco_official_tier1_20260808T213656Z/flat
```

Log shows successful completion (no traceback; "Completed 18747 case(s)
in 719.3s (0.04s/case)"; "Wrote 18747 row(s) to
.../flat/20260808T214023Z_wisco_official_tier1_flat.csv" printed —
`run_eval.py` only prints this line and returns normally on a
non-aborted run).

**Strict official hierarchical retrieval** (Part B.2): **not run.** The
flat gate (Part C.1) failed, and the task requires stopping before this
command under that condition.

## 3. WISCO package/split/hash checks and official catalogue hash

Re-verified live via `eval.audit_wisco_benchmark_leakage.recompute_dataset_hash`/
`run_full_audit` and `eval.validate_controlled_benchmark.validate_benchmark_package`,
not read from a cached summary:

| Check | Result |
|---|---|
| Dataset hash | Recomputed `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c` — matches `dataset_hash.txt` exactly |
| Package totals | `20760` total; dev `2013`, heldout `18747` (all exact) |
| Leakage audit | `leakage_found: false`; `source_family_split_check.ok: true`; `text_duplicate_check.ok: true` |
| Heldout code validation | `0` malformed ISCO-08 codes across 18,747 heldout records |
| Full-package validation | `validate_benchmark_package` returned `ok: true`, zero errors/warnings |
| Fresh heldout export | `18747` rows; SHA-256 `41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c` (byte-identical to Task 17's independently-produced export of the same canonical source, confirming determinism); required fields present; zero `industry_text`/`education_text` columns; `0` non-blank `gold_isic`/`gold_isced` values; all `case_id` unique |
| Official normalized catalogue SHA-256 | `29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3` — matches exactly |
| `eval/verified_catalogue_counts.yaml` | Unchanged; WISCO-independence statement present verbatim; official counts `10/43/130/436` |
| Task 23 success manifest | `status: "success"`, `profile: "official_ilo2021_v1"`, `embedding_model_identity: "intfloat/multilingual-e5-small"`, `embedding_vector_dim: 384` — confirmed present |

All Part A gates passed; both commands were authorized to begin (the
hierarchical command was subsequently withheld per §5's stop condition,
not because a Part A gate failed).

## 4. Pre-run, after-flat, and after-hierarchical Qdrant inventories

**Pre-run** (before any command in this task):

| Collection | Count |
|---|---:|
| `isco08_major_groups_ilo2021_v1` | 10 |
| `isco08_submajor_groups_ilo2021_v1` | 43 |
| `isco08_minor_groups_ilo2021_v1` | 130 |
| `isco08_unit_groups_ilo2021_v1` | 436 |
| `isco08_unit_groups_flat_ilo2021_v1` | 436 |
| `isco08_major_groups` (legacy) | 10 |
| `isco08_submajor_groups` (legacy) | 43 |
| `isco08_minor_groups` (legacy) | 131 |
| `isco08_unit_groups` (legacy) | 441 |
| `isco_occupations` (legacy) | 124 |

All 5 official counts matched the required table exactly; legacy counts
matched the required Task 23 inventory exactly.

**After-flat**: identical to pre-run in every collection (`diff` of the
two recorded inventory files produced no output).

**After-hierarchical**: not applicable — the hierarchical command was
never run.

## 5. Raw output and manifest paths (Git-ignored)

All under `eval/local_runs/wisco_official_tier1_20260808T213656Z/`
(confirmed Git-ignored via `.gitignore:52:eval/local_runs/` for every
path listed):

```text
wisco_package_integrity_report.json
heldout_export_fresh.csv
pre_run_qdrant_inventory.txt
post_flat_qdrant_inventory.txt
flat_stdout.log
flat/20260808T214023Z_wisco_official_tier1_flat.csv
flat_integrity_gate_report.json
failing_row_detail.json
```

No `hierarchical/` subdirectory or hierarchical manifest exists — that
command was never invoked.

## 6. Per-run wall-clock time and exit code

| Run | Wall-clock | Exit |
|---|---|---|
| Heldout export | not timed (sub-second, deterministic reformat) | 0 |
| Official flat comparator | 719.3s (0.04s/case) | 0 (no traceback; success message printed) |
| Official hierarchical retrieval | not run | not applicable |

## 7. Flat integrity-gate result (Part C.1) — FAILED

| # | Condition | Result |
|---|---|---|
| 1 | Command exits 0 | Pass |
| 2 | Exactly 18,747 rows in raw output | Pass — 18,747 |
| 3 | Every row has no row-level `error` | Pass — 0 non-blank `error` fields |
| 4 | Every `pred_method` exactly `flat_isco08_official_ilo2021_v1` | **FAIL** — 18,746/18,747 rows correct; 1 row (`WISCO-8131001300018-ar`) has `pred_method = unavailable_isco08_official_ilo2021_v1` |
| 5 | Every predicted ISCO code syntactically valid, exactly 4 digits | **FAIL** — 18,746/18,747 valid; the same 1 row has a blank `pred_isco_4digit` |
| 6 | Reranking disabled, zero reranker trace/cost/token activity | Pass — `reranker_fired` true on 0 rows; `reranker_model` blank on all rows; `estimated_cost_usd`/token fields zero on all rows |
| 7 | No ISIC/ISCED/SRE classifier/engine constructed or run | Pass — `pred_isic_section`/`pred_isced_level` blank on all rows; `sre_status="not_applicable"` on all rows |
| 8 | Read-only Qdrant post-run counts match pre-run exactly | Pass — identical (§4) |

**Gate result: FAILED** (conditions 4 and 5). 6 of 8 conditions passed;
2 failed, both caused by the same single row.

### The one failing row, preserved unmodified

```json
{
  "case_id": "WISCO-8131001300018-ar",
  "input_language": "ar",
  "gold_isco_4digit": "8131",
  "pred_isco_4digit": "",
  "pred_confidence": "0.0",
  "pred_method": "unavailable_isco08_official_ilo2021_v1",
  "pred_reasoning": "No candidates returned by the vector store.",
  "error": "",
  "end_to_end_latency_ms": "60743.26",
  "stage1_latency_ms..stage4_latency_ms": all "0.0",
  "input_order_position": "16303"
}
```

Recorded for traceability only, not as a diagnosed root cause (no fix
was attempted, per the task's explicit restrictions): this row's
`end_to_end_latency_ms` (60,743 ms) is roughly 1,500x the run's mean
per-case time (≈40 ms) and about 2x the `QDRANT_TIMEOUT_SECONDS=30`
value set for this run, while every per-stage latency field reads
`0.0` and the row's own `error` field is blank. The system's fail-closed
design correctly refused to emit a fabricated code for this case
(`pred_method="unavailable_isco08_official_ilo2021_v1"`,
`pred_confidence="0.0"`) rather than silently returning a wrong or
default 4-digit value — but this is exactly the situation the flat
gate's conditions 4 and 5 exist to catch: a raw output file containing
even one non-nominal row cannot be treated as complete, uniform
benchmark evidence.

## 8. Hierarchical integrity-gate result (Part C.2)

Not applicable. Per Part B.2's own precondition ("Run only after the
flat gate fully passes") and Part C.1's own instruction ("If any
condition fails: preserve the output, set `OFFICIAL_TIER1_COMPLETED:
no`, do not run Part B.2, do not retry, and stop"), the hierarchical
command was never started and no hierarchical gate was evaluated.

## 9. Whether each authorized command was run exactly once

| Command | Run count |
|---|---:|
| Heldout export | 1 |
| Official flat comparator | 1 |
| Official hierarchical retrieval | 0 (withheld per stop condition) |

No command was run twice. No retry, reduced subset, `--limit`, or
alternate configuration was attempted for the flat run after its gate
failed.

## 10. OFFICIAL_TIER1_COMPLETED

```text
OFFICIAL_TIER1_COMPLETED: no
```

Triggered stop condition: **Part C.1, conditions 4 and 5** — 1 of 18,747
rows in the official flat comparator's raw output has a `pred_method`
other than `flat_isco08_official_ilo2021_v1` and a predicted code that
is not a syntactically valid 4-digit ISCO-08 code. Per Part C.1's
explicit instruction, the raw flat output was preserved unmodified, the
Part B.2 hierarchical command was not run, no retry was performed, and
this task stops here.

## 11. Focused and full test results

```
python -m pytest backend/tests/test_build_official_isco08_collections.py backend/tests/test_official_isco08_catalogue.py backend/tests/test_official_isco08_profiles.py backend/tests/test_isco_classifier_official_profile.py eval/test_official_isco08_profile_evaluator.py eval/test_model_free_isco_evaluation.py eval/test_require_genuine_hierarchical.py eval/test_docs_consistency.py eval/test_wisco_leakage_audit.py eval/test_analyze_wisco_tier1.py -q
→ 156 passed in 2.01s

python -m pytest backend/tests eval/ -q
→ 2056 passed, 1 deselected, 1 warning in 301.77s (0:05:01)
```

**Zero failures.** Exact match to Task 23's baseline (2056) — this task
changed no source or test code; the flat gate's failure is a property of
one run's raw output, not a code defect this task attempted to
diagnose, repair, or hide. No unrelated test was skipped, altered, or
weakened.

## 12. Protected-branch status

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| `reviewer2-wip-snapshot-20260807` | not touched |
| `reviewer2-b2-integration-20260807` | not touched |
| every prior `reviewer2-*` task/integration branch (through `reviewer2-official-isco08-collection-build-smoke-20260809`) | not touched |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push occurred. **No PR was created.**

## 13. Evidence boundary

```text
These are raw outputs from a controlled multilingual ISCO-08 benchmark
using a public occupation-title dataset. They are not real Labour Force
Survey validation and have not yet been analysed for accuracy, statistical
comparison, or manuscript-ready conclusions. They provide no ISIC, ISCED,
SRE, reranking, cost, or real-world performance evidence.
```

This statement applies doubly here: beyond the general boundary above,
this task's own raw flat output additionally **failed its own
operational integrity gate** and is not eligible for any downstream
analysis task until a separately authorized task investigates the one
non-nominal row. No accuracy scoring, statistical comparison,
reranking, paper drafting, B1 re-freeze, or B2 sweep was run. Stopping
here, per the task's own instruction, after pushing this report.
