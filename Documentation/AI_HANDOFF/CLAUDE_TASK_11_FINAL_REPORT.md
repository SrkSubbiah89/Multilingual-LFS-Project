# Task 11 Final Report — WISCO Tier-1 Local Preflight and Path-Integrity Gate

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_11_WISCO_TIER1_PREFLIGHT.md`
(task ID `11-wisco-tier1-preflight`). **This is not the benchmark and must
never be reported as performance evidence** — it is a small (5-case),
non-manuscript-eligible local readiness check for the two planned
full-heldout, retrieval-only Tier-1 measurements.

## Branch / SHA

| | |
|---|---|
| Base branch | `reviewer2-wisco-measurement-baseline-20260808` |
| Base SHA (verified) | `4e772669449703ea09b6c3a33f32f5fa409c6a3b` |
| Working branch | `reviewer2-wisco-tier1-preflight-20260808` (new, created by this task) |
| Final SHA | commit created and pushed as this task's own closing step, containing only this report |

Start-state check passed exactly: `git fetch origin` confirmed
`origin/reviewer2-wisco-measurement-baseline-20260808` at
`4e772669449703ea09b6c3a33f32f5fa409c6a3b`; the local branch of that name
was already checked out at that exact SHA with a clean working tree
before `git switch -c reviewer2-wisco-tier1-preflight-20260808` was run.

## A. Asset and split-integrity gate

**The v2 package (`eval/local_benchmarks/wisco_isco08_v2_group_split/`)
was already present locally and was validated live, not rebuilt** — its
own `dataset_hash.txt`/`records.json` were used to independently
recompute the dataset hash and re-run the exact deterministic leakage/
integrity audit (`audit_wisco_benchmark_leakage.run_full_audit`) against
the live file contents, rather than trusting the cached
`build_summary.json` alone. All required results confirmed:

| Check | Required | Actual |
|---|---|---|
| Dataset hash | `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c` | `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c` (match) |
| Total records | 20,760 | 20,760 |
| Dev split | 2,013 | 2,013 |
| Heldout split | 18,747 | 18,747 |
| Source-family leakage | zero | `n_leaking_source_keys=0` |
| Cross-split exact-text duplicate groups | zero | `n_exact_duplicate_groups_cross_split=0` |
| Malformed ISCO codes | zero | `n_malformed_isco_codes=0` |
| `overall_ok_no_leakage` | true | `True` |

Since the package was present and every check above passed, the
deterministic rebuild command (`build_wisco_isco_benchmark_v2_group_split.py`)
was **not** invoked — v1 was correspondingly not touched either.

**Heldout export** (did not exist before this task, so it was generated
via the exact specified command):

```
python eval/export_benchmark_to_run_eval_csv.py --records eval/local_benchmarks/wisco_isco08_v2_group_split/records.json --split heldout --out eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv
→ Wrote 18747 row(s) to eval\local_benchmarks\wisco_isco08_v2_group_split\heldout_run_eval_format.csv
```

Verified directly:

- **Row count**: exactly 18,747 data rows.
- **Columns**: `case_id, input_text, input_language, gold_isco_4digit, gold_isic, gold_isced` — all four required `run_eval.py` columns present.
- **`industry_text`/`education_text`**: no such columns exist in this export at all (not merely blank) — an ISCO-only export, as expected for this benchmark.
- **ISIC/ISCED gold labels**: `gold_isic`/`gold_isced` columns exist in the schema but contain zero non-blank values across all 18,747 rows — no actual ISIC/ISCED gold label is present anywhere in this file.

## B. Non-destructive collection-readiness gate

Read-only Qdrant query only (`QdrantClient.get_collections()` /
`get_collection()` — no create/delete/recreate/upsert call made anywhere
in this task).

| | |
|---|---|
| Host | `localhost` (from `QDRANT_HOST` env default) |
| Port | `6333` (from `QDRANT_PORT` env default) |

| Collection on server | Points | Status |
|---|---|---|
| `isco08_major_groups` | 10 | green |
| `isco08_submajor_groups` | 43 | green |
| `isco08_minor_groups` | 131 | green |
| `isco08_unit_groups` | 441 | green |
| `isco_occupations` (flat) | 124 | green |

**Naming note**: the task text's Part B step 2 lists the four
hierarchical collections as `isco08_major_groups`,
**`isco08_sub_major_groups`**, `isco08_minor_groups`,
`isco08_unit_groups`. The real production retrieval path
(`backend/rag/hierarchical_store.py`'s `_COL_MAJOR/_COL_SUBMAJOR/
_COL_MINOR/_COL_UNIT` constants — the authoritative source, per this
task's own instruction to check "the real production retrieval path")
names the second collection **`isco08_submajor_groups`** (no second
underscore). No collection named `isco08_sub_major_groups` exists or is
ever queried by production code — this is a naming discrepancy in the
task text, not a missing collection. All four collections actually
required by production code are present and non-empty, as is the flat
collection `isco_occupations` required by `--system flat`. **Readiness:
PASS** for all five required collections.

## C. Five-case path-integrity smoke test

Selected the first five records in source order directly from the fixed
exported heldout CSV (all five happen to be the five language variants of
the same first WISCO occupation, `WISCO-110000200018-{en,ar,ur,hi,tl}`,
gold `0110`) — selection was purely positional (first five rows), never
based on a prediction, language, code, or expected outcome.

**Exact commands run** (both `--limit 5`, `--use-llm-reranker off`, no
`--reranker-model`):

```
python eval/run_eval.py --test-set eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv --system flat --use-llm-reranker off --limit 5 --config wisco_v2_preflight_flat_norerank --run-id wisco-v2-tier1-preflight-flat --output-dir eval/local_runs/wisco_v2_tier1_preflight/flat
→ Wrote 5 row(s) to eval\local_runs\wisco_v2_tier1_preflight\flat\20260807T233801Z_wisco_v2_preflight_flat_norerank.csv
→ Completed 5 case(s) in 1.4s (0.28s/case)

python eval/run_eval.py --test-set eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv --system hierarchical --use-llm-reranker off --limit 5 --config wisco_v2_preflight_hierarchical_norerank --run-id wisco-v2-tier1-preflight-hierarchical --output-dir eval/local_runs/wisco_v2_tier1_preflight/hierarchical
→ Wrote 5 row(s) to eval\local_runs\wisco_v2_tier1_preflight\hierarchical\20260807T233859Z_wisco_v2_preflight_hierarchical_norerank.csv
→ Completed 5 case(s) in 3.2s (0.64s/case)
```

Both runs' console output printed, before any case ran:
`No row has both industry_text and education_text -- ISICClassifier/
ISCEDClassifier/SemanticRelationEngine will not be constructed; this is
an ISCO-08-only retrieval run.` — proving these three components were
never constructed at all for this run, not merely left unused per row.

**Path-integrity checks (5/5 rows, each file), inspected directly — no
accuracy calculated or reported:**

| Check | Result |
|---|---|
| `evaluation_status` | Every row: `"measured"` — the repository's exact non-dry-run status (`CaseResult`'s dataclass default; the only other value ever used anywhere in this codebase is `"dry_run_not_measured"` for `--dry-run`). The task text's literal string `"measured_synthetic_or_operationally_realistic"` does not exist in this repository; `"measured"` is confirmed as its exact real-run equivalent. Never a real-LFS label. |
| `dataset_label` | **Column does not exist in `CaseResult`'s schema at all** in this runner version — recorded here precisely per the task's fallback instruction, not fabricated. |
| Reranker disabled | Run-level (printed, both commands): `reranker_model=none (reranking disabled)` — the exact Task 09 disabled disclosure. Per-row CSV: `reranker_fired=False`, `reranker_model=''`, `reranker_model_version=''`, `reranker_latency_ms=0.0`, `reranker_input_candidates=[]`, `reranker_output={}`, `prompt_tokens=0`, `completion_tokens=0`, `estimated_cost_usd=0.0` — uniformly across all 10 rows. No reranker trace, call, or latency of any kind. |
| ISIC/ISCED/SRE fields | Every row: `pred_isic_section=''`, `pred_isced_level=''`, `sre_status="not_applicable"`, `sre_status_reason="industry_text and/or education_text not supplied on this row"`. Confirmed blank/`not_applicable`, and (per the console message above) no such classifier was constructed at all. |
| Flat rows use a flat method | Every flat row: `pred_method="flat_semantic"` — never hierarchical. |
| Hierarchical rows use a genuine hierarchical method with stage evidence | Every hierarchical row: `pred_method="hierarchical_semantic"` (production code only ever assigns the `"hierarchical_"` prefix when `fallback_used` is False — see `backend/agents/isco_classifier.py`'s `method_prefix = "flat" if h.fallback_used else "hierarchical"`). All four stages populated: `stage1_source="semantic_retrieval"`, `stage1_candidates`/`stage2_candidates`/`stage3_candidates`/`stage4_candidates` all non-null with real code/label/score entries (stage1: 2 candidates, stage2: 4, stage3: 6-8, stage4: 13-25, per row), and `stage1_latency_ms`..`stage4_latency_ms` all populated non-empty floats. |
| No fallback/flat-only label on hierarchical rows | Confirmed — `pred_method` is `"hierarchical_semantic"` on all 5 hierarchical rows; never `"flat_semantic"`, `"flat_llm"`, or any fallback-labelled value. |
| Row-level errors | `error=''`, `degraded=False`, `invalid_output_flag=False`, `timed_out_flag=False` on all 10 rows. |

**Conclusion**: the hierarchical smoke test did not fall back, had no
missing stage evidence, produced zero row-level errors, and proved
genuine 4-stage hierarchical retrieval. No blocker condition (Part C.4)
was triggered.

**Smoke latency/environment diagnostics only (never accuracy, never a
manuscript-ready metric)**:

- Flat: 5 cases in 1.4s total (0.28s/case mean); embedding model
  `intfloat/multilingual-e5-small` loaded once for the run.
- Hierarchical: 5 cases in 3.2s total (0.64s/case mean); same embedding
  model.
- `estimated_cost_usd=0.0` for every row in both files (no LLM call was
  made to price).

## Generated artifacts (all Git-ignored, none committed)

- `eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv` (new this task, 18,747 rows)
- `eval/local_runs/wisco_v2_tier1_preflight/flat/20260807T233801Z_wisco_v2_preflight_flat_norerank.csv` (5 rows)
- `eval/local_runs/wisco_v2_tier1_preflight/hierarchical/20260807T233859Z_wisco_v2_preflight_hierarchical_norerank.csv` (5 rows)

Confirmed via `git check-ignore -v` that all three paths fall under the
existing `eval/local_benchmarks/` / `eval/local_runs/` `.gitignore`
patterns; `git status --short` was empty throughout (no tracked file was
ever modified by this task).

## Confirmation: zero Ollama, paid API, or LLM calls

Both evaluator invocations passed `--use-llm-reranker off` and no
`--reranker-model`, which (per the verified Task 09 correction on this
branch) constructs `ISCOClassifier(enable_llm=False)` — Stage 2 of
`__init__` (the only place an LLM/agent is ever constructed) is skipped
entirely; no `get_llm`, `get_llm_strict`, or CrewAI `Agent`/`Crew` object
was created. Confirmed empirically: `reranker_fired=False`,
`prompt_tokens=0`, `completion_tokens=0`, `estimated_cost_usd=0.0` on
every row of both files — zero LLM activity of any kind.

## Confirmation: no Qdrant collection was changed

Every Qdrant interaction in this task was either a read-only listing/count
call (Part B) or a read-only `query_points()` search issued by the
existing, unmodified retrieval code during the 10-case smoke test (Part
C). No `create_collection`, `recreate_collection`, `delete_collection`,
or `upsert` call was made anywhere in this task's own work. Point counts
recorded in Part B are unchanged from before this task (verifiable by
re-running the same read-only check).

## Confirmation: no full benchmark, performance analysis, or manuscript claim was produced

Exactly 5 cases per system were run (10 total), never the 18,747-case
full heldout set. No accuracy, comparison, significance, cost aggregate,
or manuscript-ready metric was calculated anywhere in this report or in
this task's own work — only raw path-integrity field values (method
labels, stage presence, status strings) were inspected and quoted
verbatim above.

## READY_FOR_TIER1_FULL_RUN: yes

**Reason**: all four gates passed cleanly with no blocker —
(A) the v2 asset package is present, its dataset hash matches the
required value exactly, and the live-rerun leakage/integrity audit found
zero source-family leakage, zero cross-split duplicate groups, and zero
malformed codes; the heldout export has the exact required row count and
schema; (B) all five Qdrant collections required by the real production
retrieval path (four hierarchical + one flat) are present and non-empty,
confirmed via a read-only listing, with one naming clarification recorded
above (no missing collection); (C) both five-case smoke tests completed
with zero errors, zero LLM/reranker activity, zero ISIC/ISCED/SRE
construction, and the hierarchical run produced genuine, non-fallback,
fully-evidenced four-stage retrieval on every row. No condition requiring
a stop (Parts A.4, B.5, or C.4) was triggered at any point.

## Protected branch status

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
| `reviewer2-wisco-measurement-baseline-20260808` | not touched (used only as the branch point; never checked out for writing) |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push was run at any point in this task. No PR was
created. No production code, test, configuration, requirements, dataset,
benchmark split, or manuscript wording was changed — verified via
`git status --short` being empty throughout except for this report.

## Working-tree status

Clean before this task started (verified) and clean again immediately
before this report's commit. No production code/test/config change was
made (per Part D.1, none was needed — no local environment issue was
encountered), so the full test suite was not re-run in this task.
