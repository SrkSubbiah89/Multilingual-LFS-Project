# Task 12 Final Report — Full WISCO Tier-1 Controlled ISCO-08 Benchmark

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_12_FULL_WISCO_TIER1_BENCHMARK.md`
(task ID `12-full-wisco-tier1-benchmark`).

## TIER1_COMPLETED: no

**Exact blocker**: the hierarchical full run produced 23 of 18,747 rows
(0.123%) with a non-hierarchical `pred_method` (`flat_semantic` instead of
`hierarchical_semantic`/`hierarchical_llm`), and at least 3 rows show
severe, anomalous single-stage latency spikes (up to ~4 hours for one
case) inconsistent with normal operation. Task 12's own Part B/C.6 rule
is explicit and unconditional: *"If ... the hierarchical run has a
fallback/flat method label or missing four-stage evidence, stop ...
Preserve the raw output and report the failure. Do not retry, alter
data, or report an incomplete comparison as a result."* That condition is
met. Per that instruction, **Part D (analysis) and Part E (manifests)
were not run** — no accuracy, Wilson interval, McNemar test, comparison
artifact, or manifest was produced. The raw output CSVs were preserved
unmodified.

## Branch / SHA

| | |
|---|---|
| Base branch | `reviewer2-wisco-tier1-preflight-20260808` |
| Base SHA (verified) | `a1ce94213e5981dc30444fd4947ff4c0dd056888` |
| Working branch | `reviewer2-wisco-tier1-results-20260808` (new, created by this task) |
| Final SHA | commit created and pushed as this task's own closing step, containing only this report |
| Code SHA used for both full runs | `a1ce942` (unchanged throughout — confirmed identical in both CSVs' `git_commit` column) |

Start-state check passed exactly: `git fetch origin` confirmed
`origin/reviewer2-wisco-tier1-preflight-20260808` at
`a1ce94213e5981dc30444fd4947ff4c0dd056888`; the local branch was already
at that exact SHA with a clean working tree before
`git switch -c reviewer2-wisco-tier1-results-20260808` was run.

## A. Pre-run gates — all passed

**WISCO v2 integrity/leakage** (live re-validation, same method as Task
11): dataset hash `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c`
matched exactly; 20,760 total records (2,013 dev / 18,747 heldout); zero
source-family leakage; zero cross-split exact-text duplicate groups; zero
malformed ISCO codes.

**Heldout export**: the existing `heldout_run_eval_format.csv` (from
Task 11) already had exactly 18,747 rows in the correct source order, so
it was reused rather than re-exported (satisfying Part A.2's "if absent
or does not match" exemption).

**Qdrant readiness** (read-only): all five required collections present
and non-empty — `isco08_major_groups` (10), `isco08_submajor_groups`
(43), `isco08_minor_groups` (131), `isco08_unit_groups` (441),
`isco_occupations` (124). Re-checked identically after both runs
completed — **point counts unchanged**, confirming zero Qdrant mutation.

## B. Full paired runs — both completed (exit code 0), but see blocker

One timestamped ignored root:
`eval/local_runs/wisco_v2_tier1_full_20260807T234643Z/`

```
python eval/run_eval.py --test-set eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv --system flat --use-llm-reranker off --config wisco_v2_tier1_full_flat_norerank --run-id wisco-v2-tier1-full-flat-norerank --output-dir eval/local_runs/wisco_v2_tier1_full_20260807T234643Z/flat

python eval/run_eval.py --test-set eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv --system hierarchical --use-llm-reranker off --config wisco_v2_tier1_full_hierarchical_norerank --run-id wisco-v2-tier1-full-hierarchical-norerank --output-dir eval/local_runs/wisco_v2_tier1_full_20260807T234643Z/hierarchical
```

Run flat first, then hierarchical, in that order, as required. Neither
`--limit`, `--reranker-model`, `--use-llm-reranker on`,
`ablation_runner.py`, nor any non-WISCO input was used.

| | Start (UTC) | End (UTC) | Elapsed | Rows written | Exit code |
|---|---|---|---|---|---|
| Flat | 2026-08-07T23:47:24Z | 2026-08-08T00:01:57Z | 816.0s (13m36s) | 18,747 | 0 |
| Hierarchical | 2026-08-08T00:01:57Z | 2026-08-08T05:24:49Z | 19,297.3s (5h21m37s) | 18,747 | 0 |
| **Total wall time** | 2026-08-07T23:47:24Z | 2026-08-08T05:24:49Z | **5h37m25s** | | |

Both commands exited 0 and wrote exactly 18,747 rows each — the "exits
non-zero / interrupted / fewer than 18,747 rows" trigger conditions were
**not** met. The trigger that **was** met is the fallback/flat-label
condition, detailed below.

Raw output paths (preserved, unmodified, Git-ignored):
- `eval/local_runs/wisco_v2_tier1_full_20260807T234643Z/flat/20260807T234757Z_wisco_v2_tier1_full_flat_norerank.csv`
- `eval/local_runs/wisco_v2_tier1_full_20260807T234643Z/hierarchical/20260808T000252Z_wisco_v2_tier1_full_hierarchical_norerank.csv`
- `eval/local_runs/wisco_v2_tier1_full_20260807T234643Z/run_log.txt` (full console log of both runs)

## C. Output-integrity results

### Checks that PASSED

- **Row count / order / gold-value consistency**: both CSVs have exactly
  18,747 data rows; `case_id` order in both is byte-identical to the
  frozen heldout export's source order; `gold_isco_4digit` values match
  the heldout export exactly in both files.
- **`evaluation_status`**: every row in both files is exactly `"measured"`
  (18,747/18,747 each) — never a real-LFS label.
- **`dataset_label`**: confirmed **absent** from the raw `CaseResult` CSV
  schema in both files — stated precisely per the task's instruction, not
  fabricated. (The label belongs in the manifest, which was not produced
  — see the TIER1_COMPLETED:no reason above.)
- **Reranker invariants**: `reranker_fired=False`,
  `reranker_input_candidates=[]`, `reranker_output={}`, `prompt_tokens=0`,
  `completion_tokens=0`, `estimated_cost_usd=0.0` — uniformly across all
  18,747 rows of both files. `reranker_model` is blank per-row (the
  per-case trace field, populated only when a reranker actually fires);
  the run-level `reranker_model_resolved` disclosure (printed at each
  run's start, not stored per-row) was `"none (reranking disabled)"` for
  both runs — the exact Task 09 disabled disclosure.
- **ISIC/ISCED/SRE invariants**: `pred_isic_section=""`,
  `pred_isced_level=""`, `sre_status="not_applicable"` uniformly across
  all rows of both files — no such classifier was constructed (confirmed
  by the printed run-start message: *"No row has both industry_text and
  education_text -- ISICClassifier/ISCEDClassifier/SemanticRelationEngine
  will not be constructed"*). Their metrics were not computed anywhere.
- **Flat output method labels**: 18,747/18,747 rows are
  `pred_method="flat_semantic"` — never hierarchical.
- **Row-level errors**: `error=""` for all 18,747 rows in **both** files —
  zero row-level errors in either run. `degraded=False`,
  `invalid_output_flag=False`, `timed_out_flag=False` uniformly.
- **Run-level uniformity**: each file has exactly one `config_hash`, one
  `embedding_model_version` (`intfloat/multilingual-e5-small`), one
  `run_id`, one `git_commit` (`a1ce942`) across all its rows.

### Check that FAILED — the blocker

**Hierarchical `pred_method` distribution**:
`{"hierarchical_semantic": 18724, "flat_semantic": 23}`.

23 of 18,747 hierarchical-run rows (0.123%) carry the flat method label,
in violation of Part C.6's "no fallback/flat label" requirement. All 23
have `error=""` (no crash) — this is a silent, non-erroring internal
fallback, not a row-level failure the pipeline itself flags. The 23 case
IDs:

```
WISCO-2310570000000-en, WISCO-4312002200018-hi, WISCO-6113040000000-en,
WISCO-6113040000000-ar, WISCO-6113990000000-ar, WISCO-6121002200018-en,
WISCO-6121070000000-en, WISCO-6129000900018-en, WISCO-6210000300018-ar,
WISCO-6221050000000-en, WISCO-6221050000000-ar, WISCO-6222010000000-en,
WISCO-6222010000000-ar, WISCO-6222020000000-en, WISCO-6222020000000-ar,
WISCO-6223000200018-en, WISCO-6223000200018-ar, WISCO-6224000500018-ar,
WISCO-6224000700018-ar, WISCO-6330010000000-ar, WISCO-6340000100016-en,
WISCO-9213010000000-ar, WISCO-9216010000000-en
```

**Diagnosis (two distinct sub-patterns, both purely observational — no
code was read beyond what was necessary to interpret existing trace
fields, and no code was changed)**:

1. **21 of the 23** rows: `stage1_source="keyword_map"`, total per-row
   latency ~50–90ms (normal/fast, no anomaly), and — materially —
   `stage1_candidates`, `stage2_candidates`, `stage3_candidates`, and
   `stage4_candidates` are **byte-identical to each other within each
   row** (the same flat-search candidate list duplicated into all four
   stage columns, not four genuinely distinct hierarchical stage
   results). `pred_reasoning` reads *"Top flat semantic match (score
   NN.NN%); LLM agent unavailable."* — the flat-path reasoning string,
   not the hierarchical path's. This is consistent with the classifier's
   existing keyword-map/major-hint stage-1 bypass interacting with a
   flat-search completion path under `--use-llm-reranker off`, for this
   specific small set of inputs — fast and non-erroring, but not genuine
   4-stage hierarchical beam search, and therefore exactly the condition
   Part C.6 requires reporting rather than silently accepting.
2. **2 of the 23** rows show severe, anomalous single-stage latency
   spikes with `stage1_source=""` (empty, not `"keyword_map"`):
   - `WISCO-4312002200018-hi`: `stage1_latency_ms=14,388,598.91`
     (**≈3h59m49s** for one case's stage 1 alone).
   - `WISCO-6121002200018-en`: `stage1_latency_ms=425,820.43`
     (**≈7m6s** for one case's stage 1 alone).

**A third row did not fall back but shows the same anomaly pattern**:
`WISCO-8160001500018-ur` completed as genuine `hierarchical_semantic`
(no fallback) but with `stage3_latency_ms=1,648,020.79`
(**≈27m28s** for stage 3 alone, `end_to_end_latency_ms=1,648,306.13`).

**These three anomalous-latency rows together account for ≈16,463s
(≈4h34m) of the hierarchical run's total 19,297.3s (5h21m37s) wall time —
approximately 85% of the entire run's duration.** By contrast, the flat
run (same machine, same benchmark, immediately prior) shows no such
anomaly anywhere: its single slowest case was 1,091.09ms
end-to-end, and its 18,747-row total latency sum (812.6s) matches its
reported wall time (816.0s) almost exactly, with no unaccounted time.

**Assessment**: the pattern (isolated, single-stage, multi-minute-to-
multi-hour stalls, at different pipeline stages across different rows,
present only in the long-running hierarchical run and absent from the
short flat run) is consistent with an intermittent environment-level
interruption during the ~5.5-hour hierarchical run — e.g. local
machine sleep/suspend, thermal throttling, or a transient local Qdrant
connectivity stall — rather than a deterministic defect in the
retrieval code triggered by specific input content. This is a
diagnostic hypothesis based on the observed data, not a confirmed root
cause; no source code was modified to investigate further, per the
task's read-only diagnostic scope.

**Stage-evidence check** (Part C.6's other required check): all 18,747
hierarchical rows, including the 23 flat-labelled ones, have non-null,
non-empty JSON in all four `stageN_candidates` columns (the 23
flat-labelled rows' four columns are, as noted, duplicates of one
another rather than four distinct genuine stages) and all four
`stageN_latency_ms` columns populated — so the "missing four-stage
evidence" trigger, read literally as "any stage column blank/null," was
not separately met; the failure is specifically the method-label/
duplicate-stage-content pattern described above.

## D and E — not run

Per the exact stop condition in Part B and Part C.6, **no analysis and no
manifest were produced.** `eval/analyze.py` was not invoked.
`eval.manifest.build_manifest()`/`write_manifest()` were not invoked. No
Wilson interval, no McNemar test, no accuracy figure, no paired
comparison artifact, no manifest file exists anywhere from this task's
own work.

## Confirmation: zero Ollama/LLM/paid API calls, zero Qdrant mutations

- Both runs used `--use-llm-reranker off` and no `--reranker-model`;
  every row of both files shows `reranker_fired=False`,
  `prompt_tokens=0`, `completion_tokens=0`, `estimated_cost_usd=0.0` —
  zero LLM activity anywhere.
- Qdrant collection point counts, checked read-only both before Part B
  and again after both runs completed, are byte-identical
  (`isco08_major_groups=10`, `isco08_submajor_groups=43`,
  `isco08_minor_groups=131`, `isco08_unit_groups=441`,
  `isco_occupations=124`) — confirming no `create`/`recreate`/`delete`/
  `upsert` call touched any collection during this task.
- No B1 re-freeze, B2 sweep, ISIC/ISCED-F collection construction, or
  ISIC/ISCED/SRE evaluation occurred. No 500-case reranking tier was run.
  No `ablation_runner.py` invocation occurred.

## Environment record (Part A.7 / C.7)

| | |
|---|---|
| Code SHA | `a1ce942` (identical throughout, confirmed in both CSVs) |
| Python | 3.11.9 |
| `qdrant-client` | 1.17.0 |
| `sentence-transformers` | 5.2.3 |
| `crewai` | 1.9.3 |
| Embedding model | `intfloat/multilingual-e5-small` (confirmed uniform in both CSVs) |
| Dataset hash | `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c` |
| Split manifest hash | `4a2d9795455b1b07c6d0ddad8843633f7c2e1fc71c13e8abd15ede9254894693` (`split_manifest.json`, unchanged from Task 11) |
| Qdrant point counts (before and after, unchanged) | `isco08_major_groups=10`, `isco08_submajor_groups=43`, `isco08_minor_groups=131`, `isco08_unit_groups=441`, `isco_occupations=124` |

No secret or credential was read, logged, or exposed at any point.

## Findings: what this run does and does not support

**Does not support** (this run produced no measurement at all, only a
diagnostic finding):

- No accuracy, latency-percentile, cost, or comparison figure from this
  run may be cited anywhere, in any form — none was computed.
- No claim that the hierarchical retrieval system "works" or "doesn't
  work" at scale — the 18,724 genuinely-hierarchical rows were never
  scored, and the run as a whole did not complete cleanly enough to
  trust its own timing data as representative operational performance.
- No real-LFS, ISIC, ISCED, SRE, or reranking conclusion — none of these
  were in scope regardless of the blocker.

**Does support**:

- The pre-run integrity gates (Part A) are solid: the WISCO v2 asset is
  valid, unleaked, and correctly exported; Qdrant collections were ready
  and remained untouched throughout.
- The retrieval-only, model-free evaluator path (Task 09/10's fix) is
  confirmed working at full scale for the flat system (18,747/18,747
  clean, fast, zero anomalies) and for 18,724/18,747 (99.877%) of
  hierarchical cases.
- A genuine, previously-undetected operational fragility exists in the
  hierarchical retrieval path (or its execution environment) that only
  manifests at full-scale, long-running invocation — Task 11's 5-case
  smoke test could not and did not catch this, which is itself a useful
  finding about the limits of a small preflight sample size.

## Manuscript-safe vs manuscript-unsafe wording

**Safe to say**: "A full-scale (18,747-case), paired, retrieval-only
WISCO Tier-1 preflight identified an intermittent execution anomaly
affecting a small fraction (0.123%) of hierarchical-retrieval cases at
full scale, which a small preflight sample did not surface; the run was
halted before any accuracy analysis per the project's evidence-integrity
protocol, and no performance claim is made from it."

**Not safe to say, from this task**: anything implying ISCO-08
hierarchical-vs-flat accuracy, latency, or comparison results; anything
implying the hierarchical retrieval path is validated at scale; anything
implying real-LFS applicability; any ISIC/ISCED/SRE/reranking claim.

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
| `reviewer2-wisco-measurement-baseline-20260808` | not touched |
| `reviewer2-wisco-tier1-preflight-20260808` | not touched (used only as the branch point) |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push was run at any point in this task. No PR was
created. No production code, test, configuration, requirements, dataset,
benchmark split, or manuscript wording was changed.

## Working-tree status

Clean before this task started (verified) and clean again immediately
before this report's commit. No raw WISCO records, exports, output CSVs,
manifests, analysis files, or local scripts were committed — all remain
Git-ignored under `eval/local_benchmarks/` and `eval/local_runs/`.

## Recommended next step (not performed by this task)

Re-run the hierarchical (and, for a clean paired comparison, likely also
the flat) full configuration in an environment/session verified not to
sleep, throttle, or lose local Qdrant connectivity for the run's full
duration, then re-apply this same Part C output-integrity gate before
proceeding to Part D/E. If the same 21 fast, non-anomalous
`keyword_map`-linked flat-label rows recur identically on a clean re-run,
that would indicate a real, reproducible code-path condition worth its
own root-cause task rather than an environment artifact; if they do not
recur, the original run's anomaly was environmental. This task does not
make that determination — it stops here, as instructed.
