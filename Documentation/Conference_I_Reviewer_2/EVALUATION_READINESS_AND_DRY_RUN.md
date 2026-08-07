# Evaluation Readiness and Dry-Run Verification

Conference I Reviewer #2 response, Step 4: "verify the evaluation, ablation,
instrumentation, and evidence-export pipeline using a synthetic-only dry
run." This document is the readiness report for that pass. It does **not**
add any new measurement — it proves the pipeline built in Steps 1-3 (and
extended here) runs end-to-end, produces correctly-shaped, honestly-labelled
output, and will not silently produce a fabricated or mislabelled result
once a real, governance-approved dataset exists.

**Nothing in this document, or in any artifact it references, may be cited
in the manuscript as a measured result.** Every number in every dry-run
artifact is `null`.

## 1. What "dry run" means here

`--dry-run` on `eval/run_eval.py` and `eval/ablation_runner.py run`:

- Parses and validates every CLI argument exactly as a real run would.
- Loads and validates the dataset-card / governance label / split-manifest
  path (the governance gate in `eval/validate_real_lfs_governance.py` runs
  in full — a dry run against a rejected `DatasetCard` is still rejected).
- Validates all 5 named ablation configurations resolve to a valid `argv`
  (`python eval/ablation_runner.py validate-configs`).
- Builds a complete `ExperimentRunManifest` and writes it (CSV + JSONL).
- Writes an empty/null-valued `CaseResult` row per input case (same CSV
  schema and column order as a real run — `pred_*` columns blank).
- Exercises the exact same output-serialization and figure-export code
  paths a real run would use.
- **Never constructs `ISCOClassifier` or `ISICClassifier`** (both connect to
  Qdrant/Ollama/an LLM reranker at `__init__` — see `eval/run_eval.py`'s
  `build_dry_run_case_result()` and its `--dry-run` branch in `main()`) and
  **never calls any classifier, embedding model, reranker, or SRE check.**
  No Qdrant, Ollama, paid LLM API, or GPU access is touched anywhere in a
  dry run.

Every dry-run manifest carries `evaluation_status="dry_run_not_measured"`.
This field is **orthogonal** to `dataset_label` (which describes whether the
*data* is synthetic, approved-real, or governance-invalid — see
`REAL_LFS_VALIDATION_DATASET_CARD_TEMPLATE.md`). A dry run against a fully
governance-compliant real dataset card can legitimately carry
`dataset_label=approved_real_lfs_validation` **and**
`evaluation_status=dry_run_not_measured` at the same time — that combination
means "yes, this is real approved data, but nothing has been measured
against it yet." In this Step 4 pass, every dry run was executed against
**only** the synthetic fixture (`dataset_label=synthetic_or_operationally_realistic`)
— see §3.

## 2. Exact commands

### 2a. Confirm all 5 ablation configs resolve (no execution, no network)

```bash
python eval/ablation_runner.py validate-configs
```

### 2b. Single dry run (one config)

```bash
python eval/ablation_runner.py run --config hierarchical_with_rerank \
  --test-set eval/fixtures/synthetic_lfs_intake_package/synthetic_test_set.csv \
  --split heldout --dry-run \
  --dataset-card eval/fixtures/synthetic_lfs_intake_package/dataset_card.json \
  --split-manifest eval/fixtures/synthetic_lfs_intake_package/split_manifest.json \
  --output-root eval/local_runs/manual_check
```

### 2c. Full readiness pass (all 5 configs + coverage audit + SRE schema
check + figure-export validation + Markdown summary — what generated the
artifacts referenced in §5)

```bash
python eval/generate_dry_run_readiness_report.py \
  --test-set eval/fixtures/synthetic_lfs_intake_package/synthetic_test_set.csv \
  --dataset-card eval/fixtures/synthetic_lfs_intake_package/dataset_card.json \
  --split-manifest eval/fixtures/synthetic_lfs_intake_package/split_manifest.json
```

Writes to `eval/local_runs/dry_run_<UTC timestamp>/` by default (Git-ignored
— see §6). Pass `--output-root <path>` to redirect.

### 2d. The future command for a genuine, approved real-LFS run

Only after `REAL_LFS_DATA_INTAKE_CHECKLIST.md` has been followed end-to-end
and a real `DatasetCard` passes `eval/validate_real_lfs_governance.py`:

```bash
python eval/ablation_runner.py run --config hierarchical_with_rerank \
  --test-set <path to the real, governance-approved test-set CSV> \
  --split heldout \
  --dataset-label approved_real_lfs_validation \
  --dataset-card <path to the completed, passing DatasetCard JSON> \
  --split-manifest <path to the real dev/heldout split manifest> \
  --reranker-model <the actual reranker model identifier to be cited> \
  --run-id <a stable, citable run id>
```

Repeat for all 5 configs (`flat_baseline`, `hierarchical_no_rerank`,
`hierarchical_with_rerank`, `no_sre`, `with_sre`), **without** `--dry-run`,
then regenerate the evaluation table:

```bash
python eval/ablation_runner.py emit-table
```

## 3. What was actually run in this Step 4 pass

`eval/fixtures/synthetic_lfs_intake_package/synthetic_test_set.csv` — 5
obviously-synthetic rows (`case_id` prefixed `SYN-`, `input_text` prefixed
`SYNTHETIC EXAMPLE -`) — was run through all 5 named ablation configs via
§2c. `dataset_card.json` and `split_manifest.json` from the same fixture
package were supplied so `dataset_card_hash` / `split_manifest_hash` are
populated (both are `sha256` hashes of the fixture files — never the
underlying case text). No other test set, and no real data of any kind, was
used anywhere in Step 4.

## 4. Measurement-field inventory (task C)

Every field below either has a real value on a completed run, or is `None`
with an explicit `<field>_unavailable_reason` (or, for cost,
`estimated_cost_method` doubling as the reason on a dry run) — see
`eval/manifest.py`'s module docstring for the full honesty rule.

| Metric (task C) | Where it lives | Status on a **dry run** | Status on a **real, completed run** |
|---|---|---|---|
| Accuracy / top-k accuracy | `eval/analyze.py` (reads `gold_*`/`pred_*` CaseResult columns) | Not applicable — dry-run CaseResult rows have blank `pred_*` columns by construction, so `analyze.py` would report 0 scored cases if pointed at one | Computed per ISCO 1/2/3/4-digit, ISIC section/division/group/class, ISCED level/broad/narrow/detailed, wherever the gold column exists in the CSV |
| Wilson CI | `eval/analyze.py::wilson_score_interval()` | N/A (needs accuracy inputs above) | Computed alongside each accuracy figure |
| McNemar comparison | `eval/analyze.py::mcnemar_test()` | N/A (needs two completed runs' per-case correctness) | Computed given two CaseResult CSVs over the same case set |
| p50/p95 latency | `ExperimentRunManifest.latency_p50_ms` / `latency_p95_ms` | `None`, `latency_unavailable_reason="no case row carried a non-empty end_to_end_latency_ms"` | Computed from real per-case `end_to_end_latency_ms` |
| Throughput | `ExperimentRunManifest.throughput_cases_per_sec` | `None` (same reason, derived from latency) | Computed |
| Process memory | `ExperimentRunManifest.peak_process_memory_mb` | `None` | `None` today even on a real run — `CaseResult.peak_memory_mb` is a declared field `eval/run_eval.py` never populates (pre-existing, honest gap, unchanged by Step 4) |
| GPU memory | `ExperimentRunManifest.peak_gpu_memory_mb` | `None`, `peak_gpu_memory_unavailable_reason` from the hardware probe | Populated only on a CUDA-visible machine; `None` + reason on this CPU-only dev environment |
| Retrieval candidate counts | `ExperimentRunManifest.mean_retrieval_candidate_pool_size` (new in Step 4) | `None`, reason given | Computed from `CaseResult.reranker_candidate_pool_size` for `--system hierarchical` |
| Reranker invocation rate | `ExperimentRunManifest.reranker_invocation_rate` (new in Step 4) | `None` (no `reranker_fired` values on a dry run) | Computed as `reranker_invocation_count / n_cases` |
| HITL escalation rate | `ExperimentRunManifest.hitl_escalation_rate` | `None`, reason given | The **evaluation harness's own** research escalation heuristic — not production `HITLQualityManager` (see `eval/manifest.py`'s docstring) |
| SRE severity outcomes | `CaseResult.sre_severity` (per case) + `eval/sre_eval_format.py` | `sre_eval_format.evaluate_sre_labels([])` → `status="no_labels_supplied"`, every metric `None` (no labelled fixture exists in this repo, dry run or not) | Same function, given a real labelled `(sre_flagged, human_judged)` set — none exists yet |
| Cost method + value | `ExperimentRunManifest.estimated_cost_usd` / `estimated_cost_method` | `None`, `estimated_cost_method="not measured -- dry run: no classifier was invoked, so no LLM/API cost was ever incurred"` (Step 4 fix — previously a dry run could show a trivial `0.0` that looked measured; see §7) | Sum of per-case `CaseResult.estimated_cost_usd` (LiteLLM cost estimation) |

## 5. Task D — evaluation discipline

- **Dev/held-out distinctness**: already enforced by
  `eval/validate_dev_set.py` (rejects any `case_id` or normalised-text
  collision between `dev_set_v1.csv` and `test_set_smoke20.csv`/
  `test_set_full130.csv`) — unchanged in Step 4, re-confirmed by reading the
  module.
- **Ablation config selection cannot use the held-out set**: enforced
  structurally by `eval/ablation_runner.py`'s `--split {dev,heldout}` ->
  `_split_output_dir()` routing (`dev` writes to
  `eval/results/dev_selection/`, `heldout` to `eval/results/raw_runs/`) —
  `eval/ablation_runner.py emit-table` only ever reads the `heldout`
  directory. Unchanged in Step 4; the new `--output-root` override
  preserves this split (it still routes to `<root>/dev/` vs `<root>/heldout/`).
- **Every run records the required fields**: `dataset_version_hash`,
  `split_manifest_hash` (Step 4), `git_commit`, `model_versions`,
  `retrieval_params` (now includes `"sre"` — see §7), `llm_params`
  (reranking status), `governance_validation_errors` (governance-validation
  result — populated only on a downgraded attempt; an
  `approved_real_lfs_validation` manifest that exists at all has, by
  construction, already passed the gate).
- **New rejection tests** (`eval/test_validate_evaluation_discipline.py`,
  13 tests) via the new `eval/validate_evaluation_discipline.py` and
  `eval/split_manifest_schema.py`:
  - missing split manifest → rejected
  - identical dev/heldout `split_id` → rejected
  - blank `dataset_version_hash` → rejected
  - `evaluation_status="measured"` with every core metric `None` → rejected
    (a dry run with every metric `None` is **not** rejected by this check —
    that combination is expected and honest)
  - `approved_real_lfs_validation` label without passing governance →
    rejected (re-exercises the Step 3 gate in `eval/manifest.py`)
  - a synthetic run's proposed manuscript wording containing a banned
    real-validation phrase → flagged (re-exercises
    `eval/validate_real_lfs_governance.check_manuscript_wording()`)

## 6. Safe vs. unsafe outputs

**Safe for infrastructure verification only** (prove the pipeline runs,
prove nothing about accuracy/performance):

- Every artifact under `eval/local_runs/dry_run_*/` (manifests, CaseResult
  CSVs, the ablation status table, the null-metric report, the discipline
  check, the coverage audit, the SRE schema check, the figure-export
  validation output, the README).
- `python eval/ablation_runner.py validate-configs` output.

**Unsafe for the manuscript** (contain no measurement and must never be
quoted as one): all of the above, without exception. Every one carries
`evaluation_status: dry_run_not_measured` and/or `dataset_label:
synthetic_or_operationally_realistic` precisely so a reader cannot mistake
them for a result. The coverage-audit output is a partial exception: it is
real, computed, in-repo code-table evidence (Section C/§5 of the earlier
matrix) — but it is a **code-coverage** count, not a classification-accuracy
measurement, and carries no `dataset_label`/`evaluation_status` of its own;
do not conflate the two when citing it.

## 7. Gaps found and closed during this pass

- **`retrieval_params` was missing SRE on/off status.** Every manifest
  before this fix recorded `system` and `use_llm_reranker` but not `sre`,
  even though `no_sre`/`with_sre` are two of the five named ablation
  configs and SRE status is one of task D's required recorded fields.
  Fixed in `eval/ablation_runner.py::run_config()`; regression-tested in
  `eval/test_ablation_runner.py::test_run_config_manifest_records_sre_status_in_retrieval_params`.
- **A dry run's `estimated_cost_usd` could read as a real `0.0`
  measurement** rather than "not measured" (the `CaseResult` dataclass
  default of `0.0` trivially passes a naive "non-empty" check). Fixed in
  `eval/manifest.py::build_manifest()` with an explicit
  `evaluation_status == "dry_run_not_measured"` override.
- **Figure exports did not surface `evaluation_status` (or, for the
  latency/scalability export, `dataset_label`)**, so a dry-run row in an
  exported figure table showed only `null` metrics with no explicit label
  explaining why. Fixed in `eval/figure_exports/export_evaluation_results.py`
  and `export_latency_scalability.py` by adding both fields to each
  script's exported column list.

## 8. Retaining raw run artifacts outside Git

`eval/local_runs/` is unconditionally Git-ignored (see `.gitignore`) — this
is deliberate for **both** dry runs (no real data, but still scratch) and a
future real run (raw artifacts must be reviewed by the dataset custodian
before anything derived from them is committed). To retain a run:

1. Copy the whole `eval/local_runs/dry_run_<timestamp>/` (or your custom
   `--output-root`) directory to storage outside this Git working tree
   (e.g. the dataset custodian's own encrypted drive, per
   `REAL_LFS_DATA_INTAKE_CHECKLIST.md`'s access-restriction requirements for
   real data — dry-run/synthetic output has no such restriction but the
   same copy step applies).
2. Never `git add` anything under `eval/local_runs/` directly. If a
   specific artifact should become permanent evidence (e.g. a completed,
   real manifest backing a manuscript figure), copy just that file into a
   tracked location under `Documentation/Conference_I_Reviewer_2/generated/`
   deliberately, by hand, after review — never by moving the `.gitignore`
   boundary.

## 9. Regenerating figures only after completed manifests exist

`eval/figure_exports/export_evaluation_results.py` and
`export_latency_scalability.py` default to reading
`eval/results/**/manifest_*.jsonl` (the **tracked** location a real,
`--split heldout`, non-dry-run invocation of `eval/ablation_runner.py run`
writes to). Both scripts already refuse to fabricate rows: with zero
manifests present they write `no_manifests_found: true` and an empty
`results` list rather than inventing figures. Do not point them at
`eval/local_runs/` for a manuscript figure — only ever regenerate
manuscript figures after a real run has written its manifest to
`eval/results/raw_runs/manifests/` (i.e. after the §2d command has been run,
without `--dry-run`, and its manifest inspected to confirm
`evaluation_status=="measured"`).
