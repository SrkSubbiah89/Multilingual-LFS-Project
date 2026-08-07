# Evaluation Protocol

Covers Sections D (evaluation/reproducibility instrumentation) and E
(ablation infrastructure) of the Reviewer #2 response.

## Two evaluation systems in this repo — which one to cite

There are genuinely two separate evaluation subsystems in this codebase.
The manuscript must not cite them interchangeably without saying which one
produced a given number.

| | `backend/evaluation/evaluate.py` | `eval/` (this work extends it) |
|---|---|---|
| Origin | Thesis Chapter 6 evaluation framework | B0–B2 hardening work + this Reviewer #2 pass |
| Test data | 100-item **synthetic** occupation-description corpus | `eval/test_set_full130.csv` / `test_set_smoke20.csv` (existing, real curated cases) + any future dev/held-out sets |
| Systems compared | BM25, flat vector, hierarchical RAG | Same 3 systems, plus the 5 named ablation configs below |
| Metrics | top-1/top-3 accuracy, Cohen's kappa, HITL rate, mean latency | Wilson-CI accuracy at every ISCO digit level, ISIC/ISCED accuracy (where gold labels exist), full experiment-run manifests (hardware, cost, latency percentiles, HITL escalation rate) |
| Reproducibility | Not manifest-tracked | Every run produces an `ExperimentRunManifest` (git commit, dataset hash, hardware, dependency versions) |

**For any number going into the manuscript, use the `eval/` harness** — it
is the one with reproducibility manifests, real (not synthetic) test data,
and Wilson confidence intervals. `backend/evaluation/evaluate.py` predates
this work and is left untouched; do not delete it (a future analysis may
still want the synthetic-corpus comparison), but don't cite its numbers as
"the" evaluation without saying which subsystem and corpus produced them.

## Dev-selects-params / held-out-eval-only discipline

- `eval/validate_dev_set.py` already enforces leakage-safety between
  `dev_set_v1.csv` and the frozen `test_set_full130.csv`/`test_set_smoke20.csv`.
- `eval/ablation_runner.py`'s `--split {dev,heldout}` flag extends the same
  discipline to every ablation run: `dev` writes to
  `eval/results/dev_selection/` (parameter selection only — e.g. choosing
  K, choosing whether to enable the SRE); `heldout` writes to
  `eval/results/raw_runs/` (the only split whose manifest may be cited as
  a confirmed result).
- **Never build the manuscript's evaluation table from
  `eval/results/dev_selection/`.** `eval/ablation_runner.py emit-table`
  only reads `eval/results/raw_runs/manifests/` for exactly this reason.

## Experiment-run manifest (`eval/manifest.py`)

One `ExperimentRunManifest` per run: `run_id`, `utc_timestamp`,
`git_commit`, `dataset_version_hash` (sha256 of the test-set CSV),
`split_name`, `dataset_label` — exactly one of
`synthetic_or_operationally_realistic` / `approved_real_lfs_validation` /
`invalid_incomplete_governance` (see
`REAL_LFS_VALIDATION_DATASET_CARD_TEMPLATE.md` and
`REAL_LFS_DATA_INTAKE_CHECKLIST.md` for the governance gate on the second
one, and `eval/dataset_card_schema.py` for the closed-vocabulary
enforcement — no other string is ever accepted), `classifier_method`, hardware
(CPU/RAM/GPU, best-effort probed), OS/Python version, a full installed-
dependency snapshot, model/embedding/retrieval/LLM params (including `sre`
on/off status), latency mean/p50/p95, throughput, HITL escalation rate,
reranker invocation count/rate, retrieval candidate pool size, and cost
(summed from `CaseResult.estimated_cost_usd` when non-zero). Also:
`evaluation_status` (`"measured"` vs. `"dry_run_not_measured"` — see below),
`dataset_card_hash` and `split_manifest_hash` (sha256 of the supplied
`DatasetCard`/split-manifest file, when supplied — never a hash of
respondent data itself).

**Known, honestly-reported gaps** (both `None` with a
`<field>_unavailable_reason` string, never a placeholder):

- `peak_process_memory_mb` — `CaseResult.peak_memory_mb` is a declared
  field in `eval/run_eval.py` that no current code path actually
  populates (verified by reading the module). `eval/dev_sweep.py`
  separately samples peak RSS via `psutil` for its own K-sweep-level
  eligibility checks, but that value is never attached to a `CaseResult`
  row, so `eval/manifest.py` cannot read it from there either. A future
  change could pass a measured value into `eval.ablation_runner.run_config`
  and override `manifest.peak_process_memory_mb` directly (the field is a
  plain mutable dataclass attribute) — not done in this pass.
- `retrieval_count` — the number of underlying Qdrant `query_points()`
  calls per case is not logged; only candidate lists and per-stage
  latency are. Estimating it from beam branching would be a guess, not a
  measurement, so it stays `None`.

## Ablation configs (`eval/ablation_runner.py`)

| Config | `--system` | `--use-llm-reranker` | `--sre` | Research question |
|---|---|---|---|---|
| `flat_baseline` | flat | on | on | Does hierarchical retrieval beat flat retrieval? |
| `hierarchical_no_rerank` | hierarchical | off | on | Does LLM reranking help? |
| `hierarchical_with_rerank` | hierarchical | on | on | (reference/default hierarchical behaviour) |
| `no_sre` | hierarchical | on | off | Does enabling the SRE change escalation/coherence outcomes? |
| `with_sre` | hierarchical | on | on | (reference/default, SRE enabled) |

Run one config:

```bash
python eval/ablation_runner.py run --config hierarchical_with_rerank \
  --test-set eval/test_set_full130.csv \
  --reranker-model anthropic/claude-3-5-sonnet-20241022 \
  --split heldout
```

Regenerate the manuscript's evaluation-table skeleton (cells read
`"not yet run"` until a real manifest exists under
`eval/results/raw_runs/manifests/`):

```bash
python eval/ablation_runner.py emit-table
```

## Accuracy analysis (`eval/analyze.py`)

Computes top-1 exact-match accuracy (with Wilson 95% CIs) at 1/2/3/4-digit
ISCO granularity, section/division/group/class ISIC accuracy, and level/
broad/narrow/detailed ISCED accuracy — each reported `status="not_measured"`
with a reason when the required gold column isn't in the CSV (true for
ISIC/ISCED-F sub-section-level gold labels today — no test set has them
yet). Also provides hand-rolled `wilson_score_interval()` and
`mcnemar_test()` (no `scipy` dependency, consistent with this project's
existing style).

## SRE evaluation format (`eval/sre_eval_format.py`)

Precision/recall/FPR/FNR/reviewer-workload-delta for the
`SemanticRelationEngine` as an incoherence detector — every field `None`
with `status="no_labels_supplied"` until a human-labelled set of
(SRE-flagged, human-judged) pairs is actually supplied. No such labelled
set exists in this repo.

## Dry-run mode (evaluation-readiness verification, no measurement)

Both `eval/run_eval.py` and `eval/ablation_runner.py run` accept `--dry-run`:
validates every argument, the governance gate, and all 5 ablation configs;
writes a full `ExperimentRunManifest` and empty/null-valued `CaseResult`
rows; never constructs a classifier and never touches Qdrant/Ollama/an
LLM/the network. Every dry-run manifest carries
`evaluation_status="dry_run_not_measured"` (orthogonal to `dataset_label`).
See `EVALUATION_READINESS_AND_DRY_RUN.md` for the full readiness report,
exact commands, the measurement-field-by-field readiness inventory, and
which outputs are safe to use for infrastructure verification only (never
as a manuscript measurement).

## Provisional methodology note

`backend/agents/semantic_relation.py`'s new LOW severity tier (Section G)
uses a deterministic boundary-gap formula (`gap<=1 -> LOW`, `gap==2 ->
MODERATE`, `gap>=3 -> HIGH`) that is a genuine engineering judgement call,
not sourced from an external publication. Review the exact thresholds
before citing them in the manuscript's SRE methodology description.
