# Step 7 Command (Prepared — Not Executed)

> **Superseded by Step 7A (2026-08-10).** The commands below target the
> `wisco_isco08_v1` package, which Step 7A's leakage audit found had 4
> cross-split exact-duplicate-text groups (see
> `WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`). **Use
> `Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`'s
> Phase D/E instead** — it targets the corrected, leakage-audited
> `wisco_isco08_v2_group_split` package and adds the no-LLM/reranking-tier
> distinction and full time/resource estimates this document does not
> have. This document is kept for its historical record of Step 6's
> reasoning; v1 itself is unchanged and retained as the pre-fix audit
> record.

Conference I Reviewer #2 response, Step 6, Phase G: "If an eligible
controlled benchmark is found, prepare a precise command file for Step 7
but do not execute it." This document is that file. **Nothing below has
been run in Step 6** — only schema validation and dataset-card validation
(deterministic, zero classifier calls) were executed; see
`CONTROLLED_BENCHMARK_AUDIT.md` and the Step 6 final report for exactly
what those were.

## What is ready

The WISCO-derived ISCO-08 controlled benchmark (`eval/build_wisco_isco_
benchmark.py`'s output), `conditionally_eligible_needs_documentation` per
`CONTROLLED_BENCHMARK_AUDIT.md` Phase B/C — real, externally-sourced,
CC-BY-4.0-licensed, DOI-anchored gold labels, never before used as an
evaluation benchmark. `dataset_label=synthetic_or_operationally_realistic`
throughout (never `approved_real_lfs_validation` — WISCO is reference
data, not Labour Force Survey respondent data).

- Package location (Git-ignored, regenerate with the build command below):
  `eval/local_benchmarks/wisco_isco08_v1/`
- Records: 20,760 (occupation × language pairs) from 4,232 source
  occupations, 5 languages (en/ar/ur/hi/tl)
- Split: 2,005 dev / 18,755 heldout (~10%/90%, deterministic per-occupation
  hash split — see `eval/build_wisco_isco_benchmark.py`'s docstring)
- `dataset_hash`: `aad7f99e7558b23323ab70ae89463d13473274e84f14ef9706c8f177628d7016`
- `eval/validate_controlled_benchmark.py` result: `ok=True, errors=0, warnings=0`

## Regenerating the package (deterministic, no classifier calls)

```bash
python eval/build_wisco_isco_benchmark.py --out-root eval/local_benchmarks/wisco_isco08_v1
python eval/export_benchmark_to_run_eval_csv.py \
    --records eval/local_benchmarks/wisco_isco08_v1/records.json \
    --split dev --out eval/local_benchmarks/wisco_isco08_v1/dev_run_eval_format.csv
python eval/export_benchmark_to_run_eval_csv.py \
    --records eval/local_benchmarks/wisco_isco08_v1/records.json \
    --split heldout --out eval/local_benchmarks/wisco_isco08_v1/heldout_run_eval_format.csv
```

## The Step 7 command itself (NOT executed in Step 6)

**Scale warning before running this**: 18,755 heldout rows with LLM
reranking (even a free local Ollama model) at the per-case latency observed
in Step 5/5.1 (roughly 15-25 seconds/case for a reranked hierarchical
config on this hardware) would take **on the order of 3-5 days of
continuous local compute** for one config, let alone all 5 ablation
configs. Before running the command below, Step 7 should decide — and
document, not silently pick — a practical `--limit` (e.g. a stratified
subsample of a few hundred cases across major groups and all 5 languages)
for a first pass, reserving the full heldout set for whichever final
config(s) the dev-split parameter selection actually settles on. This
decision belongs to Step 7, not to this preparation step.

```bash
# Confirm all 5 configs still resolve (no execution, no network) --
# safe to run any time, already exercised in Step 4:
python eval/ablation_runner.py validate-configs

# Step 7, config 1 of 5 (repeat for hierarchical_no_rerank,
# hierarchical_with_rerank, no_sre, with_sre) -- shown WITHOUT --dry-run,
# i.e. a REAL measured run, which is exactly what makes this Step 7's job
# and not Step 6's (restriction 5: "Do not run the full measured ablation
# study yet"):
python eval/ablation_runner.py run --config flat_baseline \
    --test-set eval/local_benchmarks/wisco_isco08_v1/heldout_run_eval_format.csv \
    --split heldout \
    --reranker-model ollama/llama3.2:1b \
    --dataset-label synthetic_or_operationally_realistic \
    --evaluation-status measured \
    --limit 300 \
    --output-root eval/local_runs/step7_wisco_isco08_<timestamp>
```

`--evaluation-status measured` (not `measured_synthetic_fixture_only`) is
correct here — unlike Step 5's 5-row integration-only fixture run, a
substantive run against thousands of real WISCO-sourced cases is an actual
measurement, not merely a pipeline-integration check. It is still
`manuscript_eligible=false` by construction (`dataset_label=
synthetic_or_operationally_realistic`, not `approved_real_lfs_validation`)
— see `eval/manifest.py`'s `manuscript_eligible` computation. Manuscript
wording after such a run may say "measured against a real, externally-
sourced multilingual occupation benchmark (WISCO)" but must **never** say
"validated on real Labour Force Survey data."

## Prerequisite: dev-split parameter selection first

Per this project's existing dev/heldout discipline (`eval/dev_set_schema.md`,
`eval/ablation_runner.py`'s `--split dev` routing), Step 7 must run any
parameter selection (e.g. `--reranker-candidates`/K, prompt variant choice)
against `--split dev` (`dev_run_eval_format.csv`, 2,005 rows — still large;
a further documented subsample is reasonable here too) **before** touching
`--split heldout` with the finally-chosen configuration. Do not tune after
seeing a heldout result (Step 5.1's restriction 5 applies with equal force
here).
