# Figure Data Export Guide

Covers Section I of the Reviewer #2 response, answering reviewer comment 7:
figures need better readability and resolution. The scripts here export
clean CSV/JSON — **never raster screenshots** — so figures can be rebuilt
as vector PDF/SVG (or TikZ, for a LaTeX-native paper) at any resolution.

## Running the exports

```bash
python eval/figure_exports/export_classifier_hierarchy.py --out Documentation/Conference_I_Reviewer_2/generated/figure_data/
python eval/figure_exports/export_agent_role_diagram.py --out Documentation/Conference_I_Reviewer_2/generated/figure_data/
python eval/figure_exports/export_evaluation_results.py --out Documentation/Conference_I_Reviewer_2/generated/figure_data/
python eval/figure_exports/export_latency_scalability.py --out Documentation/Conference_I_Reviewer_2/generated/figure_data/
python eval/figure_exports/export_coverage_charts.py --out Documentation/Conference_I_Reviewer_2/generated/figure_data/
```

Every script writes both `.json` and `.csv`. Scripts that depend on
evaluation runs (`export_evaluation_results.py`,
`export_latency_scalability.py`) write a structured
`"no_manifests_found": true` / `"no_coverage_reports_found": true` export
when no real run/report exists yet — never fabricated rows.

## What each export is for, and how to turn it into a figure

### `classifier_hierarchy.{json,csv}`
ISCO-08's real 4-stage configuration (collection name + confidence weight
per stage), plus the ISIC/ISCED-F hierarchical-retrieval stages marked
`status: "planned"` (not yet implemented — see
`backend/agents/classifier_methods.py`). Use this for the classifier-
hierarchy diagram: one column/branch per standard, node labels from
`stage_name`, dashed/greyed styling for `status == "planned"` rows.
Example TikZ pattern: read the CSV with `pgfplotstable`, one
`\node` per row positioned by `stage_index`, edge color keyed by `status`.

### `agent_role_diagram.json` / `_nodes.csv` / `_edges.csv`
Nodes = one row per component (`ISCOClassifier`, `LanguageProcessor`, ...)
with `method_count` and `affects_hitl_escalation`. Edges = one row per
(component, method) with `category` (deterministic/retrieval/llm/hybrid)
and `model_name`. Use `category` for edge color/style and
`is_implemented=False` for the two stub methods to render them distinctly
(e.g. dashed edge) from real, running methods.

### `evaluation_results.{json,csv}`
One row per experiment-run manifest found under `eval/results/`
(`run_id`, `classifier_method`, `split_name`, `n_cases`, latency, HITL
escalation rate, cost). Use for the manuscript's main results table/figure
— but only once real manifests exist (`eval/ablation_runner.py run ...`,
`--split heldout`).

### `latency_scalability.{json,csv}`
Latency percentiles, throughput, and hardware fields (CPU count, RAM, GPU
model) per run. Use for a latency-vs-configuration or
latency-vs-hardware scalability chart once multiple runs (ideally on
different hardware) exist.

### `coverage_charts.{json,csv}`
Per-level implemented-vs-official code counts from the most recent
`eval/coverage_audit.py` report per standard. Use for the coverage-
disclosure figure supporting reviewer comment 5 (see
`COVERAGE_AUDIT_GUIDE.md`).

## Building vector figures from these exports

Two supported paths, neither requiring a screenshot:

1. **matplotlib → PDF** (quick, works from any of the JSON/CSV exports
   directly): `plt.savefig("figure.pdf")` — PDF output is vector by
   default in matplotlib, no DPI concerns.
2. **TikZ/pgfplots** (LaTeX-native, matches Springer's typical figure
   pipeline): use `pgfplotstable` to read the CSV directly in the `.tex`
   source, so the figure regenerates from the same data file without a
   separate export step. Recommended for the paper's final figures.

Do not paste a screenshot of a Python-rendered chart into the manuscript —
regenerate as PDF/SVG/TikZ from the exported data for print resolution.

## Tests

`eval/test_figure_exports.py` — well-formed output for every script
against small fixtures, plus explicit assertion that the "no data yet"
path never fabricates rows.
