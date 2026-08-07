# Computational Analysis Guide

Covers Section D of the Reviewer #2 response, directly answering reviewer
comment 4: computational analysis (training/inference cost, memory
efficiency, latency, scalability) is missing from the manuscript.

There is no training cost to report — every classifier in this system is
either deterministic/rule-based (`ISCEDClassifier`, `ValidationAgent`
Stage 1, `HITLQualityManager` scoring) or uses a pre-trained embedding
model / pre-trained LLM via API/local inference (`intfloat/multilingual-e5-
small` for retrieval, Ollama/Claude for reranking) — nothing in this
codebase is fine-tuned. The computational analysis this guide supports is
therefore **inference cost, memory, latency, and scalability**, not
training cost.

## Producing a real manifest

```bash
python eval/ablation_runner.py run --config hierarchical_with_rerank \
  --test-set eval/test_set_full130.csv \
  --reranker-model anthropic/claude-3-5-sonnet-20241022 \
  --split heldout
```

Writes a `CaseResult` CSV plus an `ExperimentRunManifest` (CSV + JSONL)
under `eval/results/raw_runs/`. See `EVALUATION_PROTOCOL.md` for the full
manifest schema and the dev/held-out split discipline.

## What the manifest actually measures, and what it honestly doesn't

**Measured, real, computed from the run:**
- Latency mean/p50/p95 (from `CaseResult.end_to_end_latency_ms`, already
  instrumented per-case in `eval/run_eval.py`)
- Throughput (cases/sec, derived from summed per-case latency)
- Reranker invocation count (`CaseResult.reranker_fired` — a real per-case
  bool)
- HITL escalation rate (`CaseResult.escalation_triggered` — **the
  evaluation harness's own research-only heuristic**, combining
  confidence + SRE severity + stratified sampling; NOT the production
  `HITLQualityManager` decision — do not conflate the two in the
  manuscript)
- Estimated cost (summed `CaseResult.estimated_cost_usd`, real for
  API-backed rerankers, a real zero for local Ollama runs)
- Hardware (CPU model/count, RAM, best-effort GPU probe via `torch.cuda`
  or `nvidia-smi`), OS/Python version, full installed-dependency snapshot
  (`importlib.metadata`), git commit, dataset hash

**Honestly not measured (both `None` + a `<field>_unavailable_reason`
string):**
- **Peak process memory** — `CaseResult.peak_memory_mb` is a declared
  field `eval/run_eval.py` never actually populates. (A separate,
  unrelated `psutil`-based peak-RSS sampler exists in `eval/dev_sweep.py`
  for its own K-sweep eligibility checks, but its value never reaches a
  `CaseResult` row, so `eval/manifest.py` cannot read it from there.)
- **Peak GPU memory during the run** — the hardware probe reports total
  device capacity (`torch.cuda`/`nvidia-smi`), not peak usage during
  inference; there is no per-run GPU memory sampler in this codebase.
- **Retrieval count** — the number of underlying Qdrant `query_points()`
  calls per case isn't logged; only candidate lists and per-stage latency
  are, and neither is a reliable proxy for call count without guessing at
  beam branching.

Do not fill these into the manuscript from another source (e.g. a
back-of-envelope estimate) and present them as measured — if the paper
needs them, they require new instrumentation (see "Extending the
manifest" below), not estimation.

## Scalability

`eval/manifest.py`'s `latency_mean_ms`/`throughput_cases_per_sec` plus
`hardware.cpu_count`/`ram_gb` give a genuine, if limited, scalability
signal: run the same config on machines with different core counts / RAM
and compare throughput. This has not been done in this pass (only local,
single-machine runs). `eval/figure_exports/export_latency_scalability.py`
exports exactly the columns needed for a latency-vs-hardware scalability
chart, once multiple such runs exist.

## Extending the manifest (future work, not done in this pass)

To close the peak-memory/GPU/retrieval-count gaps: `eval.ablation_runner.
run_config()` returns a mutable `ExperimentRunManifest` object before
writing it — a caller can measure peak RSS via `psutil.Process().
memory_info().rss` (same pattern as `eval/dev_sweep.py`'s `_peak_rss_mb()`)
around the `subprocess.run()` call and set
`manifest.peak_process_memory_mb` directly, clearing
`peak_process_memory_unavailable_reason`. Not implemented here to avoid
scope creep beyond what Section D specified for this pass.

## Tests

`eval/test_manifest.py` — latency/throughput arithmetic, the null+reason
invariant for every honestly-unmeasured field, hardware/dependency
probing never raises, CSV+JSONL round-trip, and the real-LFS-validation
governance fail-fast gate.
