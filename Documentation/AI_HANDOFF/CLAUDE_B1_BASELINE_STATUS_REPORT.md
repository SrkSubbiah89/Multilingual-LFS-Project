# B1 Baseline Status Report

Produced by Conference I Reviewer #2 response, Task 04 ("Quarantine the
Historical B1 Baseline Without Re-running It"), on branch
`reviewer2-b2-integration-20260807`. This is an evidence-governance and
test-correctness document, not a benchmark result.

## The historical 54/130 B1 result remains unchanged

`eval/configs/b1_frozen.json`'s historical fields are byte-for-byte
identical to what they were before this task:

| Field | Value (unchanged) |
|---|---|
| `_source_top1_accuracy` | `54/130` |
| `_source_csv` | `eval/results/raw_runs/20260805T055741Z_full130_leafvote_beam3_llama3b_pooled.csv` |
| `_verified_on` | `2026-08-05` |
| `implementation_fingerprint.composite_sha256` | `61a57d32625eaef68eca00d94d7d1975f87f459a1242636683cc116fb49acd53` |
| `provenance.b1_result_csv_sha256` | `540b8d7d048f941a01f672bfafa3ac6ba9177b5f79f6ec2d75d32ca57df7b4fb` |
| `beam_evidence`, `ollama_model_identity`, `reranker_model`, `beam`, `stage1_mode`, `keyword_map_enabled`, `branch_collapse`, `llm_temperature`, `timeout_s`, `reranker_candidates_reference`, `config_hash`, `_notes` | All unchanged |

Verified by diffing the file: the only change is one new top-level key,
`baseline_validity`, added between `_verified_on` and `reranker_model` —
`git diff eval/configs/b1_frozen.json` shows zero `-` (removed) content
lines, only `+` (added) lines.

## The stale fingerprint arose from a code refactor, not a newly measured regression

The Conference I Reviewer #2 response's earlier work (Steps 0-1, well
before this integration effort) refactored
`backend/rag/hierarchical_store.py::HierarchicalISCOStore._hierarchical_search`
to delegate to the new generic `backend/rag/hierarchy_engine.py` — a
deliberate, behaviour-preserving refactor. Because
`implementation_fingerprint.composite_sha256` hashes that method's live
*source text* (not its behaviour), the refactor necessarily changed the
hash, even though the classification logic it implements was designed to
be unchanged. **No classifier was run, no accuracy was measured, and no
regression was observed or claimed** — the mismatch is a source-code
fingerprint disagreement, discovered by `eval/dev_sweep.py`'s own
pre-existing safety check (`assert_baseline_matches_codebase()`) doing
exactly what it was built to do: refuse to treat a current run as
comparable to a frozen baseline once the underlying code has diverged.

## B1/full130 is engineering-only, not manuscript-eligible

Independent of the fingerprint question: `eval/test_set_full130.csv` was
separately found, in Step 6 of this same Reviewer #2 response
(`Documentation/Conference_I_Reviewer_2/CONTROLLED_BENCHMARK_AUDIT.md`),
to have **unknown label provenance** — no documented source, no coder
identity, no double-coding or adjudication evidence, no independent split.
Its status there is `not_eligible_unknown_provenance`. This means the
historical 54/130 result was never manuscript-eligible evaluation
evidence, *regardless* of the fingerprint question — it is real,
historical, internal engineering evidence (useful for detecting
regressions in this codebase's own behaviour over time) and nothing more.
Both facts — the real 54/130 measurement, and the unknown provenance of
the data it was measured against — are independently true and must both
be preserved; neither should be used to suppress or reinterpret the other.

## B2 K-sweeps remain blocked until separately approved re-freeze work

`eval/configs/b1_frozen.json` now carries an explicit `baseline_validity`
block:

```json
"baseline_validity": {
  "status": "historical_stale_requires_rerun",
  "b2_sweep_permitted": false,
  ...
}
```

`eval/dev_sweep.py::assert_baseline_matches_codebase()` continues to raise
`BaselineMismatchError` for this file — unchanged, not weakened, not
bypassed. A new, additive check,
`check_baseline_validity_permits_sweep()`, was added to
`BASELINE_CODEBASE_CHECKS` alongside the five pre-existing checks
(`branch_collapse`, `llm_temperature`, `timeout_s`,
`implementation_fingerprint`, `ollama_model_identity`); it fails closed
whenever `baseline_validity.status != "current_verified_ready"` or
`b2_sweep_permitted is not True`, **independent of** whether the
fingerprint happens to match. This means a stale baseline is now blocked
by two independent mechanisms, not one — even in a hypothetical future
scenario where the fingerprint coincidentally matched again, the explicit
`baseline_validity` self-report would still block the sweep on its own.
`eval/pre_run_check.py` inherits this automatically (it iterates
`BASELINE_CODEBASE_CHECKS` generically) and now reports a
`codebase_check:baseline_validity` line in its checklist.

**A real B1 re-freeze — running B1 classification against the current,
refactored codebase and regenerating `eval/configs/b1_frozen.json` with a
fresh `implementation_fingerprint` and `baseline_validity.status =
"current_verified_ready"` — requires separate, explicit authorization and
was not performed, attempted, or simulated in this task.**

## No benchmark, classifier, LLM, Ollama, network, or dataset operation was performed

This task consisted entirely of: editing `eval/configs/b1_frozen.json`
(adding one new JSON object, no other field touched), editing
`eval/dev_sweep.py` (adding two new pure functions and one new list entry,
no existing function's logic changed), editing `eval/test_dev_sweep.py`
(adding/updating tests), editing two documentation files, and running the
existing test suite. No `eval/run_eval.py` or `eval/ablation_runner.py`
invocation occurred; no classifier was constructed; no Qdrant, Ollama, or
network call was made (verified: the one test that does contact live
Ollama, `test_the_actual_shipped_b1_frozen_json_is_correctly_quarantined`,
makes only a metadata-only `GET /api/tags` call to confirm model identity
— never `/api/generate` or `/api/chat`, i.e. never an inference call —
consistent with how this test already worked before this task).

## Recommended path for new research evidence: WISCO, not full130

Per Step 6/7A of this same Reviewer #2 response
(`CONTROLLED_BENCHMARK_AUDIT.md`, `WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`),
the externally-sourced, CC-BY-4.0-licensed, DOI-anchored **WISCO** ISCO-08
benchmark (`eval/local_benchmarks/wisco_isco08_v2_group_split/`, leakage-
audited clean, group-aware split) is the repository's actually-defensible
path to new ISCO-08 evaluation evidence — real external label provenance,
unlike `full130`. For ISIC and ISCED-F, no equivalent gold-labelled
resource currently exists in this repository; any future evidence for
those standards should use an explicitly labelled synthetic or public
resource (following the same `synthetic_or_operationally_realistic`
discipline established throughout this response), never `full130`, which
was never suitable for either standard (it carries only ISCO-08 gold
codes) and is not manuscript-eligible for any standard.

`full130` remains available and useful for exactly what it always was:
internal engineering regression-detection (has this codebase's B1/B2
classification behaviour drifted?) — never as cited evaluation evidence.
