# Task 37.1: Clean Offline Reproduction of Official Tier-1 Analysis

## Purpose

Independently reproduce and validate Task 37's official Tier-1 analysis in a strictly offline, no-Qdrant environment. This task exists because Task 37 disclosed one read-only Qdrant metadata call outside its own scope. That call caused no mutation and was not used by the analyzer, but it means Task 37 alone is not a fully scope-clean evidence record.

Task 37.1 does not revise history or conceal that disclosure. It provides a clean, reproducible analysis record using the identical, already-validated Task 36 raw files and Task 37 analyzer code, with no Qdrant connection or other live operation.

## Source and branch

- Base branch: `reviewer2-wisco-official-tier1-precise-deadline-analysis-20260810`
- Required base SHA: `85fcda8e7907b5dbd56780766d4642be9578e872`
- New branch: `reviewer2-wisco-official-tier1-analysis-clean-reproduction-20260810`
- Fetch `origin` and verify both local base and `origin/reviewer2-wisco-official-tier1-precise-deadline-analysis-20260810` equal the required SHA before branching.
- Do not merge, rebase, reset, clean, stash, pull, force-push, or open a PR.

## No-live-operation boundary

This task is strictly offline.

Do not:

- import, instantiate, call, or connect to Qdrant in any way;
- execute any Qdrant command, metadata check, point-count read, health check, or search;
- run an evaluator, benchmark, model, SentenceTransformer, Ollama, LLM, reranker, network request, or external API;
- mutate any dataset, raw CSV, catalogue, output artifact, source code, test, dependency, configuration, manuscript, figure, README, reviewer response, or branch;
- delete or regenerate any existing Task 36 or Task 37 artifact.

The only permitted inputs are local files already present in the checkout/output roots. The only permitted new files are fresh, Git-ignored derived artifacts under a new `analysis_task37_1_clean_reproduction/` directory beneath Task 36's existing output root, plus the final committed report.

No source-code or test modification is permitted. The only committed file must be the final report.

## Required inputs

Use exactly these Task 36 raw identities:

- Flat CSV SHA-256: `d0692b4a87db11945dbd046ead79a32dcc36d715fbaa66fcc4e4853102be5f02`
- Hierarchical CSV SHA-256: `b72193c8411b076df827abf2fa8bc6c2c72ac60f86799e435b94ff5eb237a8a4`
- Heldout export SHA-256: `41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c`

Also preserve byte-for-byte:

- Task 36 final report;
- Task 37 final report;
- Task 37 source/test changes;
- all Task 24–36 raw artifacts and required catalogue/B1/B2 safety files;
- Task 37’s existing derived analysis artifacts.

Hash every required input before and after this task. Any mismatch is a blocker.

## Mandatory eligibility validation

Before any result is calculated or written:

1. Re-run Task 37's exact eligibility gate against the three required raw inputs using `eval/analyze_official_tier1.py`.
2. Confirm the gate passes all ten conditions, including:
   - input hashes;
   - 18,747-row counts and exact heldout case-ID match;
   - uniform official flat and hierarchical method labels;
   - official four-digit catalogue membership for predictions and gold codes;
   - zero errors;
   - valid hierarchical evidence and telemetry;
   - zero exception, budget exhaustion, fallback, unavailable, or over-cap stage;
   - reranking off, zero token/cost, blank ISIC/ISCED, and `sre_status=not_applicable`;
   - unchanged Task 36 report and required historical evidence.
3. Statically inspect `eval/analyze_official_tier1.py` and its imports for prohibited live-operation dependencies. Record the exact inspection command and result. The script must not import or call Qdrant, `requests`, `httpx`, `urllib`, `socket`, a model library, or an LLM/reranker library.

If any eligibility or static-scope check fails, report:

```text
OFFICIAL_TIER1_ANALYSIS_CLEAN_REPRODUCTION_COMPLETED: no
```

Do not write analysis metrics, run an independent calculation, or begin any documentation work.

## Required clean reproduction

If the eligibility gate passes:

1. Run `eval/analyze_official_tier1.py` once with the exact Task 36 raw inputs and an output directory:

```text
.../analysis_task37_1_clean_reproduction/
```

Use its existing actual CLI only. Do not change flags, source, or configuration.

2. Independently calculate, using a short local Python process that reads only the three CSVs and official catalogue, the following values without importing Task 37 analyzer helpers:
   - flat correct count and accuracy;
   - hierarchical correct count and accuracy;
   - paired counts: both correct, flat-only correct, hierarchical-only correct, both incorrect;
   - hierarchical-minus-flat percentage-point difference;
   - exact two-sided McNemar p-value using a separate local log-space binomial-tail implementation;
   - 95% Wilson intervals using a separate direct formula.

3. Compare the independent values with both:
   - Task 37’s recorded results; and
   - the newly generated Task 37.1 analysis JSON.

All counts must match exactly. Floating values must match to a stated tight tolerance justified by floating-point arithmetic. Any mismatch is a failure.

4. Confirm the Task 37.1 analyzer output contains the same full language and ISCO-major-group descriptive tables and operational summaries required by Task 37.

## Required expected results

These are verification targets, not values to hard-code:

| Metric | Expected value |
|---|---:|
| Flat correct | 3,973 / 18,747 |
| Hierarchical correct | 1,941 / 18,747 |
| Both correct | 1,341 |
| Flat-only correct | 2,632 |
| Hierarchical-only correct | 600 |
| Both incorrect | 14,174 |
| Hierarchical minus flat | -10.8391 percentage points, subject only to displayed rounding |
| Exact two-sided McNemar p-value | approximately `1.8574e-301`, subject only to stated floating-point tolerance |

If the actual clean computation differs, fail closed. Do not choose or adjust an output to match these targets.

## Tests

Run:

```bash
python -m pytest backend/tests eval/ -q
```

Expected result: `2185 passed, 1 deselected, 1 warning`.

Because code and tests are prohibited from changing, any different result is a blocker. Report it and stop.

## Final report

Create and commit only:

`Documentation/AI_HANDOFF/CLAUDE_TASK_37_1_FINAL_REPORT.md`

The final report must include:

1. `OFFICIAL_TIER1_ANALYSIS_CLEAN_REPRODUCTION_COMPLETED: yes` or `no`;
2. branch, verified base SHA, final SHA, push confirmation, and clean working tree;
3. a direct statement that Task 37’s read-only Qdrant scope breach remains permanently disclosed and was not repeated, excused, or used in this task;
4. exact commands proving no live operations, including the static dependency scan;
5. all input and preservation hashes before/after;
6. eligibility result for all ten Task 37 gates;
7. exact Task 37.1 analyzer command and derived-artifact hashes;
8. independent-computation method, values, tolerance, and exact comparison results;
9. test command and output;
10. confirmation that no code/test/data/manuscript/README/figure/reviewer-response file was changed;
11. strict limitations:
    - controlled WISCO multilingual ISCO-08 benchmark only;
    - no real Labour Force Survey, ISIC, ISCED, SRE, cost, coverage, generalization, or production-performance claim;
    - observed latency remains local-run descriptive evidence only;
    - B1 remains stale/quarantined.

Push only the new branch and stop. Do not begin manuscript work.
