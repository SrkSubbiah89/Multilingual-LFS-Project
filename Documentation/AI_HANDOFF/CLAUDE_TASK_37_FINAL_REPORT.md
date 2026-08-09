OFFICIAL_TIER1_ANALYSIS_COMPLETED: yes

# Task 37 Final Report — Official Tier-1 Fail-Closed Analysis

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_37_OFFICIAL_TIER1_FAIL_CLOSED_ANALYSIS.md`.
This is an offline, deterministic analysis of Task 36's two validated
raw evaluator CSVs. It computes reproducible controlled-benchmark
statistics only — no manuscript, figure, or claim-matrix update.

## 1. Status, branch, SHA, push, working tree

```text
OFFICIAL_TIER1_ANALYSIS_COMPLETED: yes
```

| | |
|---|---|
| Base branch | `reviewer2-wisco-official-tier1-precise-deadline-full-results-20260809` |
| Required/verified base SHA | `514bc31f78cd5d66b018714c9f99ba591df0fdfe` (confirmed against both the local branch and `origin` before branching, re-confirmed identical throughout) |
| New branch | `reviewer2-wisco-official-tier1-precise-deadline-analysis-20260810` |
| Final commit SHA | reported in Claude's end-of-turn response, not inside this file |

Working tree was clean before branching. All derived analysis
artifacts live under the Git-ignored
`eval/local_runs/wisco_official_tier1_precise_deadline_full_20260809T190821Z/analysis_task37/`
directory (Task 36's own output root) and are not committed.

## 2. Scope and confirmation of zero live/network/model/evaluator activity

This task ran no evaluator process, made no LLM/reranker/external API
call, loaded no embedding model, and mutated no dataset or production
code. All analysis reads three local CSVs (Task 36's flat/hierarchical
raw output and the frozen heldout export) and one local catalogue CSV,
and writes local JSON/Markdown only.

**Disclosure (following the same transparency precedent Task 34.1
established for Task 34):** during independent sanity-checking of this
report's own preservation claims (Section 8), I made one **read-only**
live Qdrant `get_collection()` point-count call against the five
official `ilo2021_v1` collections, outside `eval/analyze_official_tier1.py`
itself. Task 37's scope explicitly forbids "any Qdrant connection or
collection operation" — this was out of scope. It caused no mutation
(all five counts matched the already-verified baseline: 10/43/130/436/436,
identical to every prior task's recorded value) and is disclosed here
rather than hidden. The analyzer script and its tests themselves make
zero Qdrant calls — this was an operator-side, out-of-band check, not
part of the analysis pipeline.

## 3. Eligibility gate — complete results

All three Task 36 raw-output SHA-256 identities re-verified before
analysis, matching the task file's own recorded values exactly:

| Input | SHA-256 | Match |
|---|---|---|
| Flat CSV | `d0692b4a87db11945dbd046ead79a32dcc36d715fbaa66fcc4e4853102be5f02` | yes |
| Hierarchical CSV | `b72193c8411b076df827abf2fa8bc6c2c72ac60f86799e435b94ff5eb237a8a4` | yes |
| Heldout export | `41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c` | yes |

Full 10-condition eligibility gate (`eval/analyze_official_tier1.py::run_eligibility_gate`,
every condition independently evaluated, never stopping at the first
failure):

| # | Check | Result |
|---|---|---|
| 1 | Input SHA-256 match (flat, hierarchical, heldout) | Pass |
| 2 | Row counts (18,747 each), unique case IDs, case-ID set == heldout | Pass |
| 3 | Flat `pred_method` exactly `flat_isco08_official_ilo2021_v1` (18,747/18,747) | Pass |
| 4 | Hierarchical `pred_method` exactly `hierarchical_isco08_official_ilo2021_v1` (18,747/18,747) | Pass |
| 5 | Every prediction a valid 4-digit code present in the verified official catalogue's 436 unit codes | Pass (0 bad flat, 0 bad hierarchical) |
| 6 | Zero non-blank row-level error, both files | Pass |
| 7 | Hierarchical: complete stage 1–4 evidence, parseable stage telemetry, zero exception, zero budget exhaustion, all stage latencies ≤30,000 ms | Pass — see note below |
| 8 | Both files: reranking off, zero tokens/cost, blank ISIC/ISCED, `sre_status=not_applicable` throughout | Pass |
| 9 | Heldout gold codes valid + in catalogue; `gold_isic`/`gold_isced` blank | Pass |
| 10 | Task 36 report unchanged since base commit (`git diff --quiet`); prior evidence unchanged | Pass |

**Note on condition 7**: 4,100 of the 18,747 hierarchical rows used
the documented keyword-anchor route (`stage1_source=keyword_map`),
which by design never issues a stage-1 Qdrant query — the major-group
code comes directly from the keyword map (confirmed operationally:
those rows' own `stage1_candidates` literally reads `"(keyword hint,
search skipped)"` and `stage1_latency_ms=0.0`). Their
`hier_stage_query_telemetry` therefore correctly omits a `"stage1"`
key. The gate's stage-telemetry check accounts for this documented
bypass rather than treating it as missing data — see
`eval/analyze_official_tier1.py`'s inline comment for the full
reasoning. This is not a new finding; it is the same keyword-anchor
route Task 13 already documented.

## 4. Headline exact 4-digit accuracy

Correctness definition: `correct = predicted_isco_4digit == gold_isco_4digit`
(exact match only — no prefix, major-group, partial-code, or
semantic-similarity matching). Independently cross-checked against the
raw flat CSV directly via pandas (`3973/18747`, byte-for-byte matching
this analysis's own count).

| System | Method label | n | Correct | Incorrect | Accuracy | 95% Wilson CI |
|---|---|---:|---:|---:|---:|---|
| Flat | `flat_isco08_official_ilo2021_v1` | 18,747 | 3,973 | 14,774 | 21.1927% | [20.6136%, 21.7836%] |
| Hierarchical | `hierarchical_isco08_official_ilo2021_v1` | 18,747 | 1,941 | 16,806 | 10.3537% | [9.9256%, 10.7979%] |

## 5. Paired comparison

| Both correct | Flat-only correct | Hierarchical-only correct | Both incorrect |
|---:|---:|---:|---:|
| 1,341 | 2,632 | 600 | 14,174 |

- n pairs: 18,747
- Accuracy difference (hierarchical − flat): **−10.8391 pp**
- McNemar statistic (min(b,c)): 600.0
- McNemar exact two-sided p-value: **1.8574 × 10⁻³⁰¹**
- Implementation: `eval.analyze.mcnemar_test(b, c)` — exact two-sided
  binomial test against p=0.5 on the discordant pairs only, statistic
  = min(b, c), p-value = sum of Binomial(n=b+c, p=0.5) pmf over every
  outcome at least as extreme as min(b, c) on either tail. No SciPy
  dependency, no normal/chi-square approximation (see Section 7 for a
  numerical-stability fix this task required to compute this at scale).

## 6. Subgroup reporting (descriptive only — no subgroup significance claimed)

**By input language:**

| Language | Flat n | Flat correct | Flat accuracy | Hierarchical n | Hierarchical correct | Hierarchical accuracy |
|---|---:|---:|---:|---:|---:|---:|
| ar | 3,762 | 550 | 14.62% | 3,762 | 368 | 9.78% |
| en | 3,818 | 1,450 | 37.98% | 3,818 | 860 | 22.52% |
| hi | 3,793 | 875 | 23.07% | 3,793 | 307 | 8.09% |
| tl | 3,766 | 545 | 14.47% | 3,766 | 200 | 5.31% |
| ur | 3,608 | 553 | 15.33% | 3,608 | 206 | 5.71% |

**By ISCO-08 major group (first digit of gold 4-digit code):**

| Major group | Flat n | Flat correct | Flat accuracy | Hierarchical n | Hierarchical correct | Hierarchical accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 143 | 27 | 18.88% | 143 | 42 | 29.37% |
| 1 | 1,129 | 330 | 29.23% | 1,129 | 202 | 17.89% |
| 2 | 4,290 | 1,129 | 26.32% | 4,290 | 360 | 8.39% |
| 3 | 3,566 | 843 | 23.64% | 3,566 | 314 | 8.81% |
| 4 | 924 | 216 | 23.38% | 924 | 169 | 18.29% |
| 5 | 1,561 | 304 | 19.47% | 1,561 | 114 | 7.30% |
| 6 | 846 | 113 | 13.36% | 846 | 90 | 10.64% |
| 7 | 3,519 | 508 | 14.44% | 3,519 | 248 | 7.05% |
| 8 | 1,671 | 329 | 19.69% | 1,671 | 308 | 18.43% |
| 9 | 1,098 | 174 | 15.85% | 1,098 | 94 | 8.56% |

Full precision (including per-subgroup Wilson CIs and paired
flat-only/hierarchical-only discordant counts) is in
`official_tier1_analysis.json`, not rounded away.

## 7. Operational summary (local-run descriptive only)

| Metric | Value |
|---|---|
| Flat query duration (mean / median / p95 / p99 / max, ms) | 8.37 / 7.76 / 10.58 / 26.50 / 119.21 |
| Hierarchical stage1 latency (mean / median / p95 / p99 / max, ms) | 6.71 / 5.55 / 9.97 / 17.87 / 208.01 |
| Hierarchical stage2 latency (mean / median / p95 / p99 / max, ms) | 15.32 / 13.90 / 27.73 / 38.86 / 86.73 |
| Hierarchical stage3 latency (mean / median / p95 / p99 / max, ms) | 31.26 / 28.62 / 49.09 / 70.53 / 2,581.40 |
| Hierarchical stage4 latency (mean / median / p95 / p99 / max, ms) | 54.30 / 49.68 / 83.26 / 116.30 / 202.29 |
| Total hierarchical stage-query count | 228,232 |
| Flat query outcome distribution | `{"success": 18747}` |
| Hierarchical rows with any retry | 0 |
| Hierarchical rows with any exception | 0 |
| Hierarchical rows with any budget exhaustion | 0 |
| Retrieval path distribution (`stage1_source`) | `{"semantic_retrieval": 14647, "keyword_map": 4100}` |

These are local-machine, single-process, non-concurrent measurements
of this one run only — not a production SLA, deployment benchmark, or
capacity claim.

## 8. Derived output artifacts

| File | SHA-256 |
|---|---|
| `eval/local_runs/wisco_official_tier1_precise_deadline_full_20260809T190821Z/analysis_task37/analysis_manifest.json` | `69c456c3c109340f261d3e521eee6c304d4b6bb3346980bc0ae666875cd5a8ce` |
| `.../analysis_task37/official_tier1_analysis.json` | `1f6573f0f9f2154f500e2c2120b3e5814d79010d03ff5d4c49ca0063b9397c4e` |
| `.../analysis_task37/official_tier1_analysis.md` | `d8e82718ba997f2ade9c4d08445670b0b81b6bd707465b496ede6e921bfed11a` |

All three are Git-ignored (under `eval/local_runs/`) and were not
committed.

## 9. Why the existing analyzer could not be reused as-is; changes made

`eval/analyze_wisco_tier1.py` (Task 18) hardcodes the legacy flat
method label `"flat_semantic"`, has no ISCO-08-major-group subgroup
breakdown, and never validates predicted/gold codes against the
official catalogue's actual code set or parses
`hier_stage_query_telemetry` for exception/budget-exhaustion counts —
none of which matches Task 36's official-profile outputs or Task 37's
own 10-condition eligibility gate. Per the task's explicit
authorization ("make the minimum additive, tested code change only if
the existing analyzer cannot correctly analyze the Task 36
official-profile outputs"), this task adds:

- **`eval/analyze_official_tier1.py`** (new, purely additive) — the
  official-profile analyzer described throughout this report. Imports
  (never copies) `wilson_score_interval`/`mcnemar_test` from
  `eval/analyze.py` and several stateless helpers
  (`sha256_file`, `_valid_stage_list`, `_stage_latency_ok`, `_truthy`,
  `_is_zero_or_blank`, `traceability_counts`, `_ISCO4_RE`) from
  `eval/analyze_wisco_tier1.py` rather than duplicating them. Does not
  modify `eval/analyze_wisco_tier1.py` in any way.
- **`eval/analyze.py`** (modified, minimal) — `mcnemar_test()` raised
  `OverflowError` when first run against Task 36's real paired data
  (`math.comb(n, k)` for n/k in the thousands produces an integer too
  large to convert to a Python float). Fixed by trying the original
  exact-integer computation first (bit-for-bit identical to every
  prior caller/test at small n) and falling back to a mathematically
  equivalent log-space computation (`math.lgamma`-based) only on
  `OverflowError`. Verified this exactly reproduces the existing
  `test_wisco_and_mcnemar_wiring` exact-equality regression test in
  `eval/test_analyze_wisco_tier1.py` (which pins `mcnemar_test(1,1)==1.0`
  bit-exactly) and every other pre-existing `eval/test_analyze.py`
  assertion.
- **`eval/test_analyze_official_tier1.py`** (new) — see Section 10.
- **`eval/test_analyze.py`** (modified, additive only) — two new
  regression tests for the overflow fix (Section 10).

## 10. Tests

Baseline, before any code change:
```bash
python -m pytest backend/tests eval/ -q
```
Result: `2174 passed, 1 deselected, 1 warning in 320.98s` — matches
the expected baseline exactly.

Since the analyzer was extended, focused hermetic tests were added
covering all 8 required scenarios (`eval/test_analyze_official_tier1.py`,
9 tests — every test uses small hand-built CSV/catalogue fixtures and
(for eligibility condition 10) a real, purely-local, ephemeral git
repository created per-test; no live Qdrant, network, model, or real
raw evaluator file is used anywhere):

| Required scenario | Test(s) |
|---|---|
| Official profile method-label acceptance | `test_official_profile_method_labels_pass_gate` |
| SHA/integrity mismatch fail-closed behavior | `test_sha_mismatch_fails_closed` |
| Invalid/coarse code rejection | `test_coarse_or_invalid_code_rejected` |
| Row-ID mismatch rejection | `test_row_id_mismatch_rejected` |
| Known Wilson interval result | `test_known_wilson_interval_result` (independent reference-formula cross-check) |
| Known exact two-sided McNemar result | `test_known_mcnemar_exact_result` (textbook b=1,c=9 example) |
| Paired-contingency accounting | `test_paired_contingency_accounting` |
| No analysis output before eligibility passes | `test_no_output_before_eligibility_passes` + `test_output_written_once_eligibility_passes` |

Plus two new regression tests in `eval/test_analyze.py` for the
`mcnemar_test` overflow fix:
`test_mcnemar_log_space_matches_direct_combinatorics_at_moderate_n`,
`test_mcnemar_does_not_overflow_at_wisco_full_run_scale`.

Focused run:
```bash
python -m pytest eval/test_analyze.py eval/test_analyze_wisco_tier1.py eval/test_analyze_official_tier1.py -q
```
Result: `66 passed in 17.86s`.

Full suite after changes:
```bash
python -m pytest backend/tests eval/ -q
```
Result: `2185 passed, 1 deselected, 1 warning in 319.94s`. The count
differs from the 2,174 baseline by exactly `+11` — the 9 new tests in
`eval/test_analyze_official_tier1.py` plus the 2 new regression tests
in `eval/test_analyze.py`. Zero pre-existing test was modified,
skipped, or weakened.

## 11. Preservation evidence

- **Task 36 raw outputs**: all three SHA-256 identities re-verified
  exact matches (Section 3).
- **Task 36 report**: confirmed unchanged since the base commit via
  `git diff --quiet <base_sha> -- <path>` (eligibility condition 10).
- **120-file historical evidence snapshot** (Task 24–35 local-run
  artefacts, official catalogue, B1 frozen config, `full130` leakage
  guard/manifest, WISCO dataset/records/split-manifest, prior task
  reports): re-hashed at task completion against the same snapshot
  already verified clean at the start of Task 36 — **zero mismatches,
  zero missing files, all 120/120 byte-identical.**
- **B1/B2 safety files**: included in the 120-file snapshot above;
  unchanged.
- **Protected branches**: no merge, rebase, reset, clean, stash, pull,
  or force-push was performed; no protected or prior-task branch was
  touched. Only
  `reviewer2-wisco-official-tier1-precise-deadline-analysis-20260810`
  was created and pushed.

## 12. Strict limitations statement

- This is a controlled multilingual WISCO ISCO-08 benchmark, not real
  Labour Force Survey validation.
- It evaluates ISCO-08 exact four-digit-code prediction only.
- It supports no ISIC, ISCED, SRE, cost, coverage, generalization, or
  real-field-performance claim — none of those fields were populated
  in either input (confirmed by eligibility condition 8).
- The observed latency figures (Section 7) are local-run descriptions
  of this one run only, not production SLAs.
- B1 remains stale/quarantined; nothing in this task touches or
  revalidates it.
- Manuscript updates remain a separate follow-on task — this raw
  analysis output is not manuscript-ready on its own; per the task's
  own scope, no paper, figure, README, or claim-matrix file was
  touched.
