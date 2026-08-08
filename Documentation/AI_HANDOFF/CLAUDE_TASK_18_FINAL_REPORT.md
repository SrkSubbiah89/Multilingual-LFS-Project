# Task 18 Final Report — Reproducible Controlled WISCO Tier 1 Analysis

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_18_REPRODUCIBLE_WISCO_TIER1_ANALYSIS.md`.

**Outcome: the reusable analysis utility and its hermetic tests were built,
tested, and are green. Its one authorized run against Task 17's real raw
result CSVs correctly failed closed on a genuine raw-input invariant
violation. No accuracy/statistics were computed or reported. No
manuscript-facing documentation was updated, per the task's own explicit
instruction that those updates are conditioned on the analysis gates
passing.**

## 1. Source SHA, analysis branch, final commit SHA, push confirmation

| | |
|---|---|
| Base branch | `reviewer2-wisco-tier1-strict-full-results-20260808` |
| Required SHA | `f7c40ab3675ab8a7ee239a490fbef261c2e5bc44` |
| Verified `origin` SHA | `f7c40ab3675ab8a7ee239a490fbef261c2e5bc44` — match |
| New branch | `reviewer2-wisco-tier1-analysis-20260808` |
| Final commit SHA | recorded after this report's commit (see push confirmation below) |

Working tree was clean before branching and remained clean throughout
(only the two permitted new files below were ever added; all analysis
run output/log artifacts, per the task's own instruction on a gate
failure, were written nowhere — the failed run produced zero output
files).

## 2. Exact changed tracked files

```text
eval/analyze_wisco_tier1.py            (new)
eval/test_analyze_wisco_tier1.py       (new)
Documentation/AI_HANDOFF/CLAUDE_TASK_18_FINAL_REPORT.md   (new, this file)
```

Per the task's explicit instruction on a raw-input gate failure --
*"Record the failure in the final report, commit only permitted
code/test/report documentation, and stop"* -- no other file was added or
modified. In particular, `Documentation/Conference_I_Reviewer_2/
WISCO_TIER1_CONTROLLED_RESULTS.md` was **not** created and
`REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md` / `Documentation/
Conference_I_Reviewer_2/README.md` were **not** updated, because the
task explicitly conditions any measured-result statement in those
documents on the analysis utility completing successfully with its
gates passing, which did not happen (see §4).

## 3. Raw input paths and SHA-256 values

| Input | Path | SHA-256 |
|---|---|---|
| Reference (canonical heldout) | `eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv` | `41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c` (matches the required value exactly) |
| Flat result CSV | `eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/flat/20260808T115438Z_wisco_v2_tier1_flat_model_free.csv` | `b237b56e79070e63ad49d26171c3b887bde8d43739b13dd977a049833ce9966b` |
| Hierarchical result CSV | `eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/hierarchical/20260808T153325Z_wisco_v2_tier1_hierarchical_strict_model_free.csv` | `e1456bac9fa4cc11b4c4876e8f6f3fa7268f07ca83c4a02e707e236584545a03` |

Both result CSVs are exactly the paths named in the Task 17 final
report; both were opened read-only. Post-run re-hash (see §9) confirms
neither file was modified by this task.

## 4. Raw-input validation-gate outcomes

Reference-CSV gate: **pass**. SHA-256 matched exactly; 18,747 rows;
unique, canonically-ordered case IDs; zero nonblank `gold_isic`/
`gold_isced` values.

**Flat result-CSV gate: FAIL.** First (and only) violated invariant
reported by the utility:

```text
GATE FAILURE: flat result CSV has missing/invalid pred_isco_4digit
(must be exactly 4 digits) -- 4754 affected case_id(s): [...]
No analysis output was written.
```

Root-caused (read-only inspection, no data/code changed to investigate):

- `4754 / 18747` flat rows (`25.36%`) have a `pred_isco_4digit` value
  that is **not** a 4-digit code: `1372` rows have a bare 1-digit major-
  group code (e.g. `"0"`), and `3382` rows have a 2-digit sub-major-group
  code (e.g. `"23"`). The remaining `13993` rows do have a valid 4-digit
  code.
- This is **not** a data-corruption or Task 17 execution defect. It is a
  documented, pre-existing characteristic of the `isco_occupations`
  Qdrant collection that the `--system flat` baseline searches
  (`backend/rag/vector_store.py`'s own module docstring, unchanged by
  this task): *"Holds a curated ISCO-08 dataset (~110 entries: major
  groups, sub-major groups, and the most common unit groups...)"* --
  i.e. the flat collection intentionally spans multiple hierarchy
  levels rather than indexing only 441 full unit-group codes, so its
  top-1 nearest-neighbour result can legitimately be a coarser-level
  entry when no closer unit-group match exists in that small,
  curated set.
- Task 17's own integrity checks did not (and were not scoped to) catch
  this, because Task 17 verified execution-time correctness (exit
  status, row count/order, zero row-level `error`, reranker/LLM off,
  ISIC/ISCED/SRE not constructed) -- it never asserted anything about
  the *code-length* of `pred_isco_4digit`. Task 18's stricter accuracy-
  analysis gate is the first check in this entire sequence to require
  every predicted code be a genuine 4-digit unit-group code, and it
  correctly caught a real gap.
- The hierarchical result CSV does **not** have this problem: a direct
  check found `18747/18747` hierarchical rows have a valid 4-digit
  `pred_isco_4digit` (hierarchical retrieval always terminates at the
  441-entry unit-group stage). Because the flat CSV gate failed first,
  the hierarchical CSV's full gate sequence (stage-evidence, latency,
  known-risk IDs, etc.) was never reached by the script -- gate checks
  run in the documented order (reference, then flat, then
  hierarchical) and stop at the first failure, per the tool's fail-
  closed design.

Per the task's explicit instruction, this violation was **not** worked
around: the gate was not weakened, the raw CSV was not altered, no code
under test was changed, and no partial/point-estimate result was
computed or written. `eval/analyze_wisco_tier1.py`'s CLI exited nonzero
(`sys.exit(1)`) and wrote **zero** files under `--out`.

## 5. Exact analysis command

```bash
python eval/analyze_wisco_tier1.py \
  --flat-csv eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/flat/20260808T115438Z_wisco_v2_tier1_flat_model_free.csv \
  --hierarchical-csv eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/hierarchical/20260808T153325Z_wisco_v2_tier1_hierarchical_strict_model_free.csv \
  --reference-csv eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv \
  --out eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/analysis_task18 \
  --expected-reference-sha256 41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c \
  --expected-n 18747 \
  --expected-hierarchical-method-prefix hierarchical_ \
  --max-stage-latency-ms 30000
```

Run exactly once, as authorized. Exit status: **1** (nonzero, as
required on gate failure).

## 6. Ignored output bundle path

**None was created.** The `--out` directory
(`eval/local_runs/wisco_v2_tier1_strict_full_20260808T115305Z/analysis_task18/`)
does not exist on disk -- confirmed via `ls` immediately after the run
returning "No such file or directory". This is the tool working as
designed: *"It must not produce a partially measured result bundle."*

## 7. Measured values

**None.** No accuracy, Wilson interval, per-language result, paired
contingency count, McNemar p-value, latency descriptive, or
traceability count was computed, because the run never reached the
statistics stage. Sections 3-7 of the required Markdown output
(overall accuracy, language table, paired comparison, latency,
traceability) do not exist for this run.

## 8. Focused and full test results

Both required test commands were run against the **new tool itself**
(fixture-based, fully hermetic) -- they validate that the tool behaves
correctly, independent of whether any particular real-data run happens
to pass or fail its own gates.

```
python -m pytest eval/test_analyze_wisco_tier1.py eval/test_analyze.py eval/test_run_eval_b2.py eval/test_require_genuine_hierarchical.py eval/test_docs_consistency.py -q
→ 102 passed in 149.22s (0:02:29)

python -m pytest backend/tests eval/ -q
→ 1979 passed, 1 deselected, 1 warning in 328.52s (0:05:28)
```

**Zero failures in either run.** `1979 = 1948` (Task 17's baseline
count) `+ 31` (all-new `eval/test_analyze_wisco_tier1.py` tests) --
confirming zero regressions and that every new test is additive. Both
runs ended with the project's previously-documented harmless
`crewai`/`colorama` atexit teardown traceback (`ValueError: I/O
operation on closed file`); process exit code was 0 in both cases and
all tests were already reported passed before that traceback occurred.

The 31 new tests in `eval/test_analyze_wisco_tier1.py` cover: hand-
checkable 1/2/3/4-digit accuracy, Wilson/McNemar wiring against the
reused `eval/analyze.py` functions, per-language grouping and macro-
average, order/id-set mismatch rejection, duplicate-ID rejection
(reference and result), invalid/missing 4-digit code rejection (gold
and pred, parametrized over 4 bad values), nonblank-error rejection,
flat wrong-method/reranker/cost/token rejection (4 tests), hierarchical
fallback-method rejection, hierarchical empty/malformed stage-evidence
rejection (2 tests), hierarchical over-cap/non-numeric latency
rejection (2 tests), known-risk missing/failing-row rejection (2
tests), dry-run evaluation_status rejection, reference-only gates
(nonblank ISIC/ISCED, sha256 mismatch, row-count mismatch), full-CLI
output-schema presence via `main()` (including confirming zero output
is written on a gate failure, mirroring the real outcome above), and
source-CSV-unchanged-after-analysis.

## 9. Confirmation: no evaluation, Qdrant, model, LLM, network, data-mutation, or source-prediction run occurred

- `eval/analyze_wisco_tier1.py` never imports or calls `run_eval.py`,
  `QdrantClient`, `SentenceTransformer`, CrewAI, or any LLM/network
  client -- it is pure `csv`/`json`/`hashlib`/`math`/`statistics`
  standard-library code plus one same-directory import
  (`eval/analyze.py`'s `wilson_score_interval`/`mcnemar_test`).
- The one real-data invocation (§5) read three existing local CSV files
  and made zero other I/O.
- Post-run SHA-256 re-check confirms all three source files are
  byte-identical to their pre-run hashes in §3 -- nothing was mutated:
  reference `41c20fcc9e...`, flat `b237b56e79...`, hierarchical
  `e1456bac9f...` (all unchanged).
- No Qdrant query, no model/embedding load, no Ollama/CrewAI/LLM/paid
  API call, no WISCO data rebuild/export/mutation, no B1 re-freeze, no
  B2 sweep, no ISIC/ISCED/SRE evaluation, and no `eval/run_eval.py`
  invocation occurred anywhere in this task.

## 10. Protected-branch and clean-tree confirmation

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| every prior `reviewer2-*` task/integration branch (through `reviewer2-wisco-tier1-strict-full-results-20260808`) | not touched |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push was run. No PR was created. `git status
--short` shows exactly the two new files in §2 before this report's own
commit.

## 11. Manuscript-safe wording example and unsafe-wording example

**Safe:** *"On a controlled, externally sourced multilingual ISCO-08
occupation-title benchmark (WISCO), our hierarchical retrieval pipeline
resolves every case to a genuine 4-digit unit-group prediction; a
single-stage flat baseline searched against a small curated ~110-entry
collection resolves only 74.6% of cases to a full 4-digit code, with the
remainder returning a coarser major- or sub-major-group match. Accuracy
comparison between the two systems has not yet been computed pending a
flat-baseline fix or an updated accuracy gate that can score coarser
predictions."*

**Unsafe (must not be written, and this task supports none of it):**
*"Our system achieves state-of-the-art accuracy on real Labour Force
Survey data"* / *"WISCO results validate our ISIC and ISCED
classifiers"* / *"Hierarchical retrieval is X% more accurate than flat
retrieval"* (no accuracy number of any kind was computed by this task)
/ *"Reranking with an LLM improves accuracy by X%"* (reranking was off
throughout; no such comparison exists).

## 12. Reviewer-comment evidence-status summary

| Reviewer #2 comment area | Status after Task 18 |
|---|---|
| Real-LFS validation | Still blocked -- no real LFS data exists in this project; unrelated to this task |
| Computational/latency analysis | Improved readiness only -- Task 17 produced raw per-row latency data, but Task 18 did not compute any latency descriptive because the run never reached that stage |
| ISIC/ISCED coverage disclosure | Unaffected by this task -- this run is ISCO-08-only by construction |
| Controlled multilingual ISCO-08 benchmark evidence (this task's own scope) | **Still blocked** -- the analysis utility and its tests are complete and correct, but the one authorized real-data run did not produce a citable result; a genuine data-scope gap in the flat baseline must be resolved (or the gate/comparison redesigned) in a future task before any WISCO accuracy number can be cited |

No claim is made that this comment, or any other Reviewer #2 comment, is
now fully satisfied.

## 13. Direct statement

No controlled multilingual ISCO-08 benchmark evidence, or any other
evidence, was produced by this task. The reusable analysis utility and
its hermetic test suite are complete, correct, and proven (via this very
run) to fail closed rather than fabricate a result when real data
violates a stated invariant. A future task must resolve the flat
baseline's partial-resolution behavior (or redesign the comparison to
legitimately handle it) before Task 18's tool can produce its first
real, citable WISCO Tier 1 result.
