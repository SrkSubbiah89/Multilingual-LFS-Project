# Claude B2 Integration Report

Produced in response to
`Documentation/AI_HANDOFF/PERPLEXITY_TO_CLAUDE_03_B2_INTEGRATION.md`
(task ID `03-b2-integration`), executed on `origin/reviewer2-enhancement`'s
instructions after explicit user approval in the current session.

## 1. Source branch tips and new integration-branch tip

| Branch | SHA | Role |
|---|---|---|
| `reviewer2-wip-snapshot-20260807` | `64e6ec4d1d1dac940d85242b791748ca1468050d` | Protected source (unchanged by this task) |
| `origin/conference1-b2-evaluation` | `675121bbf656dad3e611f3280d95606b5b123e92` | B2 source (unchanged by this task) |
| `reviewer2-b2-integration-20260807` | `b33a9bbd1e8daebec310214acb14095d1e0c5ad3` | **New integration branch (this task's output)** |

Start-state checks (run before any branch operation) all passed: working
tree clean, `HEAD` on `reviewer2-wip-snapshot-20260807` at exactly
`64e6ec4d1d1dac940d85242b791748ca1468050d`, both source-branch tips exactly
matching the SHAs recorded in the task file.

## 2. Exact merge commit SHA

**`1a066027b47412843bfa0fa061d7c7be33bcc79a`** — `git merge --no-ff
origin/conference1-b2-evaluation`, message: "Merge conference1-b2-evaluation
into Reviewer 2 integration branch".

A second commit, **`b33a9bbd1e8daebec310214acb14095d1e0c5ad3`**, was added
immediately after to fix a real test failure surfaced by the merge (§3,
§8) — this is the current branch tip, prior to this report's own commit.

## 3. Every conflict encountered and its resolution

### Git-level merge conflict (expected, per the merge-impact analysis)

**`eval/test_run_eval_b2.py`** — both branches had independently appended
122 lines of new test functions immediately after the same shared anchor
line (`test_retry_count_always_zero_for_standard_reranker`, line 193 in
both). Resolved by **manual concatenation**: the Reviewer #2 snapshot's
`sre_enabled`/`use_llm_reranker` test block (lines 198-315 post-resolution)
is kept first, followed immediately by B2's "exact classifier-input
regression" test block (lines 317-432 post-resolution), with the shared
193-line base above both left byte-for-byte untouched. No conflict markers
remain (verified: `grep -n "<<<<<<<\|=======\|>>>>>>>"` returns nothing);
the file parses (`ast.parse` succeeded) and both blocks' tests pass (§7).

**`requirements.txt`** — auto-merged cleanly by git with no manual
intervention (snapshot's `pyyaml` and B2's `scikit-learn==1.6.1` insert at
different, non-adjacent locations in the file). Verified post-merge: both
lines present.

### Runtime conflict discovered only after merging (not a git conflict)

Two pre-existing test failures appeared once both branches' code was
combined — neither is a git merge conflict (no conflict markers, both
files auto-applied cleanly), but both are genuine consequences of the two
branches having evolved independently:

1. **`eval/dev_sweep.py::compute_k_config_hash()`** and its own unit test
   (`eval/test_dev_sweep.py::test_compute_k_config_hash_matches_run_eval_config_hash_mechanism`)
   each construct a `SimpleNamespace` stand-in for `run_eval.py`'s CLI
   `args` object, to call into `run_eval._config_hash()`. B2 wrote both
   before the Reviewer #2 snapshot added the `--sre`/`--use-llm-reranker`
   CLI flags (Section E ablation support) that `_config_hash()` now reads
   unconditionally — causing `AttributeError: 'types.SimpleNamespace'
   object has no attribute 'sre'`. **Fixed** (commit `b33a9bb`): added
   `sre="on", use_llm_reranker="on"` to both fake-args constructions,
   matching B2's true, unmodified default behaviour (K-sweep never toggles
   either flag; the actual classification path,
   `run_eval.run_one_case()`, already defaults both to the equivalent
   `True` and was never affected — this was a hash-string-provenance gap
   only, never a classification-behaviour bug). This fix is confined
   entirely to the integration seam (two `SimpleNamespace` construction
   sites); it changes no historical result and no frozen baseline.

2. **`eval/test_dev_sweep.py::test_the_actual_shipped_b1_frozen_json_passes_shape_and_codebase_checks`**
   — **NOT fixed, reported instead**, per the task's own rule ("fix only
   if the fix is confined to the integration itself; otherwise stop and
   report it"). `eval/configs/b1_frozen.json` records
   `implementation_fingerprint.composite_sha256` for
   `HierarchicalISCOStore._hierarchical_search`, computed when B1 was
   frozen (2026-08-05). The Reviewer #2 snapshot's earlier work (Steps
   0-1, predating this integration task) refactored that exact method to
   delegate to the new generic `backend/rag/hierarchy_engine.py` — a
   behaviour-preserving refactor by design, but one that necessarily
   changes the method's *source text*, and therefore its fingerprint hash.
   The test fails exactly as designed: `dev_sweep.assert_baseline_matches_codebase()`
   raises `BaselineMismatchError` reporting the mismatch
   (`61a57d32...` in the frozen baseline vs. `f73b9a4c...` live), refusing
   to treat a future B2 run as comparable to the frozen B1 result. **The
   only correct fix is to re-run B1 classification against
   `eval/test_set_full130.csv` and re-freeze `b1_frozen.json`** — an actual
   classifier benchmark run, explicitly forbidden in this task ("Do not
   run any classifier benchmark... Do not regenerate, overwrite, delete,
   or reinterpret historical measured outputs, frozen B1/B2 results").
   Left exactly as-is; see §8/§10 for how this must be handled next.

No other conflict, of either kind, was encountered anywhere in the merge
or subsequent test runs.

## 4. Confirmation both `test_run_eval_b2.py` test blocks remain

Verified directly: `pytest eval/test_run_eval_b2.py -q` → **21 passed**
(**Correction, Task 04**: the original count above ("18 pre-existing + 3
... + 4 ...") did not arithmetically sum to 21 and has been corrected
here, using counts verified directly via `git show <ref>:eval/
test_run_eval_b2.py | grep -c "^def test_"` against the actual merge base
and both parent commits, not estimated: **12 tests shared by both branches
unchanged** (the merge base,
`git merge-base reviewer2-wip-snapshot-20260807 origin/conference1-b2-evaluation`,
has exactly 12) **+ 6 added by the Reviewer #2 snapshot** (the
`sre_enabled` block, 3 tests, and the `use_llm_reranker` block, 3 tests —
`reviewer2-wip-snapshot-20260807`'s own copy of the file has 18 total, i.e.
12 + 6) **+ 3 added by B2** (the "exact classifier-input regression" block
— `origin/conference1-b2-evaluation`'s own copy has 15 total, i.e. 12 + 3)
**= 21**. Grepped the
resolved file directly to confirm both `_kwargs_with_sre_inputs()` (snapshot)
and `_TRICKY_INPUT_TEXT`/`test_dev_sweep_csv_to_classify_chain_is_verbatim_end_to_end()`
(B2) are present with no truncation.

## 5. Dependency outcome

Both dependency additions retained in `requirements.txt`:
```
scikit-learn==1.6.1     (line 18 — B2, used by dev_sweep.py's extended statistics)
pyyaml                  (line 28 — Reviewer #2, used by eval/coverage_audit.py / eval/catalogue_importer.py)
```
`requirements-dev.txt` (B2's test-only dependency lock, new) retained
unchanged. No package version was altered beyond what the clean auto-merge
produced; no installation or test failure required a dependency-version
change, so no such change was made or requested.

## 6. B2 dev-set schema/validator/sweep decision

Per the task's explicit instruction, **B2's versions of `eval/dev_set_schema.md`,
eval/validate_dev_set.py`, and `eval/dev_sweep.py` were used as the
integration branch's implementation** (brought in by the merge unchanged,
since the Reviewer #2 snapshot never touched these three files — confirmed
zero git conflict on any of them). This gives the integration branch B2's
fail-closed ISCO-catalogue validation, hash-only `full130` leakage checks,
and normalization-fingerprint integrity checking.

**Documentation note (as requested, concise, not a forced unification)**:
B2's 7-column `eval/dev_set_v1.csv` schema
(`case_id, language, respondent_text, gold_isco_code, gold_label_source,
annotator_or_adjudication_reference, dataset_split`) is the canonical
schema for B2's K-selection dev set, used only by `eval/dev_sweep.py`/
`eval/pre_run_check.py`/`eval/validate_dev_set.py`. The Reviewer #2
snapshot's `eval/dataset_card_schema.py::DatasetCard` (real-LFS governance)
and `eval/split_manifest_schema.py::SplitManifest` +
`eval/controlled_benchmark_schema.py::BenchmarkRecord` (controlled
benchmarks, e.g. WISCO) remain the authoritative schemas for their
respective domains. These three schema families are **not** merged or
forced into one in this task — they govern three genuinely different data
categories (B2 K-selection dev set; real LFS respondent intake; controlled
reference benchmarks) and each was designed, independently, with that
specific category's governance needs in mind.

**Pre-finalisation checks** (both required by the task, both performed):
1. Confirmed no Reviewer #2 snapshot module calls `validate_dev_set()` at
   all (grep-verified across `eval/` and `backend/`) — the only caller is
   B2's own `eval/pre_run_check.py`, which already uses the correct
   keyword-argument form (`valid_isco_codes=...`) matching B2's own
   extended signature. No positional-argument breakage exists or is
   possible today.
2. Confirmed B2's validator and sweep tests pass: see §7.

## 7. Test commands and exact results

```
pytest eval/test_run_eval_b2.py -q
→ 21 passed in 123.84s

pytest eval/test_validate_dev_set.py eval/test_dev_sweep.py \
       eval/test_pre_run_check.py eval/test_full130_access_guard.py -q
→ (after both dev_sweep.py fixes) 1 failed, 84 passed in 24.04s
  FAILED eval/test_dev_sweep.py::test_the_actual_shipped_b1_frozen_json_passes_shape_and_codebase_checks
  (see §3/§8 — requires a real B1 re-freeze, explicitly out of scope)

pytest eval/test_validate_real_lfs_governance.py \
       eval/test_validate_evaluation_discipline.py \
       eval/test_validate_controlled_benchmark.py \
       eval/test_wisco_leakage_audit.py -q
→ 99 passed in 7.72s

pytest eval/test_docs_consistency.py -q
→ 7 passed in 0.05s

pytest backend/tests eval/ -q
→ 2 failed, 1826 passed, 1 deselected, 1 warning in 294.64s (0:04:54)
  FAILED backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity
  FAILED eval/test_dev_sweep.py::test_the_actual_shipped_b1_frozen_json_passes_shape_and_codebase_checks
```

**Baseline comparison**: the protected snapshot's pre-integration full-suite
result was `1656 passed, 1 known pre-existing failure, 1 deselected, 1
warning`. Post-integration: `1826 passed` (net **+170**, consistent with
B2's full test suite — `test_validate_dev_set.py`, `test_dev_sweep.py`,
`test_pre_run_check.py`, `test_full130_access_guard.py`, plus B2's 4 new
tests in the reconciled `test_run_eval_b2.py` — being merged in), same 1
deselected, same 1 warning, and **2 failed instead of 1** — but the second
failure is not a new regression in the sense the task means by that word;
see §8.

## 8. Whether any new failures were found

**Yes, initially 4** (all in `eval/test_dev_sweep.py`, none pre-existing on
either source branch alone — they only manifest once both branches'
independently-evolved code is combined):

- 3 were the `AttributeError: ... no attribute 'sre'` failures, all sharing
  the exact same root cause (§3.1) — **fixed**.
- 1 remains: `test_the_actual_shipped_b1_frozen_json_passes_shape_and_codebase_checks`
  (§3.2) — **not fixed, reported**, because the only correct fix requires
  an actual classifier benchmark run (re-freezing B1), which this task
  explicitly forbids. This is the sole test failure attributable to this
  integration beyond the pre-existing documented baseline failure.

**The known pre-existing baseline failure**
(`test_isco_classifier_extended.py::test_llm_used_for_low_similarity`) is
present, unchanged, and is **not** a new regression — it was already
failing on `reviewer2-wip-snapshot-20260807` before this integration began.

## 9. Confirmation: no benchmark, inference, historical evidence, paper, protected-branch, or new-classifier-feature change occurred

- No `eval/run_eval.py` or `eval/ablation_runner.py` invocation was made
  against real data; no classifier was constructed; no Qdrant/Ollama/
  LLM/network call was made anywhere in this task.
- No Step 7B evaluation, WISCO measurement, or any new accuracy/latency/
  cost/coverage/real-LFS claim was added, computed, or implied.
- `eval/configs/b1_frozen.json` (including its 54/130, 41.5% historical
  result) was **not** modified — brought into the integration branch
  exactly as it exists on `conference1-b2-evaluation`, byte for byte.
- The `eval/test_set_full130.csv` `not_eligible_unknown_provenance`
  conclusion (Step 6, `CONTROLLED_BENCHMARK_AUDIT.md`) was not touched,
  reinterpreted, or contradicted.
- WISCO v1/v2 benchmark records, split manifests, hashes, and the Step 7A
  leakage-audit results were not touched (no file under
  `eval/local_benchmarks/` was read or written by this task; verified via
  `git status` showing no such paths at any point).
- Existing dry-run and synthetic-run output directories
  (`eval/local_runs/`) were not touched.
- No manuscript text, dataset card, or generated evaluation-evidence file
  under `Documentation/Conference_I_Reviewer_2/generated/` was modified.
- `master`, `conference1-b2-evaluation`, `reviewer2-wip-snapshot-20260807`,
  and `reviewer2-enhancement` were fetched (read-only) but never checked
  out, merged into, rebased, or pushed to. Verified at the end of this
  task: their SHAs are unchanged from the values recorded in §1.
- No `git reset`, `git clean`, `git stash`, `git rebase`,
  `git push --force`, or `git commit --amend` was run at any point.
- ISIC/ISCED-F hierarchical retrieval was not implemented — see §10.

## 10. Explicit remaining next task

**Implement and evaluate ISIC Section→Division→Group→Class retrieval and
ISCED-F Broad→Narrow→Detailed retrieval on this integration branch**
(`reviewer2-b2-integration-20260807`), per the task specification.

**Additionally flagged** (found during this task, not part of the original
next-task instruction, but a genuine blocker worth surfacing before B2 work
resumes on this branch): `eval/configs/b1_frozen.json`'s implementation
fingerprint for `HierarchicalISCOStore._hierarchical_search` no longer
matches the live, refactored source on this integration branch (§3.2, §8).
Any future B2 K-sweep or B1 re-confirmation run on this branch will hit
`BaselineMismatchError` and refuse to proceed until B1 is deliberately
re-run and `b1_frozen.json` is re-frozen against the current, refactored
codebase — a real classifier benchmark run requiring its own explicit
approval, separate from and prior to any ISIC/ISCED-F retrieval work that
depends on B2's K-sweep machinery being usable again.
