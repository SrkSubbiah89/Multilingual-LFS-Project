# Final QA Baseline — Conference I Reviewer #2 Response

This document records the verified test-suite state and QA findings for the
uncommitted Reviewer #2 enhancement work on branch `master`, as of this QA
pass. It supersedes any earlier informal "zero regressions" / "all tests
passed" phrasing used in chat during implementation — the exact, precise
numbers below are the ones that matter.

## Exact test command and exact result

```
pytest backend/tests eval/ -q
```

```
1501 passed, 1 failed, 1 deselected, 1 warning in 225.24s (0:03:45)
```

This is the authoritative combined result for the full `backend/tests/`
and `eval/` suites together, re-verified during this QA pass. No prior
chat statement of a different pass count (e.g. intermediate figures like
"1277 passed" reported mid-implementation, before later sections added
more tests) should be read as the final state — this file is the final
state.

## Known failure — full detail

- **Test**: `backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity`
- **Command used to isolate it**:
  ```
  pytest "backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity" -q
  ```
- **Failure signature**:
  ```
  AssertionError: assert 'llm' in 'flat_semantic'
  WARNING backend.agents.isco_classifier: ISCOClassifier: LLM agent unavailable
  (clf.<locals>.<lambda>() got an unexpected keyword argument 'temperature').
  Falling back to semantic-only classification (no LLM re-ranking).
  ```
- **Evidence it reproduces on the untouched baseline**: verified in this QA
  pass by `git stash -u` (stashing every modified and untracked file from
  this work), confirming `git log -1 --oneline` showed the pre-existing
  baseline commit (`5e0ff5d gitignore Software/ and *.exe to prevent
  re-adding large local installers`), re-running the exact command above
  against that clean tree, observing the identical failure and identical
  traceback, then `git stash pop` to restore all working-tree changes
  (confirmed restored: `git status --short` shows the same 34
  modified/untracked entries before and after).
- **Why it is unrelated to this change**: the test's `clf` fixture
  (`backend/tests/test_isco_classifier_extended.py`, around line 25-26)
  patches `backend.agents.isco_classifier.get_llm` with
  `lambda t: MagicMock()` — a lambda accepting exactly one positional
  argument. `ISCOClassifier`'s LLM re-ranking construction path invokes
  this mock with a `temperature=` keyword argument, which the lambda's
  signature cannot accept, so `get_llm` raises a `TypeError` internally;
  `ISCOClassifier` catches this and falls back to semantic-only
  classification (`flat_semantic`), which is why `"llm" not in
  result.method` and the assertion fails. This is a mismatch between the
  test fixture's mock signature and `isco_classifier.py`'s LLM
  construction call in this environment/CrewAI version — it lives
  entirely inside `test_isco_classifier_extended.py`'s own fixture setup
  and `backend/agents/isco_classifier.py`'s LLM-construction call, neither
  of which this Reviewer #2 work modified. This work's only change to the
  hierarchical-retrieval path is `backend/rag/hierarchical_store.py` (an
  internal, behavior-preserving refactor onto
  `backend/rag/hierarchy_engine.py`) — `test_isco_classifier.py` and
  `test_isco_classifier_extended.py`'s other 110 tests, which exercise
  that refactored path, all pass unmodified.

## Scope of the Reviewer #2 code update

**Modified (9 files)**: `backend/agents/isced_classifier.py`,
`backend/agents/isic_classifier.py`, `backend/agents/semantic_relation.py`,
`backend/rag/hierarchical_store.py`, `backend/tests/test_isced_classifier.py`,
`backend/tests/test_isic_classifier.py`, `eval/run_eval.py`,
`eval/test_run_eval_b2.py`, `requirements.txt` (added `pyyaml`).

**New (24 files/directories)**: `backend/agents/classifier_methods.py`,
`backend/agents/method_registry.py`, `backend/rag/hierarchy_engine.py`,
`backend/tests/test_hierarchy_engine.py`,
`backend/tests/test_method_registry.py`,
`backend/tests/test_semantic_relation_evidence.py`,
`eval/ablation_runner.py`, `eval/analyze.py`, `eval/coverage_audit.py`,
`eval/dataset_card_schema.py`, `eval/manifest.py`, `eval/sre_eval_format.py`,
`eval/standards_reference.yaml`, `eval/validate_real_lfs_governance.py`,
8 `eval/test_*.py` files, `eval/figure_exports/` (5 scripts),
`eval/fixtures/` (1 synthetic fixture), `Documentation/Conference_I_Reviewer_2/`
(8 guides + `generated/`).

No API behaviour, authentication, survey flow, or frontend behaviour was
changed. `backend/rag/hierarchical_store.py`'s refactor onto
`backend/rag/hierarchy_engine.py` is verified behavior-preserving (same
211 existing ISCO classifier tests pass unmodified against it).

## Every new module has a matching test file

Verified in this QA pass by cross-referencing each new non-test module
against direct imports in a `test_*.py` file:

| Module | Test file(s) |
|---|---|
| `backend/agents/classifier_methods.py` | `test_isced_classifier.py`, `test_isic_classifier.py`, `test_method_registry.py` (constants imported and asserted directly in all three; no standalone `test_classifier_methods.py` — full public surface exercised via consumers) |
| `backend/agents/method_registry.py` | `test_method_registry.py` |
| `backend/rag/hierarchy_engine.py` | `test_hierarchy_engine.py` |
| `eval/ablation_runner.py` | `test_ablation_runner.py` |
| `eval/analyze.py` | `test_analyze.py` |
| `eval/coverage_audit.py` | `test_coverage_audit.py` |
| `eval/dataset_card_schema.py` | `test_validate_real_lfs_governance.py`, `test_manifest.py`, `test_ablation_runner.py` |
| `eval/manifest.py` | `test_manifest.py` |
| `eval/sre_eval_format.py` | `test_sre_eval_format.py` |
| `eval/validate_real_lfs_governance.py` | `test_validate_real_lfs_governance.py` |
| `eval/figure_exports/export_*.py` (5 scripts) | `test_figure_exports.py` (each imported and exercised individually) |

## Generated output audit — no fabricated values found

Verified in this QA pass by scanning every file under
`Documentation/Conference_I_Reviewer_2/generated/`:

- `grep -rl "real_lfs_validation"` across `generated/` → **no matches**. No
  generated artifact claims real-LFS-validation provenance.
- Every `official_expected_count` field across all 3 coverage-audit report
  sets (12 level-rows total: ISCO major/submajor/minor/unit, ISIC
  section/division/group/class, ISCED level/broad/narrow/detailed) is
  `null` — no official coverage total is asserted without a citation.
- `implemented_unique_count` values cited in `COVERAGE_AUDIT_GUIDE.md`
  (21 sections / 68 divisions / 118 groups / 134 classes for ISIC) were
  re-verified against the real generated JSON report in this QA pass and
  match exactly.
- `evaluation_results.json` and `latency_scalability.json` both report
  `"no_manifests_found": true, "results": []` — no run has been executed,
  and the exports say so rather than inventing rows.
- `evaluation_table_template.md` — every cell in every one of the 5
  ablation-config rows reads the literal string `"not yet run"`.
- No `%` character appears anywhere in `generated/`. The only `%`
  occurrences in the prose guides are (a) an explicit `X%` placeholder in
  `COVERAGE_AUDIT_GUIDE.md` instructing the author to fill it in from a
  future, properly-sourced official count, and (b) "Wilson 95% CIs" in
  `EVALUATION_PROTOCOL.md`, which names a statistical method, not a result.
- No benchmark/accuracy/latency number is asserted in prose outside of
  test fixtures and the verified coverage counts above.

## Remaining evidence gaps

- No ablation run has been executed (`eval/ablation_runner.py run ...`) —
  evaluation-results and latency/scalability exports are empty by design.
- No official ISCO/ISIC/ISCED-F code counts are sourced in
  `eval/standards_reference.yaml` (all `null`) — needed to compute a real
  coverage percentage for the abstract.
- No real, permissioned LFS dataset exists in this repository — Section F
  built only the intake/governance machinery (`dataset_card_schema.py`,
  `validate_real_lfs_governance.py`), never any respondent data.
- No human-labelled incoherence set exists for `eval/sre_eval_format.py`'s
  precision/recall/FPR/FNR computation.
- `backend/agents/semantic_relation.py`'s new LOW-severity boundary-gap
  thresholds are a deterministic engineering judgement call, not sourced
  from an external publication — flagged as provisional in
  `EVALUATION_PROTOCOL.md` and the implementation matrix, pending author
  review before being cited in the manuscript's SRE methodology.

## Git status

**No Git action has been performed as part of this QA review or the
preceding implementation work.** No `git add`, `git commit`, `git push`,
`git checkout`, `git reset`, or any other state-changing Git command was
run against the working tree's actual content — the only Git commands
executed were read-only inspection (`git status`, `git diff --stat`,
`git log`) and one temporary `git stash` / `git stash pop` pair used
solely to verify the pre-existing baseline failure above, which restored
the working tree to its exact prior state (confirmed by comparing
`git status --short` output before and after). All 9 modified and 24
new files/directories listed above remain uncommitted and staged for
the user's review.
