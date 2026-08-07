# Perplexity to Claude Code: B2 Integration Task

**Task ID:** `03-b2-integration`  
**Read this task from:** `origin/reviewer2-enhancement`  
**Required integration branch:** `reviewer2-b2-integration-20260807`  
**Protected source branch:** `reviewer2-wip-snapshot-20260807`  
**Protected snapshot starting commit:** `64e6ec4d1d1dac940d85242b791748ca1468050d`  
**B2 source branch:** `origin/conference1-b2-evaluation`  
**Verified B2 source commit:** `675121bbf656dad3e611f3280d95606b5b123e92`

## Purpose

Create a clean, testable integration branch that combines the protected
Reviewer #2 snapshot with the previously separate B2 reproducibility work.
This task is strictly integration and regression validation. It does not
implement new research features, run benchmark measurements, revise the
conference paper, or alter past evaluation evidence.

The completed merge-impact analysis is:

```text
Documentation/AI_HANDOFF/CLAUDE_MERGE_IMPACT_ANALYSIS.md
```

Read it before changing anything. Its verified finding is that only these
paths overlap at Git level:

```text
eval/test_run_eval_b2.py
requirements.txt
```

The first has one actual merge conflict. The second auto-merges cleanly.

## Hard safety rules

Do not modify or push to:

```text
master
conference1-b2-evaluation
reviewer2-wip-snapshot-20260807
reviewer2-enhancement
```

Do not run:

```text
git reset
git clean
git stash
git rebase
git push --force
git commit --amend
```

Do not run any classifier benchmark, WISCO measurement, Step 7B evaluation,
Ollama inference, paid API call, GPU workload, or paper-compilation task.
Do not regenerate, overwrite, delete, or reinterpret historical measured
outputs, frozen B1/B2 results, WISCO v1/v2 benchmark files, manifests, or
dataset hashes.

Do not implement ISIC or ISCED-F hierarchical retrieval in this task. That
is a separate post-integration task.

## Start-state checks

From the repository root, first run:

```bash
git fetch origin
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git rev-parse origin/reviewer2-wip-snapshot-20260807
git rev-parse origin/conference1-b2-evaluation
git show origin/reviewer2-enhancement:Documentation/AI_HANDOFF/PERPLEXITY_TO_CLAUDE_03_B2_INTEGRATION.md
```

Stop and report without changing branches if:

- the working tree is not clean;
- the checked-out branch is not `reviewer2-wip-snapshot-20260807`;
- `HEAD` does not equal `64e6ec4d1d1dac940d85242b791748ca1468050d`;
- either source branch tip differs from the SHA recorded above.

## Allowed Git operations

After all start-state checks pass, only these Git state changes are allowed:

```bash
git switch -c reviewer2-b2-integration-20260807
git merge --no-ff origin/conference1-b2-evaluation
git add <only intended integration files>
git commit -m "<accurate integration commit message>"
git push -u origin reviewer2-b2-integration-20260807
```

If a conflict appears outside the one expected test file, stop and report it
instead of improvising a resolution.

## Required integration work

### Resolve `eval/test_run_eval_b2.py`

The merge-impact analysis verified that both branches appended independent
122-line test blocks after the same existing anchor. Preserve:

1. the B2 exact classifier-input regression coverage; and
2. the Reviewer #2 SRE-enabled / SRE-disabled classification coverage.

Do not weaken or delete either assertion. Keep the shared pre-existing base
unchanged. Resolve this as a manual coexistence/concatenation of the two
independent test blocks, then run the focused test module.

### Retain complementary B2 capabilities

Retain B2-only additions unless a test proves a genuine incompatibility:

- `eval/pre_run_check.py`
- `eval/full130_access_guard.py`
- `eval/build_full130_leakage_manifest.py`
- `eval/configs/b1_frozen.json`
- `eval/configs/full130_leakage_manifest.json`
- `eval/PRE_RUN_B2_CHECKLIST.md`
- `eval/dev_set_v1.csv`
- `eval/dev_set_v1_provenance.md`
- `eval/dev_set_v1_data_dictionary.md`
- `eval/dev_set_v1_readiness_report.md`
- `requirements-dev.txt`
- their associated B2 tests

Retain the Reviewer #2 snapshot's general governance, manifest, dry-run,
WISCO, coverage, provenance, and documentation machinery.

### Requirements

Confirm that both additions are retained:

- `pyyaml` for Reviewer #2 catalogue/coverage tooling;
- `scikit-learn==1.6.1` for B2 work.

Retain `requirements-dev.txt` as a development/test dependency separation.
Do not alter package versions beyond the merge result unless required to fix
an installation or test failure, and if such a change is required, stop and
report it for approval.

### Dev-set schema, validator, and sweep

The following B2 files are not Git conflicts but must be treated as one
design unit:

```text
eval/dev_set_schema.md
eval/validate_dev_set.py
eval/dev_sweep.py
```

Use B2's versions as the integration branch implementation because they add
fail-closed catalogue validation, hash-only leakage checks for `full130`,
normalization-fingerprint checks, and the pre-run reproducibility gate.

Before finalising:

1. confirm no Reviewer #2 snapshot module calls `validate_dev_set()` with
   the old positional form;
2. confirm B2's validator and sweep tests pass;
3. add only a concise documentation note, if necessary, to the integration
   report explaining that B2's seven-column dev-set schema is canonical for
   B2 K-selection, while Reviewer #2 `DatasetCard` / `SplitManifest`
   schemas remain authoritative for controlled benchmarks and real-LFS
   governance. Do not force them into one schema in this task.

Do not change B2's frozen `full130` policy. The `full130` dataset remains
engineering/development evidence only, not manuscript-eligible evidence
because its provenance is unknown.

## Frozen evidence boundaries

Do not modify:

- `eval/configs/b1_frozen.json`, including the historical 54/130 (41.5%)
  result;
- the documented conclusion that `eval/test_set_full130.csv` is
  `not_eligible_unknown_provenance`;
- WISCO v1/v2 records, split manifests, hashes, or leakage-audit results;
- existing dry-run and synthetic-run output directories;
- any manuscript result, dataset card, or generated evaluation evidence.

The task must not add any new accuracy, latency, cost, memory, coverage, or
real-LFS claim.

## Required tests

Run these after the integration:

```bash
pytest eval/test_run_eval_b2.py -q
pytest eval/test_validate_dev_set.py eval/test_dev_sweep.py \
       eval/test_pre_run_check.py eval/test_full130_access_guard.py -q
pytest eval/test_validate_real_lfs_governance.py \
       eval/test_validate_evaluation_discipline.py \
       eval/test_validate_controlled_benchmark.py \
       eval/test_wisco_leakage_audit.py -q
pytest eval/test_docs_consistency.py -q
pytest backend/tests eval/ -q
```

The protected snapshot baseline before B2 integration was:

```text
1656 passed, 1 known pre-existing failure, 1 deselected, 1 warning
```

The known baseline failure is:

```text
test_isco_classifier_extended.py::test_llm_used_for_low_similarity
```

Do not classify it as a new regression. Any other failure is a blocker:
investigate and fix only if the fix is confined to the integration itself;
otherwise stop and report it.

## Required integration report

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_B2_INTEGRATION_REPORT.md
```

It must include:

1. exact source branch tips and new integration-branch tip;
2. exact merge commit SHA;
3. every conflict encountered and its resolution;
4. confirmation that both `test_run_eval_b2.py` test blocks remain;
5. dependency outcome;
6. B2 dev-set schema/validator/sweep decision;
7. test commands and exact results;
8. whether any new failures were found;
9. confirmation that no benchmark, inference, historical evidence, paper,
   protected source branch, or new classifier feature was changed;
10. explicit remaining next task: implement and evaluate ISIC
    Section→Division→Group→Class retrieval and ISCED-F
    Broad→Narrow→Detailed retrieval on this integration branch.

## Completion

Commit the resolved integration, the merge, and the integration report only
to `reviewer2-b2-integration-20260807`, then push that new branch.

Return:

- the exact integration branch name;
- the merge and final report commit SHAs;
- all test results;
- the final `git status --short`;
- confirmation that the three protected branches remain unchanged.
