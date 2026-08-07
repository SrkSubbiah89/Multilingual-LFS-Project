# Perplexity to Claude Code: Merge-Impact Analysis Task

**Task ID:** `02-merge-impact-analysis`  
**Working branch:** `reviewer2-wip-snapshot-20260807`  
**Protected snapshot baseline:** `b188202ea843b69dcdaf9fc6bb40dce97f709d72`

## Purpose

Determine how the protected Reviewer #2 snapshot can eventually coexist with the pre-existing `conference1-b2-evaluation` branch. This is an analysis task only. It is not permission to merge, rebase, resolve conflicts, change code, run benchmarks, or update dependencies.

## Verified starting facts

- `master` base: `5e0ff5d88c6c973f636b48cacc25e5885c11d41c`
- Protected Reviewer #2 snapshot baseline: `reviewer2-wip-snapshot-20260807` at `b188202ea843b69dcdaf9fc6bb40dce97f709d72`
- This task file is a later documentation-only commit on the same protected branch. The report must distinguish the protected snapshot baseline from the current branch tip after this task file is published.
- Existing B2 branch: `conference1-b2-evaluation`, three commits ahead of `master`
- Snapshot relative to `master`: two commits ahead, 114 changed files, with Reviewer #2 Steps 1 through 7A work
- Shared divergence is already known in at least:
  - `eval/dev_set_schema.md`
  - `eval/validate_dev_set.py`
  - `eval/dev_sweep.py`
  - `eval/test_run_eval_b2.py`
  - `requirements.txt`
- B2-only modules include `eval/pre_run_check.py` and `eval/full130_access_guard.py`.

## Safety rules

Do not run:

```bash
git merge
git rebase
git pull
git reset
git clean
git restore
git stash
git checkout <another-branch>
git switch <another-branch>
```

Do not edit application code, tests, dependencies, benchmark data, manifests, or configuration files. Do not run benchmark measurements. Do not delete, rename, or move files. Do not create an integration branch. Do not modify `master`, `conference1-b2-evaluation`, or `reviewer2-enhancement`.

`git fetch origin` and read-only Git commands are allowed. The only allowed new file is the report named below.

## Required analysis

Create exactly one report:

```text
Documentation/AI_HANDOFF/CLAUDE_MERGE_IMPACT_ANALYSIS.md
```

### Repository-state verification

Record the output of:

```bash
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git fetch origin
git rev-parse origin/master
git rev-parse origin/conference1-b2-evaluation
git rev-parse origin/reviewer2-wip-snapshot-20260807
git merge-base origin/conference1-b2-evaluation HEAD
```

Confirm that `b188202ea843b69dcdaf9fc6bb40dce97f709d72` remains an ancestor of the protected branch and identify the current branch tip after this task-file commit. Explicitly correct the earlier audit-report detail: its first `ls-remote` record reflected the first snapshot commit, while the actual snapshot baseline tip was the second audit-report commit (`b188202...`).

### Change inventory

1. Generate a file-level inventory of changes from `master` to:
   - `conference1-b2-evaluation`
   - `reviewer2-wip-snapshot-20260807`
2. Identify the exact intersection of paths changed by both branches.
3. Categorize every overlapping path:
   - text-only/nonfunctional
   - independently additive
   - potentially compatible implementation
   - likely semantic conflict
   - likely direct textual conflict
4. For B2-only files, explain whether the Reviewer #2 snapshot has an equivalent/superseding module or whether the file should likely be retained in a future integration.

### Semantic analysis of high-risk overlaps

For each high-risk overlap, compare both branch versions and document:

1. **`eval/dev_set_schema.md`**
   - Schema assumptions, development-set governance, and leakage controls.

2. **`eval/validate_dev_set.py`**
   - Validation rules, input/output contracts, error behavior, and overlap with Reviewer #2 governance validation.

3. **`eval/dev_sweep.py`**
   - Parameter-selection discipline, held-out protection, generated artifacts, and compatibility with current manifests/ablation configuration.

4. **`eval/test_run_eval_b2.py`**
   - Test assumptions that must remain true after a future integration.

5. **`requirements.txt`**
   - Dependency changes, version conflicts, security/reproducibility implications, and whether requirements-dev separation should be retained.

Also analyse the interaction between:

- B2 `eval/pre_run_check.py` and snapshot `eval/validate_real_lfs_governance.py`, `eval/validate_evaluation_discipline.py`, and `eval/validate_controlled_benchmark.py`.
- B2 `eval/full130_access_guard.py` and snapshot benchmark-governance, dataset-label, split-manifest, and leakage-validation machinery.

### Integration recommendation

Do not merge anything. Instead, provide:

1. A recommended future integration base branch.
2. A recommended order for resolving each file group.
3. Exact files that must be manually reconciled.
4. Exact files that can probably be retained unchanged from each branch.
5. A list of tests that must run after a future integration.
6. A rollback strategy.
7. A list of claims, benchmarks, and source files that must remain frozen during integration.

### Report format

The report must contain:

1. Executive summary.
2. Verified commit graph and branch state.
3. Change inventory table.
4. Overlap and conflict-risk table.
5. Detailed semantic-conflict analysis.
6. B2-to-Reviewer-#2 capability mapping.
7. Recommended future integration plan.
8. No-merge conclusion and evidence boundaries.

## Completion rules

- Run `git diff --check` after writing the report.
- Commit only the report:

  ```bash
  git add Documentation/AI_HANDOFF/CLAUDE_MERGE_IMPACT_ANALYSIS.md
  git commit -m "Add B2 and Reviewer 2 merge impact analysis"
  git push origin reviewer2-wip-snapshot-20260807
  ```

- Return the exact report commit SHA, the verified branch tips, the number of overlapping paths, the highest-risk files, and confirmation that no merge or source-code modification occurred.
