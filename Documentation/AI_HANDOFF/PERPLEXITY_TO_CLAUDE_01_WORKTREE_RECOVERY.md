# Perplexity to Claude Code: Worktree Recovery and Branch Alignment Task

**Task ID:** `01-worktree-recovery`  
**Read first:** This task supersedes all benchmark, merge, and feature work until the current local changes are safely committed and pushed.

## Verified remote state

- `master` is at commit `5e0ff5d`.
- `conference1-b2-evaluation` is three commits ahead of `master` and contains the existing B2 evaluation-hardening work.
- `reviewer2-enhancement` currently contains only the Perplexity-to-Claude audit task.
- The current local working tree reportedly contains approximately 53 changed or new files from Conference I Reviewer #2 Steps 1 through 7A.

## Critical safety rules

Do **not** run any of the following until the safety snapshot has been committed and pushed:

```bash
git reset --hard
git clean
git checkout <another-branch>
git switch <another-branch>
git restore
git stash
git pull
git merge
git rebase
```

Do not delete files, rewrite history, amend commits, run benchmark measurements, alter code, update dependencies, or resolve branch conflicts in this task.

## Objective

Create a GitHub-backed safety snapshot of the complete current Reviewer #2 working tree. The snapshot must preserve all changed and untracked work exactly as it exists today.

## Required procedure

### Record the pre-snapshot state

From the current working directory, run and save the output of:

```bash
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git diff --stat
git diff --name-status
git ls-files --others --exclude-standard
```

Do not discard or edit any file based on this output.

### Create and push a dedicated snapshot branch

Create a new local branch from the current dirty working tree:

```bash
git switch -c reviewer2-wip-snapshot-20260807
```

Then preserve every tracked modification and untracked file:

```bash
git add -A
git commit -m "WIP snapshot: Conference I Reviewer 2 work through Step 7A"
git push -u origin reviewer2-wip-snapshot-20260807
```

Do not merge this branch into `master`, `conference1-b2-evaluation`, or `reviewer2-enhancement`.

### Verify the snapshot

After push, run:

```bash
git status --short
git log -1 --oneline
git show --stat --oneline HEAD
git ls-remote --heads origin reviewer2-wip-snapshot-20260807
```

The working tree should be clean after the commit. If the commit fails, stop and report the exact error. Do not use reset, clean, stash, or branch switching as a workaround.

### Create the required audit handoff report

If `AI_HANDOFF_PROJECT_STATE.md` already exists at the repository root, preserve it. Also create:

```text
Documentation/AI_HANDOFF/CLAUDE_PROJECT_AUDIT.md
```

The report must state:

1. The snapshot branch name and exact commit SHA.
2. The original base branch and base commit.
3. The number of files captured in the snapshot.
4. The existence and purpose of `conference1-b2-evaluation`.
5. That no merge, rebase, conflict resolution, or benchmark measurement occurred.
6. That the next task must be a merge-impact analysis, not a merge.

Commit and push this report in a second commit on the same snapshot branch:

```bash
git add Documentation/AI_HANDOFF/CLAUDE_PROJECT_AUDIT.md
git commit -m "Add Reviewer 2 worktree recovery audit"
git push
```

## Required final response

Return only:

- Current branch before the snapshot.
- Snapshot branch name.
- First snapshot commit SHA.
- Second audit-report commit SHA.
- Confirmed remote branch URL or `git ls-remote` output.
- Number of tracked modified files and untracked files captured.
- Whether the working tree is clean after the snapshot.
- Confirmation that no reset, clean, stash, pull, merge, rebase, benchmark run, or source-code modification occurred.

## Stop condition

After the snapshot and report are pushed, stop. Do not inspect or merge `conference1-b2-evaluation` further in this task.
