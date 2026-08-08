# Task 16 — Integrate Verified Strict WISCO Preflight

## Purpose

Create one audited integration branch that merges the verified Task 15 strict WISCO preflight report into the Task 14 strict benchmark baseline. This is a Git integration and verification task only.

Task 15 established only operational readiness for a bounded 524-case diagnostic replay. It did **not** produce benchmark accuracy, comparative latency, scalability, cost, real-LFS, ISIC, ISCED, SRE, or manuscript-performance evidence. Preserve that distinction exactly.

Do not start, authorize, imply, or prepare a full WISCO rerun in this task.

## Verified source facts

Verify these facts directly from Git before changing anything:

| Role | Branch | Required SHA |
|---|---|---|
| Strict baseline base | `reviewer2-wisco-strict-benchmark-baseline-20260808` | `3716057db699527761127383f430b786fa21ff46` |
| Task 15 feature source | `reviewer2-wisco-strict-preflight-20260808` | `29cfa2adcf290b4ed26fdb6dcd542794ecc1cde8` |

The Task 15 feature branch must be exactly one commit ahead of the required base and that feature commit must change only:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_15_FINAL_REPORT.md
```

If any source SHA, ancestry, changed-file scope, or clean-tree check fails, do not merge. Create a final report documenting the mismatch, push the new task branch if appropriate, and stop.

## Branch and Git safety

1. Fetch remote refs and verify both source SHAs against `origin`.
2. Begin with a clean working tree.
3. Create and work only on:

```text
reviewer2-wisco-strict-preflight-integration-20260808
```

4. Create an explicit merge commit:

```bash
git merge --no-ff reviewer2-wisco-strict-preflight-20260808
```

5. Do not squash or rebase.
6. Resolve no conflict by altering source behavior. If a conflict occurs, stop and report it without improvising.
7. After the merge, add only:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_16_FINAL_REPORT.md
```

8. Commit the Task 16 final report separately, then push only:

```text
origin/reviewer2-wisco-strict-preflight-integration-20260808
```

9. Do not create a pull request.

## Required preservation checks

Before and after the merge, demonstrate all the following:

1. **Task 15 scope**: base-to-merge diff contains only `CLAUDE_TASK_15_FINAL_REPORT.md`.
2. **Task 12 audit record**: raw ignored Task 12 output artifacts remain untracked and untouched; the Task 12 final report still declares `TIER1_COMPLETED: no`.
3. **Task 13 strict safeguard**: the following remain present and unmodified from the Task 14 base:
   - unseeded retry after an unsuccessful keyword-anchored hierarchical search;
   - explicit flat fallback only after retry failure;
   - `QDRANT_TIMEOUT_SECONDS` default/override behavior;
   - `--require-genuine-hierarchical`;
   - `--max-stage-latency-ms`;
   - no partial CSV written when the strict guard fails.
4. **Task 09 model-free WISCO path**: reranker-off mode does not initialize the LLM agent, and ISIC/ISCED/SRE construction remains gated off for ISCO-only input rows.
5. **Task 05/05.1/05.2 implementation**: the ISIC/ISCED-F hierarchy-node/store/collection-builder files remain present; do not represent that code presence as benchmark evidence.
6. **B1/B2 safety**: `eval/configs/b1_frozen.json` and `eval/dev_sweep.py` are byte-identical base-to-merge.
7. **No unrelated change**: no project source, tests, dependencies, configurations, WISCO package, Qdrant data, or ignored local-run artifact is changed by the merge.

Use read-only inspection and `git diff`/`git show` checks. Do not modify code, test, documentation outside the two allowed final-report paths, data, or configuration to make a check pass.

## Tests

Run only the existing automated test suites:

```bash
pytest backend/tests eval/ -q
```

If the suite does not finish with zero failures, report the exact failure(s) and whether they are attributable to the merge. Do not fix, skip, xfail, weaken, or change any test/code in this task.

## Prohibited operations

Do not run:

- any WISCO evaluation, smoke test, preflight, selected-case replay, full benchmark, flat run, rerank run, or `eval/analyze.py`;
- any Qdrant query, collection creation, mutation, population, rebuild, or deletion;
- SentenceTransformer/model download;
- Ollama, CrewAI, LLM, paid API, or external inference call;
- B1 re-freeze, B2 sweep, ISIC/ISCED evaluation, SRE evaluation, data rebuild/export;
- a benchmark-statistics or manuscript-evidence calculation.

Do not run `git reset`, `git clean`, `git stash`, `git pull`, `git rebase`, force-push, or change any protected/prior branch.

## Final report

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_16_FINAL_REPORT.md
```

The report must include:

1. verified source branch names and SHAs;
2. integration branch name;
3. merge commit SHA, its two parent SHAs, and final report commit SHA;
4. conflict status;
5. exact base-to-merge changed-file list;
6. every preservation check and its direct evidence;
7. exact test command and output;
8. confirmation that no live evaluation, Qdrant, model, LLM, dataset, or benchmark operation occurred;
9. confirmation that protected branches were unchanged and the working tree was clean;
10. an explicit evidence boundary:

```text
Task 15 is a bounded operational-readiness diagnostic only. It provides no benchmark accuracy, comparative latency, scalability, cost, real-LFS, ISIC, ISCED, SRE, or manuscript-performance evidence. It does not authorize a full WISCO rerun.
```

After pushing the final report, stop. Do not begin any full benchmark, rerun, or next task.
