# Task 34.1 — Precise Client Deadline Correction and Scope Cleanup

## Why this correction is required

Task 34 established an important fact and supplied a useful real-transport
test: a `QdrantClient(timeout=...)` constructor timeout is a genuine
client-side HTTP deadline, whereas `query_points(timeout=...)` is only a
server-operation hint.

However, independent review found two reasons not to accept Task 34's
`CLIENT_DEADLINE_ENFORCEMENT_READY: yes` yet:

1. Task 34's final report records a read-only query of the existing local
   Qdrant collection metadata, despite Task 34 explicitly prohibiting any
   live Qdrant connection. It did not mutate anything, but it was out of
   scope and must be disclosed accurately.
2. `_QdrantClientDeadlinePool.get()` currently uses
   `ceil(deadline_seconds)`. For a remaining stage budget of 7.01 seconds,
   this can create an 8-second client deadline. That exceeds the remaining
   stage budget and does not meet Task 34's own requirement that the attempt
   deadline be no longer than the remaining budget.

This task corrects the precision issue and proves it. It does not erase,
rewrite, or hide Task 34's report or the out-of-scope metadata read.

## Required base and branch

1. Run `git fetch origin`.
2. Verify both local and `origin` resolve:

   ```text
   reviewer2-qdrant-client-deadline-enforcement-20260809
   73b167320110a9e1c41a9015814a982973a6c47a
   ```

3. Confirm a clean working tree.
4. Create and work only on:

   ```text
   reviewer2-qdrant-client-deadline-precision-20260809
   ```

5. Do not merge, rebase, reset, clean, stash, pull, force-push, open a PR,
   or modify any protected/prior-task branch.

## Strict scope

Do not:

- run `eval/run_eval.py`, WISCO export/preflight/smoke/rerun, analysis, or
  any paper/reviewer document update;
- connect to, inspect, or mutate live Qdrant in any way, including collection
  listing, counts, health checks, or metadata reads;
- load a real embedding model, call Ollama/LLMs/APIs, or access real datasets;
- alter Task 24–34 raw evidence, reports, logs, catalogues, collections, B1
  quarantine, B2 safety controls, retrieval algorithm, scoring, beam width,
  retry eligibility, retry caps, or strict guard semantics;
- weaken tests with skip, xfail, broad exception swallowing, or relaxed
  timing assertions that no longer prove the requirement.

Only hermetic local dynamically-ported HTTP test servers are allowed. They
must not use port 6333 and must be fully shut down in fixture cleanup.

## Required implementation

Correct the deadline-pool design so that every client-side timeout selected
for a stage-budget-capped attempt is **never greater** than the remaining
stage time observed immediately before the request starts.

Requirements:

1. Confirm from the installed `qdrant-client` / `httpx` source and a real
   hermetic transport test that a positive float timeout is accepted by the
   public `QdrantClient(timeout=...)` constructor and bounds the caller's
   HTTP wait.
2. Replace the `ceil()`-based selection with a precision-preserving,
   conservative representation. A cache key may be normalised, but its
   corresponding constructed client timeout must never round upward beyond the
   requested deadline.
3. If less than a documented minimum practical deadline remains, do not start
   a request. Return the existing `stage_budget_exhausted` outcome rather than
   rounding the deadline upward or fabricating a result.
4. Preserve primary-client reuse only when the requested deadline is truly
   equal to the configured primary deadline, not merely rounded to the same
   integer.
5. Preserve bounded caching, LRU eviction, public APIs only, and clean client
   closure. Do not construct an unbounded number of clients.
6. Preserve the existing server-side `query_points(timeout=...)` hint, strict
   retry allowlist, maximum three attempts, maximum two-second backoff,
   fallback behavior, strict guard, and default caller behavior.

## Required hermetic proof

Keep Task 34's real local delayed-server coverage and add/strengthen tests
that prove all of the following:

1. With a fractional remaining stage budget, the timeout requested from the
   deadline-provider and the timeout configured on the resulting real client
   are less than or equal to the remaining budget, never rounded up.
2. A deliberately delayed response returns control within a generous but
   meaningful wall-time bound derived from the fractional budget, not the
   longer server delay.
3. When the remaining time is below the documented practical minimum, the
   real transport is not called at all and telemetry is
   `stage_budget_exhausted`.
4. The pool does not incorrectly reuse the primary client for a shorter
   fractional deadline.
5. Immediate successful response parsing, retryable timeout classification,
   allowed retry when sufficient budget remains, refused retry after budget
   exhaustion, and non-timeout failures all continue to behave as before.

The tests must exercise the actual installed Qdrant REST/httpx transport for
the timeout proof. A recording provider may additionally be used for exact
deadline arguments, but it cannot replace the real delayed-server proof.

## Evidence and report requirements

Before and after the code work, prove byte-identical preservation of Task
24–34 artefacts and critical safety files using hashes/diffs only. Do not
read live Qdrant metadata this time.

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_34_1_FINAL_REPORT.md
```

The report must state:

- Task 34's factual contribution remains valid, but its prior report's
  read-only local metadata check was outside Task 34's no-live-Qdrant scope.
  It caused no mutation and is not being hidden or rewritten.
- Why `ceil()` was not precise enough for a strict shared stage budget.
- The exact float/deadline policy, minimum practical deadline policy, pool
  cache policy, and direct hermetic proof.
- Exact files changed, focused/full test commands and outputs, final SHA,
  branch, working-tree status, and protected-branch status.
- Explicitly state no live Qdrant connection, benchmark, analysis, model,
  LLM, or dataset operation occurred in Task 34.1.
- State that no accuracy, real LFS, ISIC, ISCED, SRE, reranking, cost, or
  production-performance claim is supported by this code/test task.

Use exactly one final status:

```text
CLIENT_DEADLINE_PRECISION_READY: yes
```

only if all requirements and full tests pass. Otherwise use:

```text
CLIENT_DEADLINE_PRECISION_READY: no
```

and stop.

## Tests

Run focused tests covering the existing retry/stage-budget/deadline suites,
then:

```bash
python -m pytest backend/tests eval/ -q
```

The full suite must be green.

## Commit and push

Commit only the smallest required source/test/doc changes. Push only:

```text
origin/reviewer2-qdrant-client-deadline-precision-20260809
```

## Required final response

Return only:

```text
Task 34.1 Final Report
Branch name and final commit SHA
Base branch and verified base SHA
Changed files
Precision correction and transport proof
Task 34 scope disclosure
Exact test commands and outputs
Remaining limitations
Confirmation: live operations
Confirmation: protected branches
Confirmation: working tree
```
