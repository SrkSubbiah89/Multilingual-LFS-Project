# Task 34 — Verified Client-Side Qdrant Deadline Enforcement

## Objective

Fix the remaining runtime boundary exposed by Tasks 30 and 33, then prove the
fix locally without running another WISCO benchmark.

Task 33 proved that the current strict stage-budget guard fails closed
correctly, but it also proved that an in-flight Qdrant request can consume the
shared 30,000 ms stage budget before the retry loop regains control. The
existing `query_points(timeout=...)` value is a Qdrant server-operation hint,
not proof of a client-side socket/read deadline. This task must establish and
test a genuine client-side wall-time bound through the installed
`qdrant-client` REST transport.

This is a repair and verification task only. It must not rerun WISCO, produce
accuracy results, or change paper claims.

## Required base and branch

1. Run `git fetch origin`.
2. Verify both local and `origin` resolve:

   ```text
   reviewer2-wisco-official-tier1-stage-budget-rerun-20260809
   1c053801ca49e8d48610e112355bc1d11cf20791
   ```

3. Confirm the starting working tree is clean.
4. Create and work only on:

   ```text
   reviewer2-qdrant-client-deadline-enforcement-20260809
   ```

5. Do not merge, rebase, reset, clean, stash, pull, force-push, open a PR, or
   modify any protected/prior-task branch.

## Historical evidence that must remain untouched

Before making edits, record hashes or direct verification for each item below.
Recheck them before the final commit.

- Task 24 raw artefacts.
- Task 25 diagnostic artefacts and telemetry.
- Task 26 report and its empty hierarchical-output directory.
- Task 29 preflight artefacts.
- Task 30 report, flat CSV, and empty hierarchical-output directory.
- Task 31 report and stage-budget implementation.
- Task 32 preflight artefacts.
- Task 33 report, flat CSV, hierarchical log, and empty hierarchical-output
  directory.
- Official ILO catalogue data and all five official Qdrant collections.
- B1 quarantine and B2 safety files.

No old raw CSV, log, manifest, dataset, catalogue, collection, or historical
report may be edited or regenerated.

## Non-negotiable scope boundaries

Do not:

- invoke `eval/run_eval.py`;
- run any WISCO export, preflight, smoke test, rerun, analysis, Wilson
  interval, McNemar test, or paper/document update;
- connect to or mutate live Qdrant;
- load a real embedding model;
- invoke Ollama, any LLM, or any paid API;
- build, delete, overwrite, or inspect a live Qdrant collection beyond
  source-code inspection;
- weaken, bypass, skip, xfail, or change the strict
  `--require-genuine-hierarchical` rule;
- alter the official catalogue profile, the ISCO hierarchy/search algorithm,
  labels, scoring, beam width, retry eligibility allowlist, retry hard caps,
  fallback semantics, or dataset.

All new runtime tests must be hermetic. A local in-process HTTP test server is
allowed only to exercise the installed Qdrant REST client against a delayed
response. It is not a live Qdrant service and must not use port 6333.

## Part A — Re-audit the actual transport boundary

Read, do not assume:

1. The installed `qdrant-client` version and the precise implementation of
   `QdrantClient`, its REST client, and the HTTP transport that accepts timeout
   configuration.
2. The installed `httpx` version and the public timeout API.
3. The current Task 31 code path from:

   ```text
   HierarchicalISCOStore
   → HierarchyBeamSearchEngine._query()
   → QdrantClient.query_points()
   ```

Write the exact findings into the final report, including:

- which existing timeout configuration is server-side only;
- which public client/transport API, if any, can enforce a client-side
  connect/read/write/pool timeout;
- whether a client constructed with one timeout can safely use a smaller
  per-request remaining-stage deadline;
- any conclusion that cannot be proven from the installed source.

Do not use undocumented private attributes or monkey-patch a production client
in order to claim a fix. If the installed public API cannot safely provide a
per-request deadline, use the smallest correct public-API design that you can
prove with a real delayed-response test. If no such safe design exists, stop
and report `CLIENT_DEADLINE_ENFORCEMENT_READY: no`; do not guess.

## Part B — Implement only a proven client-side deadline

Implement the smallest additive change needed to ensure every Qdrant query
issued by `HierarchyBeamSearchEngine._query()` has:

1. The existing Qdrant server-operation `timeout=` hint, preserving Task 31
   behaviour.
2. A genuine, public-API client-side deadline that bounds an in-flight REST
   request's connect/read/write/pool wait.
3. A deadline derived from the currently effective query timeout and, when a
   shared stage deadline exists, capped to the remaining stage time before the
   request begins.
4. The existing strict retry policy unchanged:
   - only direct or wrapped `httpx.TimeoutException` is retryable;
   - other exceptions remain fail-closed and are not retried;
   - maximum three attempts and maximum two-second backoff remain hard caps.
5. The existing default behaviour preserved for ordinary callers that do not
   select strict evaluation. Any new configuration must be explicitly named,
   narrowly scoped, validated, and fail safe for malformed ambient
   environment values.

The solution must not create an unbounded new client per query, leak clients
or sockets, silently increase retries, or turn a timeout into a fabricated
candidate. If a small client pool/cache is necessary, bound it, close it
cleanly, and document why it is safe. Prefer an already-supported public
per-request mechanism if the installed library provides one.

## Part C — Required hermetic proof

Add focused tests that use the actual installed `qdrant-client` REST path
against a local, intentionally delayed HTTP server. Do not replace the
transport with a `MagicMock` for this proof.

The proof must demonstrate all of the following:

1. A delayed endpoint that exceeds the configured client deadline raises a
   timeout-shaped exception through the same production query path within a
   bounded wall time. Use a generous CI tolerance, but the test must clearly
   distinguish a short configured deadline from the deliberately longer server
   delay.
2. The query attempt is classified by `_is_retryable_exception()` as
   retryable only when the wrapped cause is an `httpx.TimeoutException`.
3. With a shared stage budget, the attempt deadline is no longer than the
   remaining stage time. A delayed first attempt must return control before
   the stage can silently overrun indefinitely.
4. When a retryable timeout occurs while enough stage budget remains, the
   existing bounded retry is attempted exactly as before.
5. When no stage budget remains after a timeout, no retry starts and the
   existing `stage_budget_exhausted` outcome remains intact.
6. A successful immediate response preserves existing result parsing,
   candidate selection, method labelling, and telemetry.
7. Non-timeout responses such as HTTP validation/auth/server failures remain
   non-retryable and fail closed.
8. Existing Task 27, Task 31, and strict-guard tests remain meaningful and
   pass without weakening assertions.

Use a dynamically allocated local port, not a fixed port. Ensure every test
server and client is closed in `finally`/fixture cleanup so the full test suite
does not hang or leak a background thread.

## Part D — Required verification

Run at least:

```bash
python -m pytest \
  backend/tests/test_qdrant_retry_resilience.py \
  backend/tests/test_stage_budget_enforcement.py \
  backend/tests/test_hierarchy_engine.py \
  backend/tests/test_hierarchical_store.py \
  eval/test_require_genuine_hierarchical.py \
  -q
```

Then run:

```bash
python -m pytest backend/tests eval/ -q
```

The suite must be green. Do not hide a failure with a skip, xfail, relaxed
assertion, broad exception catch, or test-only production flag.

## Commit, push, and final report

Commit only the minimal source/test/doc changes required by this task plus:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_34_FINAL_REPORT.md
```

Push only:

```text
origin/reviewer2-qdrant-client-deadline-enforcement-20260809
```

The final report must contain:

1. Branch, base SHA, final commit SHA, push confirmation, and clean-tree
   confirmation.
2. Exact installed-library transport findings and the reason the new approach
   is a proven client-side deadline rather than another server-only hint.
3. A precise list of changed files and a concise explanation of each change.
4. Exact delayed-server test design, configured delay/deadline values,
   observed bounded timing, and why the proof is hermetic.
5. Exact focused and full test commands and outputs.
6. Direct evidence that Task 24–33 historical artefacts are byte-identical
   and protected branches unchanged.
7. A clear statement of what is still not proven:
   - the fix does not establish benchmark accuracy;
   - it does not prove Qdrant will never fail;
   - it cannot claim real Labour Force Survey validation;
   - it supports no ISIC, ISCED, SRE, reranking, cost, or production
     performance claim.
8. Exactly one final status:

   ```text
   CLIENT_DEADLINE_ENFORCEMENT_READY: yes
   ```

   only if the local delayed-response proof demonstrates a real client-side
   deadline and all tests pass. Otherwise write:

   ```text
   CLIENT_DEADLINE_ENFORCEMENT_READY: no
   ```

   and stop. Do not run a benchmark in either case.

## Required final response

Return only:

```text
Task 34 Final Report
Branch name and final commit SHA
Base branch and verified base SHA
Changed files
Transport finding and exact fix
Hermetic delayed-server proof
Exact test commands and outputs
Remaining limitations
Confirmation: live operations
Confirmation: protected branches
Confirmation: working tree
```
