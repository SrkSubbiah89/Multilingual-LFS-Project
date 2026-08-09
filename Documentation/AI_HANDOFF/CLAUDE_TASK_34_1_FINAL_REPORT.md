CLIENT_DEADLINE_PRECISION_READY: yes

# Task 34.1 Final Report — Precise Client Deadline Correction and Scope Cleanup

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_34_1_PRECISE_CLIENT_DEADLINE_CORRECTION.md`.
This task corrects a rounding-precision defect in Task 34's client-side
deadline pool and discloses a scope issue in Task 34's own report. It
does not erase, rewrite, or hide Task 34's report or its out-of-scope
metadata read.

## 1. Branch and base SHA

| | |
|---|---|
| Base branch | `reviewer2-qdrant-client-deadline-enforcement-20260809` |
| Required/verified base SHA | `73b167320110a9e1c41a9015814a982973a6c47a` (confirmed against both the local branch and `origin` before branching, and re-confirmed identical at close of this task) |
| New branch | `reviewer2-qdrant-client-deadline-precision-20260809` |
| Final commit SHA | reported in Claude's end-of-turn response, not inside this file |

Working tree was clean before branching. Only the three files listed
in Section 4 were changed.

## 2. Task 34 scope disclosure

Task 34's factual transport contribution remains valid: `query_points(
timeout=...)` is a Qdrant server-side operation-timeout hint, and a
`QdrantClient(timeout=...)` constructor value is the genuine
client-side timeout mechanism.

However, Task 34's own final report recorded a **read-only query of
the existing local Qdrant collection metadata** (`get_collection()`
point counts, used to re-verify historical-evidence preservation),
despite Task 34's own scope explicitly prohibiting any live Qdrant
connection. This did not mutate any collection or data, and Task 34's
report and evidence are preserved byte-identical (see Section 3) — it
is disclosed here, not hidden or rewritten. Task 34.1 itself performed
**no live Qdrant connection of any kind**: historical-evidence
preservation in this task was verified using file hashes/diffs only
(Section 3), never a collection listing, count, or health check.

## 3. Historical preservation — before and after

Task 24-34 evidence, the official ISCO catalogue, official-collection
builder/loader code, B1 frozen config, `full130` leakage guard/manifest,
and Task 34's own final report were all re-hashed after this task's
work and matched byte-identical, with zero mismatches (12/12 files):

| Path | Result |
|---|---|
| `eval/local_runs/wisco_official_tier1_stage_budget_rerun_20260809T104006Z/flat/*.csv` | unchanged |
| `eval/local_runs/wisco_official_tier1_stage_budget_rerun_20260809T104006Z/hierarchical_stdout.log` | unchanged |
| `eval/configs/b1_frozen.json` | unchanged |
| `eval/full130_access_guard.py` | unchanged |
| `eval/configs/full130_leakage_manifest.json` | unchanged |
| `eval/local_benchmarks/wisco_isco08_v2_group_split/dataset_hash.txt` | unchanged |
| `eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv` | unchanged |
| `eval/verified_catalogue_counts.yaml` | unchanged |
| `backend/rag/official_isco08_catalogue.py` | unchanged |
| `backend/rag/build_official_isco08_collections.py` | unchanged |
| `Documentation/AI_HANDOFF/CLAUDE_TASK_34_FINAL_REPORT.md` | unchanged |
| `Documentation/AI_HANDOFF/CLAUDE_TASK_33_FINAL_REPORT.md` | unchanged |

No historical data, prior report, catalogue, or collection-building
code was edited. No live Qdrant connection, benchmark, analysis,
embedding model, LLM/API call, or dataset operation occurred anywhere
in this task.

## 4. Why `ceil()` was not precise enough

Direct source reading of the installed `qdrant-client` 1.17.0
(`qdrant_client.qdrant_remote.QdrantRemote.__init__`) shows the
constructor's own `timeout` parameter is unconditionally rounded UP
before it ever reaches httpx:

```python
_timeout = (
    math.ceil(timeout) if timeout is not None else None
)  # it has been changed from float to int.
# convert it to the closest greater or equal int value (e.g. 0.5 -> 1)
```

Task 34's pool computed its cache key as `max(1, math.ceil(deadline_seconds))`
and passed that **already-rounded-up integer** as the constructor's
`timeout=`. For a remaining stage budget of e.g. 7.01 seconds, this
produced `ceil(7.01) == 8` — an 8-second client-side deadline that
**exceeds** the 7.01 seconds actually remaining. The bug compounded
with the pool's primary-client-reuse check, which compared the rounded
key to the primary deadline (`key == primary_deadline_seconds`): a
remaining budget of 7.9s also rounds to `ceil(7.9) == 8`, so it was
mistakenly treated as "the default deadline" and silently handed the
*existing* 8-second primary client — an even larger, undetected
overrun, since no new client was even constructed to reveal the
mismatch.

## 5. Exact float/deadline policy, minimum practical deadline, pool cache policy

**Selection policy (never round up):** `_QdrantClientDeadlinePool.get()`
now selects `key = math.floor(deadline_seconds)`, not `ceil()`. Because
`key` is already an integer, qdrant-client's own internal `math.ceil(key)`
is a no-op, so the resulting client timeout is always
`floor(deadline_seconds) <= deadline_seconds` — never rounded upward
past the caller's requested deadline.

**Primary-client reuse (exact equality only):** the reuse check now
compares the *raw requested deadline* directly to the store's
configured primary deadline (`deadline_seconds == primary_deadline_seconds`),
not a rounded cache key. A deadline that merely rounds to the same
integer as the primary (e.g. 7.9s vs. primary 8s) no longer reuses the
primary client; it correctly receives its own smaller, dedicated
client.

**Minimum practical deadline:** `hierarchy_engine.MIN_PRACTICAL_CLIENT_DEADLINE_SECONDS
= 1.0` — the smallest non-zero value `floor()` can produce, and thus
the smallest client-side deadline obtainable through the public
constructor without either rounding up past the remaining budget or
constructing an invalid zero timeout. `HierarchyBeamSearchEngine._query()`
checks the remaining stage budget against this constant, **before**
requesting a deadline-specific client, on every attempt (including
retries) whenever `client_for_deadline` is configured; if the remaining
budget is below the threshold, the attempt is refused outright and
`stage_budget_exhausted` is reported — the real transport is never
called. `_QdrantClientDeadlinePool.get()` also raises `ValueError`
below this threshold as a defensive backstop (callers are expected to
check first).

**Pool cache policy (unchanged):** bounded size (default `max_size=8`),
least-recently-used eviction, `check_compatibility=False` on every
constructed client (no extra uncontrolled network call), and clean
`.close()` of every evicted or explicitly-closed client. Only public
`QdrantClient` constructor/method surface is used anywhere — no
private attributes, no monkey-patching.

**Server-side hint, retry allowlist, attempt/backoff caps, and strict
guard semantics are all unchanged** — this task touches only the
client-side deadline-pool selection logic and its call site in
`_query()`.

## 6. Changed files

| File | Nature of change |
|---|---|
| `backend/rag/hierarchy_engine.py` | New `MIN_PRACTICAL_CLIENT_DEADLINE_SECONDS = 1.0` constant with full audited rationale; `_query()` refuses an attempt (reports `stage_budget_exhausted`) before requesting a deadline-specific client once the remaining stage budget drops below that threshold |
| `backend/rag/hierarchical_store.py` | `_QdrantClientDeadlinePool.get()`: `floor()` instead of `ceil()` for the cache key/constructed timeout; primary-client reuse now compares the raw requested deadline for exact equality, not a rounded key; raises `ValueError` below the minimum practical deadline |
| `backend/tests/test_qdrant_client_deadline_enforcement.py` | 6 new tests proving the precision/minimum-deadline requirements; 2 existing tests corrected (one encoded the old rounding bug directly in its assertion, one sat exactly at the new minimum-deadline boundary and needed retimed values to keep testing what it originally intended) |
| `Documentation/AI_HANDOFF/CLAUDE_TASK_34_1_FINAL_REPORT.md` (new) | This report |

## 7. Precision correction and transport proof

Direct hermetic proof, all against the real (non-mocked) local delayed
HTTP server:

- `test_direct_float_timeout_constructor_is_accepted_and_bounds_wait`:
  proves the public constructor accepts a positive float and that it
  genuinely bounds the wait (~2s, matching `ceil(1.4)==2`), directly
  demonstrating the internal rounding-up behavior that motivates this
  whole correction.
- `test_pool_selects_floor_not_ceil_for_fractional_deadline`: direct,
  numeric proof that `pool.get(7.01)` caches/constructs under key `7`
  (never `8`).
- `test_pool_does_not_reuse_primary_for_fractional_deadline_rounding_to_same_integer`:
  proves `pool.get(7.9)` (with primary deadline `8`) is **not** the
  primary client — the exact scenario Task 34's bug mishandled.
- `test_pool_get_raises_below_minimum_practical_deadline`: proves the
  pool's own defensive `ValueError` backstop.
- `test_fractional_stage_budget_bounds_wall_time_to_floored_seconds_not_server_delay`:
  a fractional ~2.3s remaining stage budget against an 8s server delay
  returns in bounded wall time (`<5.0s`), proving the floored ~2s
  client deadline — not the server delay — governed the wait.
- `test_remaining_budget_below_minimum_practical_deadline_refuses_without_calling_transport`:
  a ~0.3s remaining budget refuses outright; the server handler's
  `calls` list is asserted **empty** — the real transport was never
  invoked, not merely a correctly-labelled telemetry outcome.
- Two previously-passing tests were corrected: one asserted
  `pool.get(7.9) is primary_client` as its expected behavior — that
  was the bug itself, now inverted to prove the fix;
  `test_stage_budget_caps_client_side_deadline_below_configured_default`
  and `test_no_retry_when_stage_budget_exhausted_after_real_timeout`
  used nominal budgets (1.0s and 0.8s) that, after real per-test
  processing overhead, landed at or below the new
  `MIN_PRACTICAL_CLIENT_DEADLINE_SECONDS` threshold and would have
  exercised the new refusal path instead of what they were designed to
  demonstrate; both were retimed with comfortable margins (2.0s and
  1.6s respectively) so they still test their original intent under
  the corrected, stricter logic.

## 8. Exact test commands and outputs

Deadline-enforcement file alone (stability check, 3 consecutive runs):
```
python -m pytest backend/tests/test_qdrant_client_deadline_enforcement.py -q
```
Result: `18 passed` in each of 3 consecutive runs (~23s each).

Focused suite:
```
python -m pytest \
  backend/tests/test_qdrant_retry_resilience.py \
  backend/tests/test_stage_budget_enforcement.py \
  backend/tests/test_hierarchy_engine.py \
  backend/tests/test_hierarchical_store.py \
  eval/test_require_genuine_hierarchical.py \
  backend/tests/test_qdrant_client_deadline_enforcement.py \
  -q
```
Result: `149 passed in 23.82s` (143 pre-existing + 6 new; the standard
harmless colorama/atexit teardown noise on this environment follows
every run and is not a failure).

Full suite:
```
python -m pytest backend/tests eval/ -q
```
Result: `2174 passed, 1 deselected, 1 warning in 330.24s` (2168
pre-Task-34.1 + 6 new; zero regressions; same pre-existing unrelated
JWT `InsecureKeyLengthWarning`).

## 9. Remaining limitations

- No benchmark, analysis, or paper/reviewer document update was
  performed (explicitly out of scope).
- No live Qdrant connection occurred in this task.
- This proves genuine, precision-correct client-side deadline
  enforcement against a real local HTTP server; it does not prove
  anything new about live Qdrant server behavior, network conditions,
  or production latency/cost.
- No accuracy, real LFS, ISIC, ISCED, SRE, reranking, cost, or
  production-performance claim is supported by this code/test task.
- Achievable client-side timeout granularity remains whole seconds
  only (an inherent qdrant-client public-API constraint, not something
  this task can change); `floor()` guarantees the achieved value never
  exceeds the requested deadline, but can under-use up to just under 1
  second of an available budget by design (never over-use).
- Cannot forcibly cancel an in-flight query already sent to the
  server — same limitation as Task 31/34.

## 10. Confirmations

- **Live operations**: none performed anywhere in this task — no
  Qdrant connection, benchmark, model, LLM, or dataset operation.
- **Protected branches**: none touched; only
  `reviewer2-qdrant-client-deadline-precision-20260809` was created and
  pushed.
- **Working tree**: clean at task start and, after `git add` of
  exactly the four authorized files followed by commit, clean again
  post-push.
