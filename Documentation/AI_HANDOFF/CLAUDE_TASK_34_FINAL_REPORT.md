CLIENT_DEADLINE_ENFORCEMENT_READY: yes

# Task 34 Final Report — Client-Side Qdrant Deadline Enforcement

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_34_CLIENT_SIDE_QDRANT_DEADLINE_ENFORCEMENT.md`.
This task re-audits the transport layer one level deeper than Task 31,
proves that Task 31's per-request `timeout=` is a server-side hint
only, and implements + proves a genuine client-side deadline. No
benchmark was run, per the task's explicit instruction.

## 1. Branch and base SHA

| | |
|---|---|
| Base branch | `reviewer2-wisco-official-tier1-stage-budget-rerun-20260809` |
| Required/verified base SHA | `1c053801ca49e8d48610e112355bc1d11cf20791` (confirmed against `origin` before branching, and re-confirmed identical at close of this task) |
| New branch | `reviewer2-qdrant-client-deadline-enforcement-20260809` |
| Final commit SHA | reported in Claude's end-of-turn response, not inside this file |

Working tree was clean before branching. Only the four files listed in
Section 3 were changed; no historical `eval/local_runs/*` artefact, no
protected/prior branch, and no source file outside the three code
files below was touched.

## 2. Historical preservation — before and after

All 10 pre-work SHA-256 hashes (Task 24-33 evidence, official ISCO
catalogue, official-collection builder/loader code, B1 frozen config,
`full130` leakage guard/manifest) were re-hashed after this task's work
and matched byte-identical, with zero mismatches:

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

Live Qdrant collection point counts, re-checked after this task's
work, exactly matched the previously-verified baseline:

| Collection | Points |
|---|---|
| `isco08_major_groups_ilo2021_v1` | 10 |
| `isco08_submajor_groups_ilo2021_v1` | 43 |
| `isco08_minor_groups_ilo2021_v1` | 130 |
| `isco08_unit_groups_ilo2021_v1` | 436 |
| `isco08_unit_groups_flat_ilo2021_v1` | 436 |
| `isco08_major_groups` | 10 |
| `isco08_submajor_groups` | 43 |
| `isco08_minor_groups` | 131 |
| `isco08_unit_groups` | 441 |
| `isco_occupations` | 124 |

No historical data was edited, reclassified, or used for accuracy or
flat-versus-hierarchical analysis anywhere in this task.

## 3. Changed files

| File | Nature of change |
|---|---|
| `backend/rag/hierarchy_engine.py` | Additive `client_for_deadline` constructor param (default `None` = zero behaviour change); `_query()` selects a deadline-specific client per attempt only when both it and `effective_timeout_seconds` are set |
| `backend/rag/hierarchical_store.py` | New `_QdrantClientDeadlinePool` class (bounded LRU, reuses the store's existing primary client for the default deadline); wired into `HierarchicalISCOStore.__init__`; new optional `close()` method |
| `backend/tests/test_qdrant_client_deadline_enforcement.py` (new) | 12 tests, hermetic-but-real transport, no mocked transport |
| `Documentation/AI_HANDOFF/CLAUDE_TASK_34_FINAL_REPORT.md` (new) | This report |

## 4. Transport finding and exact fix

**Finding (confirmed by direct source inspection of the installed
`qdrant-client` 1.17.0 + `httpx` stack, not assumption):**

- Task 31's `query_points(timeout=N)` parameter is a Qdrant
  **server-side** operation-timeout hint, delivered as a
  `?timeout=N` REST query-string parameter
  (`qdrant_client/http/api/search_api.py::_build_for_query_points()`).
  This was already known from Task 31.
- The only genuine **client-side** timeout in effect is the
  constructor-level `httpx.Client(timeout=N)` default, set once when a
  `QdrantClient` is constructed. A bare int/float applies uniformly to
  connect/read/write/pool (`httpx.Timeout(N)`).
- qdrant-client's call chain
  (`QdrantRemote.query_points()` → `ApiClient.send_inner()` →
  `self._client.send(request)`) exposes **no public per-call override**
  of that client-side timeout anywhere in the installed public API
  surface. There is no documented way to make one already-constructed
  client's socket-level timeout track a shrinking stage budget.
- Constructing a **new** `QdrantClient` with a specific `timeout=`
  is therefore the only public-API-correct way to get a genuinely
  different client-side deadline. Doing so naively adds an extra,
  uncontrolled `httpx.get(...)` compatibility-check call per new client
  (`qdrant_client.common.version_check.get_server_version()`,
  triggered by the default `check_compatibility=True`); this is
  avoidable via the public `check_compatibility=False` constructor
  parameter, which was used throughout.

**Fix:** `_QdrantClientDeadlinePool` (`backend/rag/hierarchical_store.py`) —
a small, bounded (`max_size=8`, LRU-evicted) pool of `QdrantClient`
instances keyed by `ceil(deadline_seconds)`. Requesting the store's own
existing default deadline returns the **existing primary client**
(zero new construction, zero behaviour change for the overwhelming
majority of queries). Any other, necessarily smaller, deadline gets a
freshly constructed `QdrantClient(host=.., port=.., timeout=key,
check_compatibility=False)`, cached and LRU-evicted (with `.close()`)
once the pool exceeds its bound. `HierarchyBeamSearchEngine` gained an
optional `client_for_deadline` callable, invoked per query attempt only
when a stage budget is active; every existing caller that omits it is
unaffected. No private attributes, no monkey-patching of a production
client — only public constructor/method surface was used.

## 5. Hermetic delayed-server proof

`backend/tests/test_qdrant_client_deadline_enforcement.py` uses a real,
non-mocked `http.server.ThreadingHTTPServer` bound to `("127.0.0.1", 0)`
(dynamic port, confirmed `!= 6333` in the fixture) with a scripted
per-call delay/status-code handler, exercised through the real
`qdrant_client.QdrantClient` / `httpx` transport (never a `MagicMock`).
12 tests, all passing:

- Delayed response (3s server delay, 1s client timeout) raises in
  bounded wall time (`< 2.5s`), classified `retryable=True`,
  `exception_type="ResponseHandlingException"`, with
  `isinstance(exc.source, httpx.TimeoutException)` proven directly.
- A stage budget (~1s) caps the effective client-side deadline below
  the engine's own generous default (10s) against a 5s server delay
  (`elapsed_s < 3.0`).
- Retry succeeds when stage budget remains (2 scripted calls, 2nd
  fast); retry is correctly refused (`stage_budget_exhausted`, `hits==[]`,
  exactly 1 call made) when the budget is the binding constraint,
  independently reproducing Task 31's fail-closed behaviour on real
  transport.
- Non-retryable HTTP errors (400/401) are never retried, proven
  directly against the real raised exception type.
- 4 additional `_QdrantClientDeadlinePool`-specific tests: primary-client
  reuse for the default deadline, caching, bounded LRU eviction, and
  idempotent `close_all()`.

## 6. Exact test commands and outputs

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
Result: `143 passed in 16.13s` (plus the standard, already-established
harmless colorama/atexit teardown noise on this environment).

Full suite:
```
python -m pytest backend/tests eval/ -q
```
Result: `2168 passed, 1 deselected, 1 warning in 320.44s` (same harmless
atexit teardown noise; the pre-existing `InsecureKeyLengthWarning` from
an unrelated JWT test is expected and unrelated to this task). Zero
regressions against every pre-existing test file, including all 9 files
Task 31 previously had to widen — none needed further changes this
time, since Task 34's new parameter is purely additive/opt-in.

## 7. Remaining limitations

- No benchmark was run (explicitly out of scope for this task).
- This proves genuine client-side deadline enforcement against a real
  local HTTP server; it does not prove anything new about live Qdrant
  server behaviour, network conditions, or production latency/cost.
- Does not constitute validation against real LFS respondent data.
- Does not extend to ISIC/ISCED/semantic-relation/reranking, which are
  unaffected by this change.
- Cannot forcibly cancel an in-flight query already sent to the
  server (same limitation as Task 31's stage budget) — a pooled
  client's shorter timeout bounds how long the *caller* waits, not
  server-side execution.
- The `_QdrantClientDeadlinePool` bound (`max_size=8`) is a deliberately
  conservative default; no claim is made about optimal sizing under
  real concurrent load.

## 8. Confirmations

- **Live operations**: none performed; all evidence in this task came
  from a local, dynamically-ported HTTP test server and the existing
  local Qdrant instance's read-only collection metadata.
- **Protected branches**: none touched; only
  `reviewer2-qdrant-client-deadline-enforcement-20260809` was created
  and pushed.
- **Working tree**: clean at task start and, after `git add` of exactly
  the four authorized files followed by commit, clean again post-push.
