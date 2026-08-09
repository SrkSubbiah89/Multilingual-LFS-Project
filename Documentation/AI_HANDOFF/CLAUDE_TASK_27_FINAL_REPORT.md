# Task 27 Final Report — Bounded Qdrant Timeout Resilience and Hierarchical Query Telemetry

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_27_BOUNDED_QDRANT_TIMEOUT_RESILIENCE.md`.
This is a code/test implementation task only — no benchmark, no WISCO
rerun, no collection build, and no results analysis was performed.

## 1. Branch, base SHA, final SHA, push confirmation

| | |
|---|---|
| Base branch | `reviewer2-wisco-official-tier1-telemetry-rerun-20260809` |
| Required SHA | `6d0453a310d44c91dd911f746bfc25b5db54dffd` |
| Verified `origin` SHA | `6d0453a310d44c91dd911f746bfc25b5db54dffd` — match |
| New branch | `reviewer2-qdrant-timeout-resilience-20260809` |
| Final commit SHA | recorded after this report's commit (see push confirmation below) |

Working tree was clean before branching and clean immediately before
this report's own commit (only the files listed in §2 were staged).

## 2. Exact changed files and diff scope

```text
backend/rag/hierarchy_engine.py                              (modified, +298/-50 lines)
backend/rag/hierarchical_store.py                             (modified, +96 lines)
eval/run_eval.py                                               (modified, +46 lines)
backend/tests/test_qdrant_retry_resilience.py                  (new, 46 tests)
eval/test_qdrant_retry_resilience_serialization.py             (new, 9 tests)
Documentation/AI_HANDOFF/CLAUDE_TASK_27_FINAL_REPORT.md         (this file)
```

No official/legacy Qdrant collection, WISCO package file, catalogue
file, or historical `eval/local_runs/` artefact was touched.
`backend/agents/isco_classifier.py` needed no change (it already
threads the same `trace` dict unmodified). No B1/B2 config, ISIC,
ISCED, SRE, or LLM code was touched.

## 3. Audited Qdrant call path and installed exception taxonomy

**Call path** (confirmed by direct source reading, this session):
`eval/run_eval.py::run_one_case()` → `ISCOClassifier.classify()` →
`_classify_hierarchical()` → either `HierarchicalISCOStore.search()`
(hierarchical) or `.search_flat_only()` (flat) → both ultimately call
`HierarchicalISCOStore._query()`, a thin wrapper that delegates to
`HierarchyBeamSearchEngine._query()` — the **single** place either path
ever calls `QdrantClient.query_points()`. The hierarchical beam loop
(`HierarchyBeamSearchEngine.search()`) wraps every such call in an inner
`_timed_query()` closure (one call per explored beam branch per stage);
`_flat_search()` calls `_query()` exactly once, directly, with no retry
loop of its own (confirmed: this is why Task 24's flat failure produced
exactly one unavailable row, with no possibility of an internal retry
masking it).

**Where the client timeout is applied**: `HierarchicalISCOStore.__init__`
constructs `QdrantClient(host=_host, port=_port, timeout=_timeout)`,
`_timeout` resolved from `QDRANT_TIMEOUT_SECONDS` (default 30s, Task
13). No `prefer_grpc` is passed anywhere in this codebase (confirmed by
grep across `hierarchical_store.py`, `build_official_isco08_collections.py`,
`vector_store.py`) — every `QdrantClient` in this project uses the
default REST/httpx transport, never gRPC.

**Installed exception taxonomy** (qdrant-client 1.17.0, httpx 0.28.1,
both confirmed via `pip show`/`httpx.__mro__` inspection):

- `qdrant_client.http.exceptions.ApiException` — base class.
  - `ResponseHandlingException(source: Exception)` — read directly from
    `qdrant_client/http/api_client.py`: `ApiClient.send_inner()` wraps
    **any** exception the underlying httpx client raises during
    `self._client.send(request)` in `ResponseHandlingException(e)`;
    separately, `ApiClient.send()` also wraps a `pydantic.ValidationError`
    (a successfully-received-but-schema-unexpected response body) the
    same way. This means `ResponseHandlingException` is **not** purely a
    network/timeout signal — its `.source` must be inspected, not just
    its outer type.
  - `UnexpectedResponse` — raised for any non-2xx HTTP status code
    (`send()`'s final `raise UnexpectedResponse.for_response(response)`)
    — a real API-level error (malformed request, auth, server error),
    never a timeout.
  - `ResourceExhaustedResponse` — raised specifically for HTTP 429 with
    a `Retry-After` header — a deliberate rate-limit signal, semantically
    different from a timeout; explicitly out of this task's narrow scope
    (see §4).
- `httpx.TimeoutException` and its subclasses `ConnectTimeout`,
  `ReadTimeout`, `WriteTimeout`, `PoolTimeout` — the underlying
  transport's own timeout family, confirmed via `httpx.TimeoutException.__mro__`.
  These are what a real `.source` on a timeout-caused
  `ResponseHandlingException` will be.

**No-hit vs. exception, before this task**: both a genuine successful
zero-point Qdrant response and any caught exception produced an
identical empty list `[]` from `_query()`, with (pre-Task-25) zero
telemetry distinguishing them, and (pre-Task-27) no attempt/retry
concept existed at all — one attempt, always.

**Where Task 25 telemetry attaches**: `HierarchicalISCOStore._flat_search()`
passes a `query_telemetry` dict into `_query()` and copies its contents
into `trace["flat_query_*"]` **before** its `if not hits: return
self._empty_result()` early return, so telemetry survives on every
outcome. The hierarchical beam loop's `_timed_query()` (pre-Task-27) did
**not** attach any `query_telemetry` at all — it only measured
aggregate stage latency; a stage-level exception was logged
(`_logger.warning`) but never reached `trace` or the CSV, i.e.
hierarchical stage search **lost** exception information entirely
before this task (fixed in §5).

**What is observed vs. unproven** (per the task's explicit instruction):
observed — Task 24 and Task 26 each logged exactly one warning reading
`"...failed: timed out"`, from `HierarchyBeamSearchEngine._query()`'s
own pre-existing warning format string, on two different collections
(`isco08_unit_groups_flat_ilo2021_v1` and
`isco08_major_groups_ilo2021_v1` respectively), each during an
otherwise-clean sustained ~18,700-case run. **Unproven and not claimed
here**: the server-side or network root cause of either incident (load,
GC pause, network contention, or anything else) — this task does not
diagnose, measure, or speculate on that beyond what Task 25 already
stated for the first incident.

## 4. Retry behavior, configuration, and safety justification

**Default remains one attempt, zero retry** — `DEFAULT_MAX_QUERY_ATTEMPTS
= 1`, `DEFAULT_RETRY_BACKOFF_SECONDS = 0.0`, both set as
`HierarchyBeamSearchEngine.__init__`'s default parameter values.
Confirmed by a dedicated test
(`test_default_single_attempt_client_called_exactly_once_on_failure`)
that a retryable-shaped exception under default config still results in
exactly one `query_points()` call and the plain `"exception"` outcome —
byte-for-byte identical to Task 25's original behaviour, never
`"retry_exhausted"`, even though the exception type would be retryable
if retry were enabled.

**Opt-in configuration** — two independent env vars, resolved by
`hierarchical_store._resolve_max_query_attempts()` /
`_resolve_retry_backoff_seconds()`, mirroring
`_resolve_qdrant_timeout_seconds()`'s exact existing precedent (fail
**safe** to the default on missing/malformed/out-of-range — never
raises, never crashes an ambient misconfiguration):

| Variable | Default | Hard cap |
|---|---:|---:|
| `QDRANT_QUERY_MAX_ATTEMPTS` | 1 | 3 |
| `QDRANT_QUERY_RETRY_BACKOFF_SECONDS` | 0.0 | 2.0 |

A future benchmark harness enables retry purely through these
environment variables — zero source-code change (confirmed by
`test_store_env_var_opts_in_to_retry_without_code_change`).
`HierarchicalISCOStore(max_query_attempts=..., retry_backoff_seconds=...)`
constructor arguments are also accepted for explicit/programmatic use;
an invalid **explicit** argument raises `ValueError` immediately at
construction (fail **closed** on explicit misuse — mirroring
`UnknownISCOCatalogueProfileError`'s existing precedent in the same
file), deliberately different from the environment-variable path's
fail-safe behaviour, and documented as such in both functions'
docstrings.

**Upper bound justification** (`MAX_QUERY_ATTEMPTS_HARD_CAP = 3`,
`MAX_RETRY_BACKOFF_SECONDS_HARD_CAP = 2.0`, both enforced in
`HierarchyBeamSearchEngine.__init__` via an explicit range check that
raises `ValueError`, never merely documented): 1 initial attempt plus at
most 2 retries. A transient client-side timeout is either resolved
within a couple of attempts or reflects a real, non-transient outage
that a bounded retry cannot paper over; over an ~18,700-case run, an
unbounded or large per-case multiplier would risk materially inflating
total run time if timeouts turn out to be correlated (e.g.
sustained server-side load) rather than independent — the two
historical incidents give no basis to assume independence. The 2-second
backoff cap bounds the same worst case: at most `(3-1) × 2.0 = 4.0`
extra seconds added to any single case's own latency, never to the
run's overall structure.

**Strict eligibility allowlist** (`_is_retryable_exception()`,
type-based only, never message-substring matching):

- `httpx.TimeoutException` (direct) — retryable.
- `qdrant_client.http.exceptions.ResponseHandlingException` whose
  `.source` is an `httpx.TimeoutException` — retryable (the wrapper
  cause is inspected explicitly, per the task's requirement).
- Everything else — **not** retryable, including: a bare
  `ResponseHandlingException` wrapping a non-timeout cause (e.g. a
  `pydantic.ValidationError` or `ConnectionResetError`), `UnexpectedResponse`
  (any non-2xx status — malformed request/auth/server error),
  `ResourceExhaustedResponse` (429 rate-limiting — a different signal,
  deliberately out of this narrow task's scope), any other `ApiException`
  subclass, generic `Exception`, and Python's builtin `TimeoutError`
  (confirmed explicitly NOT allowlisted — it is not what the installed
  qdrant-client/httpx stack actually raises for a real timeout; test
  `test_is_retryable_exception_allowlist_directly` asserts this
  directly).

**Unknown and non-retryable exceptions still fail closed** — explicit
statement: any exception not matching the allowlist above is retried
zero additional times regardless of `max_query_attempts`, produces the
existing `"exception"` outcome and empty-result/fallback behaviour
unchanged, and is never silently converted into a fabricated code or a
no-hit result. Confirmed by
`test_non_retryable_exception_not_retried_even_with_attempts_available`
(parametrized over four distinct non-retryable exception shapes, each
constructed so that a retry — if wrongly attempted — would have
succeeded, proving the rejection is genuine and not a coincidence of
the fixture).

**Bounded execution / no mutation**: each retry is only a repeat of the
exact same `client.query_points(...)` call with identical arguments —
no collection create/upsert/delete call exists anywhere in the retry
path (confirmed by inspection: `_query()`'s only Qdrant call is
`query_points`). No unbounded sleep, no exponential backoff, no
automatic rerun of a whole case — retry is scoped to exactly one
low-level query call.

## 5. Telemetry schema and privacy/sanitization limits

**`query_telemetry` dict** (internal, passed into
`HierarchyBeamSearchEngine._query()`): `outcome` (one of `"success"`,
`"success_after_retry"`, `"exception"`, `"retry_exhausted"`), `attempts`
(int), `duration_ms` (float, **total** across every attempt and any
backoff — for the default single-attempt case this is byte-identical to
Task 25's original single-query duration), `attempt_durations_ms`
(list[float], one entry per attempt), and — only when the final outcome
is `"exception"`/`"retry_exhausted"` — `exception_type` (last attempt's
class name), `exception_message` (sanitized), `retryable` (bool).

**Flat-path CSV fields** (`eval/run_eval.py::CaseResult`) — Task 25's
four original fields are **unmodified in name and default**; only the
**set of values** `flat_query_outcome` may take is widened (to include
`"success_after_retry"`/`"retry_exhausted"`), and only when retry is
explicitly enabled — under the default (no opt-in), only `"success"`/
`"exception"` ever appear, exactly as before. Two new additive columns:
`flat_query_attempts` (`Optional[int] = None`) and
`flat_query_attempt_durations_ms` (`str = "[]"`, JSON list).

**Hierarchical-stage telemetry** — deliberately **separate** from the
flat fields (never overloaded, never read by the strict guard as stage
evidence): one new additive column,
`hier_stage_query_telemetry: str = "{}"` — a JSON object keyed
`"stage1"`.."stage4"`, present only for stages that genuinely issued a
live Qdrant query (absent for a keyword-anchor-seeded or leaf_vote-
overridden stage 1, which never calls `_query()`). Each stage's value:
`{"queries": int, "any_retry": bool, "any_exception": bool,
"max_attempts_used": int, "exception_types": [str, ...]}` — aggregated
(never overwritten) across every beam-branch query issued at that
stage, so a single genuinely-retried or genuinely-failed query anywhere
in a stage's branches is never hidden by an average or a last-write-
wins overwrite (confirmed by
`test_stage_query_telemetry_aggregates_across_branches_without_overwriting`).

**Privacy/sanitization** (`_sanitize_exception_message()`, unchanged
from Task 25, reused as-is for retry-path messages): collapses all
whitespace to single spaces, truncates to 300 characters plus an
explicit `"...(truncated)"` suffix. Derived **only** from the last
attempt's `str(exc)` — never `traceback.format_exc()`, the query
vector, the raw input text, or any request/response payload. Confirmed
by `test_retry_exhausted_exception_message_never_contains_query_vector_or_text`
using a deliberately secret-shaped marker string that never appears in
the recorded message.

## 6. Strict-guard and latency semantics after retry

- `--require-genuine-hierarchical` (`check_strict_hierarchical()`) is
  **unmodified** by this task — it still only inspects `pred_method` and
  `stage{1..4}_candidates`/`_latency_ms` by exact field name. This is a
  deliberate design choice, not an oversight: when a stage's retries are
  all exhausted, the existing (untouched) fallback logic in
  `HierarchicalISCOStore` fires exactly as it did before Task 27 (an
  empty `hits` list still propagates to `search()` returning `None`,
  which still triggers the existing flat-fallback path), so the guard's
  existing `pred_method`-prefix check rejects it automatically, with no
  new special-casing required. Confirmed by
  `test_strict_guard_rejects_flat_fallback_caused_by_retry_exhaustion`.
- The new `flat_query_*`/`hier_stage_query_telemetry` fields are never
  read by `check_strict_hierarchical()` — confirmed directly by
  `test_strict_guard_ignores_hier_stage_query_telemetry_field_entirely`
  (a genuine, complete hierarchical case with retry telemetry attached
  still passes purely on its existing stage-evidence/method checks).
- `--max-stage-latency-ms` continues to measure the **full** stage time
  including any retry attempts and backoff, with **zero code change**
  required to achieve this: `_timed_query()`'s `elapsed_ms` wraps the
  entire `self._query(...)` call, and the retry loop (including any
  `time.sleep(backoff)`) lives entirely *inside* that single call —
  so retry/backoff time is included in `stageN_latency_ms` by
  construction, never reset or hidden. Confirmed by
  `test_stage_latency_includes_retry_and_backoff_time` and
  `test_strict_guard_rejects_excessive_total_stage_latency_including_retry_time`.

## 7. No broad behaviour change — confirmed

Scoring, candidate ranking, hierarchy traversal (`search()`'s beam-loop
structure is unmodified beyond the added telemetry hook), the official
catalogue, WISCO data, output row schema (only additive columns),
default `QDRANT_TIMEOUT_SECONDS` (untouched), B1/B2 gate files
(untouched — no `_config_hash()`/`dev_sweep.py` change was needed since
no new CLI flag was added), ISIC, ISCED, SRE, and LLM/reranker behaviour
are all unchanged. No official or legacy Qdrant collection was created,
deleted, or modified by this task — this was a source/test-only change;
no live Qdrant instance was contacted at any point.

## 8. Focused and full test results

```
python -m pytest backend/tests/test_qdrant_retry_resilience.py eval/test_qdrant_retry_resilience_serialization.py backend/tests/test_flat_query_telemetry.py eval/test_flat_query_telemetry_serialization.py backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchical_store.py backend/tests/test_standard_hierarchical_store.py backend/tests/test_isco_classifier_official_profile.py backend/tests/test_official_isco08_catalogue.py backend/tests/test_official_isco08_profiles.py backend/tests/test_build_official_isco08_collections.py eval/test_require_genuine_hierarchical.py eval/test_official_isco08_profile_evaluator.py eval/test_model_free_isco_evaluation.py eval/test_run_eval_b2.py eval/test_docs_consistency.py eval/test_wisco_leakage_audit.py eval/test_analyze_wisco_tier1.py -q
→ 321 passed in 4.81s

python -m pytest backend/tests eval/ -q
→ 2129 passed, 1 deselected, 1 warning in 271.56s (0:04:31)
```

**Zero failures.** `2129 = 2074` (Task 26 baseline) `+ 55` net new
tests (46 in `backend/tests/test_qdrant_retry_resilience.py` + 9 in
`eval/test_qdrant_retry_resilience_serialization.py`) — an exact match
confirming zero regressions. No existing test was modified, skipped, or
weakened.

The 11 required coverage items map to these tests: (1)
`test_engine_default_max_query_attempts_is_one_no_retry`,
`test_default_single_attempt_client_called_exactly_once_on_failure`,
`test_default_single_attempt_success_unaffected`; (2)
`test_retryable_timeout_then_success_reports_success_after_retry`,
`test_bare_httpx_timeout_exception_directly_is_retryable`; (3)
`test_retryable_timeout_exhausts_attempts_returns_empty_with_telemetry`;
(4) `test_non_retryable_exception_not_retried_even_with_attempts_available`
(×4 exception shapes), `test_is_retryable_exception_allowlist_directly`;
(5) `test_genuine_zero_hit_response_not_retried`; (6)
`test_engine_rejects_invalid_max_query_attempts` (×7 bad values),
`test_engine_rejects_invalid_retry_backoff_seconds` (×4 bad values),
`test_engine_accepts_hard_cap_boundary_values`,
`test_resolve_max_query_attempts_env_var_fails_safe` (×7),
`test_resolve_retry_backoff_seconds_env_var_fails_safe` (×6),
`test_store_rejects_invalid_explicit_max_query_attempts`; (7)
`test_retry_exhausted_exception_message_never_contains_query_vector_or_text`;
(8) `test_flat_default_no_retry_attempts_field_is_one`,
`test_flat_success_after_retry_serializes_new_outcome_value`,
`test_flat_retry_exhausted_serializes_correctly_and_no_fabricated_code`,
`test_hierarchical_row_leaves_new_flat_fields_blank`; (9)
`test_stage_query_telemetry_aggregates_across_branches_without_overwriting`,
`test_hier_stage_query_telemetry_serializes_and_is_distinct_from_flat`;
(10) `test_strict_guard_rejects_flat_fallback_caused_by_retry_exhaustion`,
`test_strict_guard_ignores_hier_stage_query_telemetry_field_entirely`,
`test_strict_guard_rejects_excessive_total_stage_latency_including_retry_time`,
`test_strict_guard_rejects_missing_stage_evidence_unchanged`; (11) the
full focused set above plus the full suite, both green.

## 9. Confirmation: zero live operation

No live Qdrant connection, embedding-model load, LLM/CrewAI/Ollama/paid
API call, WISCO file read/write, or evaluation run against real data
occurred anywhere in this task. Every test in
`backend/tests/test_qdrant_retry_resilience.py` and
`eval/test_qdrant_retry_resilience_serialization.py` uses a scripted
fake Qdrant client (`ScriptedQdrantClient`/`EmptyHitQdrantClient`), a
`FakeEmbedder`, and/or a `MagicMock`-based fake `ISCOClassifier` —
confirmed by direct authorship and by the test run producing no
network-related warnings/errors.

## 10. Historical artefact and protected-branch confirmation

Re-verified after implementation and testing completed: all four Task
24 raw-artefact SHA-256 checksums (flat CSV, `flat_stdout.log`,
`flat_integrity_gate_report.json`, `failing_row_detail.json`) matched
their Task 25/26-recorded values exactly. Task 25's diagnostic output
root and Task 26's report (`OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED:
no`) remain present and unmodified. No `eval/local_runs/` path was
opened for writing by this task.

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| `reviewer2-wip-snapshot-20260807` | not touched |
| `reviewer2-b2-integration-20260807` | not touched |
| every prior `reviewer2-*` task/integration branch (through `reviewer2-wisco-official-tier1-telemetry-rerun-20260809`) | not touched |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push occurred. No PR was created.

## 11. Limitations

This task produces no benchmark evidence and no paper claim. It
implements and hermetically tests a narrow, opt-in resilience mechanism
only; it does not run, rerun, or validate any WISCO evaluation, and it
does not establish whether the two historical timeouts will or will not
recur, or whether enabling retry would have changed Task 24's or Task
26's outcome — that would require a new, separately authorized
benchmark run using this mechanism, which this task does not perform.
WISCO remains a controlled multilingual ISCO-08 benchmark, not real
Labour Force Survey validation, ISIC/ISCED/SRE evidence, or a
real-world performance claim.

Stopping here, per the task's own instruction, after pushing this
report. No smoke test, preflight, rerun, analysis, or manuscript work
was started.
