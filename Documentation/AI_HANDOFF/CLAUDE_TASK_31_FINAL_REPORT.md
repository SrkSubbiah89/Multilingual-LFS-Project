STAGE_BUDGET_ENFORCEMENT_READY: yes

# Task 31 Final Report — Enforce Per-Request Transport Timeouts and Stage-Level Deadline Budgets

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_31_STAGE_BUDGET_ENFORCEMENT.md`.
This is a code/test implementation task only — no live Qdrant
operation, no WISCO preflight or rerun, no collection build, no
analysis, and no manuscript work was performed.

## 1. Branch and push confirmation

| | |
|---|---|
| Base branch | `reviewer2-wisco-official-tier1-retry-rerun-20260809` |
| Required SHA | `d3d46ac95992ad7699c4a18eabbb342139698056` |
| Verified `origin` SHA | `d3d46ac95992ad7699c4a18eabbb342139698056` — match |
| New branch | `reviewer2-qdrant-stage-budget-enforcement-20260809` |

Working tree was clean before branching and clean immediately before
this report's own commit (only the files listed in §4 were staged).
**The final pushed branch tip SHA is reported in Claude's end-of-turn
response, not inside this file.**

## 2. Audited transport/client configuration path and package API facts

Traced directly from the installed packages (qdrant-client 1.17.0,
httpx 0.28.1) and this repository's source, this session:

**Client construction path**: `HierarchicalISCOStore.__init__` resolves
`_timeout` (from `timeout_seconds=` or `QDRANT_TIMEOUT_SECONDS`, default
30s) and constructs `QdrantClient(host=_host, port=_port,
timeout=_timeout)`. `QdrantClient.__init__` forwards `timeout` to
`QdrantRemote.__init__`, which computes `_timeout = math.ceil(timeout)`
and sets `self._rest_args["timeout"] = _timeout`, then constructs
`SyncApis(host=self.rest_uri, **self._rest_args)` → `ApiClient(**kwargs)`
→ `self._client = httpx.Client(**kwargs)`. **This is a client-WIDE
default** (httpx applies a bare int/float uniformly to its connect,
read, write, and pool sub-timeouts — confirmed directly:
`httpx.Timeout(8)` produces `connect=8, read=8, write=8, pool=8`).

**Per-call `query_points(..., timeout=...)` parameter**: qdrant-client's
own `QdrantRemote.query_points()` (REST branch) forwards a `timeout`
argument into `self.http.search_api.query_points(..., timeout=timeout,
...)`, whose generated `_build_for_query_points()` implementation
(`qdrant_client/http/api/search_api.py`) does exactly:
```python
if timeout is not None:
    query_params["timeout"] = str(timeout)
```
**This is a server-side operation-timeout hint, delivered as an HTTP
query-string parameter (`?timeout=N`) on the request URL — not a
client-side httpx socket/connect/read timeout.** Confirmed by direct
source reading, not inference.

**Direct answer to the five audit questions**: the configured
`QDRANT_TIMEOUT_SECONDS` value is (a) a client-wide default (applied to
every request through that `httpx.Client` unless overridden per-call)
via the constructor path; it is (b) **not** independently used as a
client-side per-request timeout by this codebase prior to this task
(the per-call `timeout=` kwarg was never passed at all — `_query()`
called `query_points()` with no `timeout=` argument); it **is** (c)
passed through to an underlying transport, but as a server-side hint,
not a client transport override; it was (d) **not overridden** by any
call site prior to this task — simply omitted; and this combination
(client-wide default only, no per-call value, no server-side hint) is
(e) insufficient to bound observed synchronous wall time — directly
demonstrated by two independent, unexplained observations from Task 30:
one retried flat row's first attempt took 29,540.302 ms (vs. the
configured 8,000 ms), and one hierarchical case's accumulated stage-4
latency reached 1,155,613.62 ms.

**Stage accumulation**: `HierarchyBeamSearchEngine.search()`'s stage
loop issues one `_timed_query()` call per explored beam branch at each
stage (`for br in frontier: ... hits = _timed_query(...)`), and
`stage{i}_latency_ms` accumulates (`trace[lk] = trace.get(lk, 0.0) +
elapsed_ms`) across every one of those branch calls — never a single
query's time. With the default `beam=2` and 4 ISCO stages, up to
`2^3 = 8` distinct branch queries can each independently contribute to
the FINAL stage's (`stage4`) accumulated total. Task 30's own per-query
theoretical bound (`3 attempts × 8s + 2 × 0.5s backoff = 25s`) applies
to **one** such query; it was never a bound on the stage's accumulated
total across up to 8 of them, and — per the observation above — even a
single query's actual duration was not reliably bounded by the
per-attempt timeout either. **Both facts together (unbounded
accumulation across branches AND an unexplained single-query duration
anomaly) are what made Task 30's 1,155,613.62 ms figure possible; this
report does not determine which contributed how much, and does not
claim a server/network root cause for either.**

## 3. Design: what is provably fixed vs. what remains an open question

**Reliable per-request timeout propagation** (implemented,
unconditional): every `query_points()` call now explicitly passes
`timeout=<int>`, derived from `HierarchyBeamSearchEngine.query_timeout_seconds`
(propagated from `HierarchicalISCOStore`'s own already-resolved
`_timeout`) and, when a stage budget is active, capped by the remaining
budget. This closes the previously-missing server-side deadline signal
(condition (b)/(d) above) and is directly, hermetically provable via a
test-observable call argument (§7). **It does not, and this report does
not claim, that it fixes or explains the observed single-query-duration
anomaly** (a server-side `?timeout=N` hint instructs Qdrant when to give
up server-side; it says nothing about why an httpx client-side read
might itself take longer than its own configured socket timeout — that
remains unproven and is not addressed by this parameter).

**Strict stage-level deadline budget** (implemented, opt-in): a
monotonic wall-clock deadline (`time.monotonic()`-based), established
once per stage, shared by every branch query/retry/backoff at that
stage, checked before every one of them. This is the primary, provable
fix for Task 30's actual failure mode — it bounds the *stage's total*
regardless of how many branch queries or retries occur, independent of
whatever caused any single query to run long. **Explicit limitation,
stated per the task's own requirement**: the budget check happens
*before* starting a new query or sleep — it cannot forcibly abort a
query already in flight. No thread/future/process-level cancellation
was implemented or claimed; doing so without proof would be exactly the
"unproven pseudo-timeout" the task forbids. In the worst case, a single
stage with only one branch remaining could still see that one query's
own actual duration exceed the budget before the *next* check ever
runs — the budget prevents *compounding* (many branches/retries each
independently adding unbounded time) and prevents *wasted* work
(no new query or sleep starts once budget is gone); it does not
guarantee an absolute ceiling against one already-started call.

## 4. Exact changed files

```text
backend/rag/hierarchy_engine.py                       (modified)
backend/rag/hierarchical_store.py                       (modified)
backend/agents/isco_classifier.py                        (modified)
eval/run_eval.py                                          (modified)
backend/tests/test_stage_budget_enforcement.py            (new, 22 tests)
eval/test_stage_budget_enforcement_serialization.py       (new, 5 tests)
backend/tests/test_flat_query_telemetry.py                (modified -- fixture fix)
backend/tests/test_hierarchical_store.py                  (modified -- fixture fix)
backend/tests/test_official_isco08_profiles.py            (modified -- fixture fix)
backend/tests/test_qdrant_retry_resilience.py             (modified -- fixture fix)
eval/test_flat_query_telemetry_serialization.py           (modified -- fake-classify signature)
eval/test_official_isco08_profile_evaluator.py            (modified -- fake-classify signature)
eval/test_qdrant_retry_resilience_serialization.py        (modified -- fake-classify signature)
eval/test_require_genuine_hierarchical.py                 (modified -- fake-classify signature)
eval/test_run_eval_b2.py                                  (modified -- fake-classify signature)
eval/test_sre_isic_isced_coupling_fix.py                  (modified -- fake-classify signature)
Documentation/AI_HANDOFF/CLAUDE_TASK_31_FINAL_REPORT.md    (this file)
```

**Why the fixture/fake-signature files needed changing**: making the
per-request timeout unconditional (§3) means `HierarchicalISCOStore`
now always calls `query_points(..., timeout=...)` by default (not only
under a strict budget) — four hermetic `FakeQdrantClient.query_points()`
fixtures that previously omitted a `timeout` parameter needed it added
(accepted, and in one file recorded for assertions) to keep working,
exactly the same class of fixture evolution as Task 21's
`_config_hash()` field addition and Task 22's AST-scoping fix. Six test
files' fake `ISCOClassifier.classify()` stand-ins needed
`max_stage_latency_ms=None` added to their signature for the same
reason — `run_one_case()` now always passes that kwarg. No test's
actual assertions, fixtures' data, or behaviour under test was
weakened; only call/fixture signatures were widened to match the new
(additive, default-preserving) parameters.

## 5. New parameters and semantics

| Location | New parameter | Default | Effect |
|---|---|---|---|
| `HierarchyBeamSearchEngine.__init__` | `query_timeout_seconds: Optional[float]` | `None` | Explicit per-request timeout value; `None` omits the `timeout=` kwarg entirely (byte-identical prior behaviour) |
| `HierarchyBeamSearchEngine.search()` | `max_stage_latency_ms: Optional[float]` | `None` | Opt-in stage deadline budget; `None` never establishes a deadline (byte-identical prior behaviour, hermetically proven by a test that makes `time.monotonic()` itself raise if called) |
| `HierarchyBeamSearchEngine._query()` | `stage_deadline_monotonic: Optional[float]` | `None` | Internal: the absolute deadline for the current stage, passed by `search()` |
| `HierarchicalISCOStore.search()` / `_hierarchical_search()` | `max_stage_latency_ms: Optional[float]` | `None` | Passed straight through to the engine, for both the keyword-anchor attempt and its retry (Task 13) |
| `ISCOClassifier.classify()` / `_classify_hierarchical()` | `max_stage_latency_ms: Optional[float]` | `None` | Passed straight through; has no effect on the flat retrieval path |
| `eval/run_eval.py::run_one_case()` | `max_stage_latency_ms: Optional[float]` | `None` | Passed straight through to `clf.classify()` |
| `eval/run_eval.py::main()` | (internal) `effective_stage_budget_ms` | — | `= args.max_stage_latency_ms` **only** when `args.system == "hierarchical"` **and** `args.require_genuine_hierarchical` are both true; `None` otherwise — the exact same relationship `--max-stage-latency-ms` already had to `--require-genuine-hierarchical` for the (unmodified) post-hoc `check_strict_hierarchical()` check |
| `HierarchicalISCOStore.__init__` | (internal) engine construction | — | Always passes `query_timeout_seconds=_timeout` (the store's own already-resolved Qdrant client timeout) — this half of the feature is unconditional, not gated on strict mode |

**Retry allowlist preservation**: `_is_retryable_exception()` is
completely unmodified by this task — still exactly `httpx.TimeoutException`
directly, or `ResponseHandlingException` whose `.source` is an
`httpx.TimeoutException`; no message-substring matching; every other
exception type (including a `ResponseHandlingException` wrapping
something else, `UnexpectedResponse`, `ResourceExhaustedResponse`,
generic `Exception`) remains non-retryable. Confirmed unchanged by
re-running every Task 27 allowlist test (§7) and by direct diff review.

**Stage-budget exhaustion semantics**: on exhaustion (before a query
starts, or before a retry/backoff proceeds), `_query()` returns `[]`
and sets `query_telemetry["outcome"] = "stage_budget_exhausted"` —
never a fabricated candidate. This propagates through `search()`'s
existing zero-hits handling (`if not hits0: return None` /
`if not hits: continue`) into the **same, unmodified** flat-fallback
path already used for a genuine zero-hit response, an exception, or
retry exhaustion. **No change was made to `check_strict_hierarchical()`**
— its existing `pred_method`-prefix and `stageN_candidates`-non-empty
checks already reject any state that is not genuine, complete
hierarchical evidence, regardless of which of these four causes
produced it; this was verified, not merely assumed (§7, item 8).
`--max-stage-latency-ms`'s continued full-latency semantics: `stageN_latency_ms`
still wraps the entire `_query()` call including every attempt and any
backoff, by construction — unchanged from Task 13/27, now also
correctly including any time spent before a budget-exhaustion
short-circuit.

## 6. Telemetry schema and privacy limits

**`stage{i}_query_telemetry`** (internal `trace` key; JSON-serialized
into the existing `hier_stage_query_telemetry` CSV column, unchanged
column name from Task 27/28) additive fields:

```text
{
  "queries": int,                                  # unchanged from Task 27
  "any_retry": bool,                                # unchanged
  "any_exception": bool,                            # unchanged
  "max_attempts_used": int,                         # unchanged
  "exception_types": [str, ...],                    # unchanged
  "stage_budget_exhausted": bool,                   # NEW -- true if ANY query at this stage hit budget exhaustion
  "configured_query_timeout_seconds": float | None, # NEW -- constant per stage
  "initial_stage_budget_ms": float | None,          # NEW -- the max_stage_latency_ms value active for this stage; None when no budget was requested
  "queries_detail": [                                # NEW -- one entry per branch query issued at this stage
    {"outcome": str, "attempts": int, "attempt_durations_ms": [float, ...],
     "remaining_stage_budget_ms_at_entry": float | None},
    ...
  ]
}
```

**Privacy/sanitization**: every new field is a count, a duration, a
constant-per-stage configuration value, or an `outcome` enum string
(one of `"success"`, `"success_after_retry"`, `"exception"`,
`"retry_exhausted"`, `"stage_budget_exhausted"`) — never raw query
text, a query vector, credentials, or a stack trace.
`exception_type`/`exception_message` (unchanged from Task 25/27) remain
bounded/sanitized and are populated only when a final error genuinely
remains. `queries_detail` is bounded in practice by the number of
explored beam branches at that stage (at most 8 for ISCO's 4-stage,
`beam=2` configuration) — not unbounded growth.

**Flat Task 25 telemetry**: `flat_query_outcome`,
`flat_query_duration_ms`, `flat_query_exception_type`,
`flat_query_exception_message`, `flat_query_attempts` (Task 27),
`flat_query_attempt_durations_ms` (Task 27) are **completely
untouched** by this task — no new column, no new value, no renamed
field. `_flat_search()`'s call into `_query()` gains the same
unconditional `timeout=` propagation as every other call (§3), but this
changes only the wire-level request, not any flat CSV field's name,
default, or existing value semantics. Confirmed by every pre-existing
flat-telemetry test remaining green unmodified (§7).

## 7. Focused and full test results

```
python -m pytest backend/tests/test_stage_budget_enforcement.py eval/test_stage_budget_enforcement_serialization.py backend/tests/test_qdrant_retry_resilience.py eval/test_qdrant_retry_resilience_serialization.py backend/tests/test_flat_query_telemetry.py eval/test_flat_query_telemetry_serialization.py backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchical_store.py backend/tests/test_standard_hierarchical_store.py backend/tests/test_isco_classifier_official_profile.py backend/tests/test_official_isco08_catalogue.py backend/tests/test_official_isco08_profiles.py backend/tests/test_build_official_isco08_collections.py eval/test_require_genuine_hierarchical.py eval/test_official_isco08_profile_evaluator.py eval/test_model_free_isco_evaluation.py eval/test_run_eval_b2.py eval/test_docs_consistency.py eval/test_wisco_leakage_audit.py eval/test_analyze_wisco_tier1.py eval/test_sre_isic_isced_coupling_fix.py backend/tests/test_isco_classifier.py backend/tests/test_isco_classifier_extended.py -q
→ 486 passed in 5.17s

python -m pytest backend/tests eval/ -q
→ 2156 passed, 1 deselected, 1 warning in 316.50s (0:05:16)
```

**Zero failures.** `2156 = 2129` (Task 30 baseline) `+ 27` net new
tests (22 in `backend/tests/test_stage_budget_enforcement.py` + 5 in
`eval/test_stage_budget_enforcement_serialization.py`) — an exact match
confirming zero regressions. No existing test's assertions were
weakened; only fixture/fake-signature widening was needed (§4), all
verified passing with their original assertions intact.

The 14 required coverage items map to: (1)
`test_configured_query_timeout_passed_to_query_points`,
`test_no_query_timeout_configured_omits_timeout_kwarg`,
`test_store_propagates_its_own_resolved_timeout_to_the_engine`; (2)
`test_effective_timeout_capped_by_remaining_stage_budget`,
`test_effective_timeout_uses_configured_value_when_budget_is_looser`;
(3) `test_search_without_max_stage_latency_ms_never_establishes_a_deadline`
(makes `time.monotonic()` itself raise if ever called),
`test_default_engine_construction_accepts_omitted_query_timeout`,
`test_plain_hierarchical_run_without_strict_flag_passes_no_budget`,
`test_flat_run_never_receives_a_stage_budget`; (4)
`test_second_branch_query_skipped_once_stage_budget_exhausted`; (5)
`test_backoff_capped_by_remaining_stage_budget`; (6)
`test_retry_not_started_when_stage_budget_exhausted_after_failure`; (7)
`test_stage_budget_exhausted_at_entry_returns_no_fabricated_hits`; (8)
`test_strict_guard_rejects_stage_budget_exhaustion_no_csv_written`; (9)
`test_success_after_retry_within_budget_has_real_final_evidence`; (10)
`test_non_retryable_error_fails_immediately_even_with_budget_active`;
(11) `test_genuine_zero_hit_not_retried_with_budget_active`; (12) full
Task 27 allowlist suite re-run green, unmodified; (13)
`test_stage_query_telemetry_includes_new_task31_fields`,
`test_stage_budget_fields_absent_when_no_budget_requested`,
`test_hier_stage_query_telemetry_serializes_task31_fields`; (14) the
full focused set above plus the full suite, both green.

## 8. Confirmation of zero live operations

No live Qdrant connection, embedding-model load, LLM/CrewAI/Ollama/paid
API call, WISCO file read/write, or evaluation run against real data
occurred anywhere in this task. Every new test uses
`RecordingQdrantClient` (a hermetic fake recording every `timeout=`
kwarg received), a `FakeEmbedder`, a controllable `_MonotonicClock`
fake, monkeypatched `time.sleep`, and/or a `MagicMock`-based fake
`ISCOClassifier` — confirmed by direct authorship and by the runs
producing no network-related warnings or errors.

## 9. Historical artifact and protected-branch confirmation

Re-verified after implementation and testing completed, all matching
their previously-recorded values exactly: Task 24's four raw-artefact
SHA-256 checksums; Task 25's diagnostic root (18 artefacts, unmodified);
Task 26's report (`OFFICIAL_TIER1_TELEMETRY_RERUN_COMPLETED: no`) and
its empty (0-file) failed hierarchical output directory; Task 29's
report (`RETRY_CONFIGURATION_PREFLIGHT_READY: yes`); Task 30's report
(`OFFICIAL_TIER1_RETRY_RERUN_COMPLETED: no`), its complete flat CSV
(SHA-256 re-verified byte-identical), and its empty (0-file) failed
hierarchical output directory. No `eval/local_runs/` path was opened
for writing by this task.

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| `reviewer2-wip-snapshot-20260807` | not touched |
| `reviewer2-b2-integration-20260807` | not touched |
| every prior `reviewer2-*` task/integration branch (through `reviewer2-wisco-official-tier1-retry-rerun-20260809`) | not touched |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push occurred. No PR was created.

## 10. Mandatory limitation

No new benchmark evidence, accuracy/statistics, or paper result exists
as a result of this task. This is a code/test correction and hermetic
proof only — it does not run, rerun, or validate any WISCO evaluation,
and it does not establish whether enabling the stage budget would have
prevented Task 30's specific 1,155,613.62 ms observation (the
structural fix — bounding stage-total accumulation — is proven; the
unexplained single-query-duration anomaly noted in §2/§3 is not
diagnosed or claimed fixed). WISCO remains a controlled multilingual
ISCO-08 benchmark, not real Labour Force Survey validation, ISIC/ISCED/
SRE evidence, or a real-world performance claim. A new, separately
authorized live preflight/rerun using this mechanism would be required
to produce any such evidence, and this task does not perform one.

Stopping here, per the task's own instruction, after pushing this
report. No preflight or rerun was started.
