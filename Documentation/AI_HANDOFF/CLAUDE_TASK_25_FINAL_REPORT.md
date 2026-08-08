# Task 25 Final Report — Diagnose the Official Flat Unavailability and Add Fail-Closed Retrieval Telemetry

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_25_OFFICIAL_FLAT_UNAVAILABILITY_DIAGNOSTICS.md`.
This task diagnoses Task 24's single unavailable row without rerunning,
repairing, or scoring Task 24's benchmark, and adds narrow, additive
retrieval telemetry so a future run can distinguish a genuine zero-hit
Qdrant response from a swallowed query exception.

## 1. Branch, base SHA, final SHA, push, clean-tree status

| | |
|---|---|
| Base branch | `reviewer2-wisco-official-tier1-raw-results-20260809` |
| Required SHA | `2e6647c5413b9d8a4ca14e84aa8104f43f5a488d` |
| Verified `origin` SHA | `2e6647c5413b9d8a4ca14e84aa8104f43f5a488d` — match |
| New branch | `reviewer2-official-flat-unavailability-diagnostics-20260809` |
| Final commit SHA | recorded after this report's commit (see push confirmation below) |

Working tree was clean immediately before branching, and immediately
before this report's own commit only the files listed in §2 were
staged (all Task 25 diagnostic run artefacts live under the
Git-ignored `eval/local_runs/official_flat_unavailability_diagnostics_20260808T221507Z/`
output root — confirmed via `git check-ignore -v`).

## 2. Exact changed files and why each was necessary

```text
backend/rag/hierarchy_engine.py                        (modified)
backend/rag/hierarchical_store.py                       (modified)
eval/run_eval.py                                         (modified)
backend/tests/test_flat_query_telemetry.py               (new)
eval/test_flat_query_telemetry_serialization.py           (new)
Documentation/AI_HANDOFF/CLAUDE_TASK_25_FINAL_REPORT.md   (this file)
```

- **`backend/rag/hierarchy_engine.py`**: `HierarchyBeamSearchEngine._query()`
  gained an optional `query_telemetry: Optional[dict] = None` param
  (default `None` reproduces prior behaviour exactly for every existing
  caller that omits it). When provided, it is populated as a side effect
  (mirroring the existing `trace` dict convention already used
  throughout this module) with `"outcome"` (`"success"` | `"exception"`),
  `"duration_ms"`, and — only on `"exception"` — `"exception_type"` and a
  sanitized `"exception_message"`. Necessary because this is the single
  place (confirmed by direct inspection, Part A.4) where a raised
  exception and a genuine empty response were previously collapsed into
  an identical `[]` return with zero telemetry. Also adds the new
  `_sanitize_exception_message()` module function.
- **`backend/rag/hierarchical_store.py`**: `HierarchicalISCOStore._flat_search()`
  now times its one `self._query()` call via `query_telemetry` and
  writes `trace["flat_query_outcome"]` /
  `trace["flat_query_duration_ms"]` /
  `trace["flat_query_exception_type"]` /
  `trace["flat_query_exception_message"]` **before** the existing
  `if not hits: return self._empty_result()` early return, so telemetry
  survives on every outcome, including the unavailable-result path
  Task 24 hit. `HierarchicalISCOStore._query()` (the thin wrapper that
  delegates to the engine) gained the same passthrough param. Necessary
  because `_flat_search()` previously never timed its query at all
  (unlike the legacy `_classify_flat()` path in `isco_classifier.py`,
  which already records `stage4_latency_ms`) — this is the exact gap
  behind every official flat row showing `stage1_latency_ms`
  through `stage4_latency_ms` as `0.0`.
- **`eval/run_eval.py`**: `CaseResult` gained four additive fields
  (`flat_query_outcome`, `flat_query_duration_ms`,
  `flat_query_exception_type`, `flat_query_exception_message`), each
  with a harmless blank/`None` default, and `run_one_case()` copies them
  from `trace` after the existing stage-latency block. Necessary to
  thread the new telemetry into the evaluation CSV (Part C.4) — the CSV
  header is derived from `CaseResult.__dataclass_fields__.keys()`
  (confirmed by inspection), so this is a purely additive schema change;
  no existing column was removed, renamed, or reinterpreted.
- **`backend/tests/test_flat_query_telemetry.py`** /
  **`eval/test_flat_query_telemetry_serialization.py`**: new hermetic
  regression tests (Part D; see §9).

No other file was touched. `backend/agents/isco_classifier.py` needed
no change — it already threads the same `trace` dict object unchanged
through `search_flat_only(..., trace=trace)`.

## 3. Task 24 artefact immutability — before/after SHA-256

| Artefact | SHA-256 (before this task) | SHA-256 (after this task) | Match |
|---|---|---|---|
| Flat raw CSV (`.../flat/20260808T214023Z_wisco_official_tier1_flat.csv`) | `775bc46d9fe83858471f7e559b84bf1a5607be4b905c8e1b5f2f723d6307c1fa` | `775bc46d9fe83858471f7e559b84bf1a5607be4b905c8e1b5f2f723d6307c1fa` | Yes |
| `flat_stdout.log` | `83540c5ebc281e74ca98e5f8903fd34c78a177ce29caaedc57b6d7361850c078` | `83540c5ebc281e74ca98e5f8903fd34c78a177ce29caaedc57b6d7361850c078` | Yes |
| `flat_integrity_gate_report.json` | `55dbea2a59308a87cac09b61a99ce8ba68b9df149e27fc698da5fa9a1df40349` | `55dbea2a59308a87cac09b61a99ce8ba68b9df149e27fc698da5fa9a1df40349` | Yes |
| `failing_row_detail.json` | `c8136262da53f0c50c88b2da2908bf7b7c6cec362424e1b7ecbbca8b7ceb785c` | `c8136262da53f0c50c88b2da2908bf7b7c6cec362424e1b7ecbbca8b7ceb785c` | Yes |

All four checksums were computed before this task made any code or
diagnostic change, and recomputed after all Part A-D work completed.
All four are byte-identical. Task 24's WISCO package, split manifest,
official ILO catalogue/metadata, and prior local-run artefacts were
also not opened for writing at any point in this task.

## 4. Task 24 log/raw artefact evidence — fact vs. hypothesis

**Observed facts** (directly read from the preserved Task 24 artefacts):

- The failing row (`WISCO-8131001300018-ar`) has `pred_method=
  unavailable_isco08_official_ilo2021_v1`, blank `pred_isco_4digit`,
  `pred_reasoning="No candidates returned by the vector store."`,
  blank `error`, and `end_to_end_latency_ms=60743.26`.
- `flat_stdout.log` contains **exactly one** line matching
  `WARNING backend.rag.hierarchy_engine`, at line 16211:
  `HierarchyBeamSearchEngine: query on 'isco08_unit_groups_flat_ilo2021_v1'
  failed: timed out`. `isco08_unit_groups_flat_ilo2021_v1` is the exact
  collection the official flat comparator queries. No other
  warning/exception/timeout/retry text of any kind appears anywhere
  else in the 18,759-line log (grep for
  `warning|exception|traceback|timeout|retry|error`, case-insensitive,
  excluding TensorFlow import banners, returns only this one line).
- The full CSV contains **exactly one** row with a non-nominal
  `pred_method` (`flat_integrity_gate_report.json`'s own
  `bad_code_count: 1`).
- `_flat_search()`'s code, as it existed at the time of the Task 24 run
  (confirmed by reading the pre-Task-25 source), calls `self._query()`
  **exactly once** per case for the flat path — there is no retry loop
  for flat retrieval (unlike the hierarchical keyword-anchor path).
  Every exception inside `_query()` is caught, logged via
  `_logger.warning(...)`, and converted to an empty list, with no
  telemetry recorded before this task.
- The warning's line position (16211) precedes the printed case-index
  line for case `[16204/18747]`, ~100 cases before the failing case
  `[16304/18747]` at line 16312 — the raw text positions are **not**
  adjacent.

**Hypothesis, clearly labelled as such** (not directly observed): the
apparent ~100-case positional gap is explained by Python's I/O
buffering, not by the warning belonging to a different case. `eval/run_eval.py`
configures `logging.basicConfig(level=logging.WARNING, ...)`, whose
default handler writes to `sys.stderr`; the per-case `print(f"[{i+1}/...")`
line (`eval/run_eval.py:1304-1307`) writes to `sys.stdout` with no
`flush=True`. When stdout is redirected to a file (as it was, via
`nohup ... > flat_stdout.log 2>&1`), Python's stdout becomes fully
block-buffered while stderr remains line-buffered/unbuffered — so a
`logging.warning()` call is typically flushed to the merged log file
well before the many already-buffered `print()` lines that preceded it
in real time. This is a standard, well-documented CPython I/O behaviour
(not evaluation-tool-specific), offered as the most plausible
explanation for the observed offset — it is not independently proven
from the log alone.

**A structural (reasoning-based, not directly observed) corroborating
point**: `_flat_search()` has no retry, and an unfiltered
`limit=5` query against a populated 436-point collection (confirmed
non-empty by both the pre- and post-run Task 24 Qdrant inventories)
returns nearest-neighbour hits regardless of similarity score — Qdrant
does not refuse to return points for a well-formed, non-degenerate
query vector. A "genuine successful zero-hit" response against this
specific collection shape is therefore highly implausible as an
alternative explanation, though this task did not attempt to
mathematically prove it impossible.

**No exception evidence exists for any other row.** This is stated
explicitly, per the task's own instruction not to infer an exception
merely because an output was unavailable: only one row is unavailable,
and only one row has any corresponding log evidence at all.

## 5. Diagnostic classification

```text
confirmed_exception
```

**Evidence supporting this classification**: (a) a specific, logged
exception message naming the exact collection used by the official
flat comparator, whose text ("failed: timed out") is characteristic of
a client-side query timeout; (b) an exact 1:1 correspondence between
the single logged warning and the single unavailable row, given that
`_flat_search()` calls `_query()` exactly once per case with no retry,
making "one exception → one unavailable row" and "one unavailable row →
one exception" the same fact, not merely a coincidence of counts; (c)
no evidence anywhere in the log of any other failure mode (parsing
error, malformed response, connection refusal, etc.). This is
classified as `confirmed_exception` rather than `confirmed_zero_hits`
because a genuine zero-hit response would not produce a `_logger.warning`
call at all (the non-exception path in `_query()` never logs), and rather
than `intermittent_unresolved` because the classification concerns the
one, single, already-completed Task 24 event, for which direct log
evidence exists — `intermittent_unresolved` describes Part B's
diagnostic reproducibility (see §6), not the original event itself.

**What is not established**: the exact exception class (the pre-Task-25
code only logged `str(exc)`, not `type(exc).__name__`, so Task 24's log
alone cannot name the Python exception type — e.g. `httpx.ReadTimeout`,
`grpc.RpcError`, or a qdrant-client wrapper); why the timeout occurred at
that specific point in a ~12-minute continuous run (server-side load,
network contention, GC pause, or another transient cause were not
diagnosed and this task makes no claim about which); and whether it
would recur on a different case or a different run.

## 6. Part B — single-case diagnostic attempts

Diagnostic root: `eval/local_runs/official_flat_unavailability_diagnostics_20260808T221507Z/`
(Git-ignored — confirmed via `git check-ignore -v`).

Materialised input: a one-row CSV containing only
`WISCO-8131001300018-ar` (`input_text`, `input_language`,
`gold_isco_4digit` copied read-only from Task 24's own fresh heldout
export; the source WISCO package was not opened for writing). Each
attempt used `QDRANT_TIMEOUT_SECONDS=30`, the official profile, and no
reranker.

**Exact command (identical for all three attempts, only `--config`/`--output-dir` varied):**

```bash
QDRANT_TIMEOUT_SECONDS=30 python eval/run_eval.py \
  --test-set eval/local_runs/official_flat_unavailability_diagnostics_20260808T221507Z/single_case_input.csv \
  --system flat \
  --use-llm-reranker off \
  --isco-catalogue-profile official_ilo2021_v1 \
  --config task25_diag_attemptN \
  --output-dir eval/local_runs/official_flat_unavailability_diagnostics_20260808T221507Z/attemptN
```

| Attempt | Duration | Exit | `pred_method` | `pred_isco_4digit` | `flat_query_outcome` | `flat_query_duration_ms` |
|---|---|---:|---|---|---|---:|
| 1 | 33.0s (wall; model load dominates — per-case time 0.34s) | 0 | `flat_isco08_official_ilo2021_v1` | `8211` | `success` | 91.207 |
| 2 | 29.4s (per-case time 0.30s) | 0 | `flat_isco08_official_ilo2021_v1` | `8211` | `success` | 92.097 |
| 3 | 28.4s (per-case time 0.26s) | 0 | `flat_isco08_official_ilo2021_v1` | `8211` | `success` | 70.680 |

All three attempts succeeded (`flat_query_outcome=success`) with
per-query durations of 70-92 ms — nowhere near the 30-second
`QDRANT_TIMEOUT_SECONDS` bound — and identical predictions
(`8211`, confidence `0.8442`). None of the three attempts reproduced
Task 24's unavailable result. Per the task's own instruction, this
result set (0 failures out of 3) does not itself trigger the
`intermittent_unresolved` category (that category is defined for "one
attempt succeeds, another fails"); it is consistent with — and does not
contradict — a single transient, load-dependent timeout during Task
24's sustained ~12-minute, 18,747-query run that does not reproduce
under an isolated single-case query.

**These attempts are explicitly not a Task 24 retry**: they ran against
a one-row temporary CSV under a new, separate output root, used
separately-timestamped `--config`/`--run-id` values, and their output
was never compared against or merged into Task 24's raw CSV. Task 24's
raw output (§3) was never touched.

## 7. Qdrant before/after inventories — zero mutation

**Before diagnostics** (identical to Task 24's post-run inventory):

| Collection | Count |
|---|---:|
| `isco08_major_groups_ilo2021_v1` | 10 |
| `isco08_submajor_groups_ilo2021_v1` | 43 |
| `isco08_minor_groups_ilo2021_v1` | 130 |
| `isco08_unit_groups_ilo2021_v1` | 436 |
| `isco08_unit_groups_flat_ilo2021_v1` | 436 |
| `isco08_major_groups` (legacy) | 10 |
| `isco08_submajor_groups` (legacy) | 43 |
| `isco08_minor_groups` (legacy) | 131 |
| `isco08_unit_groups` (legacy) | 441 |
| `isco_occupations` (legacy) | 124 |

**After diagnostics**: identical in every collection (`diff` of the two
recorded inventory files, taken before attempt 1 and after attempt 3,
produced no output). No builder execution, collection creation,
deletion, or repair occurred at any point in this task.

## 8. Additive telemetry schema and sanitization

New `CaseResult` fields (`eval/run_eval.py`), all additive with harmless
defaults, positioned immediately after `stage4_latency_ms`:

| Field | Type/default | Populated when |
|---|---|---|
| `flat_query_outcome` | `str = ""` | `"success"` or `"exception"` whenever `_flat_search()`'s one Qdrant call runs (official-profile flat runs); blank for hierarchical runs and legacy-profile flat runs (separate `VectorStore` code path, untouched) |
| `flat_query_duration_ms` | `Optional[float] = None` | The single flat query's wall-clock duration, rounded to 3 decimal places, on both success and exception |
| `flat_query_exception_type` | `str = ""` | `type(exc).__name__` (e.g. `"TimeoutError"`), only when `flat_query_outcome == "exception"` |
| `flat_query_exception_message` | `str = ""` | Sanitized `str(exc)`, only when `flat_query_outcome == "exception"` |

**Sanitization** (`hierarchy_engine._sanitize_exception_message()`):
collapses all whitespace (including newlines/tabs) to single spaces,
then truncates to 300 characters with an explicit `"...(truncated)"`
suffix if longer. The message is derived **only** from the exception's
own `str()` — never from `traceback.format_exc()`, the query vector, the
raw input text, or any request/response payload — so it structurally
cannot contain a stack trace, and the query text/vector are never
in scope to leak (confirmed by a dedicated test, §9, using a
deliberately secret-shaped query string that does not appear in the
recorded telemetry).

## 9. Focused and full test results

Hermetic (`FakeQdrantClient`/`_RaisingQdrantClient`/`FakeEmbedder`
only; no live Qdrant/model/LLM dependency in any new test):

- `backend/tests/test_flat_query_telemetry.py` (13 tests): zero-hit
  success records no exception telemetry; a generic exception
  (`ConnectionError`) and a timeout-shaped exception (`TimeoutError`)
  are each captured with the correct `exception_type`; duration
  telemetry is recorded on both the hit and zero-hit paths; a
  successful official flat retrieval still returns a valid 4-digit
  code; an exception still returns the existing empty/unavailable
  result (no fabricated code); the legacy profile's `_flat_search()`
  (shared implementation) also gains telemetry without changing its
  existing return value; `_sanitize_exception_message()` bounds length
  and flattens newlines/tabs; a deliberately secret-shaped query string
  never appears in the recorded exception message; `query_telemetry`
  defaults to `None` and is fully backward compatible for every
  existing `_query()` caller.
- `eval/test_flat_query_telemetry_serialization.py` (5 tests): the new
  fields serialize correctly to the written CSV on both the exception
  and success paths; a hierarchical run's CSV leaves all four new
  fields blank/empty; `check_strict_hierarchical()` still rejects a
  flat/unavailable `pred_method` exactly as before and still passes a
  genuine complete hierarchical case, confirming the new fields are
  never inspected by and cannot be mistaken for stage evidence.

```
python -m pytest backend/tests/test_flat_query_telemetry.py eval/test_flat_query_telemetry_serialization.py backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchical_store.py backend/tests/test_standard_hierarchical_store.py backend/tests/test_isco_classifier_official_profile.py backend/tests/test_official_isco08_catalogue.py backend/tests/test_official_isco08_profiles.py backend/tests/test_build_official_isco08_collections.py eval/test_require_genuine_hierarchical.py eval/test_official_isco08_profile_evaluator.py eval/test_model_free_isco_evaluation.py eval/test_run_eval_b2.py eval/test_docs_consistency.py -q
→ 213 passed in 2.17s

python -m pytest backend/tests eval/ -q
→ 2074 passed, 1 deselected, 1 warning in 307.45s (0:05:07)
```

**Zero failures.** `2074 = 2056` (Task 24 baseline) `+ 18` net new
tests (13 + 5, exactly matching the two new test files' counts) — an
exact match confirming zero regressions and no unrelated test was
repaired, skipped, or altered.

## 10. Protected-branch status and no-PR confirmation

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| `reviewer2-wip-snapshot-20260807` | not touched |
| `reviewer2-b2-integration-20260807` | not touched |
| every prior `reviewer2-*` task/integration branch (through `reviewer2-wisco-official-tier1-raw-results-20260809`) | not touched |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push occurred. **No PR was created.**

## 11. Technical decision

- **Is the cause diagnosed sufficiently to propose a narrow fix?**
  Sufficiently to explain the *mechanism* (an uncaught Qdrant client
  exception on the flat collection, converted to an unavailable result
  with no prior telemetry) — this task's Part C change is exactly that
  narrow fix, but scoped to observability only, per the task's explicit
  restriction against remediation/rerun/repair. It is **not** diagnosed
  to the level of a specific network/server root cause, and no retry or
  resilience logic was added or proposed as a code change in this task.
- **Is a future clean full official-tier rerun technically justified?**
  Yes, with a caveat: the evidence (one logged timeout-shaped exception
  in an 18,747-query, ~12-minute sustained run; three isolated
  single-case retries all succeeding quickly) is consistent with a rare,
  transient, load-dependent event rather than a systematic defect in
  the official flat retrieval path or the official collections
  themselves (which passed every Task 23/24 integrity check). A future
  rerun is technically reasonable. However, because this task
  deliberately added no retry/resilience logic (out of scope), a future
  rerun could in principle encounter a similar rare transient failure on
  a different row — if it does, the telemetry added in this task
  (§8) will make the outcome (`exception` vs genuine zero-hit),
  exception type, message, and duration explicit in that row itself,
  removing the need for a Task-25-style follow-up investigation.
- **Why does Task 24 remain ineligible regardless of the diagnostic
  result?** Task 24's raw flat CSV (§3) is unmodified and still
  contains exactly one non-nominal row; per Task 24's own fail-closed
  gate rule, any unavailable prediction fails the flat integrity gate
  outright, regardless of cause. Diagnosing *why* a row failed does not
  retroactively make that row's output correct or complete, and this
  task explicitly did not weaken, bypass, or reinterpret Task 24's gate
  criteria. Only a fresh, complete run that independently passes both
  Task 24 gates in full could ever be eligible for downstream analysis
  — and this task was explicitly forbidden from performing that rerun.

## Evidence boundary

```text
Task 24 remains an incomplete, fail-closed raw run and is not eligible for
accuracy analysis or manuscript claims. Task 25 single-case probes are
diagnostics only, not a benchmark retry or validation result. No conclusion
about real Labour Force Survey performance, ISIC, ISCED, SRE, reranking, cost,
or real-world accuracy is supported.
```

No new full benchmark, accuracy analysis, statistical comparison, paper
draft, B1 re-freeze, or B2 sweep was run. Stopping here, per the task's
own instruction, after pushing this report.
