# Task 13 Final Report — Eliminate Silent Hierarchical Fallbacks and Add Bounded Evaluation Guards

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_13_HIERARCHICAL_INTEGRITY_HARDENING.md`
(task ID `13-hierarchical-integrity-hardening`).

## Branch / SHA

| | |
|---|---|
| Base branch | `reviewer2-wisco-tier1-results-20260808` |
| Base SHA (verified) | `6710192ea255c2cfb00599db9e55f65f9a53ca53` |
| Working branch | `reviewer2-hierarchical-integrity-hardening-20260808` (new, created by this task) |
| Final SHA | commit created and pushed as this task's own closing step, containing all changes below |

Start-state check passed exactly: `git fetch origin` confirmed
`origin/reviewer2-wisco-tier1-results-20260808` at
`6710192ea255c2cfb00599db9e55f65f9a53ca53`; the local branch was already
at that exact SHA with a clean working tree before
`git switch -c reviewer2-hierarchical-integrity-hardening-20260808` was
run.

## Exact changed files

`git diff --stat` → **3 files modified, 2 files new, 351 insertions(+), 8 deletions(-)**

- `backend/rag/hierarchical_store.py` (modified) — Part A (keyword-anchor
  retry recovery) and Part B (bounded Qdrant timeout), plus an expanded
  module docstring documenting the resulting four-state retrieval-outcome
  model.
- `eval/run_eval.py` (modified) — Part C (`--require-genuine-hierarchical`,
  `--max-stage-latency-ms`, `check_strict_hierarchical()`), plus two new
  additive `CaseResult` fields (`keyword_anchor_retry_used`,
  `keyword_anchor_original_hint`) and their population in `run_one_case()`.
- `backend/tests/test_hierarchical_store.py` (new) — 18 hermetic tests
  covering Part A/B (this file did not exist before Task 13; no
  `HierarchicalISCOStore`-level test file previously existed).
- `eval/test_require_genuine_hierarchical.py` (new) — 19 hermetic tests
  covering Part C.
- `Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`
  (modified) — new "Post-execution update: Task 12 and Task 13" section.

No other file was touched — `git status --short` before staging showed
exactly these 5 paths. The Task 12 raw output CSVs and report were **not**
modified: `md5sum -c` against checksums recorded at the start of this
task confirms both
`eval/local_runs/wisco_v2_tier1_full_20260807T234643Z/{flat,hierarchical}/*.csv`
are byte-identical before and after this task's work.

## A. Recovered hierarchical retrieval after a failed keyword anchor

**Verified root cause** (direct code inspection, exactly as Task 13
described): `ISCOClassifier.classify()` computes
`major_hint = "" if self._disable_keyword_map else _keyword_major_hint(job_title)`
by default and passes it to `HierarchicalISCOStore.search(major_hint=...)`.
`_hierarchical_search()` turned a non-empty hint into a single-branch
`SeedSpec`, bypassing semantic stage 1 entirely. If that one anchored
branch reached no stage-4 result, `HierarchyBeamSearchEngine.search()`
returned `None`, and `_hierarchical_search()` returned `None` straight
through — `search()`'s existing code then fell through to
`_flat_search()` with no attempt at genuine hierarchical recovery. This
is the confirmed, code-level cause of the 21 "fast" fallback rows (not
the 2-3 severe-latency rows — see below) in Task 12's full run.

**Fix** (`_hierarchical_search()`, minimum connected code only): when
`major_hint` was supplied and the seeded engine call returns `None`, the
method now retries the same `HierarchyBeamSearchEngine.search()` exactly
once with `seed=None` (normal semantic stage-1 retrieval) before
`_hierarchical_search()` returns `None` to its caller. Each attempt
(seeded, and — only if needed — the unseeded retry) is run with its own
throwaway trace dict; only the winning attempt's dict is ever copied into
the caller-visible `trace` via `trace.update(attempt_trace)` — a failed
seeded attempt's stage evidence is discarded entirely, never merged with
the retry's. Two new trace keys — `keyword_anchor_retry` (bool) and
`keyword_anchor_original_hint` (the failed hint string) — are set
whenever a keyword anchor was attempted at all (success or failure),
giving transparent, separate metadata without overloading
`stage1_source` (which continues to report the true final path:
`"keyword_map"` on first-try success, `"semantic_retrieval"` after a
recovery retry). Two additive `CaseResult` fields
(`keyword_anchor_retry_used`, `keyword_anchor_original_hint`, both
safe-defaulted) surface this in `eval/run_eval.py`'s CSV output. If the
unseeded retry also fails, the existing, unmodified flat-fallback code
path runs exactly as before — `fallback_used=True`,
`ISCOClassification.method` gets the existing `"flat_"` prefix, never
silently relabelled as hierarchical.

Default production behaviour for a successful keyword hint is byte-for-
byte unchanged (verified by
`test_successful_keyword_seed_performs_one_engine_search`: exactly one
engine search, correct candidate calls, correct trace shape).

## B. Bounded Qdrant request timeout

Added `QDRANT_DEFAULT_TIMEOUT_SECONDS = 30` and
`_resolve_qdrant_timeout_seconds()` (reads `QDRANT_TIMEOUT_SECONDS`;
fails safe to the 30s default on missing, non-integer, or non-positive
values — never raises, never crashes startup). `HierarchicalISCOStore.
__init__` gained an additive `timeout_seconds: Optional[int] = None`
parameter (explicit override for tests/callers; `None` resolves from the
environment) and now constructs `QdrantClient(..., timeout=<resolved>)`.

**This does not claim to fix or explain the Task 12 multi-minute-to-
multi-hour stalls** — that root cause remains unconfirmed. It only bounds
a *class* of indefinitely-blocked Qdrant requests: a future stall now
raises a catchable exception at the client level, which the existing,
unmodified `HierarchyBeamSearchEngine._query()` already catches (`except
Exception: return []`) — routing straight into the same zero-hits →
fallback/retry path Part A hardened, rather than blocking forever. This
is stated explicitly, not implied, in the code's own docstring and in
this report.

Dependency injection/testability preserved: no constructor change
requires a live Qdrant connection; all timeout tests monkeypatch
`hierarchical_store.QdrantClient`/`SentenceTransformer` at module level
(the same pattern already used throughout this codebase) and never touch
a real client.

## C. Strict evaluation-only guard (`--require-genuine-hierarchical`)

Added to `eval/run_eval.py`:

- `--require-genuine-hierarchical` (opt-in flag) — valid only with
  `--system hierarchical`; `parser.error()` rejects any other `--system`
  value at argument-validation time, before any classifier is
  constructed.
- `--max-stage-latency-ms` (opt-in float) — no effect unless
  `--require-genuine-hierarchical` is also passed; `parser.error()`
  rejects zero/negative/invalid values.
- `check_strict_hierarchical(result, max_stage_latency_ms)` — pure,
  side-effect-free function returning `None` for a genuine, complete,
  (optionally) latency-bounded hierarchical result, or a human-readable
  reason string naming the exact case, stage, and observed value
  otherwise. Checks, in order: `pred_method` must start with
  `"hierarchical_"` (a `"flat_"` prefix means the fallback fired);
  `stage1_candidates`..`stage4_candidates` must each be non-empty,
  valid-JSON, non-empty lists; and (only if a threshold was supplied)
  every `stageN_latency_ms` must not exceed it.
- Wired into the main classification loop immediately after each
  `run_one_case()` call: on the first violating case, the run prints the
  exact reason to stderr and calls `sys.exit(1)` **before** the
  CSV-writing code is ever reached — so no result CSV is written for a
  contaminated run, matching the task's explicit "do not write a normal
  successful result CSV that can be mistaken for valid evidence"
  requirement literally.
- Purely evaluative: `check_strict_hierarchical()` never mutates its
  input, never retries, never overwrites `pred_method`, and the guard
  itself never alters classification behaviour — it only decides whether
  `main()` proceeds to write output.
- Default runs (flag omitted) are provably unaffected:
  `test_ordinary_run_without_strict_flag_preserves_fallback_labelling`
  and `test_dry_run_unaffected_by_require_genuine_hierarchical` both
  construct a fallback-labelled/dry-run scenario and confirm the CSV is
  still written with the explicit `"flat_semantic"` label intact, and
  that `--dry-run` still constructs zero classifiers even when
  `--require-genuine-hierarchical` is also passed.

## Verified silent-fallback code path vs. unconfirmed environment-stall hypothesis

These are two **distinct** findings, and this report (like the code
comments and doc update) keeps them separate:

| | Verified by direct code inspection | Root cause confirmed? |
|---|---|---|
| 21 fast `flat_semantic` fallback rows | **Yes** — the single-branch seeded-search dead-end described in Part A, reproduced exactly by `test_failed_seed_retries_once_unseeded_and_recovers`'s fixture shape | Yes — fixed in Part A |
| 2-3 severe-latency (minutes-to-hours) stall rows | No — only the *absence* of a client-side timeout was identified as a contributing structural gap | **No** — Part B only bounds future occurrences; it does not explain what caused Task 12's specific stalls |

## Test commands and exact outputs

```
pytest backend/tests/test_hierarchical_store.py eval/test_require_genuine_hierarchical.py backend/tests/test_hierarchy_engine.py backend/tests/test_isco_classifier.py backend/tests/test_isco_classifier_extended.py eval/test_model_free_isco_evaluation.py eval/test_run_eval_b2.py eval/test_sre_isic_isced_coupling_fix.py -q
→ 238 passed in 6.65s

pytest backend/tests eval/ -q
→ 1948 passed, 1 deselected, 1 warning in 305.84s (0:05:05)
```

**Zero failures.** Full-suite pass count went from 1911 (this branch's
starting baseline, per Task 10's final state) to 1948 (+37 — exactly the
18 new `test_hierarchical_store.py` tests plus 19 new
`test_require_genuine_hierarchical.py` tests), with the same `1
deselected` and no new failure. Existing tests were not weakened,
skipped, xfailed, or deleted anywhere in this task.

## Proof of the 11 required Part D properties

1. **Successful keyword seed = one engine search, prior behaviour
   preserved**: `test_successful_keyword_seed_performs_one_engine_search`.
2. **Failed seed retries exactly once unseeded, then succeeds**:
   `test_failed_seed_retries_once_unseeded_and_recovers` (asserts the
   exact 5-call sequence: 1 failed seeded query + 4 genuine unseeded
   stage queries).
3. **Retry trace shows only winning-attempt evidence, plus separate
   retry metadata**: `test_retry_trace_shows_only_winning_stage_evidence`.
4. **Both paths fail → explicit, distinguishable flat fallback**:
   `test_both_seeded_and_unseeded_fail_falls_back_to_flat_explicitly`.
5. **Timeout default/valid override/invalid fallback/client construction,
   no live connection**: `test_timeout_default_is_30_seconds`,
   `test_timeout_resolves_to_default_when_env_absent`,
   `test_timeout_valid_env_override`,
   `test_timeout_invalid_env_falls_back_to_default` (parametrized over 6
   malformed values), `test_client_construction_passes_resolved_default_timeout`,
   `test_client_construction_honours_explicit_timeout_seconds_param`,
   `test_client_construction_honours_valid_env_override`.
6. **Query exceptions never appear as hierarchical success**:
   `test_query_exception_never_appears_as_hierarchical_success`.
7. **`--require-genuine-hierarchical` rejects non-hierarchical systems**:
   `test_cli_rejects_require_genuine_hierarchical_with_flat_system`,
   `test_cli_rejects_require_genuine_hierarchical_with_bm25_system`.
8. **Strict run fails non-zero on fallback / missing evidence / exceeded
   latency**: `test_strict_guard_aborts_run_and_writes_no_csv_on_fallback`,
   `test_strict_guard_aborts_run_on_exceeded_stage_latency`, plus the
   unit-level `check_strict_hierarchical()` tests for each trigger.
9. **A valid genuine four-stage result passes the strict guard**:
   `test_strict_guard_passes_genuine_four_stage_result`,
   `test_check_strict_hierarchical_passes_genuine_complete_result`.
10. **Ordinary runs preserve fallback labelling and dry-run behaviour**:
    `test_ordinary_run_without_strict_flag_preserves_fallback_labelling`,
    `test_dry_run_unaffected_by_require_genuine_hierarchical`.
11. **Task 09 model-free behaviour intact**: proven by regression — the
    unmodified `eval/test_model_free_isco_evaluation.py` (16 tests) and
    `backend/tests/test_isco_classifier.py::TestEnableLlmFalse`/
    `TestEnableLlmTrueUnchanged` suites continue to pass unchanged in the
    focused-verification run above; this task touched neither
    `enable_llm` nor the ISIC/ISCED/SRE construction-gating logic.

All tests use fakes/mocks only (`FakeQdrantClient`, `FakeEmbedder`,
`MagicMock`-based classifier stand-ins, `monkeypatch`). No Qdrant,
SentenceTransformer download, Ollama, LLM/API call, or benchmark command
occurred in any test.

## Confirmation: Task 12 raw artifacts unmodified

```
md5sum -c task13_task12_artifact_checksums_before.txt
→ eval/local_runs/wisco_v2_tier1_full_20260807T234643Z/hierarchical/20260808T000252Z_wisco_v2_tier1_full_hierarchical_norerank.csv: OK
→ eval/local_runs/wisco_v2_tier1_full_20260807T234643Z/flat/20260807T234757Z_wisco_v2_tier1_full_flat_norerank.csv: OK
```

Both files are byte-identical to their state before this task began.
`Documentation/AI_HANDOFF/CLAUDE_TASK_12_FINAL_REPORT.md` was not opened
for writing (not in `git status`, not in `git diff`).

## Confirmation: zero live Qdrant/model/LLM/benchmark use

- Every test in both new files monkeypatches
  `QdrantClient`/`SentenceTransformer`/`ISCOClassifier`/`ISICClassifier`/
  `ISCEDClassifier`/`SemanticRelationEngine` to fakes/mocks — several
  additionally use `MagicMock(side_effect=AssertionError(...))` to prove
  a component is never constructed when it shouldn't be.
- No `eval/run_eval.py`, `ablation_runner.py`, or
  `build_wisco_isco_benchmark*.py` invocation occurred outside of the
  hermetic test suite (where every classifier is mocked).
- No WISCO dataset was rebuilt, exported, or re-evaluated. No B1
  re-freeze, no B2 sweep, no Qdrant collection build/mutation of any kind.

## Protected branch status

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| `reviewer2-wip-snapshot-20260807` | not touched |
| `reviewer2-b2-integration-20260807` | not touched |
| `reviewer2-isic-iscedf-hierarchical-rag-20260808` | not touched |
| `reviewer2-isic-iscedf-hierarchical-resilience-20260808` | not touched |
| `reviewer2-isic-iscedf-hierarchical-model-init-resilience-20260808` | not touched |
| `reviewer2-isic-iscedf-integration-20260808` | not touched |
| `reviewer2-isic-iscedf-test-green-20260808` | not touched |
| `reviewer2-pre-evaluation-baseline-20260808` | not touched |
| `reviewer2-model-free-isco-evaluation-20260808` | not touched |
| `reviewer2-wisco-measurement-baseline-20260808` | not touched |
| `reviewer2-wisco-tier1-preflight-20260808` | not touched |
| `reviewer2-wisco-tier1-results-20260808` | not touched (used only as the branch point) |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push was run at any point in this task. No PR was
created.

## Working-tree status

Clean before this task started (verified) and clean again immediately
before this report's commit — `git status --short` showed exactly the 5
paths listed under "Exact changed files" plus this new report, matching
the described scope with nothing else pending.

## Residual manuscript limits

Unchanged from Task 12 — this task fixes code and adds guards, it
produces no new benchmark evidence:

- No Tier-1 accuracy, latency, or comparison figure exists anywhere as of
  this task's completion — Task 12's run was halted before analysis, and
  this task did not run any new benchmark.
- No claim that the hierarchical retrieval system's Task 12 stalls are
  understood or resolved — Part B only bounds future occurrences of a
  *class* of blocking behaviour; it is explicitly not a root-cause fix.
- No claim that the 21-row keyword-anchor fallback issue is the *only*
  remaining integrity risk — it is the one issue this task could verify
  and fix via direct code inspection.
- No real-LFS, ISIC, ISCED, SRE, or reranking conclusion is supported by
  this task, consistent with every prior WISCO-related task.
- Per the stop condition, no WISCO rerun, no collection build, and no new
  benchmark metric were produced here. A separate task will independently
  audit and integrate this fix before a stricter preflight and rerun are
  considered.
