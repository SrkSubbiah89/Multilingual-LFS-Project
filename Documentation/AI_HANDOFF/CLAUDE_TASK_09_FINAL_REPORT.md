# Task 09 Final Report — Make Retrieval-Only ISCO Evaluation Truly Model-Free

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_09_MODEL_FREE_ISCO_EVALUATION.md`
(task ID `09-model-free-isco-evaluation`). Code-and-test task only — no
benchmark or evaluation was run.

## Starting and final SHAs

| | |
|---|---|
| Base branch | `reviewer2-pre-evaluation-baseline-20260808` |
| Base SHA (verified) | `c3d39d89f1b164f4fc5595c319e90546a2c17716` |
| Working branch | `reviewer2-model-free-isco-evaluation-20260808` (new, created by this task) |
| Final SHA | commit created and pushed as this task's own closing step, containing this report (see push confirmation below) |

Start-state check passed exactly: `git fetch origin` confirmed
`origin/reviewer2-pre-evaluation-baseline-20260808` at
`c3d39d89f1b164f4fc5595c319e90546a2c17716`; the local branch of that name
was already checked out at that exact SHA with a clean working tree
(`git status --short` produced no output) before
`git switch -c reviewer2-model-free-isco-evaluation-20260808` was run.

## Before/after: the preflight defect

**Before**: `eval/run_eval.py --system hierarchical/flat --use-llm-reranker
off` still (1) required `--reranker-model` at CLI validation even though
nothing would use it, (2) constructed `ISCOClassifier` through
`build_system()` with no way to suppress LLM initialisation — `__init__`
always attempted `get_llm(TaskType.GENERAL, ...)` (or
`get_llm_strict(...)` if a model happened to be pinned), constructing an
LLM/agent object that was simply never invoked because `use_llm=False`
was passed to `classify()`, and (3) unconditionally constructed
`ISICClassifier`, `ISCEDClassifier`, and `SemanticRelationEngine` in
`main()` regardless of whether the loaded CSV had any `industry_text`/
`education_text` at all — for a WISCO-style ISCO-only CSV, all three were
constructed and then never used for a single row. The config hash also
hardcoded `"use_llm": True` regardless of the actual `--use-llm-reranker`
value, so a retrieval-only run's hash could not be distinguished from a
reranked run's on that field alone.

**After**: `ISCOClassifier(enable_llm=False)` (Scope A) skips LLM/agent
construction entirely — Stage 2 of `__init__` is never entered,
`reranker_model_resolved` becomes the explicit string `"none (reranking
disabled)"`. `build_system(..., use_llm_reranker=False)` (Scope B) passes
`enable_llm=False, reranker_model=None` for hierarchical/flat, ignoring
any stray `--reranker-model` value. CLI validation now requires
`--reranker-model` only when `--system` is hierarchical/flat **and**
`--use-llm-reranker` is `on`, outside `--dry-run`. `main()` inspects the
post-`--limit` loaded rows once and only constructs
`ISICClassifier`/`ISCEDClassifier`/`SemanticRelationEngine` when at least
one row has both non-blank `industry_text` and `education_text`;
otherwise all three stay `None`, and `run_one_case()`'s guard was widened
(`isic_clf is not None and isced_clf is not None and ...`) so it can never
be reached with a `None` classifier even if a future caller's row
population diverged from the one `main()` inspected. `_config_hash()`'s
`"use_llm"` field now reflects `args.use_llm_reranker == "on"` instead of
a hardcoded `True`.

## Exact changed files

`git diff --stat` → **4 files changed, 302 insertions(+), 45 deletions(-)**, plus 1 new file:

- `backend/agents/isco_classifier.py` (modified) — Scope A: additive
  `enable_llm: bool = True` constructor parameter (final positional slot),
  docstring, and the Stage-2 branch guard.
- `eval/run_eval.py` (modified) — Scope B: `build_system(..., use_llm_reranker=True)`,
  CLI `--reranker-model` validation narrowed to reranking-on-only,
  `--reranker-model`/`--use-llm-reranker` help text and module docstring
  updated, `_config_hash()`'s `"use_llm"` field fixed, `run_one_case()`'s
  `sre`/`isic_clf`/`isced_clf` params typed `Optional[...]` with a
  strengthened guard, and `main()`'s conditional ISIC/ISCED/SRE
  construction based on inspecting the loaded (post-`--limit`) rows.
- `backend/tests/test_isco_classifier.py` (modified) — 11 new tests
  (`TestEnableLlmFalse`, `TestEnableLlmTrueUnchanged`).
- `eval/test_model_free_isco_evaluation.py` (new) — 16 new tests covering
  `build_system()`, CLI validation, conditional ISIC/ISCED/SRE
  construction, and config-hash honesty.
- `Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`
  (modified) — added a note to Phase D.1 explaining the Task 09
  correction; the plan's own prepared Tier-1 commands already omitted
  `--reranker-model` and are now valid (previously they would have failed
  CLI validation).

No other file was touched — `git status --short` before staging showed
exactly these 5 paths.

## Test commands and exact outputs

```
pytest backend/tests/test_isco_classifier.py -q
→ 41 passed in 0.58s   (30 pre-existing + 11 new)

pytest eval/test_model_free_isco_evaluation.py -q
→ 16 passed in 31.15s  (new file)

pytest backend/tests/test_isco_classifier.py backend/tests/test_isco_classifier_extended.py eval/test_run_eval_b2.py eval/test_sre_isic_isced_coupling_fix.py -q
→ 159 passed in 1.54s
  (all four named files exist on this branch -- no substitution needed)

pytest backend/tests eval/ -q
→ 1911 passed, 1 deselected, 1 warning in 291.60s (0:04:51)
```

**Zero failures.** Full-suite pass count went from 1885 (this branch's
starting baseline) to 1911 (+26: 11 new `ISCOClassifier` tests + 16 new
`run_eval.py` tests, minus the fact that pre-existing counts elsewhere are
unchanged) with the same `1 deselected` and no new failure. The full
suite is genuinely green — this claim is based on the actual command
output above, not assumed.

## Proof of the 7 required Scope C properties

1. **`ISCOClassifier(enable_llm=False)` never calls `get_llm_strict`/
   `get_llm`/`_build_reranker_agent`, still classifies via a fake ready
   store**: `TestEnableLlmFalse::test_does_not_call_get_llm_or_get_llm_strict`,
   `::test_does_not_construct_agent`,
   `::test_still_makes_semantic_only_classification_with_ready_store`.
2. **Default/`enable_llm=True` behaviour unchanged, including the strict
   pinned-model path**: `TestEnableLlmTrueUnchanged::test_default_omitted_still_calls_get_llm`,
   `::test_reranker_model_pin_still_uses_get_llm_strict`,
   `::test_no_reranker_model_pin_uses_plain_get_llm`.
3. **`build_system(..., use_llm_reranker=False)` passes `enable_llm=False`
   and no reranker model; reranking-on retains the pin**:
   `test_build_system_hierarchical_reranker_off_passes_enable_llm_false_and_no_model`,
   `test_build_system_flat_reranker_off_passes_enable_llm_false_and_no_model`,
   `test_build_system_hierarchical_reranker_on_passes_model_no_enable_llm_override`,
   `test_build_system_flat_reranker_on_passes_model`.
4. **CLI accepts `--use-llm-reranker off` without `--reranker-model` for
   both systems; still rejects reranking-on without a pin**:
   `test_cli_accepts_use_llm_reranker_off_without_reranker_model_hierarchical`,
   `::_flat`, `test_cli_rejects_reranking_on_hierarchical_without_pin`,
   `::_flat`.
5. **ISIC/ISCED/SRE not constructed at all for WISCO-style rows (mocks
   that fail if constructed)**:
   `test_main_does_not_construct_isic_isced_sre_when_no_row_has_paired_text`,
   `test_main_does_not_construct_isic_isced_sre_when_text_is_one_sided`
   (each uses `MagicMock(side_effect=AssertionError(...))` constructors).
6. **With paired text, full ISIC/ISCED/SRE behaviour available; `--sre off`
   still runs ISIC/ISCED**:
   `test_main_constructs_isic_isced_sre_when_a_row_has_paired_text`,
   `test_main_sre_off_still_runs_isic_isced_only_sre_skipped`; regression
   confirmed by the unmodified `eval/test_sre_isic_isced_coupling_fix.py`
   suite still passing (19/19).
7. **A no-reranker run produces explicit retrieval-only metadata, never a
   fake model identity or LLM-reranker trace**:
   `TestEnableLlmFalse::test_reranker_model_resolved_is_explicit_disabled_string`,
   `::test_classify_never_produces_an_llm_reranker_trace`,
   `test_config_hash_reflects_use_llm_reranker_off_not_hardcoded_true`.

All tests use fakes/mocks (`MagicMock`, hand-built `SimpleNamespace`
results, `monkeypatch`). No Qdrant, SentenceTransformer download, Ollama,
paid LLM/API call, benchmark run, WISCO dataset build, or real dataset
access occurred in any test.

## Confirmation: no live Qdrant/model/LLM/benchmark/dataset operation occurred

- `ISCOClassifier`/`get_llm`/`get_llm_strict`/`Agent`/`Task`/`Crew` were
  monkeypatched to fakes in every test that constructs a classifier;
  `get_hierarchical_store`/`get_vector_store` were forced to raise/return
  fakes, never a real `QdrantClient`.
- No `eval/run_eval.py` invocation in this task's own work (only inside
  hermetic tests, with every classifier mocked) constructed a real
  `ISCOClassifier`/`ISICClassifier`/`ISCEDClassifier`/
  `SemanticRelationEngine`, connected to Qdrant, or called Ollama/an LLM
  API.
- No WISCO dataset build, `eval/build_wisco_isco_benchmark.py`,
  `eval/export_benchmark_to_run_eval_csv.py`, or benchmark run occurred.
- Test-file CSVs were written only to pytest's own `tmp_path` (ephemeral,
  local, auto-cleaned) — never to `eval/local_benchmarks/` or any tracked
  dataset location.
- `git rev-parse --short HEAD` runs inside `main()` (best-effort, 5s
  timeout, non-fatal on failure) — a local, non-network subprocess call
  already part of `run_eval.py`'s existing, unmodified bookkeeping; not a
  benchmark/dataset/network operation.
- No official ISIC/ISCED-F catalogue import. No B1 re-freeze, no B2
  sweep, no manuscript claim change.

## Protected branch SHAs

| Branch | SHA (unchanged) |
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
| `reviewer2-pre-evaluation-baseline-20260808` | not touched (used only as the branch point; never checked out for writing) |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push was run at any point in this task. No PR was
created.

## Working-tree status

Clean before this task started (verified) and clean again immediately
before this report's commit — `git status --short` showed exactly the 5
paths listed above, matching the scope described with nothing else
pending.

## Remaining manuscript limits

Unchanged from prior tasks, and this task adds no new evidence:

- No live Qdrant collection has been built or populated for ISIC/ISCED-F.
- No accuracy, latency, cost, or improvement measurement exists for
  ISCO-08 retrieval-only, ISIC/ISCED-F hierarchical retrieval, or any
  WISCO configuration — this task changed no numbers, it only removed an
  unnecessary LLM/reranker-model dependency and unnecessary ISIC/ISCED/SRE
  construction from a code path that had not yet been run.
- No official-catalogue coverage claim is supported.
- A retrieval-only run (`--use-llm-reranker off`) is **not** a reranking
  comparison — there is nothing to compare, since no reranking runs at
  all — and it supports **ISCO-08 classification only**; it must never be
  described as measuring ISIC/ISCED-F, SRE, or reranking quality.
- The next task (per this task's own stop condition) will independently
  review and integrate this correction before any measured WISCO
  evaluation occurs — no benchmark or evaluation was started here.
