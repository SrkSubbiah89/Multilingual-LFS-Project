# Task 07 Final Report — Repair the Legacy ISCO LLM Mock-Signature Test Defect

Produced in response to
`Documentation/AI_HANDOFF/PERPLEXITY_TO_CLAUDE_07_REPAIR_LEGACY_ISCO_MOCK_SIGNATURE.md`
(task ID `07-repair-legacy-isco-mock-signature`). Test-fixture repair only
— no classifier behavior change.

## Branch / SHA

| | |
|---|---|
| Base branch | `reviewer2-isic-iscedf-integration-20260808` |
| Base SHA (verified) | `53c09e40a2383b102bfa8e432e60426e6c26bda7` |
| Working branch | `reviewer2-isic-iscedf-test-green-20260808` (new, created by this task) |
| Final SHA | commit created and pushed as this task's own closing step (see push confirmation below; this report is committed in the same commit) |

Start-state check passed exactly: `git fetch origin` confirmed
`origin/reviewer2-isic-iscedf-integration-20260808` at
`53c09e40a2383b102bfa8e432e60426e6c26bda7`; the local branch of that name
was already checked out at that exact SHA with a clean working tree
(`git status --short` produced no output) before
`git switch -c reviewer2-isic-iscedf-test-green-20260808` was run.

## Exact root cause

`backend/tests/test_isco_classifier_extended.py`'s shared `clf` fixture
patched `backend.agents.isco_classifier.get_llm` with:

```python
monkeypatch.setattr("backend.agents.isco_classifier.get_llm", lambda t: MagicMock())
```

Production code (`backend/agents/isco_classifier.py:591`) calls it as:

```python
self._llm = get_llm(TaskType.GENERAL, temperature=self._llm_temperature)
```

— one positional argument plus a `temperature=` keyword. The fixture's
`lambda t: MagicMock()` accepts only one positional argument and no
keywords, so the call raised `TypeError: <lambda>() got an unexpected
keyword argument 'temperature'`. That exception is caught by `__init__`'s
own `try/except Exception` (lines 590–600, unrelated production-safe
graceful-degradation logic, not touched by this task), which logs a
warning and leaves `self._agent_available = False`. Every test built on
the `clf` fixture was therefore silently running with the LLM agent
disabled — including `test_llm_used_for_low_similarity`, whose entire
purpose is to exercise the LLM-reranking path (`method` ending in `_llm`),
which can only fire when `self._agent_available is True`. With the LLM
agent always disabled, `classify()` always took the semantic-only
fallback (`method == "flat_semantic"`), so `assert "llm" in result.method`
always failed.

## Exact fixture change

`backend/tests/test_isco_classifier_extended.py` — two changes, both
confined to the test file, no production code touched:

1. **New `get_llm_mock` fixture** (replaces the inline lambda):
   ```python
   @pytest.fixture
   def get_llm_mock():
       return MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
   ```
   Accepts any positional/keyword arguments (matching the real
   `get_llm(TaskType.GENERAL, temperature=...)` call shape) while still
   returning a `MagicMock()`, exactly as the original lambda intended.
   Extracted into its own fixture (rather than inlined in `clf`) so
   `test_llm_used_for_low_similarity` can request it separately and assert
   on the recorded call, without changing what the shared `clf` fixture
   returns (still just an `ISCOClassifier()` instance — every other test
   using `clf` is unaffected).
2. **`clf` fixture** now takes `get_llm_mock` as a parameter and patches
   `get_llm` with it instead of the broken lambda. No other line in `clf`
   changed.
3. **`test_llm_used_for_low_similarity`** now also requests `get_llm_mock`
   and adds a narrowly scoped assertion proving the mock was actually
   called with the `temperature` keyword — direct proof the fix addresses
   the real root cause, not just an incidental pass:
   ```python
   assert get_llm_mock.called
   _, call_kwargs = get_llm_mock.call_args
   assert "temperature" in call_kwargs
   ```

No skip, xfail, exception swallowing, weakened assertion, or hard-coded
result was used anywhere. The original assertion (`assert "llm" in
result.method`) is unchanged and now passes because the intended LLM
re-ranking code path genuinely executes.

## Changed files

**Modified (1 file, 21 insertions / 3 deletions per `git diff --stat`):**

- `backend/tests/test_isco_classifier_extended.py`

No other file was touched — `backend/agents/isco_classifier.py`,
production LLM configuration, thresholds, retrieval logic, classifier
outputs, and every other test file are byte-identical to the base commit.
`git status --short` before staging showed exactly this one file.

## Test commands and exact outputs

```
pytest backend/tests/test_isco_classifier_extended.py -q
→ 80 passed in 0.71s

pytest backend/tests/test_hierarchy_engine.py backend/tests/test_hierarchy_nodes.py backend/tests/test_standard_hierarchical_store.py backend/tests/test_isic_classifier.py backend/tests/test_isced_classifier.py backend/tests/test_method_registry.py -q
→ 131 passed in 1.88s

pytest eval/test_run_eval_b2.py eval/test_dev_sweep.py eval/test_pre_run_check.py eval/test_validate_evaluation_discipline.py eval/test_docs_consistency.py -q
→ 169 passed in 82.57s (0:01:22)

pytest backend/tests eval/ -q
→ 1885 passed, 1 deselected, 1 warning in 295.87s (0:04:55)
```

## Explicit proof the full suite is green

**Zero failed tests.** The full `pytest backend/tests eval/ -q` run
reports `1885 passed, 1 deselected` with no `FAILED` lines anywhere in the
output — the previously-remaining
`backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity`
failure, tracked across every prior task's final report since Task 04.1,
is gone. Pass count went from 1884 (Task 06's integrated state) to 1885
(+1, the same test now passing instead of failing) — confirming the fix
resolved exactly the one known failure and introduced no new one.

## Confirmation: no live network, Qdrant, LLM, model download, dataset, or evaluation operation occurred

- The fix only changes a `unittest.mock.MagicMock` test double's call
  signature — no real LLM, Ollama, or network call is or was ever made by
  this test file; `get_llm`, `Agent`, `Task`, and `Crew` are all mocked,
  exactly as before.
- No `python -m backend.rag.build_standard_hierarchical_collections
  --execute` was run. No Qdrant collection was created or populated.
- No `eval/run_eval.py`, `ablation_runner.py`, or benchmark script was
  invoked. No dataset file, evaluation manifest, or synthetic-data
  evaluation was read or written.
- No official ISIC/ISCED-F catalogue was imported.
- `eval/configs/b1_frozen.json` and B1/B2 configuration were not opened
  for writing (not in `git status`, not in `git diff`) — B1 remains
  quarantined and B2 sweep remains blocked exactly as before.

## Confirmation: protected branches unchanged

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| `reviewer2-wip-snapshot-20260807` | not touched |
| `reviewer2-b2-integration-20260807` | not touched |
| `reviewer2-isic-iscedf-hierarchical-rag-20260808` | not touched |
| `reviewer2-isic-iscedf-hierarchical-resilience-20260808` | not touched |
| `reviewer2-isic-iscedf-hierarchical-model-init-resilience-20260808` | not touched |
| `reviewer2-isic-iscedf-integration-20260808` | not touched (this task branched from it, did not merge back into it) |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push was run at any point in this task.

## Confirmation: working tree

Clean before this task started (verified) and clean again immediately
before this report's commit — `git status --short` showed exactly the one
modified file (`test_isco_classifier_extended.py`) plus this new report,
matching the scope described above with nothing else pending.
