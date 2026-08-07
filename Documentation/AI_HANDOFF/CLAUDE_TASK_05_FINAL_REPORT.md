# Task 05 Final Report — Genuine Hierarchical Retrieval for ISIC and ISCED-F

Produced in response to
`Documentation/AI_HANDOFF/PERPLEXITY_TO_CLAUDE_05_ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL.md`
(task ID `05-isic-iscedf-hierarchical-retrieval`), executed on a new branch
per the task's explicit instruction.

## Branch / SHA

| | |
|---|---|
| Base branch | `reviewer2-b2-integration-20260807` |
| Base SHA | `2ababd56b1b5e50349e8f7556de4639c4b0bc3fa` |
| Working branch | `reviewer2-isic-iscedf-hierarchical-rag-20260808` (new, created by this task) |
| Final SHA | see the commit this report ships in — this file is committed together with all code/doc changes below, as the task's own closing step |

Start-state check passed exactly: `git switch reviewer2-b2-integration-20260807`
was clean and at the expected base SHA before `git switch -c
reviewer2-isic-iscedf-hierarchical-rag-20260808` created the working branch.

## What was built

Genuine, parent-filtered hierarchical retrieval for ISIC Rev.4 and
ISCED-F 2013, reusing the existing generic
`backend/rag/hierarchy_engine.py::HierarchyBeamSearchEngine` verbatim (no
new search algorithm). Full architecture, collection names, stage weights,
implemented node counts, explicit fallback semantics, operator build
commands, and the required safe/unsafe manuscript wording are documented
in **[ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md](../Conference_I_Reviewer_2/ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md)**
— this report summarizes and points there rather than duplicating it.

### New files

- `backend/rag/hierarchy_nodes.py` — deterministic node derivation from
  `_ISIC_DATA` / `_ISCED_FIELDS`, fail-closed validation.
- `backend/rag/standard_hierarchical_store.py` — `StandardHierarchicalStore`,
  `get_isic_hierarchical_store()` / `get_iscedf_hierarchical_store()`.
- `backend/rag/build_standard_hierarchical_collections.py` — operator-only
  CLI (`--dry-run` / `--execute --recreate`).
- `Documentation/Conference_I_Reviewer_2/ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md`
- `backend/tests/test_hierarchy_nodes.py`,
  `backend/tests/test_standard_hierarchical_store.py`,
  `backend/tests/test_build_standard_hierarchical_collections.py`

### Changed files

- `backend/agents/classifier_methods.py` — removed the now-obsolete
  `NOT_IMPLEMENTED_METHODS` / `NOT_IMPLEMENTED_REASON` stub machinery
  (both entries it ever held are now genuinely implemented); added
  `ISIC_HIERARCHICAL_FALLBACK_KEYWORD`, `ISIC_HIERARCHICAL_FALLBACK_LLM`,
  `ISCEDF_HIERARCHICAL_FALLBACK_KEYWORD`.
- `backend/agents/isic_classifier.py` — `classify()` split into
  `_classify_legacy()` (byte-for-byte the old default pipeline) and
  `_classify_hierarchical()` (new); `ISICClassification` gained
  `hierarchy_path`, `stage_confidences`, `hitl_required`, `fallback_used`,
  `fallback_reason` (all safe-defaulted); removed the now-dead
  `_not_implemented()` stub method; added per-level title lookup dicts
  built from `_ISIC_DATA` (`_SECTION_TITLES`/`_DIVISION_TITLES`/
  `_GROUP_TITLES`/`_ENTRY_BY_CLASS`).
- `backend/agents/isced_classifier.py` — same split
  (`_classify_legacy()`/`_classify_hierarchical()`); `ISCEDClassification`
  gained the same new safe-defaulted fields plus `top_candidates`; removed
  `_not_implemented()`; added `_BROAD_TITLES`/`_NARROW_TITLES` lookups.
  ISCED 2011 level is computed by the unchanged, independent
  `_score_level()` on every path, including the hierarchical one.
- `backend/agents/method_registry.py` — the two existing
  `isic_hierarchical_retrieval` / `iscedf_hierarchical_retrieval` rows
  rewritten from "NOT YET IMPLEMENTED" stubs to real-but-unevaluated
  entries (real collection names/weights/embedding model in
  `output_schema`/`decoding_config`, `evaluated=False` preserved,
  `affects_hitl_escalation=False` preserved, fallback labels documented in
  `fallback_behaviour` text).
- `backend/tests/test_isic_classifier.py`, `test_isced_classifier.py`,
  `test_method_registry.py` — the old "returns structured not-implemented"
  stub tests replaced with hermetic tests against the real hierarchical
  path (FakeQdrantClient/FakeEmbedder dependency-injected via
  monkeypatching `get_isic_hierarchical_store`/`get_iscedf_hierarchical_store`
  at the point `_classify_hierarchical()` imports them); default-path
  (`method=None`) regression tests kept and passing unchanged.
- `eval/figure_exports/export_agent_role_diagram.py`,
  `export_classifier_hierarchy.py`, `eval/test_figure_exports.py` — these
  Section I exporters read `backend.agents.classifier_methods` /
  `method_registry.REGISTRY` directly and would have broken (import error /
  stale "not yet implemented" diagram data) once the stub constants were
  removed and the registry rows changed; updated to reflect the real
  implementation status (`is_implemented=True` for all rows now;
  `IMPLEMENTED_UNEVALUATED_STAGES` replaces the old `PLANNED_STAGES` with
  real collection names/weights and an `"implemented_unevaluated"` status,
  distinct from ISCO's evaluated `"implemented"` status). Regenerated
  `Documentation/Conference_I_Reviewer_2/generated/classifier_method_registry.{json,md}`
  and `generated/figure_data/{agent_role_diagram,classifier_hierarchy}.*`
  via their documented CLI commands (never hand-edited).
- `Documentation/Conference_I_Reviewer_2/CLASSIFIER_METHOD_REGISTRY.md`,
  `README.md` — updated the stale "stub"/"not yet implemented"/"planned"
  language to describe the real, unevaluated implementation and link the
  new implementation doc.

## Exact hierarchy counts (verified against the live embedded tables)

| Standard | Level | Count |
|---|---|---|
| ISIC Rev.4 | sections | 21 |
| ISIC Rev.4 | divisions | 68 |
| ISIC Rev.4 | groups | 118 |
| ISIC Rev.4 | classes | 134 |
| ISCED-F 2013 | broad fields | 11 |
| ISCED-F 2013 | narrow fields | 25 |
| ISCED-F 2013 | detailed fields | 63 |

Verified by `backend/tests/test_hierarchy_nodes.py`'s exact-count assertions
and independently by `backend/rag/build_standard_hierarchical_collections.py
--dry-run` for both standards. These are **not** an official-catalogue
coverage claim — see the implementation doc's coverage section.

## Collection names / stage weights / explicit fallback labels

See the "Collection names", "Stage weights", and "Explicit fallback
semantics" sections of
[ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md](../Conference_I_Reviewer_2/ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md).
In short: `isic_rev4_{sections,divisions,groups,classes}` (weights
0.10/0.20/0.25/0.45) and `iscedf2013_{broad,narrow,detailed}_fields`
(weights 0.20/0.30/0.50); fallback labels
`isic_hierarchical_fallback_{keyword,llm}` and
`iscedf_hierarchical_fallback_keyword`.

## Proof that real parent-filtered query paths are covered by hermetic tests

`backend/tests/test_standard_hierarchical_store.py::test_isic_search_follows_real_parent_filtered_path`
and `::test_iscedf_search_follows_real_parent_filtered_path` assert the
exact `(collection, parent_code, limit)` call sequence a `FakeQdrantClient`
recorded — `A -> 01 -> 011 -> 0111` for ISIC and `06 -> 061 -> 0613` for
ISCED-F — proving each stage's query was genuinely filtered by the
previous stage's chosen code, not a flat lookup.
`backend/tests/test_isic_classifier.py::test_hierarchical_retrieval_runs_real_parent_filtered_search`
and the ISCED-F equivalent repeat this proof through the actual
`ISICClassifier.classify()` / `ISCEDClassifier.classify()` entry points
(via `monkeypatch` on `get_isic_hierarchical_store` /
`get_iscedf_hierarchical_store`, resolved at call time since both
classifiers import the store module locally, inside the hierarchical
branch only).

## Test commands and results

```
pytest backend/tests/test_hierarchy_engine.py backend/tests/test_isic_classifier.py backend/tests/test_isced_classifier.py backend/tests/test_method_registry.py -q
→ 93 passed in 0.35s

pytest backend/tests -q
→ 1 failed, 1314 passed, 1 deselected in 225.22s
  FAILED backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity

pytest backend/tests eval/ -q
→ 1 failed, 1871 passed, 1 deselected in 301.77s
  FAILED backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity
```

Only the known pre-existing, unrelated failure remains (a CrewAI LLM mock
signature mismatch in an ISCO-08 test, present before this task and
untouched by it — same failure identified in Task 04.1's final report).
No other failure, no skip/xfail, no hard-coded pass was introduced.

## Confirmation: no live Qdrant/embedding/network/benchmark/dataset operation

- No `QdrantClient(...)` was ever constructed against a real host during
  this task's own test runs or code execution — every hermetic test
  dependency-injects a `FakeQdrantClient`/fake embedder.
  `backend/rag/build_standard_hierarchical_collections.py --dry-run` was
  run directly (twice, once per standard) and its own output confirms "No
  Qdrant connection, embedding model load, or network request was made" —
  verified by `test_execute_run_not_imported_at_module_load_time`, which
  asserts `QdrantClient`/`SentenceTransformer` are not present in the CLI
  module's namespace after import (they're only imported lazily inside
  `execute_run()`, never called by this task).
- `--execute` was never invoked. No live Qdrant collection was created or
  populated by this task.
- No Ollama/LLM inference call was made — the ISIC LLM re-ranking path
  (`_llm_rerank`) is unaffected by this task's changes and was not
  exercised beyond its pre-existing mocked tests.
- No `eval/run_eval.py`, `ablation_runner.py`, or benchmark script was
  invoked. No dataset file (`eval/test_set_full130.csv`, WISCO records,
  `eval/local_benchmarks/`, `eval/local_runs/`) was read or written.
- The `python -m backend.agents.method_registry` and
  `python -m eval.figure_exports.export_*` commands run to regenerate the
  `generated/` documentation artifacts read only in-process Python
  constants (`REGISTRY`, `ISIC_STAGE_WEIGHTS`, etc.) — no network call, no
  Qdrant connection, no dataset access. `sentence_transformers`/
  `qdrant_client` were imported as Python packages (an existing,
  unavoidable module-level import in `standard_hierarchical_store.py`,
  matching `hierarchical_store.py`'s pre-existing pattern) but no client
  was ever constructed or connected during this step.

## Confirmation: protected branches unchanged

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| `reviewer2-wip-snapshot-20260807` | not touched |
| `reviewer2-enhancement` | fetched read-only only, to read the task-handoff file |
| `reviewer2-b2-integration-20260807` | not touched (this task branched from it, did not merge back into it) |

No `git reset`, `git clean`, `git stash`, `git rebase`, `git pull --no-ff`,
`git merge`, or force-push was run at any point in this task. No B1/B2/
WISCO/full130 evaluation manifest, frozen output, or historical result was
modified. No official ISIC/ISCED-F catalogue was imported, downloaded, or
scraped. No coverage percentage was calculated or claimed.

## What is, and is not, manuscript-safe right now

See the implementation doc's closing section for the exact required
wording. In short: it is now safe to say the prototype implements real
parent-filtered hierarchical retrieval *code* for both standards over the
repository's currently embedded records; it is **not** safe to claim full
official-catalogue coverage, any accuracy/latency/cost/improvement number,
validation on real LFS data, or that ISIC/ISCED-F hierarchical retrieval
"was evaluated" or "was run" — none of that has happened yet.
