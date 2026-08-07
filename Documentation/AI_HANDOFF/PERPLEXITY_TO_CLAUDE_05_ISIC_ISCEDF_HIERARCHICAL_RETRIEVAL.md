# Task 05: Implement Genuine Hierarchical Retrieval for ISIC and ISCED-F

## Purpose

Implement actual, parent-filtered, multi-stage Qdrant retrieval for:

1. **ISIC Rev.4**: Section → Division → Group → Class
2. **ISCED-F 2013**: Broad field → Narrow field → Detailed field

Reuse the existing generic:

```text
backend/rag/hierarchy_engine.py::HierarchyBeamSearchEngine
```

This is implementation work, not an evaluation or benchmark run. The result must be technically real and callable when collections have been built, but documentation must remain precise: the repository currently embeds only a bounded subset, and no measured ISIC/ISCED-F performance evidence exists.

## Required start state

Start from:

```text
base branch: reviewer2-b2-integration-20260807
expected base HEAD: 2ababd56b1b5e50349e8f7556de4639c4b0bc3fa
new feature branch: reviewer2-isic-iscedf-hierarchical-rag-20260808
```

Before editing:

```bash
git fetch origin
git switch reviewer2-b2-integration-20260807
git status --short
git rev-parse HEAD
```

Stop and report if the tree is not clean or HEAD differs from the expected SHA.

Then create and work only on:

```bash
git switch -c reviewer2-isic-iscedf-hierarchical-rag-20260808
```

## Protected branches

Do not modify, merge into, rebase, reset, clean, stash, pull, or switch to:

```text
master
conference1-b2-evaluation
reviewer2-wip-snapshot-20260807
reviewer2-enhancement
reviewer2-b2-integration-20260807
```

Do not run `git pull`, `git merge`, `git rebase`, `git reset`, `git clean`, or `git stash`.

## Facts that must remain true

- `HierarchyBeamSearchEngine` already implements generic multi-stage parent-filtered beam retrieval. Reuse it; do not copy or fork the beam-search algorithm.
- ISCO-08 remains the only currently measured classification dimension. Do not alter its result or evaluation path.
- Current embedded ISIC corpus is exactly:
  - 21 sections
  - 68 divisions
  - 118 groups
  - 134 classes
- Current embedded ISCED-F corpus is exactly:
  - 11 broad fields
  - 25 narrow fields
  - 63 detailed fields
- The above are **implemented-code counts**, not official-coverage claims. Existing coverage safeguards remain in force; no percentage may be calculated from unverified denominators.
- ISCED 2011 attainment level (0–8) is not a parent-child ISCED-F field hierarchy. It must remain an independent deterministic/rule component.
- No Qdrant collections for these two standards are currently populated. This task creates runtime code and an explicit builder; it does not create or populate a live collection.

## Required architecture

### 1. Add standard-specific hierarchical store support

Add a testable module under `backend/rag/` for standard-specific hierarchy configuration and search. Exact file/class names are flexible, but it must provide separate callable stores/factories equivalent to:

```python
get_isic_hierarchical_store()
get_iscedf_hierarchical_store()
```

The store must:

- construct `HierarchyBeamSearchEngine` with the existing Qdrant client;
- use the E5 convention already established in the project:
  - `"passage: "` for indexing;
  - `"query: "` for search;
  - `intfloat/multilingual-e5-small`, 384 dimensions;
- accept dependency injection of a fake Qdrant client and fake embedder in tests;
- perform an upfront collection-readiness check;
- expose an explicit unavailable/not-ready condition when any required collection is absent;
- never silently report a successful hierarchical result when the collection set is absent;
- return engine-derived code, hierarchy path, stage confidences, top candidates, confidence, and HITL indication when ready.

Use stable, unambiguous collection names:

```text
isic_rev4_sections
isic_rev4_divisions
isic_rev4_groups
isic_rev4_classes

iscedf2013_broad_fields
iscedf2013_narrow_fields
iscedf2013_detailed_fields
```

Use parent payloads:

```text
ISIC:   section root; division.parent_code=section;
        group.parent_code=division; class.parent_code=group
ISCED-F: broad root; narrow.parent_code=broad;
         detailed.parent_code=narrow
```

Use explicit, documented stage weights that sum to 1.0. Suggested values:

```text
ISIC: 0.10 / 0.20 / 0.25 / 0.45
ISCED-F: 0.20 / 0.30 / 0.50
```

You may change these only if the reason is documented and tests validate the actual arithmetic. Do not claim that weights were tuned by measured evaluation.

### 2. Add deterministic catalogue-to-node derivation and validation

Create pure functions that derive unique hierarchy nodes from the existing embedded tables:

```text
backend/agents/isic_classifier.py::_ISIC_DATA
backend/agents/isced_classifier.py::_ISCED_FIELDS
```

Do not add a downloaded official catalogue, scrape a source, change the embedded entries, or claim full coverage.

Derived nodes must carry, at minimum:

```text
code
parent_code (empty or absent only for root nodes)
label_en
label_ar (may be empty if source table has no Arabic text)
description/index_text
```

For internal nodes, construct index text deterministically from that node’s label plus descendant keywords/titles available in the existing bounded table. Document that this is an indexing representation derived from existing embedded records, not an official catalogue expansion.

Fail closed on:

- duplicate code with conflicting parent/title;
- malformed code length/pattern;
- a non-root node whose parent does not exist;
- impossible/empty hierarchy.

### 3. Add an opt-in collection builder

Add an explicit builder module/CLI under `backend/rag/`, for example:

```text
python -m backend.rag.build_standard_hierarchical_collections --standard isic --dry-run
python -m backend.rag.build_standard_hierarchical_collections --standard iscedf --dry-run
```

Requirements:

- `--dry-run` must run node derivation and validation only, print counts/hashes/collection plan, and must not connect to Qdrant, load an embedding model, make a network request, or write data.
- A non-dry run may create/upsert collections only through an explicit opt-in flag; it must fail safely if collections already exist unless an explicit replacement flag is supplied.
- Non-dry mode must be clearly documented as an operator action and must not be executed in this task.
- Do not commit extracted standard catalogues or collection dumps.

### 4. Wire classifiers without breaking default behavior

#### ISIC

Update `ISICClassifier.classify()` so:

```python
classify(text)  # unchanged legacy keyword + optional LLM pipeline
classify(text, method="isic_hierarchical_retrieval")  # real hierarchical mode
```

For explicit hierarchical mode:

- run the ISIC hierarchy store;
- populate section, division, group, class, confidence, alternatives, hierarchy path, stage confidences, and HITL metadata from the real engine result;
- set `method == "isic_hierarchical_retrieval"` when the engine actually runs;
- if collections are unavailable or the hierarchy returns no usable result, preserve operational continuity by using the existing legacy classifier **only with a clearly labeled result**, such as `isic_hierarchical_fallback_keyword` or `isic_hierarchical_fallback_llm`, plus:
  - `fallback_used=True`
  - a nonempty `fallback_reason`
  - empty/appropriate hierarchy trace fields that cannot be mistaken for a completed H-RAG path.
- do not silently label a keyword/LLM result as H-RAG.

#### ISCED-F

Update `ISCEDClassifier.classify()` so:

```python
classify(text)  # unchanged legacy independent level + keyword field pipeline
classify(text, method="iscedf_hierarchical_retrieval")  # real ISCED-F field H-RAG
```

For explicit ISCED-F hierarchical mode:

- retain the current independent `_score_level()` outcome and level title;
- run the ISCED-F hierarchy store only for broad/narrow/detailed field classification;
- set `method == "iscedf_hierarchical_retrieval"` only when the field engine actually runs;
- follow the same explicit fallback labeling/status requirements if collections are unavailable or no path is returned;
- never represent ISCED 2011 levels as a hierarchical retrieval result.

Add backward-compatible optional result metadata with safe defaults to the existing dataclasses as needed:

```text
hierarchy_path
stage_confidences
top_candidates or alternatives
hitl_required
fallback_used
fallback_reason
```

Default calls must retain their current output/method behavior.

### 5. Update method labels, registry, and reviewer documentation

Update:

```text
backend/agents/classifier_methods.py
backend/agents/method_registry.py
Documentation/Conference_I_Reviewer_2/CLASSIFIER_METHOD_REGISTRY.md
Documentation/Conference_I_Reviewer_2/REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md
```

Remove ISIC and ISCED-F hierarchical retrieval from the “not implemented” stub set only after live code paths and tests are complete.

The method registry must state all of the following:

- ISIC and ISCED-F hierarchical retrieval is **implemented in code**, using parent-filtered Qdrant collections and the generic engine.
- It is **not measured/evaluated**.
- Collections have **not been populated/run** by this task.
- It covers only the repository’s embedded records: ISIC 134 classes and ISCED-F 63 detailed fields; neither count is an official coverage percentage.
- ISCED 2011 attainment level remains rule/keyword based and separate from ISCED-F H-RAG.
- Explicit hierarchical fallback is labeled, not silent.

Regenerate the method registry artifacts only if the generator is fully local and deterministic. Do not hand-edit generated artifacts.

Create:

```text
Documentation/Conference_I_Reviewer_2/ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md
```

It must include architecture, collection names, current implemented counts, the no-measurement/no-full-coverage statement, operator-only build commands, and manuscript-safe vs. unsafe wording.

Safe wording after this task:

> “The prototype implements parent-filtered hierarchical retrieval code for ISIC Rev.4 and ISCED-F 2013 using the repository’s currently embedded classification records. Collection population and controlled performance evaluation remain pending.”

Unsafe wording after this task:

- “All ISIC/ISCED codes are covered.”
- Any ISIC/ISCED-F accuracy, latency, cost, or improvement claim.
- “Validated on real LFS data.”
- “ISIC/ISCED-F H-RAG was evaluated” or “was run” unless a future manifest supports it.

## Tests required

All tests must be hermetic: fake Qdrant and fake embedding models only. No live Qdrant, SentenceTransformer download, Ollama, LLM, network, benchmark, or dataset operation.

Add/extend tests covering at least:

1. deterministic ISIC node derivation with exact 21/68/118/134 counts;
2. deterministic ISCED-F node derivation with exact 11/25/63 counts;
3. malformed/conflicting/parent-missing hierarchy records fail closed;
4. real standard-specific stage configuration uses the required collection names and weights sum to 1.0;
5. ISIC path `A → 01 → 011 → 0111` demonstrates actual parent-filtered query calls through a fake Qdrant client;
6. ISCED-F path `06 → 061 → 0613` demonstrates actual parent-filtered query calls;
7. successful explicit classifier methods expose hierarchy path/stage confidences and correct method labels;
8. ISCED-F H-RAG retains independently classified ISCED level;
9. missing collections/no result follows explicitly labeled fallback, never a silent default method;
10. default legacy ISIC/ISCED calls remain behavior-compatible;
11. stub tests and method-registry tests now assert implementation honesty rather than stale “not implemented” behavior;
12. builder `--dry-run` validates without constructing live Qdrant or embedding objects.

Run:

```bash
pytest backend/tests/test_hierarchy_engine.py backend/tests/test_isic_classifier.py backend/tests/test_isced_classifier.py backend/tests/test_method_registry.py -q
pytest backend/tests -q
pytest backend/tests eval/ -q
```

Expected: only the pre-existing unrelated failure may remain:

```text
backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity
```

If any other failure occurs, stop and report it exactly.

## Prohibited actions

Do not:

- execute a live Qdrant collection build, connect to Qdrant, download/load an embedding model, run Ollama/LLM inference, call a network service, run an evaluation/benchmark, or use dataset records;
- import/download/scrape/commit an official ISIC or ISCED-F catalogue;
- alter B1, B2, WISCO, full130, evaluation manifests, frozen outputs, historical results, the paper/manuscript, or protected branches;
- calculate or claim an official coverage percentage;
- use skip/xfail or hard-coded pass logic to hide an error;
- modify the generic engine’s core behavior or ISCO’s current production behavior.

## Commit and final report

Commit and push only:

```text
reviewer2-isic-iscedf-hierarchical-rag-20260808
```

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_05_FINAL_REPORT.md
```

Report:

- base and final SHA;
- changed files;
- exact ISIC and ISCED-F hierarchy counts;
- collection names, stage weights, and explicit fallback semantics;
- proof that real parent-filtered query paths are covered by hermetic tests;
- test commands/results;
- confirmation no live collection build, inference, network, benchmark, or dataset operation occurred;
- confirmation protected branches were unchanged;
- remaining known test failure;
- precisely what is and is not now manuscript-safe.

Stop after the report is committed and pushed.
