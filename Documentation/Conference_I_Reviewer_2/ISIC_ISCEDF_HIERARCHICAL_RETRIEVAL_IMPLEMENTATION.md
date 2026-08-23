# ISIC Rev.4 / ISCED-F 2013 Hierarchical Retrieval — Implementation

Task 05 of the Conference I Reviewer #2 response. Implements genuine,
parent-filtered hierarchical retrieval for ISIC Rev.4 (industry) and
ISCED-F 2013 (education field), reusing the same generic beam-search engine
ISCO-08's hierarchical RAG already uses. This document is the authoritative
description of what exists, what doesn't, and the exact wording that is and
is not safe to use in the manuscript.

## Architecture

Both standards are built on the pre-existing, generic
`backend/rag/hierarchy_engine.py::HierarchyBeamSearchEngine` — the same
engine `HierarchicalISCOStore` (ISCO-08) uses. No new search algorithm was
written; only new, standard-specific configuration and wiring:

```
backend/rag/hierarchy_nodes.py
    Deterministic derivation of per-level indexable nodes from the
    existing embedded classifier tables (_ISIC_DATA / _ISCED_FIELDS).
    Fails closed (HierarchyValidationError) on a malformed code, a
    duplicate code with conflicting parent/title, a missing parent, or
    an empty level. No Qdrant/embedding/network dependency.

backend/rag/standard_hierarchical_store.py
    StandardHierarchicalStore -- one instance per standard, parameterized
    by StageConfig list + stage weights. get_isic_hierarchical_store() /
    get_iscedf_hierarchical_store() are lazily-constructed, cached
    singletons in production; both accept optional client=/embedder= for
    dependency injection in tests (never touches/warms the production
    singleton when either is passed). search() has an explicit
    ready / unavailable_reason contract -- it never fabricates or
    silently reports a completed result when the required collections are
    absent, or when a query genuinely finds nothing.

backend/rag/build_standard_hierarchical_collections.py
    Operator-only CLI. --dry-run runs node derivation + validation only
    (no Qdrant/embedding-model/network dependency -- verified by test).
    --execute is the separate, explicit, destructive action that actually
    creates and populates the live Qdrant collections. Run for real on
    2026-08-23 for both standards (see "Operator build commands" below) --
    the 7 collections listed under "Collection names" are now live and
    populated; classify(method=<hierarchical constant>) has been confirmed
    to genuinely fire against them (fallback_used=False on a real query).

backend/agents/isic_classifier.py / isced_classifier.py
    classify(text, method="isic_hierarchical_retrieval" /
    "iscedf_hierarchical_retrieval") runs the real store. classify(text)
    (no method=) is byte-for-byte unchanged legacy behaviour.
```

## Embedding convention

Identical to ISCO-08's own store: `intfloat/multilingual-e5-small`
(384-dim), `"query: "` prefix at search time, `"passage: "` prefix at
index time (only used by the operator-only `--execute` path).

## Collection names

| Standard | Level | Collection |
|---|---|---|
| ISIC Rev.4 | section | `isic_rev4_sections` |
| ISIC Rev.4 | division | `isic_rev4_divisions` |
| ISIC Rev.4 | group | `isic_rev4_groups` |
| ISIC Rev.4 | class | `isic_rev4_classes` |
| ISCED-F 2013 | broad field | `iscedf2013_broad_fields` |
| ISCED-F 2013 | narrow field | `iscedf2013_narrow_fields` |
| ISCED-F 2013 | detailed field | `iscedf2013_detailed_fields` |

## Stage weights

Engineering defaults (later stages weighted higher, following ISCO's own
shape), **not tuned by any measured evaluation** — no ISIC/ISCED-F accuracy
evidence exists yet. Do not describe these as tuned/optimal in any
manuscript-facing text.

- ISIC Rev.4: section 0.10 / division 0.20 / group 0.25 / class 0.45
- ISCED-F 2013: broad 0.20 / narrow 0.30 / detailed 0.50

## Implemented node counts (this repository's currently embedded records)

| Standard | Level | Count |
|---|---|---|
| ISIC Rev.4 | sections | 21 |
| ISIC Rev.4 | divisions | 68 |
| ISIC Rev.4 | groups | 118 |
| ISIC Rev.4 | classes | 134 |
| ISCED-F 2013 | broad fields | 11 |
| ISCED-F 2013 | narrow fields | 25 |
| ISCED-F 2013 | detailed fields | 63 |

**These counts are NOT an official-catalogue coverage claim.** They reflect
exactly what `backend/agents/isic_classifier.py::_ISIC_DATA` and
`backend/agents/isced_classifier.py::_ISCED_FIELDS` already contain — the
same tables the pre-existing keyword classifiers use — deterministically
re-derived into a per-level, Qdrant-indexable shape. No official ISIC
Rev.4 or ISCED-F 2013 publication was imported, scraped, or compared
against. See `COVERAGE_AUDIT_GUIDE.md` / `STANDARDS_SOURCE_PROVENANCE.md`
for how a real, citable coverage percentage would be produced.

## Explicit fallback semantics

`classify(text, method=<hierarchical constant>)` only ever reports the
hierarchical-retrieval method id (`isic_hierarchical_retrieval` /
`iscedf_hierarchical_retrieval`) when the engine actually ran and returned
a usable result (`ready=True`, `unavailable_reason==""`, a non-empty
`code`). In every other case — required collections missing, Qdrant
unreachable or erroring at readiness-check time (Task 05.1), an embedding
failure (including the embedding *model itself* failing to initialize —
Task 05.2, see below), an engine-search failure, or the search returning
nothing — it falls back to the existing legacy keyword/LLM (ISIC) or
keyword/rule (ISCED-F) pipeline and reports one of:

- `isic_hierarchical_fallback_keyword` / `isic_hierarchical_fallback_llm`
- `iscedf_hierarchical_fallback_keyword`

with `fallback_used=True` and a non-empty `fallback_reason`. All of these
operational failure modes are caught narrowly at their external-call
boundary in `StandardHierarchicalStore` (the Qdrant readiness check, the
embedding call — which now also covers lazy model construction — and the
engine-search call) — never as a blanket catch around unrelated classifier
logic — and fold into the same `ready=False` / non-empty
`unavailable_reason` contract the classifier already branches on, so no
separate handling was needed in the classifiers themselves. The
hierarchical-trace fields (`hierarchy_path`, `stage_confidences`,
`top_candidates`) are left at their empty defaults on every fallback
result, so a fallback can never be mistaken for a completed hierarchical
retrieval path. ISCED 2011 attainment **level** is always computed by the
independent `_score_level()` scorer regardless of which path runs — it is
never part of the ISCED-F hierarchical retrieval and is never zeroed or
blanked by a fallback.

## Operator build commands (--execute run 2026-08-23)

```bash
# Safe, offline, no Qdrant/embedding-model/network dependency:
python -m backend.rag.build_standard_hierarchical_collections --standard isic --dry-run
python -m backend.rag.build_standard_hierarchical_collections --standard iscedf --dry-run

# Operator-only, live Qdrant write -- creates and populates the real
# collections listed above. Requires a running Qdrant instance and
# downloads the embedding model if not already cached. Run for real
# 2026-08-23: isic -> 341 nodes (21/68/118/134 sections/divisions/
# groups/classes); iscedf -> 99 nodes (11/25/63 broad/narrow/detailed
# fields). Both confirmed live via a direct classify(method=...) call.
python -m backend.rag.build_standard_hierarchical_collections --standard isic --execute
python -m backend.rag.build_standard_hierarchical_collections --standard iscedf --execute
```

## Hermetic test coverage

`backend/tests/test_hierarchy_nodes.py`, `test_standard_hierarchical_store.py`,
`test_build_standard_hierarchical_collections.py`, and the extended
`test_isic_classifier.py` / `test_isced_classifier.py` / `test_method_registry.py`
cover, entirely offline (FakeQdrantClient/FakeEmbedder, no live Qdrant, no
embedding-model load, no network call):

1. Deterministic node derivation with the exact counts above (ISIC and
   ISCED-F).
2. Fail-closed validation on malformed codes, duplicate/conflicting codes,
   and missing parents.
3. Stage configuration uses the required collection names and weights that
   sum to 1.0.
4. A real parent-filtered query chain for ISIC (`A -> 01 -> 011 -> 0111`)
   and ISCED-F (`06 -> 061 -> 0613`), verified against the fake client's
   recorded `(collection, parent_code, limit)` call log.
5. Successful `classify(method=...)` calls expose `hierarchy_path`,
   `stage_confidences`, and the correct method label; ISCED-F retains its
   independently classified level.
6. Missing collections and zero-hit searches both produce an explicit,
   correctly-labeled fallback — never a silent default or a mislabeled
   hierarchical result.
7. Default `classify(text)` (no `method=`) is unchanged for both
   classifiers.
8. The collection-builder CLI's `--dry-run` path never imports
   `qdrant_client`/`sentence_transformers` classes into scope and produces
   deterministic, well-formed output.
9. (Task 05.1) A Qdrant readiness-check failure (e.g. connection error)
   yields an explicit unavailable result, never runs a hierarchy query,
   and never loads the embedding model; both classifiers fall back with
   the correct explicit label in this condition, and ISCED-F still
   reports its independently classified level. An embedding failure and
   an engine-search failure are each surfaced the same way — an explicit,
   non-fabricated `unavailable_reason`, never a raised exception or a
   silently fabricated result.
10. (Task 05.2) `SentenceTransformer(MODEL_NAME)` construction is proven
    lazy (never attempted in `__init__`, only on first real `search()`
    use) and proven covered by the same protected boundary as query
    encoding: a construction failure yields the same explicit,
    embedding-related `unavailable_reason` and correct classifier fallback
    label as any other embedding failure, for both ISIC and ISCED-F (with
    ISCED-F's level still independently classified); an injected fake
    embedder is proven to bypass model construction entirely, even when
    the real constructor is made to always fail.

## What is, and is not, manuscript-safe right now

**Safe wording:**

> "The prototype implements parent-filtered hierarchical retrieval for
> ISIC Rev.4 and ISCED-F 2013 using the repository's currently embedded
> classification records. The live Qdrant collections were populated and
> confirmed to serve real queries on 2026-08-23; controlled performance
> evaluation against a labelled test set remains pending."

**Unsafe wording — must NOT appear as a claim:**

- "All ISIC/ISCED codes are covered."
- Any ISIC/ISCED-F accuracy, latency, cost, or improvement claim.
- "Validated on real LFS data."
- "ISIC/ISCED-F H-RAG was evaluated" or "was run" — unless a future
  manifest supports it.
