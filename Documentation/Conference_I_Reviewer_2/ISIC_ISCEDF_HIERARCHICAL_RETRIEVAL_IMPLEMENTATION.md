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

Default profile (`e5_small`) is identical to ISCO-08's own store:
`intfloat/multilingual-e5-small` (384-dim), `"query: "` prefix at search
time, `"passage: "` prefix at index time (only used by the operator-only
`--execute` path).

**Added 2026-08-25 — a second, additive `e5_large` profile**, mirroring
ISCO-08's own `E5LARGE_PROFILE` pattern exactly (see `CLAUDE.md`'s
"Environment & deployment" section for the ISCO-08 e5-large story this
follows). `backend/rag/standard_hierarchical_store.py` gained
`PROFILE_MODEL_CONFIG` (`e5_small` → `intfloat/multilingual-e5-small`/384,
`e5_large` → `intfloat/multilingual-e5-large`/1024),
`ISIC_COLLECTIONS_BY_PROFILE` / `ISCEDF_COLLECTIONS_BY_PROFILE`, and a
`profile=` parameter threaded through `isic_stages()` / `iscedf_stages()`
and `get_isic_hierarchical_store()` / `get_iscedf_hierarchical_store()`.
`ISIC_COLLECTIONS` / `ISCEDF_COLLECTIONS` (no profile suffix) remain exact
aliases of the `e5_small` profile — every existing caller, including
`ISICClassifier`/`ISCEDClassifier`'s own `_classify_hierarchical()` (which
calls the factories with no `profile=` argument), is byte-for-byte
unaffected. **The classifiers were not changed and do not use the
e5_large profile in production** — this is infrastructure parity with
ISCO-08, not a production switch, same discipline as every ISCO-08
profile addition. 26 new tests
(`backend/tests/test_standard_hierarchical_store_e5large_profile.py` +
2 in `test_build_standard_hierarchical_collections.py`).

**Real, disclosed limitation this profile addition does NOT resolve**:
unlike ISCO-08, there is still no labelled evaluation set for ISIC/ISCED-F
(see "What is, and is not, manuscript-safe right now" below) — so, unlike
ISCO-08's e5-large result (+8.50pp at full 18,747-case scale), there is no
way to measure whether `e5_large` actually improves ISIC/ISCED-F accuracy.
This profile exists so that question is answerable the moment a real
evaluation set exists — it does not answer it today. Live-verified only
in the same limited sense the e5-small collections were on 2026-08-23:
`get_isic_hierarchical_store(profile="e5_large").search(...)` /
`get_iscedf_hierarchical_store(profile="e5_large").search(...)` against
real query text return `ready=True`, `unavailable_reason==""`, and a
non-empty code — proof the collections are live and serving, not an
accuracy claim.

## Collection names

| Standard | Level | `e5_small` (default) | `e5_large` (additive, 2026-08-25) |
|---|---|---|---|
| ISIC Rev.4 | section | `isic_rev4_sections` | `isic_rev4_sections_e5large` |
| ISIC Rev.4 | division | `isic_rev4_divisions` | `isic_rev4_divisions_e5large` |
| ISIC Rev.4 | group | `isic_rev4_groups` | `isic_rev4_groups_e5large` |
| ISIC Rev.4 | class | `isic_rev4_classes` | `isic_rev4_classes_e5large` |
| ISCED-F 2013 | broad field | `iscedf2013_broad_fields` | `iscedf2013_broad_fields_e5large` |
| ISCED-F 2013 | narrow field | `iscedf2013_narrow_fields` | `iscedf2013_narrow_fields_e5large` |
| ISCED-F 2013 | detailed field | `iscedf2013_detailed_fields` | `iscedf2013_detailed_fields_e5large` |

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

# Additive e5_large profile, run for real 2026-08-25 (see "Embedding
# convention" above): same node counts as e5_small (341 / 99), separate
# _e5large-suffixed collections, e5_small collections untouched:
python -m backend.rag.build_standard_hierarchical_collections --standard isic --profile e5_large --execute
python -m backend.rag.build_standard_hierarchical_collections --standard iscedf --profile e5_large --execute
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

## Flat retrieval — ISCO-08 best-tested-config parity (added 2026-08-25)

Prompted directly by a request to give ISIC/ISCED-F "the same
implementation ISCO-08 uses" as the project's novel-contribution
architecture. Checked what that actually means before building anything:
ISCO-08's own **best-tested, headline configuration is FLAT retrieval**
(40.95% — flat + rich catalogue text + `multilingual-e5-large`), not
hierarchical — ISCO-08's hierarchical retrieval measurably
**underperformed** its own flat retrieval (10.35% vs 21.19% at baseline,
McNemar p≈1.86×10⁻³⁰¹, see `CLAUDE.md`). So the hierarchical-only
implementation this document described until now, on its own, would NOT
have actually mirrored ISCO-08's real, best implementation — it mirrored
ISCO-08's *worse* one. This section closes that gap: ISIC Rev.4 and
ISCED-F 2013 now also have a flat retrieval path, architecturally
identical to ISCO-08's own.

**What "flat" means here, concretely**: a single direct Qdrant query
against one leaf-level collection (ISIC's 134 classes / ISCED-F's 63
detailed fields), no parent-chain beam traversal — reusing the exact
same leaf-level derived nodes (`hierarchy_nodes.py`'s "classes"/
"detailed_fields" `index_text`, already keyword-rich, same richness
ISCO-08's catalogue only gained after its 2026-08-24 enrichment fix) as
the hierarchical build's own final stage, just written into a
separately-named collection. This mirrors ISCO-08's own architecture
exactly: its "flat" and hierarchical "unit" collections are built from
identical source records (`official_isco08_catalogue.py`'s
`_target_builds()`), just queried differently.

**New code, all additive, zero change to any existing collection or
default `classify()` behaviour**:

- `backend/rag/standard_hierarchical_store.py`: `ISIC_FLAT_COLLECTIONS_BY_PROFILE`
  / `ISCEDF_FLAT_COLLECTIONS_BY_PROFILE`, `StandardFlatStore` (same
  ready/`unavailable_reason` contract as `StandardHierarchicalStore`, but
  a single direct query — deliberately NOT built on
  `HierarchyBeamSearchEngine`, which requires ≥2 stages by its own
  module docstring), `get_isic_flat_store()` / `get_iscedf_flat_store()`
  factories (same DI/singleton-per-profile contract as the hierarchical
  factories).
- `backend/rag/build_standard_hierarchical_collections.py`: new `--flat`
  flag, `dry_run_flat()` / `execute_run_flat()`, reusing the same
  leaf-level node derivation (no new node-derivation logic).
- `backend/agents/classifier_methods.py`: `ISIC_FLAT_RETRIEVAL` /
  `ISCEDF_FLAT_RETRIEVAL` + fallback labels, same explicit-fallback
  contract as the hierarchical constants.
- `backend/agents/isic_classifier.py` / `isced_classifier.py`:
  `classify(text, method=ISIC_FLAT_RETRIEVAL / ISCEDF_FLAT_RETRIEVAL)` —
  **hardcoded to `profile="e5_large"`** internally (not a caller-facing
  choice), because that is specifically the recipe being mirrored — flat
  + e5-large, ISCO-08's actual winning combination, not just "flat with
  whatever the default embedding model happens to be." `classify(text)`
  (no `method=`) is byte-for-byte unchanged. Since the flat store's
  result only ever carries the single leaf code (no parent chain was
  traversed), the full section/division/group (ISIC) or broad/narrow
  (ISCED-F) ancestry is resolved via `_ENTRY_BY_CLASS` /
  `_ENTRY_BY_DETAILED` (new small lookup dicts, same pattern already
  used for `alternatives`). ISCED 2011 attainment level remains
  independently classified either way, same as the hierarchical path.

**Live-built and verified, 2026-08-25** (`e5_large` profile only — see
"What was deliberately not built" below):

| Standard | Collection | Nodes |
|---|---|---|
| ISIC Rev.4 | `isic_rev4_classes_flat_e5large` | 134 |
| ISCED-F 2013 | `iscedf2013_detailed_fields_flat_e5large` | 63 |

34 new tests across `test_standard_flat_store.py`,
`test_isic_classifier.py`, `test_isced_classifier.py`, and
`test_build_standard_hierarchical_collections.py`; full suite re-run
with zero regressions.

**What was deliberately not built**: the `e5_small` flat collections
(`isic_rev4_classes_flat`, `iscedf2013_detailed_fields_flat`) exist in
code (`ISIC_FLAT_COLLECTIONS_BY_PROFILE["e5_small"]` etc., and
`--flat --profile e5_small --execute` builds them) but were not run live
— nothing calls that profile today, since `ISIC_FLAT_RETRIEVAL`/
`ISCEDF_FLAT_RETRIEVAL` always request `e5_large` specifically (the
actual best-tested recipe). Building an unused collection live would
have been effort spent proving nothing.

**What this still does NOT resolve — same limitation as the
hierarchical-only version of this document, restated because it applies
here identically**: whether flat retrieval is actually more accurate
than the keyword/LLM pipeline, for ISIC or ISCED-F specifically, is
**untested**. No labelled evaluation dataset exists for either standard
(see "Why this remains the one gap infrastructure cannot close" above).
Giving ISIC/ISCED-F the SAME IMPLEMENTATION as ISCO-08 is a real,
completed architecture-parity fix — it directly answers "why does the
architecture diagram show ISCO-08 with a best-tested RAG configuration
and ISIC/ISCED-F with only 'keyword match'," which is a legitimate
manuscript-consistency concern. It is not, and cannot yet be, an
accuracy claim.

## Real official-source enrichment + a genuine magnet-effect regression (2026-08-25)

Prompted by a direct instruction to double-check the "already rich, no fix
needed" claim in the section above, before treating it as settled. Verified
rather than re-asserted: measured `_ISIC_DATA`/`_ISCED_FIELDS`'s keyword
text objectively (mean 11.4 / 9.7 words per entry) against what ISCO-08's
own real, official, POST-enrichment text actually looks like (50-150+ word
prose definitions + real example lists, straight from the official ILO
workbook) — not against ISCO-08's ORIGINAL bug state, which is what the
earlier comparison had actually done. The richness gap was real.

Checked something that hadn't been checked before dismissing the idea:
does an equivalent official primary source even exist for ISIC/ISCED-F?
**Yes.** Two real, public documents:

- UN Statistics Division, *International Standard Industrial
  Classification of All Economic Activities (ISIC), Revision 4* —
  the official structure-and-explanatory-notes publication.
- UNESCO Institute for Statistics, *International Standard Classification
  of Education: Fields of education and training 2013 (ISCED-F 2013) —
  Detailed field descriptions* (2015).

Both downloaded into `eval/local_catalogues/isic_rev4_2008/` and
`eval/local_catalogues/iscedf_2013/`. A reproducible parser,
`eval/parse_official_isic_iscedf_definitions.py` (uses `pdfplumber`,
already a project dependency), extracts real per-code definitions and
examples from each PDF's narrative "Detailed structure" section — 419
ISIC classes (the full standard) and 92 ISCED-F fields parsed correctly,
verified by direct comparison against the raw extracted PDF text, not
just a handful of spot checks.

### A second, larger, unplanned finding

Cross-checking the already-embedded `_ISIC_DATA`/`_ISCED_FIELDS` codes
against the real official document surfaced **13 ISIC codes and 2
ISCED-F codes that do not exist in the official standard at all**:

| Catalogue's code/title | Real official code/title |
|---|---|
| ISIC `7311` "Advertising agencies" | `7310` "Advertising" |
| ISIC `7430` "Translation and interpretation activities" | no dedicated class — folded elsewhere |
| ISIC `7739` "Renting and leasing of other machinery..." | `7730` "Renting and leasing of other machinery..." |
| ISIC `8424` "Public order and safety activities" | `8423` "Public order and safety activities" |
| ISIC `8531`/`8559`/`8560` (education) | no exact match found |
| ISIC `8621`/`8622`/`8623` (medical practice) | no exact match found |
| ISIC `8899` "Other social work activities without accommodation n.e.c." | `8890` "Other social work activities without accommodation" |
| ISIC `9001` "Performing arts" / `9003` "Artistic creation" | both fold into the single real class `9000` "Creative, arts and entertainment activities" |
| ISCED-F `0224` "History, philosophy and related subjects" | no exact match found |
| ISCED-F `0919` "Health (not elsewhere classified)" | no exact match found |

This is the same class of bug Task 20/21's primary-source audit already
found and fixed in the ISCO-08 catalogue (19 non-standard codes there) —
genuinely new here, previously unknown, and **not fixed in this pass**:
determining the correct replacement code for each (and any downstream
consistency implications) is a distinct, careful task, out of scope for a
text-enrichment change. Disclosed precisely, not silently absorbed, via
`backend/rag/official_source_enrichment.py`'s `NON_STANDARD_ISIC_CODES` /
`NON_STANDARD_ISCEDF_CODES` frozensets.

### The enrichment itself

`backend/rag/official_source_enrichment.py::build_enriched_text()` builds
`"{code} {title}. {definition} Examples: {examples}."` for every code with
a real official match, and falls back to the existing
`"{code} {title} {keywords}"` text (unchanged) for the 15 non-standard
codes above — never fabricated. Wired into a new `"enriched_e5large"`
profile on the **flat** collections only (the hierarchical collections
were left at plain `e5_large` — see below for why). `ISIC_FLAT_RETRIEVAL`
/ `ISCEDF_FLAT_RETRIEVAL` now use this profile, changed from the plain
`e5_large` profile the prior section built — this is genuinely the same
implementation ISCO-08 uses (flat + real enriched text + e5-large), not
merely the same algorithm family.

### A real regression, caught by live-testing before declaring this done

The first live build (all 134/63 codes included, the 15 non-standard ones
kept with their thin fallback text) was tested against a 15-query smoke
set spanning diverse occupations before being written up as finished —
and failed: **"I build mobile apps at a software company" and
"construction labourer on a residential building site" both wrongly
matched non-standard code `8899`** instead of their real codes (`6201`,
`4100`). 3 of the 13 non-standard ISIC codes were actively capturing
unrelated queries. This is the exact same "magnet effect" mechanism found
in ISCO-08's own pre-enrichment catalogue (see `CLAUDE.md`'s 32.55%
enrichment finding) — once every OTHER code's text became much richer,
the already-thin non-standard entries stood out as disproportionately
generic-looking near-matches for a wide variety of unrelated queries.

**Fix**: exclude `NON_STANDARD_ISIC_CODES` / `NON_STANDARD_ISCEDF_CODES`
from this collection entirely, rather than keep them with thin fallback
text. Not a coverage loss in any meaningful sense — these codes were
already confirmed to not correspond to any real official code, so a query
that would have hit one now correctly falls through to its real
neighbouring code instead (e.g. excluding `7311` means "advertising
agency" queries now land on `7310`, the actual official code for the same
concept). Rebuilt both collections (121 / 61 points, down from 134 / 63);
re-ran the identical 15-query smoke test — **zero repeated codes across
all 15 queries**, every prior magnet resolved. Remaining differences from
the "expected" code in that smoke test are ordinary adjacent-category
ambiguity (e.g. "nurse" → `8690` Other human health activities vs the
expected `8610` Hospital activities) — ordinary classification judgment
calls between genuinely similar categories, not a systemic bug. Live-
verified end-to-end through the real classifiers afterward, not just the
store layer directly.

60 new tests (`test_official_source_enrichment.py` plus additions to
`test_build_standard_hierarchical_collections.py`); full suite re-run,
zero regressions.

### What this still does not, and cannot, resolve

Whether flat retrieval with real enriched text actually beats the
existing keyword/LLM pipeline for ISIC or ISCED-F was, at the time this
section was written, still **untested** — no labelled evaluation dataset
existed for either standard. What changed here is that
`ISIC_FLAT_RETRIEVAL`/`ISCEDF_FLAT_RETRIEVAL` became genuinely, not just
architecturally, the same implementation ISCO-08's own best-tested
configuration uses.

**Update, 2026-08-26/27 — a real, if synthetic and incomplete, answer now
exists.** `eval/generate_synthetic_isic_iscedf_benchmark.py` +
`eval/run_synthetic_isic_iscedf_eval.py` (see `CLAUDE.md`'s "Knowledge
base construction" log for the full writeup, including the real
generation-quota constraint that capped coverage at 372/1,092 target
rows) found `flat_retrieval` (enriched_e5large) at **83.06%** vs. the
legacy keyword/LLM pipeline's **13.98%** overall (n=372, McNemar
p≈9.4×10⁻⁶⁸), consistent across all 6 project languages. This is a real,
LLM-generated-but-officially-grounded synthetic result, not WISCO-
equivalent, not pilot-validated — see that script's own docstring for the
full disclosure before citing this number anywhere. It is still the
first real, computed evidence behind what this document's architecture
had only argued for by analogy until now.

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
- "Flat retrieval outperforms keyword matching for ISIC/ISCED-F" or any
  framing implying ISCO-08's flat-vs-hierarchical or flat-vs-keyword
  results generalize here — neither has been tested, in either
  direction, for these two standards. The flat implementation exists;
  its accuracy does not have a measured answer.
- "The e5-large profile improves ISIC/ISCED-F accuracy" or any framing
  implying the ISCO-08 e5-large result (+8.50pp) generalizes here — it has
  not been tested, in either direction, for these two standards.
- "Real official-source enrichment improves ISIC/ISCED-F accuracy" or any
  framing implying the ISCO-08 enrichment result (+11.36pp) generalizes
  here — same as above, not tested in either direction. The magnet-effect
  fix corrected a real regression the enrichment work itself introduced;
  it is not evidence the enrichment improves accuracy on genuine queries
  beyond that.
- "ISIC/ISCED-F's catalogues are now fully audited/verified against the
  official standard" — 121/134 ISIC classes and 61/63 ISCED-F fields are
  confirmed against the real official document; the 13 + 2 non-standard
  codes are a real, disclosed, NOT-yet-resolved gap (see
  `NON_STANDARD_ISIC_CODES` / `NON_STANDARD_ISCEDF_CODES`), not audited
  and fixed the way ISCO-08's catalogue was in Task 20/21.

## Why this remains the one gap infrastructure cannot close (2026-08-25)

Prompted directly by a request to close the gap between ISCO-08 (which now
has a real, full-scale, statistically decisive best-tested accuracy number
— see `CLAUDE.md`) and ISIC/ISCED-F (which still only show "method used").
Checked directly, not assumed, before doing anything:

1. **ISIC/ISCED-F catalogue text is already rich — the ISCO-08 "magnet
   effect" fix does not apply here.** `backend/agents/isic_classifier.py`'s
   `_ISIC_DATA` and `backend/agents/isced_classifier.py`'s `_ISCED_FIELDS`
   already carry real multilingual keyword strings per entry (English +
   Arabic terms), and `backend/rag/hierarchy_nodes.py` already aggregates
   descendant keywords bottom-up into every internal node's `index_text` —
   the exact richness ISCO-08's catalogue was missing before its
   2026-08-24 enrichment fix. There was no equivalent thin-text bug to fix.
2. **The infrastructure gap that did exist has been closed**: ISIC/ISCED-F
   had no e5-large embedding option at all (ISCO-08's own store already
   had one). Added the same additive profile pattern (see "Embedding
   convention" above) — both standards now have live, verified e5-large
   collections, at parity with ISCO-08's infrastructure.
3. **The one gap that infrastructure genuinely cannot close: no labelled
   evaluation dataset exists for ISIC/ISCED-F.** WISCO (the dataset behind
   every ISCO-08 accuracy number in this document) is occupation-only — it
   carries no industry or field-of-study gold labels. An extensive web
   search this project already ran (see `CLAUDE.md`'s Module G / IPUMS
   correspondence) found no open multilingual industry- or
   education-field-classification benchmark comparable to WISCO. The only
   labelled data in this repo touching ISIC/ISCED-F is
   `eval/fixtures/synthetic_lfs_intake_package/synthetic_test_set.csv` — 5
   rows, explicitly prefixed `"SYNTHETIC EXAMPLE"` in every field, built
   for Module F's pre-fill stress test, and missing ISCED-F **field** gold
   labels entirely (only the independent ISCED **level** dimension is
   present). Too small and too narrow to support any accuracy claim, and
   it was never built for this purpose — using it as one would be exactly
   the kind of fabricated-looking evidence this document's own discipline
   exists to prevent. **This is a data-availability problem, not an
   engineering one** — no further code change closes it. The real path
   forward is the same one already open: a genuine multilingual
   ISIC/ISCED-labelled benchmark, most plausibly via the pending IPUMS
   International correspondence, or a properly-scoped, disclosed pilot
   (Module E).

   **Correction, 2026-08-27**: both halves of that last sentence are now
   stale. The IPUMS International correspondence is no longer pending —
   it concluded with a definitive negative answer (IPUMS does not have
   access to, and cannot distribute, original verbatim occupation/
   industry/education text for any sample, including the three this
   project asked about specifically); full transcript at
   `Documentation/Phase_2/Week_1/ipums_correspondence_log.md`. And the
   synthetic-benchmark update earlier in this document (2026-08-26/27, at
   "Update ... a real, if synthetic and incomplete, answer now exists")
   already produced a real, if synthetic, evaluation number in the
   meantime. The one fully real (non-synthetic) path left is Module E.
