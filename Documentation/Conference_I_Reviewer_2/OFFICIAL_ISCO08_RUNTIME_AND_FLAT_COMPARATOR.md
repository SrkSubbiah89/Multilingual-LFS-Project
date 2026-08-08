# Official ISCO-08 Runtime and Full-Unit-Group Flat Comparator

Task 21: implements the official-source runtime/builder path and a
separately named, genuinely flat, four-digit-only ISCO-08 comparator,
adopting the approved catalogue policy:

```text
Adopt the verified primary ILO English ISCO-08 catalogue exactly:
10 major groups, 43 sub-major groups, 130 minor groups, and 436 unit
groups, with the official code, parent, and English title values.
```

**This task builds no Qdrant collection, connects to no Qdrant instance,
executes no classifier call, runs no WISCO evaluation, and produces no
benchmark result.** It is source/runtime/builder implementation and a
dry-run planning path only.

```text
OFFICIAL_ISCO08_RUNTIME_IMPLEMENTATION_READY: yes
```

## 1. Verified official source and local-hash requirement

The only authoritative official-source metadata is the tracked file
`eval/verified_catalogue_counts.yaml` (Task 20). The official runtime
path (`backend/rag/official_isco08_catalogue.py`) never trusts a
supplied catalogue file on its own — it:

1. requires an explicit local normalized-catalogue path (never fetches
   or parses a raw ILO workbook itself — that remains Task 20's
   `eval/normalize_ilo_isco08_catalogue.py` job);
2. verifies the supplied file's SHA-256 exactly matches
   `verified_catalogue_counts.yaml`'s `isco08.normalized_catalogue_sha256`;
3. verifies the observed per-level record counts match both the fixed
   official figures (10/43/130/436, `OFFICIAL_EXPECTED_COUNTS`) and the
   metadata file's own `verified_counts` — a caller cannot use a lenient
   caller-supplied expectation to bypass what the trusted metadata
   records;
4. validates code format, per-level uniqueness, parent-link consistency
   (top-down order required), and nonblank English titles;
5. raises `OfficialISCO08CatalogueError` (never returns a partial record
   list) on any violation of 1-4;
6. never falls back to the legacy hand-authored `_MAJOR`/`_SUBMAJOR`/
   `_MINOR`/`_UNIT` lists when the requested official catalogue is
   unavailable — the caller gets an explicit error, not a silently
   different dataset.

The loader takes only the two file paths a caller supplies as input. It
contains no WISCO import, path, or reference anywhere in its source
(enforced by a dedicated hermetic test using AST inspection of its
actual import statements, not a prose/docstring scan).

## 2. Versioned official collection names

For profile `official_ilo2021_v1` (the only registered profile so far,
in `backend/rag/official_isco08_catalogue.py`'s `PROFILE_COLLECTION_NAMES`
— the single source of truth both the retrieval store and the builder
import, so they can never drift apart):

| Collection | Level | Record count |
|---|---|---|
| `isco08_major_groups_ilo2021_v1` | major | 10 |
| `isco08_submajor_groups_ilo2021_v1` | submajor | 43 |
| `isco08_minor_groups_ilo2021_v1` | minor | 130 |
| `isco08_unit_groups_ilo2021_v1` | unit (hierarchical stage 4) | 436 |
| `isco08_unit_groups_flat_ilo2021_v1` | unit (flat, direct) | 436 |

The flat collection is a **separate collection identity** over the same
436 unit-group records — not a duplicate data source, a distinct
retrieval entry point with zero hierarchy/parent filtering.

## 3. Legacy vs. official profile distinction

`backend/rag/hierarchical_store.py`'s `HierarchicalISCOStore` gained a
`profile` constructor parameter (default `"legacy"`, byte-identical to
every prior version of this class — existing callers that omit it see
zero behavioural change, including continuing to share the module-level
`get_hierarchical_store()` singleton).

| | Legacy (`profile="legacy"`, default) | Official (`profile="official_ilo2021_v1"`) |
|---|---|---|
| Hierarchical collections | `isco08_major_groups`, ..., `isco08_unit_groups` | `isco08_major_groups_ilo2021_v1`, ..., `isco08_unit_groups_ilo2021_v1` |
| Flat/fallback collection | `isco_occupations` (124-entry curated, mixed-granularity — see `FLAT_BASELINE_COVERAGE_AUDIT.md`) | `isco08_unit_groups_flat_ilo2021_v1` (436-entry, four-digit-only) |
| Hierarchical method label | `hierarchical_semantic` / `hierarchical_llm` | `hierarchical_isco08_official_ilo2021_v1` (single label; reranker firing is still recorded separately in trace/instrumentation) |
| Flat method label | `flat_semantic` / `flat_llm` | `flat_isco08_official_ilo2021_v1` — **never** `flat_semantic` |
| Total-unavailability label | `flat_semantic` (pre-existing sentinel quirk, unchanged) | `unavailable_isco08_official_ilo2021_v1` — explicit, never conflated with a real flat/hierarchical result |
| Cross-fallback to the other profile's collections | N/A | **Never** — an official profile's own flat fallback always targets its own versioned flat collection; it never queries `isco_occupations` |
| Coarse (non-4-digit) code from the flat collection | Allowed (documented, audited limitation) | Rejected at runtime (`_ISCO4_RE` check) — returns the explicit unavailable sentinel instead of a coarse code |

`ISCOClassifier` gained an additive `isco_catalogue_profile: str =
"legacy"` constructor parameter (default preserves every existing
caller's behaviour exactly, including which module-level singleton is
used). A non-legacy profile always constructs a dedicated
`HierarchicalISCOStore(profile=...)` instance — never the shared legacy
singleton — and never falls back to the legacy flat `VectorStore`
(`get_vector_store()`) if the official collections are unavailable; the
classifier's own construction leaves `_hierarchical_store = None` and
`classify()` raises `RuntimeError` rather than silently using a legacy
result (verified by a hermetic test that makes `get_vector_store()`
raise an assertion error if ever called for a non-legacy profile).

## 4. The four-digit-only flat comparator

`ISCOClassifier(isco_catalogue_profile="official_ilo2021_v1",
force_flat=True)` selects the **full-unit-group flat comparator**:
direct, unfiltered nearest-neighbour retrieval against only
`isco08_unit_groups_flat_ilo2021_v1` (`HierarchicalISCOStore.
search_flat_only()`, reusing the existing `_flat_search`/`_embed_query`
implementations unchanged — no new retrieval algorithm), with zero
hierarchy beam or parent filter. Its method label is always
`flat_isco08_official_ilo2021_v1`, distinct from every legacy label. A
runtime check rejects any candidate code that is not exactly `^[0-9]{4}$`
(defensive — the builder's own record-count contract already guarantees
the collection contains nothing else) rather than ever returning a
coarse code as a genuine four-digit prediction.

## 5. Exact future operator commands (dry-run form; nothing was executed)

**Catalogue validation + collection plan (implemented, run in this
task against real Task 20 local artifacts):**

```bash
python -m backend.rag.build_official_isco08_collections \
  --catalogue eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv \
  --metadata eval/verified_catalogue_counts.yaml \
  --profile official_ilo2021_v1
```

Prints the validated 5-entry collection plan (names, per-level counts,
source hash) and exits 0. **No Qdrant collection was created, connected
to, or modified.**

**`--execute` (explicitly refused by this task):**

```bash
python -m backend.rag.build_official_isco08_collections \
  --catalogue <path> --metadata <path> --profile official_ilo2021_v1 --execute
```

Prints `REFUSED: --execute is not implemented by this task (Task 21)
and requires a separately approved future task.` and exits nonzero,
unconditionally — this module contains no Qdrant/SentenceTransformer
import at all (verified by AST inspection in a hermetic test), so no
code path in this file can reach either even if `--execute` succeeded
in bypassing this refusal.

**A future, separately-approved evaluation run (NOT executed in this
task — no collection exists to run it against yet):**

```bash
python eval/run_eval.py \
  --test-set <canonical WISCO v2 heldout CSV, unchanged> \
  --system hierarchical --isco-catalogue-profile official_ilo2021_v1 \
  --use-llm-reranker off --require-genuine-hierarchical \
  --max-stage-latency-ms 30000 \
  --output-dir <new ignored output root>

python eval/run_eval.py \
  --test-set <same canonical WISCO v2 heldout CSV> \
  --system flat --isco-catalogue-profile official_ilo2021_v1 \
  --use-llm-reranker off \
  --output-dir <new ignored output root>
```

## 6. No official collection has yet been built or populated

`isco08_major_groups_ilo2021_v1`, `isco08_submajor_groups_ilo2021_v1`,
`isco08_minor_groups_ilo2021_v1`, `isco08_unit_groups_ilo2021_v1`, and
`isco08_unit_groups_flat_ilo2021_v1` do not exist in any Qdrant
instance. This task performed zero Qdrant connection, query, count,
build, populate, or delete operation of any kind — confirmed by (a) the
builder module's own source containing no `qdrant`/
`sentence_transformers` import at all (AST-verified in a hermetic test)
and (b) every retrieval-path test in this task using
`FakeQdrantClient`/`FakeEmbedder` monkeypatched at module level.

## 7. No post-correction WISCO evaluation or accuracy exists yet

No `eval/run_eval.py`, `eval/analyze.py`, or `eval/analyze_wisco_tier1.py`
invocation against real data occurred in this task. No accuracy,
coverage, latency, or scalability number was produced.

## 8. Task 17's raw outputs remain superseded for standard-compliant accuracy

Task 17's hierarchical and flat WISCO raw CSVs were produced entirely
under the **legacy** profile (`isco_catalogue_profile` did not exist at
the time). They remain integrity-checked, non-accuracy artifacts (per
their own final report) and now, additionally, are known to have used
collections whose catalogue identity diverges from the official ILO
standard in 20 non-standard + 14 missing + numerous title-mismatched
codes (`ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md`). A standard-
compliant accuracy result requires the staged, separately-approved
process in that document's §6, now extended by this task's §9 below —
still not started.

## 9. WISCO remains controlled multilingual data; ISIC/ISCED/SRE/reranking out of scope

Nothing in this task changes WISCO's status: it remains an externally
sourced, controlled multilingual ISCO-08 occupation-title benchmark, not
real Labour Force Survey respondent data (`FLAT_BASELINE_COVERAGE_AUDIT.md`,
`ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md`). This task implements
ISCO-08 retrieval infrastructure only — no ISIC, ISCED, SRE, or reranker-
accuracy claim is made or supported by anything built here.

## 10. Updated staged plan (still not started)

1. Human review of the full mismatch list from Task 20 (unchanged from
   `ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md` §6, step 1).
2. A dedicated source-data correction task (step 2, unchanged).
3. Local official-source availability check: confirm the normalized
   catalogue + `eval/verified_catalogue_counts.yaml` are present and
   still hash-valid, then a separately approved collection build using
   `backend.rag.build_official_isco08_collections --execute` (not
   implemented by this task's `--execute` path — a future task must add
   the actual build logic under a fresh, explicit approval).
4. A smoke gate: a handful of known cases run against the newly-built
   collections to confirm basic sanity before any full run.
5. A fresh, full controlled WISCO evaluation — both the official
   full-unit-group flat comparator and the official hierarchical
   profile — reusing Task 17's already-validated, unchanged canonical
   heldout CSV.
6. A new fail-closed analysis task (Task 18-style) over those fresh
   results.

None of steps 3-6 were performed in this task.
