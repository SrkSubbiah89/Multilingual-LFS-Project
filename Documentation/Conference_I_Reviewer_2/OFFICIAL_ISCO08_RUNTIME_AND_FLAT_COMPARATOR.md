# Official ISCO-08 Runtime and Full-Unit-Group Flat Comparator

Task 21: implements the official-source runtime/builder path and a
separately named, genuinely flat, four-digit-only ISCO-08 comparator,
adopting the approved catalogue policy:

```text
Adopt the verified primary ILO English ISCO-08 catalogue exactly:
10 major groups, 43 sub-major groups, 130 minor groups, and 436 unit
groups, with the official code, parent, and English title values.
```

**Neither this task nor Task 22 (which added the real, guarded
`--execute` implementation) builds a Qdrant collection, connects to a
Qdrant instance, executes a classifier call, runs a WISCO evaluation, or
produces a benchmark result.** Task 21 shipped source/runtime/builder
implementation and a dry-run planning path. Task 22 replaced the
unconditional `--execute` refusal with a real, hermetically-tested (26
tests, fakes only), still-never-invoked-against-a-live-service
execution path — see §5-§6 below.

```text
OFFICIAL_ISCO08_RUNTIME_IMPLEMENTATION_READY: yes
OFFICIAL_ISCO08_COLLECTION_BUILD_IMPLEMENTATION_READY: yes
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

## 5. Exact future operator commands (dry-run and execution forms; nothing was executed)

**Catalogue validation + collection plan (dry-run, unchanged since
Task 21; safe to run any time — no Qdrant/embedder import occurs):**

```bash
python -m backend.rag.build_official_isco08_collections \
  --catalogue eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv \
  --metadata eval/verified_catalogue_counts.yaml \
  --profile official_ilo2021_v1
```

Prints the validated 5-entry collection plan (names, per-level counts,
source hash) and exits 0. **No Qdrant collection was created, connected
to, or modified.**

**`--execute` (Task 22: now a real, guarded implementation — still
never invoked by any command, script, or test in this task; a future
task must independently authorize and run this):**

```bash
python -m backend.rag.build_official_isco08_collections \
  --catalogue eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv \
  --metadata eval/verified_catalogue_counts.yaml \
  --profile official_ilo2021_v1 \
  --output-manifest <ignored local manifest path> \
  --execute \
  --confirm-profile official_ilo2021_v1 \
  --allow-local-qdrant-mutation
```

Both `--confirm-profile official_ilo2021_v1` (exact string match, and
must equal `--profile`) and `--allow-local-qdrant-mutation` are
mandatory; omitting either fails immediately with `ACKNOWLEDGEMENT
REQUIRED`, before any Qdrant/embedder import or connection. Passing
`--dry-run` alongside `--execute` always forces the dry-run path
regardless of the other flags (a deliberate safety override).
`--output-manifest` is required whenever `--execute` is passed (the CLI
refuses to run without a place to write the ignored local build
manifest).

Once acknowledgements pass, execution: re-validates the catalogue via
`load_official_catalogue()` (same hash/count/format checks as dry-run);
connects to a **local-only** Qdrant target (`QDRANT_HOST`/`QDRANT_PORT`
env vars, default `localhost`/`6333` — no CLI flag or environment
variable anywhere in this module accepts a remote URL or token); refuses
if **any** of the five target collection names already exists (empty or
not — no auto-replace/delete/recreate/upsert-into-existing/alias-swap);
then creates and verifies the five collections in the fixed order
major → submajor → minor → unit (hierarchical) → unit (flat), verifying
each one's exact point count and full payload identity immediately
after writing it, before moving to the next. A success manifest is
written only after all five are verified; any error writes a **failure**
manifest naming exactly which target(s) were created/verified/partial
and states plainly that remediation needs a separate explicit task — no
automatic deletion or overwrite ever occurs.

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
instance. Task 21 performed zero Qdrant connection, query, count,
build, populate, or delete operation because the module contained no
Qdrant/embedder import at all. **Task 22 implemented the real execution
path but still performed zero live Qdrant operation of any kind** — no
Task 22 test, script, or command ever passes `--execute` against a real
service; every test that reaches the execution code path injects a
`FakeQdrantClient`/`FakeEmbedder` via `execute_build()`'s
`qdrant_client_factory`/`embedder_factory` dependency-injection
parameters. The two module-level lazy-import functions
(`_default_qdrant_client_factory()`, `_default_embedder_factory()`)
that would perform a real import/connection are never called in this
task — confirmed by (a) a hermetic test asserting the module's
*top-level* (module-body) statements contain no `qdrant`/
`sentence_transformers` import, (b) a companion test confirming those
imports genuinely exist, but only nested inside the three lazy-import
function bodies (a positive control against the check in (a) being
vacuous), and (c) every execution-path test supplying fakes for both
dependencies.

### Execution preflight / acknowledgement contract

| Check | Enforced by | Result if missing/wrong |
|---|---|---|
| `--confirm-profile` exactly `official_ilo2021_v1` | `check_execution_acknowledgements()` | `BuildAcknowledgementError`, before any import |
| `--confirm-profile` equals `--profile` | same | same |
| `--allow-local-qdrant-mutation` present | same | same |
| Catalogue hash/count/format valid | `load_official_catalogue()` (reused, not reimplemented) | `OfficialISCO08CatalogueError`, before any Qdrant/embedder import |
| All 5 target names absent | live `get_collections()` call (first Qdrant call made) | `BuildPreflightError`, before any `create_collection`/`upsert` |

### Target collections, payload fields, and verification

Same five names as Task 21 (§2 above). Every payload written includes
`code`, `level`, `parent_code`, `title_en`, `profile`,
`source_catalogue_sha256`, `collection_role` (`major` / `submajor` /
`minor` / `unit_hierarchical` / `unit_flat`), and the record's own
deterministic `embedding_text` — nothing else, no WISCO title, benchmark
label, prediction, or hand-authored correction. The flat collection
receives the exact same 436 unit records as the hierarchical unit
collection, tagged `collection_role="unit_flat"` instead of
`"unit_hierarchical"`, and every code in it is verified to match
`^[0-9]{4}$` after writing. Verification after each collection checks
the exact planned point count, that every payload's profile/source-hash/
role/level matches the plan, and (for the flat target specifically) the
four-digit-only rule — before moving on to the next collection.

### Manifest contract

Written only after all five collections verify successfully (`status:
"success"`), or immediately on any failure (`status:
"preflight_failed_existing_target"` or `"failed_partial_build"`, always
including `targets_created_or_partial` and, for a build-time failure, a
`remediation_note` stating plainly that fixing a partial build needs a
separate explicit task). Fields: UTC build timestamp, builder script
SHA-256, catalogue/metadata paths and hashes, profile, local Qdrant
host/port, resolved embedding model identity (`intfloat/multilingual-
e5-small`, reused from `backend/rag/hierarchical_store.py`/
`backend/rag/load_full_isco.py`'s existing configuration — not a new
dimension), payload-schema version, both acknowledgements, planned
targets, and (on success) each target's verified count.

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
   `ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md` §6, step 1). **Still not
   started.**
2. A dedicated source-data correction task (step 2, unchanged). **Still
   not started.**
3. Local official-source availability check, then a separately approved
   live collection build. **Still not started — but the build
   implementation itself is now real** (Task 22): `backend.rag.
   build_official_isco08_collections --execute --confirm-profile
   official_ilo2021_v1 --allow-local-qdrant-mutation --output-manifest
   <path>`, hermetically tested against fakes only (26 tests, §6 above).
   A future task must (a) independently review this execution code, (b)
   perform the local availability check (confirm the normalized
   catalogue + `eval/verified_catalogue_counts.yaml` are present and
   still hash-valid against a real local Qdrant instance), and (c) run
   the command for real, under its own explicit approval — none of that
   happened here.
4. A smoke gate: a handful of known cases run against the newly-built
   collections to confirm basic sanity before any full run. **Still not
   started.**
5. A fresh, full controlled WISCO evaluation — both the official
   full-unit-group flat comparator and the official hierarchical
   profile — reusing Task 17's already-validated, unchanged canonical
   heldout CSV. **Still not started.**
6. A new fail-closed analysis task (Task 18-style) over those fresh
   results. **Still not started.**

None of steps 3-6 were performed in this task; step 3's tooling is now
implementation-ready but was never executed.
