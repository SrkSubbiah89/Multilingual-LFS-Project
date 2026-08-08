# Flat Baseline Coverage Audit and Future Unit-Group Comparator Specification

Read-only source/provenance audit produced for Task 19, in response to
Task 18's finding that Task 17's `--system flat` raw CSV cannot support a
four-digit ISCO-08 accuracy comparison: 4,754 of 18,747 rows (25.36%)
have a predicted code shorter than 4 digits (1,372 one-digit, 3,382
two-digit). This document explains *why*, with exact file/line evidence,
and separately investigates whether a fair, standards-conformant
unit-group-only flat comparator can be built. **No code was written,
built, run, or benchmarked to produce this document** — every claim
below is either a direct read of existing source files, a direct
(harmless, read-only) Python import of existing static data structures
to count them, or a citation of existing project documentation.

## Headline conclusion

```text
FLAT_COMPARATOR_IMPLEMENTATION_READY: no
```

**Updated by Task 20** (see
[ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md](ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md)):
the catalogue-*identity* blocker described in this document's original
text (below) is now closed — every one of the 33 mismatched codes (plus
one previously-undocumented mismatched minor-group code) is individually
identified against a primary ILO source, and `eval/
verified_catalogue_counts.yaml` now exists. The comparator remains not
implementation-ready for a different reason: the catalogue itself has
not been *corrected*, and Task 20 additionally found 84 title mismatches
among the codes both catalogues share, a handful of which look like
content misalignment, not spelling. See that document's §6 for the
staged (not started) correction plan.

Original finding (superseded, kept for history): the legacy flat
collection's coverage gap is fully explained (see below). But the
*count discrepancy* between this project's own unit-group catalogue (441
records) and the true ILO ISCO-08 standard (436 unit groups) was only
explained at the aggregate level — the catalogue's exact code-level
identity was **not** reconciled: 15 of 19 "extra" codes and 10 of 14
"missing" codes had never been individually verified against the primary
ILO ISCO-08 structure document. Building a "clean" unit-group-only
comparator directly from the (uncorrected) 441-entry table would still
silently bake 19 known-non-standard codes (and the absence of 14 real
ones) into a supposedly standards-conformant baseline.

---

## 1. Legacy flat baseline audit

### 1.1 Collection name and intended role

`isco_occupations` (`backend/rag/vector_store.py:53`,
`COLLECTION_NAME = "isco_occupations"`). Per the module's own docstring
(`vector_store.py:1-14`): *"Holds a curated ISCO-08 dataset (~110
entries: major groups, sub-major groups, and the most common unit
groups...)"* and *"On first run, embeds the dataset ... and upserts all
vectors into a 'isco_occupations' Qdrant collection."* Its documented
role is as `ISCOClassifier`'s fallback path — used when the hierarchical
collections aren't populated, or deliberately via `force_flat=True`
(`backend/agents/isco_classifier.py:851-866`, docstring: *"Use the flat
VectorStore when the hierarchical store failed to initialise ... or when
__init__(force_flat=True) was used to request this path deliberately
(the eval harness's 'Flat RAG' baseline...)"*). It was never designed or
documented as a standalone, standards-conformant four-digit accuracy
baseline — it is an **integration fallback / eval-harness convenience
baseline**, and its own docstring says so.

### 1.2 Exact source of its records

`backend/rag/vector_store.py:79-610`, module-level literal Python list
`_ISCO_DATA: list[dict]`. Every entry is hand-authored inline (code,
level, `title_en`, `title_ar`, `description`) — there is no file read,
no import from another module's data, no WISCO reference anywhere in
this file (`grep -in wisco backend/rag/vector_store.py` → zero matches).

### 1.3 Record counts by code length and declared hierarchy level

Directly counted from `_ISCO_DATA` (124 total entries):

| Declared `level` | Code length | Count |
|---|---|---|
| 1 (major group) | 1 digit | 10 |
| 2 (sub-major group) | 2 digits | 32 |
| 4 (unit group) | 4 digits | 82 |

Zero entries at `level=3` (minor group, 3-digit codes) exist in this
collection at all. This exactly matches the live Qdrant point count for
`isco_occupations` recorded in Tasks 15/17/19 (124 points).

### 1.4 Contains 1-, 2-, 3-, and 4-digit records?

**1-digit: yes (10). 2-digit: yes (32). 3-digit: no (0). 4-digit: yes
(82, "the most common unit groups" per the docstring — a subset, not the
full 441/436).**

### 1.5 Search behavior and why it can return a coarse code

`VectorStore.search()` (`vector_store.py:672-718`) embeds the query,
calls `self._client.query_points(collection_name=COLLECTION_NAME,
query=query_vec, limit=top_k, with_payload=True)` against this single
mixed-granularity collection, and returns whatever the nearest
neighbours are — a major-group entry, a sub-major-group entry, or a
unit-group entry, entirely dependent on which is semantically closest.
`ISCOClassifier._classify_flat()` (`backend/agents/isco_classifier.py:
851-925`) takes `best = candidates[0]` unconditionally (line 896) and
uses `best.code` as the primary prediction with **no level/length
check anywhere in this function**.

### 1.6 Are returned results filtered to 4-digit codes?

**No.** Confirmed by reading the full body of `_classify_flat()`
(`isco_classifier.py:851-950+`) — there is no `len(code) == 4` check, no
level filter passed to `search()`, and no post-hoc rejection of a
coarse-level top-1 result anywhere on this path.

### 1.7 Precise relationship to Task 18's 4,754 invalid predictions

Traced end-to-end. `eval/run_eval.py:578` sets
`result.pred_isco_4digit = clf_result.primary.code` — the **raw**
classifier output copied verbatim, whatever its length (this is
different from the separately-derived `pred_isco_1digit`/`2digit`/
`3digit` columns at lines 579-581, which use `_digits(code, n)`
(`run_eval.py:210-213`), a helper that safely returns `""` for a code
shorter than `n`). When `_classify_flat()`'s nearest-neighbour hit is a
`level=1` or `level=2` entry from `_ISCO_DATA`, `pred_isco_4digit`
becomes a bare `"0"`-`"9"` (1 digit) or a two-character code like `"23"`
(2 digits) — exactly the `1,372` and `3,382` counts Task 18's gate
reported, confirmed again independently in this task via a direct count
of the real Task 17 flat CSV: `Counter({4: 13993, 2: 3382, 1: 1372})`
over `pred_isco_4digit` string lengths, summing to the 18,747 total.

### 1.8 Why this is a baseline-coverage limitation, not corruption or a Task 17 defect

Task 17's own integrity checks verified execution-time correctness
(exit status, row count/order, zero row-level `error`, reranker/LLM off,
ISIC/ISCED/SRE not constructed) — none of those checks were ever scoped
to assert anything about predicted-code *length*, and Task 17 never
claimed to. The hierarchical CSV, evaluated identically, has **zero**
such rows (`18747/18747` valid 4-digit `pred_isco_4digit`, confirmed by
direct count) — because hierarchical retrieval always terminates at the
dedicated `isco08_unit_groups` stage (see §2). The flat baseline's
partial resolution is therefore a direct, deterministic, and now fully
explained consequence of `isco_occupations`' documented design as a
small, mixed-granularity, ~110-124-entry curated set — not a data
corruption, a random failure, or anything Task 17 did wrong.

```text
The legacy `isco_occupations` collection is a curated mixed-granularity
integration resource. It must not be called a four-digit flat ISCO-08
accuracy baseline, and its Task 17 raw output must not be used for a
four-digit flat-versus-hierarchical accuracy calculation.
```

---

## 2. Hierarchical unit-group source audit

### 2.1 Source file(s) and transformation path

`backend/rag/load_full_isco.py`, module-level literal Python list
`_UNIT: list[tuple[str, str]]` (declared at line 267), hand-authored
`(code, label_en)` tuples (Arabic labels and enriched descriptions are
generated at build time by `_unit_ar()`/`_unit_desc()`,
`load_full_isco.py:755-871` — not read from any external file). The
transformation path is `_UNIT` → `_build_points()`
(`load_full_isco.py:885-906`, builds one payload + one embedding text
per entry) → `_upsert()` (`load_full_isco.py:909-939`, embeds and
upserts into Qdrant) → `main()` (`load_full_isco.py:946+`, iterates all
four `(collection, entries, has_ar)` tuples including
`(COL_UNIT, _UNIT, True)`). No file I/O, no external dataset read, no
WISCO import anywhere in this module (`grep -in wisco
backend/rag/load_full_isco.py` → zero matches; the module's only
imports are `argparse`, `os`, `uuid`, `dotenv`, `qdrant_client`,
`sentence_transformers` — confirmed via `grep -n "^import\|^from"`).

### 2.2 Expected code format; is every node exactly 4 digits?

**Yes, verified directly**, not merely assumed:

```python
from backend.rag.load_full_isco import _UNIT
codes = [c for c, *_ in _UNIT]
len(_UNIT) == 441
len(set(codes)) == 441      # zero duplicates
{len(c) for c in codes} == {4}   # every code is exactly 4 digits
```

This exactly matches the existing `Documentation/Conference_I_Reviewer_2/
generated/coverage_audit_isco08_20260806T214548Z.md` report's own
figures for the `unit` level: `Implemented = 441`, `Duplicates = 0`,
`Malformed = 0`.

### 2.3 Declared vs. observed/count-tested totals (all four levels)

| Level | Docstring/ILO-declared | Directly counted from source list | Match? |
|---|---|---|---|
| major (`_MAJOR`) | 10 | 10 | yes |
| submajor (`_SUBMAJOR`) | 43 | 43 | yes |
| minor (`_MINOR`) | 130 | **131** | **no — previously undocumented, newly found in this audit** |
| unit (`_UNIT`) | 436 | **441** | **no — see §3, previously documented (Phase 2 Week 1)** |

`load_full_isco.py`'s own module docstring (lines 5-9) states
`isco08_minor_groups (130 entries — 3-digit codes)` and
`isco08_unit_groups (436 entries — 4-digit codes)`, both **stale** —
neither was updated after the corresponding list grew. The 131-vs-130
minor-group gap does not appear to have been previously investigated
anywhere in this repository (a search of `Documentation/Phase_2/Week_1/`
and `Documentation/Conference_I_Reviewer_2/` for `"131"` or `"130
minor"` found no prior discussion) — flagged here as a new, smaller,
equally unresolved discrepancy for the same future reconciliation work.

### 2.4 Payload schema

From `_build_points()` (`load_full_isco.py:885-906`), every point's
Qdrant payload is exactly:

```python
{"code": code, "label_en": label_en, "label_ar": label_ar,
 "parent_code": parent, "description": desc}
```

`parent_code = code[:-1]` (empty string for 1-digit codes) — a purely
derived field, not a source/provenance field. **There is no source,
provenance, citation, or lineage field of any kind in this payload
schema** — code, two labels, a derived parent code, and a generated
description are the entire schema. Embedding text (not stored in the
payload, only used to compute the vector):
`f"{_PREFIX}{code} {enriched} {label_ar}"` where `enriched =
_unit_desc(code, label_en)` for the unit level (synonym-enriched
description) or `label_en` directly for the other three levels
(`_build_points()` line 898).

### 2.5 Read directly by hierarchical stage-4 retrieval?

**Yes.** `backend/rag/hierarchical_store.py:143`:
`_COL_UNIT = "isco08_unit_groups"`, wired into the stage list at line
314: `StageConfig(name="unit", collection=_COL_UNIT, weight=_W4)`. This
is the same collection name `load_full_isco.py` populates
(`COL_UNIT = "isco08_unit_groups"`, line 51 of that file) — confirmed
identical string literal in both files.

### 2.6 Is a direct, unfiltered query over only the 4-digit nodes technically feasible?

**Yes, trivially.** `isco08_unit_groups` is already a dedicated,
separate Qdrant collection containing only the 441 unit-group entries
(no major/sub-major/minor entries are ever upserted into it — confirmed
by `main()`'s per-collection loop, each `(collection, entries)` pair
strictly separated). A direct `client.query_points(collection_name=
"isco08_unit_groups", query=query_vec, limit=top_k, with_payload=True)`
call, bypassing the beam-search parent-traversal machinery entirely,
would return only unit-group hits by construction — no new code
architecture is required to make such a query technically possible; see
§4 for why this is not yet being proposed as ready to build.

### 2.7 Is any WISCO source file, title, code, or WISCO-derived mapping imported into the unit-group nodes?

**No import path found — search was broad, not limited to filename
matching.** Checked: (a) `grep -in wisco backend/rag/*.py` → zero
matches in any RAG source file, including `load_full_isco.py`,
`hierarchical_store.py`, `vector_store.py`, `hierarchy_engine.py`,
`standard_hierarchical_store.py`; (b) `load_full_isco.py`'s import
statements (§2.1) touch no WISCO module, no `backend/evaluation/wisco/`
path, and no JSON/CSV data file at all — `_MAJOR`/`_SUBMAJOR`/`_MINOR`/
`_UNIT` are 100% inline Python literals; (c) the only two consumers of
these four lists anywhere in the repository are `eval/coverage_audit.py`
(read-only audit) and `eval/validate_dev_set.py` (read-only dev-set
validator) — both downstream readers, neither a source; (d) direct
verification that the one *specific, documented* fix WISCO's comparison
surfaced (moving codes `6161`-`6164` to `6310`-`6340`,
`module_a_week1_report.md` §5.3) was **never applied** — `_UNIT` still
contains `6161`/`6162`/`6163`/`6164` today and does **not** contain
`6310`/`6320`/`6330`/`6340` (verified by direct import and set
membership check in this task). This confirms, rather than merely
assumes, that WISCO's role relative to this catalogue has so far been
**read-only external comparison, never a write-back or data source** —
consistent with "no WISCO leakage into the collection's construction,"
though it also means the specific fix WISCO's comparison identified
still hasn't been made.

---

## 3. Count discrepancy audit

Three numbers, precisely distinguished:

| Number | What it counts | Source | Measure type |
|---|---|---|---|
| **436** | The true ILO ISCO-08 standard's unit-group count | `isco.ilo.org/en/isco-08`, fetched directly and quoted verbatim in `Documentation/Conference_I_Reviewer_2/STANDARDS_SOURCE_PROVENANCE.md` (*"a four-level hierarchically structured classification ... 436 unit groups ..."*) — a **primary-source, directly-fetched** confirmation, independently corroborated by the WISCO academic dataset covering exactly 436 distinct unit groups in all 5 languages (`Documentation/Phase_2/Week_1/module_a_week1_report.md` §4) | Official source-catalogue count (primary-source-confirmed, but **not** run through this project's own `eval/catalogue_importer.py` — no `eval/verified_catalogue_counts.yaml` file exists in this repository, confirmed by direct filesystem check) |
| **~436 / "436 entries"** appearing in `load_full_isco.py`'s own module docstring and (per `module_a_week1_report.md` §5) the original thesis proposal | An **intended target**, never re-synchronised after the actual `_UNIT` list grew | `load_full_isco.py:9` (stale docstring); thesis proposal (cited secondhand in `module_a_week1_report.md`, not independently re-verified in this task) | Stale/approximate declared-target count, not re-measured against the real list |
| **441** | This project's own static-record count for `_UNIT`, loaded 1:1 into Qdrant | `backend/rag/load_full_isco.py`'s `_UNIT` list (directly counted in this task: 441 entries, 441 unique codes, all exactly 4 digits, 0 duplicates, 0 malformed per the existing `coverage_audit_isco08` report); live Qdrant `isco08_unit_groups` point count recorded in Tasks 11/15/17 (441, unchanged before/after every evaluation run to date) | Static-record count == Qdrant-point count (proven identical, not merely assumed — `_upsert()` writes exactly `len(payloads)` points with sequential integer IDs, one point per `_UNIT` entry, no aliasing or duplication mechanism exists in `_build_points()`/`_upsert()`) |

### Do duplicates, aliases, extra metadata points, or a documented revision explain the 441-vs-436 gap?

**No** — ruled out with direct evidence, not assumed away:

- **Not duplicates**: `len(set(codes)) == len(_UNIT) == 441` (§2.2).
- **Not aliases/extra metadata points**: the payload schema (§2.4) has no
  alias or metadata-only record type; every one of the 441 points is a
  normal `{code, label_en, label_ar, parent_code, description}` unit-group
  record.
- **Not a documented revision**: no changelog, version note, or ISCO-08
  revision reference exists anywhere in `load_full_isco.py` or its
  surrounding documentation explaining a deliberate expansion beyond the
  standard.
- **It is a genuine, partially-root-caused catalogue-quality defect**,
  already discovered and documented before this Reviewer #2 response
  effort began (`Documentation/Phase_2/Week_1/module_a_week1_report.md`,
  dated 2026-08-02): comparing this project's 441 codes against WISCO's
  436 (which independently matches the official standard exactly) found
  **19 codes in `_UNIT` that do not exist in the real ISCO-08 standard**
  (listed individually in that report's §5.1) and **14 real standard
  unit-group codes missing from `_UNIT`** (§5.2). Arithmetic check:
  `441 - 19 = 422` correctly-overlapping codes; `436 - 14 = 422` —
  consistent. A specific root cause is verified for (at least) 4 of these
  33 mismatches: this project's `6161`-`6164` ("Subsistence Crop/
  Livestock/Mixed/Fishers Farmers") are the same real-world occupational
  content as the official/WISCO codes `6310`-`6340`, misfiled under the
  wrong sub-major group (`61`, "Market-oriented Skilled Agricultural
  Workers," instead of the correct `63`, "Subsistence Farmers, Fishers,
  Hunters and Gatherers"). **This specific fix has never been applied**
  (§2.7). The remaining 15 of 19 extra codes and 10 of 14 missing codes
  have **no confirmed 1:1 correspondence** and, per that report's own
  words, a *"full cross-check against the ILO's official ISCO-08
  structure document"* was explicitly out of that module's scope — and,
  per this task's own check (no `eval/verified_catalogue_counts.yaml`
  exists), **still has never been performed**.

### Verdict — updated by Task 20

```text
RESOLVED (identity level) by Task 20's primary-source reconciliation;
NOT YET CORRECTED in production code.
```

**Update (Task 20):** the catalogue-identity blocker described below was
closed by importing the official ILO ISCO-08 structure directly from a
primary, machine-readable source and diffing it code-by-code against
`_MAJOR`/`_SUBMAJOR`/`_MINOR`/`_UNIT`. Every one of the 33 previously-
unidentified mismatched codes is now individually identified (an exact,
code-for-code match to the list below, confirmed against the primary
source rather than inferred via WISCO), the previously-undocumented
131-vs-130 minor-group gap is now traced to exactly one fabricated code
(`913`, "Building and Related Caretakers"), and a new, larger issue was
found: 84 of the 422 codes present in both catalogues have a mismatched
title, a handful of which look like genuine content misalignment rather
than spelling. See
[ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md](ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md)
for the full evidence trail and `eval/verified_catalogue_counts.yaml`
for the machine-readable verified counts (10/43/130/436). **The
catalogue itself has not been corrected** — this remains a blocker for
the future comparator in §4 below, now with a fully specified fix list
instead of an open-ended one.

Original finding (superseded numerically, kept for history): the
*numeric magnitude* of the gap (441 vs 436, net +5) was well explained
at the aggregate level from a WISCO-based comparison; the *catalogue
identity* was not — 25 of the 33 known mismatched codes had never been
individually checked against the primary ILO ISCO-08 structure document.
Task 20 closed that gap.

---

## 4. Future comparator specification — still blocked

**Updated by Task 20.** Blockers 1 and 3 below are now closed. The
comparator is still not implementation-ready because the catalogue
itself has not been corrected (blocker 4 is now the operative one), and
Task 20 surfaced additional title-level mismatches (§3 update above)
that widen blocker 1's scope beyond what was known when this list was
first written.

### Blockers

1. ~~**Catalogue identity reconciliation.**~~ **Closed by Task 20**: a
   full, code-by-code cross-check of the primary ILO ISCO-08 structure
   document against `_MAJOR`/`_SUBMAJOR`/`_MINOR`/`_UNIT` is complete —
   see `ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md` §4 for the exact 19
   extra / 14 missing / 1 extra-minor code lists and the 84 newly-found
   title mismatches (a strict superset of what this blocker originally
   asked for). The verified `6161`-`6164` → `6310`-`6340` fix is now
   confirmed by primary-source title identity, not just WISCO
   corroboration. **Not yet done**: applying any of these corrections to
   production code (that is blocker 4 below).
2. **WISCO-independence of any catalogue fix** — still required, and
   already honored in Task 20's own process (zero WISCO input). Any
   correction must be sourced from the **primary ILO ISCO-08 structure
   document** (already used directly in Task 20), never from WISCO's own
   code list.
3. ~~**A verified (not merely unverified) official count**~~ **Closed by
   Task 20**: `eval/verified_catalogue_counts.yaml` now exists, produced
   by a clean `eval/catalogue_importer.py` validation of the
   Task-20-normalized official catalogue. `eval/coverage_audit.py` can
   now compute a real `coverage_percentage` for ISCO-08 (not yet run in
   this task — that would be a new evaluation-adjacent operation, out of
   scope here).
4. **Catalogue correction and an explicit, approved decision on final
   scope** — now the primary remaining blocker. `_MAJOR`/`_SUBMAJOR`/
   `_MINOR`/`_UNIT` still contain the same 19+1 non-standard codes, are
   still missing the same 14 real codes, and still carry the 84 newly-
   found title mismatches (two clusters of which look like content
   misalignment, not spelling — `ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md`
   §4.3). This requires the staged, separately-approved plan in that
   document's §6, starting with human review of the full mismatch list.

**Update (Task 21): the specification below is no longer deferred — it
is implemented, but as a dry-run-only builder gated on blocker 4 above.**
See [OFFICIAL_ISCO08_RUNTIME_AND_FLAT_COMPARATOR.md](OFFICIAL_ISCO08_RUNTIME_AND_FLAT_COMPARATOR.md)
for the full detail. In short: `backend/rag/official_isco08_catalogue.py`
(fail-closed loader, hash/count-verified against `eval/
verified_catalogue_counts.yaml`), `backend/rag/
build_official_isco08_collections.py` (dry-run collection planner;
`--execute` unconditionally refused), the five versioned collection
names from item 1 below, direct unfiltered retrieval (item 5), a hard
runtime four-digit assertion (item 6), and a distinct method label
`flat_isco08_official_ilo2021_v1` (item 7, never `flat_semantic`) are
all built and hermetically tested (item 9). **No collection has been
built or populated** (items 8, 10 remain genuinely future work) — the
builder's `--execute` path exists only as a CLI shape and refuses to run
in this task, pending the catalogue correction in blocker 4.

### What such a specification will eventually need to contain (deferred, not written now)

For context only — once the above blockers close, a future task's
specification will need to address, at minimum, the ten elements this
task's brief enumerates: a separately-named collection distinct from
`isco_occupations` (e.g. `isco08_unit_groups_flat_v1`); one record per
*accepted* four-digit code only; an immutable catalogue/source hash and
exact expected count checked before any build; a payload/embedding-text
design; direct unfiltered nearest-neighbour retrieval over unit-group
leaves only (technically straightforward per §2.6); a hard runtime
four-digit assertion; a distinct method label (never `flat_semantic`);
a separately-approved collection-build command; hermetic tests for
source counts/format/payload/WISCO-non-dependence/runtime filtering/
collection identity; and a future evaluation protocol reusing Task 17's
already-validated hierarchical raw output unchanged. None of this is
implementation-ready today.

---

## 5. Paper/reviewer implications — evidence-status table

| Claim area | Current status | What is safe now | What remains blocked |
|---|---|---|---|
| Architecture / implementation | Implemented | Describing the 4-stage hierarchical pipeline, the generic `HierarchyBeamSearchEngine`, and the legacy flat fallback's existence and documented purpose | Describing the legacy flat path as a rigorous four-digit baseline |
| Hierarchical strict integrity | Verified (Task 15/17) | Citing that the strict hierarchical path produces genuine, non-fallback, 4-digit retrieval on 18,747/18,747 WISCO heldout cases with bounded per-stage latency | Any accuracy claim from this integrity result alone |
| WISCO controlled four-digit hierarchical output integrity | Verified (Task 17) | Citing the raw hierarchical CSV as integrity-clean and available for a future analysis task | Treating raw integrity as an accuracy result |
| Fair flat-vs-hierarchical accuracy comparison | **Blocked** | Nothing — no fair comparison exists | Any flat-vs-hierarchical number until a reconciled, WISCO-independent unit-group comparator (§4) is built and run |
| Controlled multilingual ISCO-08 accuracy (any system) | **Blocked** (Task 18 produced no metric output) | Nothing | Any accuracy percentage for either system |
| Latency / computational evidence | Partial raw data exists (Task 17 per-row latency), not analyzed | Noting raw per-row latency data exists | Any latency distribution, comparison, or production-readiness claim |
| ISIC / ISCED / SRE evidence | Out of scope for this evidence line | Nothing from this line | Any claim — this run is ISCO-08-only by construction |
| Real LFS validation | Not attempted | Nothing | Any real-LFS claim; WISCO remains externally sourced controlled benchmark data, not respondent data |

No claim is made that Reviewer #2 is fully satisfied by this audit or
by any prior task in this evidence line. This document produces zero
new accuracy, latency, or coverage-percentage numbers — it only
explains, with file/line evidence, why the flat baseline cannot yet
support one, and what must happen before a fair replacement can be
built.
