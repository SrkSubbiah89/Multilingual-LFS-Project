# Task 21 Final Report — Official ISCO Source Correction and Full Flat Comparator Implementation

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_21_OFFICIAL_ISCO_SOURCE_AND_FULL_FLAT_COMPARATOR.md`.
Implementation and hermetic-test task only — no Qdrant collection was
built, connected to, or mutated; no classifier call, WISCO read, or
benchmark run occurred.

```text
OFFICIAL_ISCO08_RUNTIME_IMPLEMENTATION_READY: yes
```

## 1. Source SHA, branch, final commit, push, clean-tree status

| | |
|---|---|
| Base branch | `reviewer2-isco08-official-catalogue-reconciliation-20260808` |
| Required SHA | `d161b3cc8b433744191a05d7e8c2698bc7025ba3` |
| Verified `origin` SHA | `d161b3cc8b433744191a05d7e8c2698bc7025ba3` — match |
| New branch | `reviewer2-official-isco08-runtime-and-flat-comparator-20260808` |
| Final commit SHA | recorded after this report's commit (see push confirmation below) |

Working tree was clean before branching and remained clean throughout
except for the files listed in §2, all tracked deliverables this task
is permitted to add/modify.

## 2. Exact changed tracked files

**New:**

```text
backend/rag/official_isco08_catalogue.py
backend/rag/build_official_isco08_collections.py
backend/tests/test_official_isco08_catalogue.py
backend/tests/test_build_official_isco08_collections.py
backend/tests/test_official_isco08_profiles.py
backend/tests/test_isco_classifier_official_profile.py
eval/test_official_isco08_profile_evaluator.py
Documentation/Conference_I_Reviewer_2/OFFICIAL_ISCO08_RUNTIME_AND_FLAT_COMPARATOR.md
Documentation/AI_HANDOFF/CLAUDE_TASK_21_FINAL_REPORT.md (this file)
```

**Modified:**

```text
backend/rag/hierarchical_store.py         (additive profile= parameter)
backend/agents/isco_classifier.py         (additive isco_catalogue_profile= parameter)
eval/run_eval.py                          (additive --isco-catalogue-profile flag)
eval/dev_sweep.py                         (integration-seam fix, see §7)
eval/test_dev_sweep.py                    (matching fixture fix, see §7)
eval/test_model_free_isco_evaluation.py   (matching fixture fix, see §7)
Documentation/Conference_I_Reviewer_2/ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md
Documentation/Conference_I_Reviewer_2/FLAT_BASELINE_COVERAGE_AUDIT.md
Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md
Documentation/Conference_I_Reviewer_2/REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md
Documentation/Conference_I_Reviewer_2/README.md
```

`eval/configs/b1_frozen.json` was **not** touched (`git diff --stat` on
that path returns no output). No production hierarchy data
(`backend/rag/load_full_isco.py`'s `_MAJOR`/`_SUBMAJOR`/`_MINOR`/
`_UNIT`), Qdrant data, or WISCO package file was changed.

## 3. Official source metadata path and runtime hash/count validation contract

`backend/rag/official_isco08_catalogue.py`'s `load_official_catalogue()`
takes an explicit local normalized-catalogue path and validates it
against the sole authoritative metadata file, `eval/
verified_catalogue_counts.yaml` (Task 20), before returning any record:

1. catalogue file must exist;
2. its SHA-256 must exactly match `isco08.normalized_catalogue_sha256`
   in the metadata file;
3. observed per-level counts must match **both** the fixed official
   figures (`OFFICIAL_EXPECTED_COUNTS = {major: 10, submajor: 43, minor:
   130, unit: 436}`) and the metadata file's own `verified_counts`;
4. every row's code format, per-level uniqueness, parent-link order,
   and nonblank title are validated;
5. any violation of 1-4 raises `OfficialISCO08CatalogueError` — no
   partial record list is ever returned;
6. there is no code path back to the legacy `_MAJOR`/`_SUBMAJOR`/
   `_MINOR`/`_UNIT` lists anywhere in this module.

The loader reads only the two paths it is given — confirmed by a
hermetic test that monkeypatches `Path.read_bytes`/`Path.read_text` to
assert against an explicit allow-list, and a second test that AST-walks
the module's actual `import`/`from...import` statements to confirm no
WISCO module is ever imported (a prose/docstring mention of "WISCO
independence" is expected and explicitly excluded from that check).

## 4. Collection plan names/counts and proof no collection was built

`backend/rag/build_official_isco08_collections.py --catalogue <path>
--metadata <path> --profile official_ilo2021_v1` (no `--execute`)
validates via the loader above and returns a 5-entry plan:

| Collection | Level | Count |
|---|---|---|
| `isco08_major_groups_ilo2021_v1` | major | 10 |
| `isco08_submajor_groups_ilo2021_v1` | submajor | 43 |
| `isco08_minor_groups_ilo2021_v1` | minor | 130 |
| `isco08_unit_groups_ilo2021_v1` | unit | 436 |
| `isco08_unit_groups_flat_ilo2021_v1` | unit (flat) | 436 |

**`--execute` is unconditionally refused** by `main()` — prints
`REFUSED: --execute is not implemented by this task (Task 21) and
requires a separately approved future task.` and exits nonzero, before
any planning or catalogue read occurs. Independently, the module
contains **no `qdrant`/`sentence_transformers` import anywhere in its
source** (verified by AST inspection in a hermetic test, not a
substring scan of prose) — no code path in this file can reach either
library even if the `--execute` refusal were somehow bypassed. No
Qdrant collection was created, connected to, queried, counted, or
modified anywhere in this task.

## 5. Hierarchy and flat method labels / profile identity

| | Legacy (default) | Official (`profile="official_ilo2021_v1"`) |
|---|---|---|
| Hierarchical, non-fallback | `hierarchical_semantic` / `hierarchical_llm` | `hierarchical_isco08_official_ilo2021_v1` |
| Flat / fallback-used | `flat_semantic` / `flat_llm` | `flat_isco08_official_ilo2021_v1` (never `flat_semantic`) |
| Total unavailability | `flat_semantic` (pre-existing sentinel, unchanged) | `unavailable_isco08_official_ilo2021_v1` |

`HierarchicalISCOStore.__init__` gained an additive `profile: str =
"legacy"` parameter; `_PROFILE_COLLECTIONS` maps it to either the
unchanged legacy collection-name constants or the five versioned names
from §4 (`PROFILE_COLLECTION_NAMES`, defined once in `backend/rag/
official_isco08_catalogue.py` and imported by both the store and the
builder, so the two can never drift apart). An unrecognized profile
raises `UnknownISCOCatalogueProfileError` immediately. A new
`search_flat_only()` method reuses the existing `_flat_search`/
`_embed_query` implementations unchanged to implement the direct,
unfiltered "full-unit-group flat comparator" — no new retrieval
algorithm was written anywhere in this task.

`ISCOClassifier.__init__` gained an additive `isco_catalogue_profile:
str = "legacy"` parameter. Non-legacy profiles always construct a
dedicated (non-singleton) `HierarchicalISCOStore(profile=...)` and never
touch `get_hierarchical_store()` or `get_vector_store()` (the legacy
singleton/flat-store paths) — verified by hermetic tests that make both
raise an `AssertionError` if ever called for a non-legacy profile.

## 6. Compatibility and no-silent-fallback evidence

- **Default behaviour unchanged**: `test_legacy_profile_default_unaffected`,
  `test_default_profile_is_legacy_and_uses_singleton_getter`, and
  `test_default_cli_profile_is_legacy` each confirm omitting the new
  parameter/flag entirely reproduces exact prior collection names,
  singleton usage, and CLI behaviour. All 121 pre-existing
  `test_isco_classifier*.py` tests and all 18 pre-existing
  `test_hierarchical_store.py` tests continue to pass unmodified.
- **No cross-profile leakage**: `test_official_hierarchical_profile_selects_only_official_collections`
  and `test_official_flat_fallback_uses_only_official_flat_collection`
  assert the exact set of Qdrant collection names a `FakeQdrantClient`
  was queried with contains zero legacy names (and vice versa for the
  legacy-profile compatibility tests).
- **No coarse-code leakage into the official flat path**:
  `test_official_flat_rejects_non_four_digit_code` confirms a
  (synthetic, adversarial) non-4-digit candidate from the flat
  collection is refused rather than returned, while
  `test_legacy_flat_still_allows_coarse_code` confirms the legacy
  profile's documented, audited coarse-code behaviour is completely
  unaffected.
- **Explicit unavailability, never silent legacy**:
  `test_official_profile_total_unavailability_is_explicit` and
  `test_official_profile_construction_failure_never_falls_back_to_legacy`
  confirm a fully-unavailable official profile returns the explicit
  `unavailable_isco08_official_ilo2021_v1` sentinel (store level) and
  raises `RuntimeError` rather than silently using `get_vector_store()`
  (classifier level).
- **Strict-guard compatibility**: because the official hierarchical
  label (`hierarchical_isco08_official_ilo2021_v1`) still starts with
  the literal prefix `"hierarchical_"`, Task 13's
  `check_strict_hierarchical()` / `--require-genuine-hierarchical`
  required **zero code changes** to remain compatible — confirmed by
  `test_require_genuine_hierarchical_accepts_official_profile_method_label`
  (passes) and `test_require_genuine_hierarchical_rejects_official_flat_fallback`
  (a `flat_isco08_official_...` row is correctly still treated as a
  non-hierarchical fallback and aborts the run, exactly like a legacy
  `flat_semantic` row would).
- **Config-hash/manifest identity**: `_config_hash()` gained an
  `isco_catalogue_profile` field; `test_config_hash_differs_between_legacy_and_official_profile`
  confirms two runs differing only in `--isco-catalogue-profile` always
  hash differently.

## 7. Focused and full test results

```
python -m pytest eval/test_analyze_wisco_tier1.py eval/test_model_free_isco_evaluation.py eval/test_require_genuine_hierarchical.py eval/test_docs_consistency.py -q
→ 73 passed in 29.33s

python -m pytest backend/tests eval/ -q
→ 2037 passed, 1 deselected, 1 warning in 507.43s (0:08:27)
```

**Zero failures in the final state.** `2037 = 1995` (Task 20 baseline)
`+ 42` new tests across the 5 new test files (17 + 7 + 7 + 6 + 5).

The first full-suite run surfaced 3 pre-existing failures unrelated to
new test files: `eval/test_dev_sweep.py`'s `compute_k_config_hash()`
(and its own test) build a `SimpleNamespace` "fake args" object to call
`run_eval._config_hash()` directly — adding the new
`isco_catalogue_profile` field to that function's payload (§6) left
those fake-args objects missing the attribute. This is the **identical
integration-seam class of issue, fixed the identical way**, as an
earlier, already-committed fix (`git log` on `eval/dev_sweep.py` shows
commit `b33a9bb`, "Fix integration-seam gap: dev_sweep.py's config-hash
fake args predated --sre/--use-llm-reranker") — adding
`isco_catalogue_profile="legacy"` (the value every B2 K-sweep case
actually ran with; B2 predates this task) to `compute_k_config_hash()`'s
own fake-args construction and to `eval/test_dev_sweep.py`'s matching
test fixture, plus `eval/test_model_free_isco_evaluation.py`'s own
similar `_hash_args()` helper for two `_config_hash` unit tests. **No
assertion was weakened, no test was skipped/xfailed, and `eval/
configs/b1_frozen.json` was not touched** — confirmed by `git diff
--stat` on that path returning no output. All 2037 tests pass after
this fix, with zero remaining failures.

## 8. Confirmation: no official raw workbook/WISCO/Qdrant/model/LLM/evaluation operation occurred

- No official ILO workbook was downloaded, parsed, or copied in this
  task (Task 20's already-downloaded, git-ignored local artifacts were
  the only inputs to the loader/builder validation runs in §3/§4).
- No WISCO file, path, code, title, split, or output was read anywhere
  — confirmed by the loader's AST-based no-WISCO-import test and by
  every retrieval-path test in this task using `FakeQdrantClient`/
  `FakeEmbedder` monkeypatched at module level.
- No Qdrant instantiation, connection, query, count, build, populate,
  mutation, or deletion occurred at any point in this task, in either
  the implementation or the test suite.
- No `SentenceTransformer` model was loaded or downloaded.
- No Ollama, CrewAI, LLM, or paid API call was made.
- No `eval/run_eval.py` invocation against real data, and no
  `eval/analyze.py`/`eval/analyze_wisco_tier1.py` invocation, occurred.
  Every `run_eval.main()` call in this task's own test suite used a
  monkeypatched `ISCOClassifier`/fake store.
- No B1 re-freeze, B2 sweep, or ISIC/ISCED/SRE evaluation occurred.

## 9. Protected-branch confirmation

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| every prior `reviewer2-*` task/integration branch (through `reviewer2-isco08-official-catalogue-reconciliation-20260808`) | not touched |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push occurred. No PR was created.

## 10. Precise next step (needs separate approval)

Per `OFFICIAL_ISCO08_RUNTIME_AND_FLAT_COMPARATOR.md` §10 and
`ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md` §6: (1) human review of the
full Task 20 mismatch list and an explicit, approved decision on final
catalogue scope; (2) a dedicated source-data correction task for
`backend/rag/load_full_isco.py`; **then, and only then**, (3) a local
official-source availability check followed by a separately approved
`backend.rag.build_official_isco08_collections --execute` run (this
task's `--execute` path still refuses unconditionally — a future task
must implement the actual build logic under its own explicit approval);
(4) a smoke gate against a handful of known cases; (5) a fresh, full
controlled WISCO evaluation using `--isco-catalogue-profile
official_ilo2021_v1` for both the hierarchical and full-unit-group flat
systems, reusing Task 17's unchanged canonical heldout CSV; (6) a new
fail-closed analysis task (Task 18-style) over those results. **None of
these six steps were performed in this task.**
