# Task 22 Final Report — Implement Auditable Official ISCO Collection Builder

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_22_IMPLEMENT_AUDITABLE_OFFICIAL_COLLECTION_BUILDER.md`.
Code and hermetic-test implementation only — no Qdrant connection, no
embedding model load, no WISCO read, no build execution.

```text
OFFICIAL_ISCO08_COLLECTION_BUILD_IMPLEMENTATION_READY: yes
```

## 1. Source SHA, branch, final commit, push, clean-tree status

| | |
|---|---|
| Base branch | `reviewer2-official-isco08-runtime-and-flat-comparator-20260808` |
| Required SHA | `d784a45bafa3a1c87cd090f39457511b5effb303` |
| Verified `origin` SHA | `d784a45bafa3a1c87cd090f39457511b5effb303` — match |
| New branch | `reviewer2-official-isco08-collection-builder-20260809` |
| Final commit SHA | recorded after this report's commit (see push confirmation below) |

Working tree was clean before branching and remained clean throughout
except for the files listed in §2.

## 2. Exact changed tracked files

```text
backend/rag/build_official_isco08_collections.py   (rewritten: real, guarded --execute added)
backend/tests/test_build_official_isco08_collections.py   (expanded 7 -> 26 tests)
Documentation/Conference_I_Reviewer_2/OFFICIAL_ISCO08_RUNTIME_AND_FLAT_COMPARATOR.md
Documentation/Conference_I_Reviewer_2/ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md
Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md
Documentation/Conference_I_Reviewer_2/REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md
Documentation/Conference_I_Reviewer_2/README.md
Documentation/AI_HANDOFF/CLAUDE_TASK_22_FINAL_REPORT.md (this file)
```

`backend/rag/official_isco08_catalogue.py` (the sole source-validation
authority) was **not** modified — `execute_build()` calls its existing
`load_official_catalogue()`/`records_by_level()` unchanged, per the
task's own preservation requirement. No static hand-authored hierarchy
list, legacy collection name, or `eval/configs/b1_frozen.json` was
touched (confirmed: `git diff --stat` on that path returns no output).

## 3. Execution preflight/acknowledgement contract

Enforced by `check_execution_acknowledgements()`, called first, before
any other code in `execute_build()`:

| Check | Failure |
|---|---|
| `--confirm-profile` exactly `"official_ilo2021_v1"` | `BuildAcknowledgementError` |
| `--confirm-profile` equals `--profile` | `BuildAcknowledgementError` |
| `--allow-local-qdrant-mutation` present | `BuildAcknowledgementError` |

Then, still before any Qdrant/embedder import: catalogue re-validation
via the unmodified `load_official_catalogue()` (hash, 10/43/130/436
counts, format, parent links, titles). Only after all of that succeeds
does `execute_build()` lazily construct the Qdrant client/embedder and
make its **first** live call — `get_collections()` — to preflight-check
that all five target names are absent; any conflict raises
`BuildPreflightError` before any `create_collection`/`upsert` call.

## 4. Lazy-import/dependency-isolation evidence

`qdrant_client`/`sentence_transformers` are imported **only** inside
`_default_qdrant_client_factory()`, `_default_embedder_factory()`, and
`_create_and_populate_collection()` (for `qdrant_client.models`) — three
function bodies, never at module top level. Two hermetic tests prove
this precisely: `test_dry_run_module_top_level_has_no_qdrant_or_embedder_import`
walks only `tree.body` (module-level AST nodes, not `ast.walk()`) and
asserts zero qdrant/sentence_transformers references there;
`test_lazy_imports_are_confined_inside_function_bodies` is the positive
control, confirming those imports genuinely exist, but only nested
inside exactly those three `FunctionDef` nodes (so the first test can't
be vacuously passing because the feature doesn't exist). A third test,
`test_dry_run_never_constructs_qdrant_or_embedder_object`, monkeypatches
both factory functions to raise if ever called and confirms a dry-run
plan still succeeds. Every execution-path test in this task supplies
`qdrant_client_factory`/`embedder_factory` fakes to `execute_build()` —
`_default_qdrant_client_factory`/`_default_embedder_factory` (the only
functions that would ever perform a real import/connection) are never
invoked by any test, script, or command in this task.

## 5. Exact target collection names/counts/payload fields

Unchanged from Task 21: `isco08_major_groups_ilo2021_v1` (10),
`isco08_submajor_groups_ilo2021_v1` (43), `isco08_minor_groups_ilo2021_v1`
(130), `isco08_unit_groups_ilo2021_v1` (436, `collection_role=
"unit_hierarchical"`), `isco08_unit_groups_flat_ilo2021_v1` (436,
`collection_role="unit_flat"`, the exact same 436 records as the
hierarchical unit collection, verified to contain zero 1/2/3-digit
codes). Every payload written by `_build_payload()` includes exactly:
`code`, `level`, `parent_code`, `title_en`, `profile`,
`source_catalogue_sha256`, `collection_role`, `embedding_text` — nothing
else (no WISCO title, benchmark label, prediction, or hand-authored
correction; embedding vectors are computed only from each record's own
deterministic `embedding_text`, prefixed `"passage: "` per the existing
E5 convention). Legacy collection names (`isco08_major_groups`, ...,
`isco_occupations`) never appear as targets — confirmed by
`test_target_builds_use_only_official_versioned_names`.

## 6. No-overwrite/partial-build behavior

`test_existing_target_collection_blocks_before_any_mutation` confirms:
if even one of the five target names already exists (empty or not),
`BuildPreflightError` is raised, `client.created_order` remains empty,
and a `preflight_failed_existing_target` manifest is written — zero
collections are ever created. `test_partial_failure_writes_failure_manifest_and_raises`
simulates a write failure on the third target (after two were already
created): the failure propagates as `BuildExecutionError`, a
`failed_partial_build` manifest names exactly the created/partial
targets and states plainly that remediation requires a separate
explicit task, and nothing already created is deleted, recreated, or
overwritten. `test_failed_verification_is_not_marked_success` confirms a
corrupted post-write payload is caught by `_verify_collection()` and
also produces a failure manifest, never a success one.

## 7. Manifest contract

Written by `_write_manifest()` only after acknowledgement, catalogue
validation, and preflight all pass. Success manifest (`status:
"success"`) includes: `build_timestamp_utc`, `builder_script_sha256`,
`catalogue_path`/`catalogue_sha256`, `metadata_path`, `profile`,
`qdrant_host`/`qdrant_port`, `embedding_model_identity`
(`intfloat/multilingual-e5-small`, reused from `backend/rag/
hierarchical_store.py`/`backend/rag/load_full_isco.py`'s existing
configuration — no new dimension invented), `embedding_vector_dim`
(384), `payload_schema_version`, both acknowledgements,
`planned_targets`, `targets_created_or_partial`, and `verified_targets`
(per-target expected/observed counts). A failure manifest additionally
carries `failure_reason` and, for a mid-build failure,
`targets_verified_before_failure` and an explicit `remediation_note`.

## 8. Focused and full test results

```
python -m pytest backend/tests/test_build_official_isco08_collections.py backend/tests/test_official_isco08_catalogue.py backend/tests/test_official_isco08_profiles.py backend/tests/test_isco_classifier_official_profile.py eval/test_official_isco08_profile_evaluator.py eval/test_model_free_isco_evaluation.py eval/test_require_genuine_hierarchical.py eval/test_docs_consistency.py -q
→ 103 passed in 1.51s

python -m pytest backend/tests eval/ -q
→ 2056 passed, 1 deselected, 1 warning in 299.16s (0:04:59)
```

**Zero failures.** `2056 = 2037` (Task 21 baseline) `+ 19` net new
tests (`test_build_official_isco08_collections.py` expanded from 7 to
26). Two pre-existing tests in that file were replaced rather than kept
verbatim, both because their premise was the exact behavior this task
was instructed to change, not because of weakening: `test_builder_module_does_not_import_qdrant_or_embedder`
used `ast.walk()` (recursive), which would have falsely flagged this
task's own deliberately-nested lazy imports — replaced by the
correctly-scoped top-level-only check in §4 plus its positive control;
`test_cli_execute_flag_refused_and_raises_before_any_planning` asserted
Task 21's literal unconditional-refusal message, which Task 22's own
brief explicitly authorized replacing with a real, guarded
implementation — replaced by the acknowledgement-gated CLI tests in
§3/§6. No other existing test in the repository was modified, skipped,
or weakened.

## 9. Confirmation: no Qdrant/model/WISCO/evaluation/official-workbook operation occurred

- No `QdrantClient`, `SentenceTransformer`, `qdrant_client.models`
  object backed by a real service, ISCOClassifier, ISICClassifier,
  ISCEDClassifier, SemanticRelationEngine, CrewAI agent, or reranker was
  ever constructed with real dependencies in this task — every test that
  reaches `execute_build()`'s post-acknowledgement code supplies a
  `FakeQdrantClient`/`FakeEmbedder` via dependency injection (§4).
- No official ILO workbook was downloaded, parsed, or copied — the two
  already-downloaded, git-ignored Task 20 local artifacts were the only
  catalogue inputs used, read-only, by the test fixtures' own small
  synthetic CSVs (the real Task 20 files were not even read in this
  task's tests, which use small hand-built fixtures per the task's own
  hermetic-test requirement).
- No WISCO file, path, code, title, split, or output was read —
  confirmed by `test_builder_module_has_no_wisco_evaluation_or_classifier_imports`
  (AST-based, extended from Task 21's check to also cover
  evaluation/classifier modules).
- No `eval/run_eval.py`, `eval/analyze.py`, or `eval/analyze_wisco_tier1.py`
  invocation against real data occurred.
- No Ollama, CrewAI, LLM, or paid API call was made.
- `--execute` was never passed to the real CLI in this task — every
  test that exercises the execute code path calls `execute_build()`
  directly with injected fakes, or calls `main()` with the default
  `_default_qdrant_client_factory`/`_default_embedder_factory`
  monkeypatched to raise if ever invoked.

## 10. Protected-branch status

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| every prior `reviewer2-*` task/integration branch (through `reviewer2-official-isco08-runtime-and-flat-comparator-20260808`) | not touched |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push occurred. No PR was created.

## 11. Precise next separately approved action

1. **Local official-source availability preflight**: on a machine with a
   real local Qdrant instance, confirm the normalized catalogue and
   `eval/verified_catalogue_counts.yaml` are present and still
   hash-valid, and confirm none of the five target collection names
   already exist.
2. **Actual collection build**: a future task must independently review
   this execution code (§3-§7), then run `python -m backend.rag.
   build_official_isco08_collections --catalogue <path> --metadata
   <path> --profile official_ilo2021_v1 --output-manifest <path>
   --execute --confirm-profile official_ilo2021_v1
   --allow-local-qdrant-mutation` for real, under its own explicit
   approval — not performed here.
3. **A five-case official-profile smoke gate**: after a successful
   build, a small number of known cases run against the new collections
   to confirm basic sanity before any full evaluation.

None of these three steps were performed in this task. The catalogue-
correction staged plan (`ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md`
§6) remains the logically prior gate to all of them.
