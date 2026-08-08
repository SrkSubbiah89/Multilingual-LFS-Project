# Task 22 — Implement Auditable Official ISCO Collection Builder

## Purpose

Implement and hermetically test the actual local Qdrant collection-build capability for the verified official ISCO-08 profile. This task changes code only. It must not connect to Qdrant, load an embedding model, read WISCO, or execute a build.

The user-approved official scope is:

```text
ILO English ISCO-08 exactly:
10 major groups, 43 sub-major groups, 130 minor groups, 436 unit groups,
with official code, parent, and English title values.
```

Task 21 intentionally made `--execute` refuse unconditionally. Task 22 may replace that refusal with a real, guarded execution implementation, but no Task 22 command may invoke it against a live service. A separate future task will independently inspect this code before authorizing execution.

## Required source state

Fetch refs and verify:

| Role | Branch | Required SHA |
|---|---|---|
| Task 21 implementation base | `reviewer2-official-isco08-runtime-and-flat-comparator-20260808` | `d784a45bafa3a1c87cd090f39457511b5effb303` |

Require a clean working tree. Create and work only on:

```text
reviewer2-official-isco08-collection-builder-20260809
```

Push only that branch. Do not create a PR. Do not merge, rebase, reset, clean, stash, pull, force-push, or change protected/prior branches.

## Scope and non-negotiable constraints

1. Preserve `backend/rag/official_isco08_catalogue.py` as the sole source-validation authority.
2. Preserve legacy resources and unversioned collection names unchanged.
3. Preserve the official profile/flat comparator labels introduced in Task 21.
4. Do not change any static hand-authored list to appear official.
5. Do not commit an official workbook, normalized official catalogue, raw catalogue rows, vectors, Qdrant dump, WISCO data, or build artifact.
6. Do not change the default legacy profile behavior.
7. Do not touch `eval/configs/b1_frozen.json` or weaken the B2 gate.

## Actual builder execution design

Refactor:

```text
backend/rag/build_official_isco08_collections.py
```

to support a future real build only with all safeguards below.

### Required CLI contract

Support:

```text
--catalogue <local-normalized-catalogue-path>
--metadata <verified-catalogue-counts-yaml-path>
--profile official_ilo2021_v1
--output-manifest <ignored-local-path>
--dry-run
--execute
--confirm-profile official_ilo2021_v1
--allow-local-qdrant-mutation
```

Rules:

1. No `--execute` means dry-run behavior only. It must validate source and print/write a collection plan, but never import/instantiate Qdrant, an embedder, or a classifier.
2. `--execute` must require both:
   - exact `--confirm-profile official_ilo2021_v1`;
   - `--allow-local-qdrant-mutation`.
3. Missing either acknowledgement must fail before any Qdrant/embedder import or local connection.
4. Before any Qdrant/embedder import, the builder must:
   - load the local official catalogue through `load_official_catalogue()`;
   - validate the exact recorded SHA-256;
   - validate 10/43/130/436;
   - validate source/profile identity;
   - derive the five collection plans.
5. The execution path may only target:

```text
isco08_major_groups_ilo2021_v1
isco08_submajor_groups_ilo2021_v1
isco08_minor_groups_ilo2021_v1
isco08_unit_groups_ilo2021_v1
isco08_unit_groups_flat_ilo2021_v1
```

6. It must reject every other profile or target name.
7. It must reject an existing target collection, whether empty or nonempty. No auto-replace, delete, recreate, upsert-into-existing, alias swap, or overwrite is permitted.
8. It must accept only a local Qdrant target. Default host/port must be explicitly documented and no remote URL/token CLI option may be introduced.
9. It must use the project’s established local embedding-model configuration and record its resolved model identity/version in the manifest. Do not introduce a paid embedding service or external inference path.
10. It must never construct an ISCOClassifier, ISICClassifier, ISCEDClassifier, SRE, CrewAI agent, or reranker.

### Lazy dependency boundaries

At module import and dry-run time, do not import:

- `qdrant_client`;
- `sentence_transformers`;
- `backend.agents.isco_classifier`;
- any WISCO or evaluation module.

Import Qdrant and the embedding dependency lazily only inside the guarded live execution function, after all CLI acknowledgements and catalogue validation succeed.

### Collection records and payloads

Build records only from `OfficialCatalogueRecord` values returned by the verified loader. For each vector payload, include:

- `code`;
- `level`;
- `parent_code`;
- `title_en`;
- `profile`;
- `source_catalogue_sha256`;
- explicit `collection_role` equal to one of `major`, `submajor`, `minor`, `unit_hierarchical`, or `unit_flat`;
- deterministic `embedding_text`.

The flat collection must contain exactly the same 436 official unit records as the hierarchical unit collection, but under its distinct `unit_flat` collection role and identity. It must never contain one-, two-, or three-digit records.

Vectors must be created only from the source-validated record’s deterministic `embedding_text`. Do not add WISCO titles, benchmark labels, prediction data, or any hand-authored correction.

### Build ordering and validation

Execution must:

1. perform all preflight checks before mutating Qdrant;
2. check that all five target names are absent before creating any target;
3. resolve and record the local embedding model identity before beginning any write;
4. create/write level-specific collections in deterministic collection-name and code order;
5. verify after each completed collection:
   - exact planned point count;
   - every payload profile/source hash/role matches the plan;
   - all codes match that collection’s expected hierarchy level;
   - flat collection has exactly 436 valid four-digit codes;
6. write an ignored local JSON manifest only after all five collections have been verified;
7. include in the manifest:
   - UTC build timestamp;
   - builder code SHA-256;
   - catalogue/metadata paths and hashes;
   - profile;
   - local Qdrant host/port;
   - resolved embedding model identity;
   - collection names;
   - planned/observed counts;
   - payload-schema version;
   - execution acknowledgements;
   - verification status;
   - any failure reason.

If an execution error occurs after one or more new target collections were created, the builder must:

- fail nonzero;
- write an ignored local failure manifest identifying exactly which targets were created/verified/partial;
- never delete or overwrite automatically;
- clearly state that later remediation requires a separate explicit task.

Do not label a partial build as successful.

## Code organization

Use small dependency-injected helpers for Qdrant client and embedder creation so that all live dependencies can be replaced with fakes in tests. Reuse existing project vector/Qdrant utilities where technically compatible, but do not route the official builder through legacy curated data.

If vector dimension or collection-schema setup must be inferred from an existing local implementation, reuse the existing configuration rather than inventing a new embedding dimension. Document the reused source.

Do not duplicate hierarchy search logic. This task is build logic only.

## Required hermetic tests

Expand/create tests under:

```text
backend/tests/test_build_official_isco08_collections.py
```

and add focused tests only where necessary. Every test must use synthetic temporary catalogue/metadata files, a fake Qdrant client, and a fake embedder. No test may access:

- a real official workbook;
- a real normalized official catalogue;
- WISCO;
- a live Qdrant instance;
- network;
- a real embedding model;
- Ollama/CrewAI/LLM/paid API.

Test at minimum:

1. dry run validates source and has no Qdrant/embedder import or construction;
2. execute missing either acknowledgement fails before dependency construction;
3. execute with wrong profile fails before dependency construction;
4. invalid/missing/hash-mismatched catalogue fails before dependency construction;
5. target collection names are exactly the five versioned official names;
6. a non-absent target collection causes a preflight failure before any create/write;
7. all five collections are planned before the first mutation;
8. code ordering and collection ordering are deterministic;
9. level-specific counts and flat 4-digit-only rule;
10. payload schema includes all required identity/provenance fields;
11. embeddings use only deterministic official record embedding text;
12. successful fake build verifies exact counts/payloads and writes a success manifest;
13. fake partial failure returns nonzero, writes a failure manifest, and performs no automatic deletion/overwrite;
14. a failed collection verification is not marked success;
15. builder source remains free of WISCO/evaluation/classifier imports;
16. legacy collection names are never targets;
17. no remote Qdrant URL/token configuration is accepted;
18. official runtime/flat profile compatibility remains intact;
19. B1 frozen config remains untouched;
20. current full test suite is green.

Do not weaken existing tests, mark tests xfail, skip tests, or change unrelated files to force success.

## Documentation

Update:

```text
Documentation/Conference_I_Reviewer_2/OFFICIAL_ISCO08_RUNTIME_AND_FLAT_COMPARATOR.md
Documentation/Conference_I_Reviewer_2/ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md
Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md
Documentation/Conference_I_Reviewer_2/REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md
Documentation/Conference_I_Reviewer_2/README.md
```

Document:

1. the exact future dry-run and execution command forms;
2. both mandatory execution acknowledgements;
3. local-only Qdrant boundary;
4. target-absence/no-overwrite policy;
5. partial-build behavior and remediation rule;
6. manifest fields and verification requirements;
7. that Task 22 did not execute a build;
8. that no official collection exists yet from this code;
9. that no new benchmark/accuracy result exists.

## Strictly prohibited this task

Do not:

- run the builder with `--execute`;
- connect to, query, count, create, mutate, rebuild, or delete any Qdrant collection;
- instantiate/download a real embedding model;
- read any WISCO file/path/output;
- invoke `eval/run_eval.py`, analysis, benchmark, or classifier inference;
- invoke Ollama, CrewAI, LLM, paid API, or external inference;
- download/parse/copy the official raw ILO workbook;
- change project static hierarchy lists;
- touch B1/B2 source/configuration except new test compatibility if a narrowly justified API seam requires it;
- create a PR.

## Tests and final report

Run focused tests for the builder, official catalogue/profile, model-free evaluator, strict hierarchy guard, and docs. Then run:

```bash
pytest backend/tests eval/ -q
```

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_22_FINAL_REPORT.md
```

Include:

1. source SHA, branch, final commit SHA, push confirmation, and clean-tree status;
2. exact changed tracked files;
3. execution preflight/acknowledgement contract;
4. lazy-import/dependency-isolation evidence;
5. exact target collection names/counts/payload fields;
6. no-overwrite/partial-build behavior;
7. manifest contract;
8. focused and full test results;
9. confirmation that no Qdrant/model/WISCO/evaluation/official-workbook operation occurred;
10. protected-branch status;
11. one status:

```text
OFFICIAL_ISCO08_COLLECTION_BUILD_IMPLEMENTATION_READY: yes
```

or:

```text
OFFICIAL_ISCO08_COLLECTION_BUILD_IMPLEMENTATION_READY: no
```

12. a precise next separately approved action: local official-source availability preflight, actual collection build, and a five-case official-profile smoke gate.

After pushing the final report, stop. Do not perform a live build or any evaluation.
