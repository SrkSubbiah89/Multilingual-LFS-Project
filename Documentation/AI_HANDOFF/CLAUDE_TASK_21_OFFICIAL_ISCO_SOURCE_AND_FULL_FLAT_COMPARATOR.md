# Task 21 — Official ISCO Source Correction and Full Flat Comparator Implementation

## Approved catalogue policy

The approved policy is:

```text
Adopt the verified primary ILO English ISCO-08 catalogue exactly:
10 major groups, 43 sub-major groups, 130 minor groups, and 436 unit groups,
with the official code, parent, and English title values.
```

The project’s existing hand-authored `_MAJOR`, `_SUBMAJOR`, `_MINOR`, and `_UNIT` lists are no longer acceptable as the authority for any future official ISCO-08 retrieval profile or benchmark. They contain 20 nonstandard codes, omit 14 official unit-group codes, and include title mismatches.

This task implements the official-source runtime/builder path and a separately named full-unit-group flat comparator. It does **not** build a collection, connect to Qdrant, execute a classifier, run WISCO, or produce a benchmark result.

## Required source state

Fetch refs and verify before changing anything:

| Role | Branch | Required SHA |
|---|---|---|
| Task 20 verified primary-source base | `reviewer2-isco08-official-catalogue-reconciliation-20260808` | `d161b3cc8b433744191a05d7e8c2698bc7025ba3` |

Require a clean working tree, then create and work only on:

```text
reviewer2-official-isco08-runtime-and-flat-comparator-20260808
```

Push only this new branch. Do not create a PR. Do not modify, merge into, rebase, reset, clean, stash, pull, force-push, or switch protected/prior branches.

## Provenance and licensing constraints

The official ILO workbooks and normalized catalogue are local, ignored source inputs. Do not commit them, copy their raw rows into a tracked file, or generate a hand-authored replacement catalogue in repository source.

The only authoritative official-source metadata is:

```text
eval/verified_catalogue_counts.yaml
```

The official runtime/builder path must:

1. take an explicit local normalized-catalogue path;
2. verify the input file’s SHA-256 matches `normalized_catalogue_sha256` in `eval/verified_catalogue_counts.yaml`;
3. verify exact counts 10/43/130/436;
4. validate format, uniqueness, parent links, and nonblank English titles before any node/collection definition is produced;
5. fail closed with an actionable `OfficialISCO08CatalogueError` if a file is missing, hash mismatches, metadata is missing/malformed, counts differ, code format is invalid, parent links fail, or source rows are ambiguous;
6. never fall back silently to the old static lists when the requested official profile is unavailable.

The official runtime/builder path must not read or import WISCO paths, titles, codes, labels, splits, or outputs. Tests must prove this by exercising the source-data dependency boundary and by static/path checks where practical.

## Legacy compatibility and claim controls

Preserve the existing curated `isco_occupations` and hand-authored hierarchy resources only as an explicitly labelled legacy/integration path for backward compatibility. Do not delete them in this task.

However:

1. legacy paths must not be selectable by an official profile;
2. no official-profile method may emit `flat_semantic`;
3. all official-profile methods must use a separate collection identity and explicit method label;
4. documentation must state that legacy resources are not eligible for four-digit ISCO benchmark accuracy;
5. a missing official catalogue/collection must be surfaced explicitly, not silently described as official retrieval.

Do not change the default public `ISCOClassifier.classify(text)` result behavior without a compatibility test. Add official selection only through explicit, clearly named configuration/profile arguments and CLI choices. This preserves existing integration behavior while preventing the evaluator from calling the legacy baseline an official four-digit comparator.

## Official catalogue loader

Add a focused production module, with a name appropriate to the existing package layout, for example:

```text
backend/rag/official_isco08_catalogue.py
```

It must provide typed or structured records for all four levels, including at least:

- code;
- hierarchy level;
- parent code where applicable;
- official English title;
- deterministic embedding text;
- source/profile identity;
- source-catalogue SHA-256.

It must validate the normalized local official catalogue against `eval/verified_catalogue_counts.yaml` before returning records.

It must not hard-code a complete catalogue row list. Small synthetic record lists are allowed only in test files.

Use project dependencies already present. Do not add a new dependency unless there is no viable existing/standard-library route and document the reason.

## Versioned official collection definitions

Implement a new operator-only builder path, adapting existing patterns rather than duplicating search algorithms. A suggested file is:

```text
backend/rag/build_official_isco08_collections.py
```

The builder must accept:

```text
--catalogue <normalized-official-catalogue-path>
--metadata <verified-catalogue-counts-yaml-path>
--profile official_ilo2021_v1
--execute
```

Without `--execute`, it must dry-run and print/return only the validated collection plan and source hashes. It must not instantiate Qdrant, an embedding model, or a classifier.

With `--execute`, it may be used only in a later separately approved task.

For profile `official_ilo2021_v1`, define distinct collection names:

```text
isco08_major_groups_ilo2021_v1
isco08_submajor_groups_ilo2021_v1
isco08_minor_groups_ilo2021_v1
isco08_unit_groups_ilo2021_v1
isco08_unit_groups_flat_ilo2021_v1
```

Requirements:

1. the four hierarchical collections contain only their respective official level;
2. the flat collection contains exactly one record per official four-digit unit group and no 1/2/3-digit record;
3. all collection payloads include code, level, parent, title, profile identity, and source-catalogue hash;
4. collection plan counts must be exactly 10/43/130/436/436;
5. dry-run must fail closed if the input does not validate;
6. existing unversioned legacy collections must never be mutated by this builder;
7. no Qdrant operation occurs during Task 21.

## Official retrieval profiles

Adapt the existing generic hierarchy engine and stores. Do not write a new search algorithm.

### Hierarchical profile

Add an explicit official profile, such as:

```text
official_ilo2021_v1
```

that maps hierarchical ISCO retrieval to the four versioned official collection names. It must have a method label distinct from legacy hierarchy, such as:

```text
hierarchical_isco08_official_ilo2021_v1
```

It must retain the Task 13 integrity properties:

- parent-filtered staged retrieval;
- unseeded retry after a failed keyword-anchored attempt;
- no merged failed/successful traces;
- explicit unavailable/fallback labels only after the allowed retry path fails;
- configurable Qdrant timeout;
- no silent downgrade labelled as official hierarchy.

### Full-unit-group flat profile

Add a direct, unfiltered nearest-neighbor official flat profile that queries only:

```text
isco08_unit_groups_flat_ilo2021_v1
```

It must have a distinct method label:

```text
flat_isco08_official_ilo2021_v1
```

It must:

1. query only the versioned official four-digit unit-group flat collection;
2. return a prediction only if the chosen candidate code matches exactly `^[0-9]{4}$`;
3. expose a clear unavailable/empty/error method reason rather than return a coarse code;
4. never emit `flat_semantic`;
5. use no hierarchy beam or parent filter, so it is genuinely a flat comparator;
6. include profile/source hash in trace/metadata where current result schema permits;
7. preserve legacy flat behavior under an explicit legacy profile only.

### Evaluator wiring

Modify the evaluation harness only as needed to select a profile explicitly. Provide a fail-closed CLI/config choice such as:

```text
--isco-catalogue-profile official_ilo2021_v1
```

Requirements:

1. official-profile selection must reject a missing/invalid profile;
2. it must make legacy versus official profile unambiguous in the manifest/config hash and per-row method metadata;
3. it must not alter default legacy behavior unless a caller explicitly chooses the official profile;
4. it must remain model-free when `--use-llm-reranker off`;
5. it must still avoid constructing ISIC/ISCED/SRE for ISCO-only rows;
6. it must be compatible with Task 13 strict guard and `--require-genuine-hierarchical`;
7. it must never make the legacy Task 17 raw outputs appear to have used the official profile.

If a clean integration requires a different precise argument or module shape, use the existing project conventions, document the reason, and preserve every constraint above.

## Required tests

Add focused hermetic tests. Use only synthetic temporary catalogue files, fake Qdrant clients, and fake embedders. No official workbook, normalized official catalogue, WISCO artifact, live Qdrant, network, model, Ollama, CrewAI, or paid API may be used in tests.

At minimum cover:

1. valid synthetic official-style catalogue load with 10/43/130/436-equivalent parameterized expectations;
2. metadata missing/malformed;
3. input hash mismatch;
4. missing source file;
5. wrong counts;
6. malformed, duplicate, and non-four-digit unit codes;
7. invalid parent links;
8. nonblank title requirement;
9. loader never imports/reads WISCO;
10. builder dry-run plan has level-specific counts and versioned names;
11. builder dry-run never constructs Qdrant/embedder;
12. builder rejects invalid source before planning;
13. official hierarchical profile selects only versioned official collection names;
14. official flat profile selects only the versioned four-digit collection;
15. official flat rejects coarse candidate code rather than returning it;
16. official flat method label is distinct from `flat_semantic`;
17. official profile unavailable/error is explicit and never silently legacy;
18. legacy profile/default compatibility remains covered;
19. model-free evaluation profile selection does not construct LLM/reranker/ISIC/ISCED/SRE for ISCO-only rows;
20. strict hierarchical guard remains effective with the official profile;
21. manifest/config hash records profile and source identity.

Do not weaken existing tests. Do not modify an unrelated test to make the new implementation pass.

## Documentation

Add:

```text
Documentation/Conference_I_Reviewer_2/OFFICIAL_ISCO08_RUNTIME_AND_FLAT_COMPARATOR.md
```

Update:

```text
Documentation/Conference_I_Reviewer_2/ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md
Documentation/Conference_I_Reviewer_2/FLAT_BASELINE_COVERAGE_AUDIT.md
Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md
Documentation/Conference_I_Reviewer_2/REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md
Documentation/Conference_I_Reviewer_2/README.md
```

Document:

1. the verified official ILO source and local-hash requirement;
2. the versioned official collection names;
3. legacy versus official profile distinction;
4. the four-digit-only flat comparator definition;
5. exact future operator commands in dry-run form;
6. that no official collection has yet been built/populated;
7. that no post-correction WISCO evaluation/accuracy exists yet;
8. that Task 17 outputs remain superseded for standard-compliant accuracy;
9. WISCO remains controlled multilingual ISCO-08 title data, not real LFS validation;
10. ISIC/ISCED/SRE/reranking claims remain out of scope.

## Prohibited operations

Do not:

- download, parse, or copy the official raw workbook during this task;
- read WISCO data or outputs;
- execute any builder with `--execute`;
- instantiate/query/mutate/count/rebuild/delete Qdrant;
- instantiate/download SentenceTransformer;
- invoke Ollama, CrewAI, LLM, paid API, or external inference;
- invoke `eval/run_eval.py` or any benchmark/analysis command;
- modify an existing raw output or historical report;
- run a B1 re-freeze, B2 sweep, ISIC/ISCED/SRE evaluation;
- change legacy source data to look official;
- claim an accuracy, coverage, latency, scalability, or real-LFS result.

## Tests and final report

Run focused tests covering the new loader/builder/profiles plus:

```bash
pytest eval/test_analyze_wisco_tier1.py eval/test_model_free_isco_evaluation.py eval/test_require_genuine_hierarchical.py eval/test_docs_consistency.py -q
```

Then run:

```bash
pytest backend/tests eval/ -q
```

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_21_FINAL_REPORT.md
```

The final report must include:

1. source SHA, branch, final commit SHA, push confirmation, and clean-tree status;
2. exact changed tracked files;
3. official source metadata path and the runtime hash/count validation contract;
4. collection plan names/counts and proof no collection was built;
5. hierarchy and flat method labels/profile identity;
6. compatibility and no-silent-fallback evidence;
7. focused and full test results;
8. confirmation no official raw workbook/WISCO/Qdrant/model/LLM/evaluation operation occurred;
9. protected-branch confirmation;
10. clear status:

```text
OFFICIAL_ISCO08_RUNTIME_IMPLEMENTATION_READY: yes
```

or:

```text
OFFICIAL_ISCO08_RUNTIME_IMPLEMENTATION_READY: no
```

11. a precise next step that requires a separate approval: local official-source availability check, versioned collection build, smoke gate, then fresh full controlled WISCO flat and hierarchical evaluation.

After pushing the final report, stop. Do not build a collection or run a benchmark.
