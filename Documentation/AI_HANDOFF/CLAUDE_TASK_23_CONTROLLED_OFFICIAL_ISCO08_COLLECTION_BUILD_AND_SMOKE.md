# Task 23 — Controlled Official ILO ISCO-08 Collection Build and Five-Case Smoke Gate

## Authorisation and purpose

This is a separately authorised, controlled local operation. It authorises:

1. one guarded build of the five new official ILO ISCO-08 Qdrant collections; and
2. only after a fully successful build, two five-case operational smoke runs using
   the official profile.

It does **not** authorise a full WISCO benchmark, accuracy analysis, statistical
comparison, manuscript-result claim, re-freeze of B1, B2 sweep, reranking,
Ollama/LLM use, paid API calls, ISIC, ISCED, or SRE evaluation.

The intended result is operational readiness evidence only: the primary-source
ILO catalogue can be safely indexed locally and the new official flat and
hierarchical ISCO paths can be exercised on five controlled cases.

## Exact source and branch requirements

1. Fetch `origin` and verify the exact source branch and commit before branching:

   ```text
   reviewer2-official-isco08-collection-builder-20260809
   1bfe64fb4ee6d5e2268272e7fde710c86fd48018
   ```

2. Confirm the working tree is clean before creating a branch.
3. Create exactly this new branch:

   ```text
   reviewer2-official-isco08-collection-build-smoke-20260809
   ```

4. Do not use `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
   `git pull`, force-push, or create a PR.
5. Do not alter any protected or prior-task branch, including `master`,
   `conference1-b2-evaluation`, `reviewer2-wip-snapshot-20260807`,
   `reviewer2-b2-integration-20260807`, or any earlier `reviewer2-*` branch.

## Non-negotiable evidence and safety boundaries

- Use only the exact `official_ilo2021_v1` profile.
- Use only the verified local primary-source artefacts created in Task 20:

  ```text
  catalogue:
    eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv

  expected catalogue SHA-256:
    29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3

  metadata:
    eval/verified_catalogue_counts.yaml
  ```

- Verify that the metadata still records official counts `10/43/130/436`, and
  confirms that WISCO is not a catalogue source.
- The five target collection names and exact required point counts are:

  | Target collection | Expected count |
  | --- | ---: |
  | `isco08_major_groups_ilo2021_v1` | 10 |
  | `isco08_submajor_groups_ilo2021_v1` | 43 |
  | `isco08_minor_groups_ilo2021_v1` | 130 |
  | `isco08_unit_groups_ilo2021_v1` | 436 |
  | `isco08_unit_groups_flat_ilo2021_v1` | 436 |

- Do not mutate legacy collections. Do not delete, drop, overwrite, recreate,
  repair, or clean up any Qdrant collection.
- Every generated manifest, temporary five-row input, and smoke output must
  remain beneath a new Git-ignored `eval/local_runs/` directory. Never commit
  raw CSVs, manifests, embeddings, collection exports, or benchmark results.
- No WISCO source package, split manifest, or prior raw run artefact may be
  modified.

## Mandatory stop conditions

Stop immediately, do not retry, and do not repair or delete anything if:

1. the source SHA, metadata validation, or expected official counts do not
   match;
2. localhost Qdrant is unavailable;
3. even one of the five target collection names already exists, whether empty
   or non-empty;
4. the dry-run plan does not list exactly the five names and counts above;
5. the actual builder returns a non-success status, its manifest is not a
   success manifest, or post-build verification does not match exactly;
6. either smoke command fails any required pass condition;
7. an unexpected LLM, Ollama, paid API, ISIC, ISCED, or SRE operation is
   attempted.

If a real build has already completed before a later smoke failure, do not
delete or rebuild the collections. Record that fact clearly and stop.

## Part A — Independent read-only preflight

Before any live mutation:

1. Independently inspect
   `backend/rag/build_official_isco08_collections.py` and its tests.
2. Confirm all of the following directly from the code/tests:
   - `--dry-run` overrides `--execute`;
   - both exact acknowledgements are required before dependency construction:
     `--confirm-profile official_ilo2021_v1` and
     `--allow-local-qdrant-mutation`;
   - all five target names are checked for absence before the first create or
     upsert call;
   - no overwrite path exists;
   - a partial-build failure is reported with a failure manifest and no
     automatic remediation.
3. Verify the catalogue file's SHA-256, the metadata counts/source field, and
   that local Qdrant is reachable at `localhost:6333`.
4. Obtain a read-only pre-build inventory of Qdrant collections, including
   point counts where available. Confirm that every one of the five targets is
   absent.
5. Create one new ignored output root:

   ```text
   eval/local_runs/official_isco08_collection_build_<UTC timestamp>/
   ```

6. Run the builder once in dry-run mode, using the same source inputs, profile,
   and output-manifest path planned for the real command. Confirm the plan
   exactly matches the five target names and expected counts. The dry run must
   not load a real embedder or instantiate a Qdrant client.

Use `python -m backend.rag.build_official_isco08_collections --help` to inspect
the exact supported dry-run syntax before invoking it. Do not invent CLI flags.

## Part B — One authorised local build

Only if every Part A gate passes, run exactly one real builder command. Use the
new ignored output root and save `build_manifest.json` there:

```bash
python -m backend.rag.build_official_isco08_collections \
  --catalogue eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv \
  --metadata eval/verified_catalogue_counts.yaml \
  --profile official_ilo2021_v1 \
  --output-manifest eval/local_runs/official_isco08_collection_build_<UTC timestamp>/build_manifest.json \
  --execute \
  --confirm-profile official_ilo2021_v1 \
  --allow-local-qdrant-mutation
```

After the command:

1. Validate that the manifest status is exactly success.
2. Confirm it records:
   - profile `official_ilo2021_v1`;
   - the required catalogue SHA-256;
   - model identity `intfloat/multilingual-e5-small`;
   - embedding vector dimension 384;
   - the exact five planned/created/verified targets.
3. Use read-only Qdrant inspection to confirm that all five targets now exist
   with exact counts `10/43/130/436/436`.
4. Confirm legacy collection names and their point counts did not change.

Do not run the command a second time for any reason.

## Part C — Five-case official-profile smoke gate

Perform Part C only if Part B succeeds fully.

### Selection

1. Select exactly five heldout WISCO rows by a deterministic documented rule.
2. The five rows must span at least three languages and at least three ISCO
   major groups.
3. Record only the five non-sensitive benchmark IDs and selection rule in the
   final report.
4. Materialise a temporary five-row input CSV only under the ignored Task 23
   output root. Do not change the original WISCO package or split manifest.

### Commands

First inspect `python eval/run_eval.py --help`. Use only supported flags and
the existing Task 17/21 evaluator conventions. Do not invent flags.

Run exactly these two types of smoke run against the same five rows:

1. **Official flat comparator**

   ```text
   --system flat
   --use-llm-reranker off
   --isco-catalogue-profile official_ilo2021_v1
   ```

2. **Official hierarchical retrieval**

   ```text
   --system hierarchical
   --use-llm-reranker off
   --isco-catalogue-profile official_ilo2021_v1
   --require-genuine-hierarchical
   --max-stage-latency-ms 30000
   ```

Write both outputs only under the same ignored Task 23 output root. Do not run
any reranker, and do not pass a reranker model.

### Required pass conditions

The smoke gate passes only when all of the following are true:

1. Exactly five clean output rows exist for each system.
2. Every flat row has method exactly
   `flat_isco08_official_ilo2021_v1` and a syntactically valid four-digit
   predicted ISCO code.
3. Every hierarchical row has method exactly
   `hierarchical_isco08_official_ilo2021_v1`, valid non-empty stage 1 through
   stage 4 evidence, no fallback, and no stage over the 30,000 ms cap.
4. There are no row errors.
5. Reranking is disabled, with no reranker trace, cost, or token activity.
6. ISIC, ISCED, and SRE are not constructed or run for these ISCO-only rows.
7. Read-only post-smoke Qdrant inspection confirms all five official
   collections still have exact counts `10/43/130/436/436`.

This is operational smoke evidence only. Do not compute or report accuracy,
precision, recall, Wilson intervals, McNemar tests, latency comparisons, or a
full WISCO result.

## Part D — Tests, report, and stopping point

1. Run the relevant focused builder, official-profile, and evaluator tests,
   then run:

   ```bash
   pytest backend/tests eval/ -q
   ```

2. Do not repair unrelated tests or change production behavior merely to make
   tests pass.
3. Commit only:

   ```text
   Documentation/AI_HANDOFF/CLAUDE_TASK_23_FINAL_REPORT.md
   ```

4. Push only the new Task 23 branch. Do not create a PR.
5. The final report must include:
   - exact base SHA, branch, final SHA, push confirmation, and clean-tree
     status;
   - all exact commands executed;
   - input paths and verified hashes;
   - dry-run result;
   - pre-build and post-build Qdrant inventories;
   - whether the actual build was performed and the success/failure manifest
     status plus ignored path;
   - the five smoke IDs and deterministic selection rule;
   - method-label, four-digit-code, hierarchical-stage, no-fallback, no-error,
     no-reranker, and no-ISIC/ISCED/SRE checks;
   - pre- and post-smoke official collection counts;
   - focused and full test output;
   - protected branch status;
   - a precise list of any stop condition encountered.
6. Include this exact evidence statement:

   ```text
   A successful build proves only that the official ILO catalogue was locally
   indexed and verified. A five-case smoke run proves operational path integrity
   only. Neither establishes WISCO accuracy, real-LFS validation,
   ISIC/ISCED/SRE performance, latency claims, or manuscript-ready comparative
   results.
   ```

7. Stop after pushing the report. Do not begin a full official-profile
   benchmark, reranking run, analysis, paper draft, B1 re-freeze, or B2 sweep.
