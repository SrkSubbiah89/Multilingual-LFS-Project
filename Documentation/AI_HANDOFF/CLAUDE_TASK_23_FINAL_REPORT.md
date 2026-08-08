# Task 23 Final Report — Controlled Official ILO ISCO-08 Collection Build and Five-Case Smoke Gate

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_23_CONTROLLED_OFFICIAL_ISCO08_COLLECTION_BUILD_AND_SMOKE.md`.
This is the first task in this evidence line that performs a real,
authorized, live local Qdrant mutation.

**Zero stop conditions were encountered anywhere in this task.** Every
Part A-C gate passed on the first attempt; nothing was retried, deleted,
or repaired.

```text
A successful build proves only that the official ILO catalogue was locally
indexed and verified. A five-case smoke run proves operational path integrity
only. Neither establishes WISCO accuracy, real-LFS validation,
ISIC/ISCED/SRE performance, latency claims, or manuscript-ready comparative
results.
```

## 1. Source SHA, branch, final commit, push, clean-tree status

| | |
|---|---|
| Base branch | `reviewer2-official-isco08-collection-builder-20260809` |
| Required SHA | `1bfe64fb4ee6d5e2268272e7fde710c86fd48018` |
| Verified `origin` SHA | `1bfe64fb4ee6d5e2268272e7fde710c86fd48018` — match |
| New branch | `reviewer2-official-isco08-collection-build-smoke-20260809` |
| Final commit SHA | recorded after this report's commit (see push confirmation below) |

Working tree was clean before branching and remained clean throughout —
`git status --short` was empty immediately before this report's own
commit (every artifact this task produced lives under the git-ignored
`eval/local_runs/official_isco08_collection_build_20260808T211235Z/`
output root, confirmed via `git check-ignore -v`).

## 2. Part A — Independent read-only preflight

**Code inspection** (`backend/rag/build_official_isco08_collections.py`,
re-read fresh in this task): confirmed directly from source, all five
required properties —

1. `--dry-run` overrides `--execute`: `main()`'s `if args.execute and
   not args.dry_run:` guard.
2. Both exact acknowledgements required before any dependency
   construction: `check_execution_acknowledgements()` is the first call
   inside `execute_build()`, raising `BuildAcknowledgementError` before
   catalogue validation or any import.
3. All five target names checked for absence before the first
   create/upsert: `client.get_collections()` (the first live Qdrant
   call) is checked against all five planned names before any
   `create_collection` call.
4. No overwrite path exists: `_create_and_populate_collection()` only
   ever calls `create_collection` (never delete/recreate), and the
   preflight check blocks entirely if any target already exists.
5. Partial-build failure produces a failure manifest with no automatic
   remediation: the `except Exception` block in `execute_build()` always
   writes `status: "failed_partial_build"` plus a `remediation_note`
   before re-raising; nothing is deleted.

`python -m backend.rag.build_official_isco08_collections --help` was run
first to confirm the exact supported flags before any invocation — no
CLI flag was invented.

**Input verification:**

| Check | Result |
|---|---|
| Catalogue SHA-256 | `29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3` — exact match to the required value |
| Metadata `verified_counts` | `{major: 10, submajor: 43, minor: 130, unit: 436}` — exact match |
| Metadata WISCO-independence statement | Present verbatim: *"This verified catalogue record is based exclusively on a primary official ILO import... No WISCO file, title, code, split, or output was read..."* |
| Qdrant reachable at `localhost:6333` | Yes — `GET /collections` returned HTTP 200 |

**Pre-build Qdrant inventory** (`qdrant_pre_build_inventory.txt`):

| Collection | Points |
|---|---|
| `isco08_major_groups` (legacy) | 10 |
| `isco08_submajor_groups` (legacy) | 43 |
| `isco08_minor_groups` (legacy) | 131 |
| `isco08_unit_groups` (legacy) | 441 |
| `isco_occupations` (legacy) | 124 |
| `isco08_major_groups_ilo2021_v1` | **absent** |
| `isco08_submajor_groups_ilo2021_v1` | **absent** |
| `isco08_minor_groups_ilo2021_v1` | **absent** |
| `isco08_unit_groups_ilo2021_v1` | **absent** |
| `isco08_unit_groups_flat_ilo2021_v1` | **absent** |

All five targets confirmed absent — stop condition 3 did not trigger.

**Dry-run** (exact command, output root
`eval/local_runs/official_isco08_collection_build_20260808T211235Z/`):

```bash
python -m backend.rag.build_official_isco08_collections \
  --catalogue eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv \
  --metadata eval/verified_catalogue_counts.yaml \
  --profile official_ilo2021_v1 \
  --output-manifest eval/local_runs/official_isco08_collection_build_20260808T211235Z/dry_run_manifest.json
```

Exit 0. Plan listed exactly the five required names and counts
(`isco08_major_groups_ilo2021_v1`=10, `isco08_submajor_groups_ilo2021_v1`=43,
`isco08_minor_groups_ilo2021_v1`=130, `isco08_unit_groups_ilo2021_v1`=436,
`isco08_unit_groups_flat_ilo2021_v1`=436), each entry's
`source_catalogue_sha256` matching the required hash exactly. Printed
*"--dry-run: no Qdrant collection was created, connected to, or
modified."* — stop condition 4 did not trigger.

## 3. Part B — The one authorized local build

**Exact command run (once):**

```bash
python -m backend.rag.build_official_isco08_collections \
  --catalogue eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv \
  --metadata eval/verified_catalogue_counts.yaml \
  --profile official_ilo2021_v1 \
  --output-manifest eval/local_runs/official_isco08_collection_build_20260808T211235Z/build_manifest.json \
  --execute \
  --confirm-profile official_ilo2021_v1 \
  --allow-local-qdrant-mutation
```

Exit **0**. `build_manifest.json` status: **`"success"`** (exact
string). Manifest contents confirmed: `profile: "official_ilo2021_v1"`,
`catalogue_sha256` matching the required hash exactly,
`embedding_model_identity: "intfloat/multilingual-e5-small"`,
`embedding_vector_dim: 384`, all five `planned_targets` and
`verified_targets` present with `expected_count == observed_count` for
every one (10/43/130/436/436) and `verified: true` on all five. Stop
condition 5 did not trigger.

**Post-build Qdrant inventory** (`qdrant_post_build_inventory.txt`):

| Collection | Points |
|---|---|
| `isco08_major_groups` (legacy) | 10 (unchanged) |
| `isco08_submajor_groups` (legacy) | 43 (unchanged) |
| `isco08_minor_groups` (legacy) | 131 (unchanged) |
| `isco08_unit_groups` (legacy) | 441 (unchanged) |
| `isco_occupations` (legacy) | 124 (unchanged) |
| `isco08_major_groups_ilo2021_v1` | **10** |
| `isco08_submajor_groups_ilo2021_v1` | **43** |
| `isco08_minor_groups_ilo2021_v1` | **130** |
| `isco08_unit_groups_ilo2021_v1` | **436** |
| `isco08_unit_groups_flat_ilo2021_v1` | **436** |

Legacy collection point counts confirmed byte-for-byte unchanged
(`diff` of the pre/post legacy portion produced no output). The build
command was **not** run a second time.

## 4. Part C — Five-case official-profile smoke gate

### Selection

**Deterministic rule** (documented verbatim, as required): sort all
18,747 canonical heldout rows
(`eval/local_benchmarks/wisco_isco08_v2_group_split/
heldout_run_eval_format.csv`, unmodified) by `sha256(case_id)` ascending
— a fixed, seedless, deterministic total order. Select the first row
for each of the first 3 distinct `input_language` values encountered
(exactly 3 picks, guaranteeing ≥3 distinct languages by construction).
Continuing the same walk from where that left off (never restarting),
select the next 2 rows whose ISCO major group (`gold_isco_4digit[0]`)
had not already been selected, to maximize major-group diversity.

**Five selected benchmark IDs** (selection order):
`WISCO-2144000700018-hi`, `WISCO-3115002500018-en`,
`WISCO-9216000100018-ur`, `WISCO-6224000600018-ar`,
`WISCO-7233040000000-ar`.

**Diversity achieved**: 4 distinct languages (`ar, en, hi, ur`) and 5
distinct ISCO major groups (`2, 3, 6, 7, 9`) — both exceed the required
minimum of 3. The temporary 5-row input CSV was written only to
`eval/local_runs/official_isco08_collection_build_20260808T211235Z/
smoke_input_5rows.csv` (git-ignored); the original WISCO package and
split manifest were not touched.

### Commands (each run exactly once)

```bash
python eval/run_eval.py \
  --test-set eval/local_runs/official_isco08_collection_build_20260808T211235Z/smoke_input_5rows.csv \
  --system flat --use-llm-reranker off --isco-catalogue-profile official_ilo2021_v1 \
  --output-dir eval/local_runs/official_isco08_collection_build_20260808T211235Z/smoke_flat

python eval/run_eval.py \
  --test-set eval/local_runs/official_isco08_collection_build_20260808T211235Z/smoke_input_5rows.csv \
  --system hierarchical --use-llm-reranker off --isco-catalogue-profile official_ilo2021_v1 \
  --require-genuine-hierarchical --max-stage-latency-ms 30000 \
  --output-dir eval/local_runs/official_isco08_collection_build_20260808T211235Z/smoke_hierarchical
```

Both exited **0**. No `--reranker-model` was passed to either.

### Required pass conditions — all met

| # | Condition | Result |
|---|---|---|
| 1 | Exactly 5 clean rows per system | flat: 5, hierarchical: 5 |
| 2 | Every flat row `pred_method == "flat_isco08_official_ilo2021_v1"`, valid 4-digit code | 5/5 pass |
| 3 | Every hierarchical row `pred_method == "hierarchical_isco08_official_ilo2021_v1"`, valid non-empty stage 1-4 evidence, no fallback, no stage over 30,000 ms | 5/5 pass; **max observed stage latency: 497.13 ms** |
| 4 | No row errors | 0 non-blank `error` fields, either system |
| 5 | Reranking disabled, no reranker trace/cost/token activity | `reranker_fired=0`, `reranker_model` blank, `estimated_cost_usd`/token fields zero, both systems |
| 6 | ISIC/ISCED/SRE not constructed | Both runs printed *"No row has both industry_text and education_text -- ISICClassifier/ISCEDClassifier/SemanticRelationEngine will not be constructed"*; `pred_isic_section`/`pred_isced_level` blank, `sre_status="not_applicable"` on all 10 rows |
| 7 | Post-smoke official collection counts unchanged | `10/43/130/436/436`, identical to post-build inventory |

Stop condition 6 did not trigger. No accuracy, precision, recall,
Wilson interval, McNemar test, or latency comparison was computed —
predicted-vs-gold codes were observed only as a byproduct of reading
the raw CSVs for the checks above, never compared or scored.

## 5. Focused and full test results

```
python -m pytest backend/tests/test_build_official_isco08_collections.py backend/tests/test_official_isco08_catalogue.py backend/tests/test_official_isco08_profiles.py backend/tests/test_isco_classifier_official_profile.py eval/test_official_isco08_profile_evaluator.py eval/test_model_free_isco_evaluation.py eval/test_require_genuine_hierarchical.py eval/test_docs_consistency.py -q
→ 103 passed in 1.48s

python -m pytest backend/tests eval/ -q
→ 2056 passed, 1 deselected, 1 warning in 294.32s (0:04:54)
```

**Zero failures.** Exact match to Task 22's own final-state count
(2056) — this task changed no production/test code, only executed
already-implemented, already-tested tooling for real. No unrelated test
was repaired, skipped, or altered.

## 6. Protected-branch status

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| `reviewer2-wip-snapshot-20260807` | not touched |
| `reviewer2-b2-integration-20260807` | not touched |
| every prior `reviewer2-*` task/integration branch (through `reviewer2-official-isco08-collection-builder-20260809`) | not touched |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push occurred. No PR was created.

## 7. Stop conditions encountered

**None.** All seven listed stop conditions were checked at their
respective gates and none triggered:

1. Source SHA/metadata/counts all matched exactly.
2. Localhost Qdrant was reachable throughout.
3. All five target collections were confirmed absent before the build.
4. The dry-run plan matched exactly.
5. The build returned status `"success"` with exact post-build
   verification.
6. Both smoke commands passed every required condition.
7. No unexpected LLM, Ollama, paid API, ISIC, ISCED, or SRE operation
   was attempted — confirmed directly in both smoke runs' own printed
   guards and raw output.

## Evidence boundary

```text
A successful build proves only that the official ILO catalogue was locally
indexed and verified. A five-case smoke run proves operational path integrity
only. Neither establishes WISCO accuracy, real-LFS validation,
ISIC/ISCED/SRE performance, latency claims, or manuscript-ready comparative
results.
```

No further evaluation, reranking run, analysis, paper draft, B1
re-freeze, or B2 sweep was begun. Stopping here, per the task's own
instruction, after pushing this report.
