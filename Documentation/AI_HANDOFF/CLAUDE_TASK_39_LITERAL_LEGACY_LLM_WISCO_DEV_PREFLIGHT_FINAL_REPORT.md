LITERAL_LEGACY_LLM_WISCO_DEV_PREFLIGHT_READY: no

# Task 39 Final Report — Literal Legacy ISCO Conditional-LLM WISCO Development Preflight

Produced in response to
`Documentation/AI_HANDOFF/PERPLEXITY_TO_CLAUDE_03_LEGACY_LLM_WISCO_DEV_PREFLIGHT.md`.
Every Phase A gate through the legacy-index build passed. The task
stops before any live Anthropic call because a real, unavoidable
dependency-compatibility blocker was found: the currently-installed
`qdrant-client` (1.17.0, shared globally by this entire project) no
longer has the `QdrantClient.search()` method the historical
`VectorStore.search()` literally calls. Per explicit operator decision
(recorded below), this is reported as a stop, not worked around.

## 1. Task branch, final SHA, verified base branch/SHA, historical source SHA

| | |
|---|---|
| Base branch | `reviewer2-phase1-conference1-final-evidence-alignment-20260810` |
| Required/verified base SHA | `1c62efaec4495f183fcfbe9b011ce5a73e7e846e` (confirmed via `git rev-parse origin/$BASE_BRANCH` before branching) |
| New branch | `reviewer2-literal-legacy-llm-wisco-dev-preflight-20260810` |
| Final commit SHA | reported in Claude's end-of-turn response, not inside this file |
| Historical source commit (LEGACY_SHA) | `824fcf235ae2f8787706cf479a07620519c914de` — verified present (`git cat-file -e`) and subject confirmed exactly `"Add two-stage ISCO-08 classifier agent"` |

Required Git procedure executed exactly as specified in the task file:
`git remote get-url origin` → matched `https://github.com/SrkSubbiah89/Multilingual-LFS-Project.git`;
`git status --porcelain` → clean; `git fetch origin "$BASE_BRANCH"` and
`git fetch origin "$LEGACY_SHA"` → both succeeded;
`git rev-parse origin/$BASE_BRANCH"` → matched `BASE_SHA` exactly;
`git ls-remote --heads origin "$TASK_BRANCH"` → empty (branch did not
already exist); `git switch --create "$TASK_BRANCH" "origin/$BASE_BRANCH"`
→ succeeded, working tree clean immediately after.

## 2. Direct code evidence for every historical policy value

Every value below was read directly from `git show LEGACY_SHA:<path>`
(quoted verbatim), not inferred or assumed:

| Historical property | Required literal value | Verified evidence |
|---|---|---|
| Classifier flow | direct `VectorStore.search(job_title, top_k=top_k)` then conditional LLM | `backend/agents/isco_classifier.py`: `candidates = self._store.search(job_title, top_k=top_k)` |
| Candidate count | 5 | `def classify(self, job_title: str, context: str = "", top_k: int = 5) -> ...`; `backend/rag/vector_store.py`: `TOP_K_DEFAULT = 5` |
| High-confidence condition | top ≥ 0.92 | `_HIGH_CONFIDENCE_THRESHOLD = 0.92`; `if candidates[0].confidence >= _HIGH_CONFIDENCE_THRESHOLD:` |
| High-confidence result | semantic top candidate, no LLM call | returns `candidates[0]` with `method="semantic"` inside the `if` branch above; nothing else is called |
| Low-confidence result | one LLM call via `get_llm(TaskType.CRITICAL)`, select only from supplied candidates | `self._llm = get_llm(TaskType.CRITICAL)` (built once in `__init__`); `_llm_select()` issues exactly one `crew.kickoff()` per `classify()` call; `_parse_llm_response()`'s `if selected_code in code_map:` restricts selection to the five supplied candidates |
| Legacy model route | `anthropic/claude-3-5-sonnet-20241022` | `backend/llm/llm_client.py`: `MODEL_CRITICAL = "anthropic/claude-3-5-sonnet-20241022"` |
| Temperature | 0.0 | `_TEMP_CRITICAL = 0.0` |
| Prompt evidence | job title, language/context, candidate code, EN/AR title, level, score, description | `_llm_select()`'s `candidate_block` includes `f"{i+1}. [{c.code}] {c.title_en} / {c.title_ar}\n   Level {c.level} \| Semantic score: {c.confidence:.2%}\n   {c.description}"`; job title, `lang_note`, `context_line` all included in the `Task.description` |
| Accepted LLM response | JSON with `selected_code` and `reasoning` | `'{"selected_code": "<isco_code>", "reasoning": "<one sentence>"}'` prompt instruction; `data.get("selected_code")` / `data.get("reasoning")` parsing |
| Validation | selected code must be one of the five supplied | `code_map = {c.code: c for c in candidates}`; `if selected_code in code_map:` |
| Malformed/invalid response path | semantic top candidate + legacy fallback reasoning | `return (candidates[0], "Fallback to top semantic match (LLM response could not be parsed).")` |
| Method labels | `"semantic"` and `"llm_ranked"` | exact string literals at both `classify()` return sites |
| Legacy index | `isco_occupations`, historical VectorStore + embedding config | `backend/rag/vector_store.py`: `COLLECTION_NAME = "isco_occupations"`, `MODEL_NAME = "intfloat/multilingual-e5-large"`, `VECTOR_DIM = 1024` |

Automated confirmation: `eval/legacy824/historical_source_identity.py::verify_historical_source_identity()`
re-checks every one of these exact substrings against a live `git show`
of `LEGACY_SHA` — see Section 7 for its test coverage.

## 3. Changed files and why none change the historical classifier

**New files only** (nothing in `backend/`, `eval/run_eval.py`, or any
existing evaluator/classifier file was touched):

```
eval/legacy824/__init__.py                            (21 lines)
eval/legacy824/historical_source_identity.py          (102 lines)
eval/legacy824/historical_loader.py                    (110 lines)
eval/legacy824/isolation.py                             (55 lines)
eval/legacy824/dataset_gate.py                          (83 lines)
eval/legacy824/adapter.py                              (236 lines)
eval/legacy824/build_legacy_collection.py               (68 lines)
eval/legacy824/test_historical_source_identity.py      (131 lines)
eval/legacy824/test_historical_loader.py                (70 lines)
eval/legacy824/test_historical_classifier_behavior.py  (189 lines)
eval/legacy824/test_isolation.py                        (36 lines)
eval/legacy824/test_dataset_gate.py                    (100 lines)
eval/legacy824/test_adapter.py                         (226 lines)
```

None of these files import, alter, or execute anything from
`backend/`, `eval/run_eval.py`, `backend/rag/hierarchy_engine.py`,
`backend/rag/hierarchical_store.py`, `backend/rag/official_isco08_catalogue.py`,
or the ISIC/ISCED/SRE classifiers (statically enforced by
`test_adapter.py::test_adapter_module_imports_no_forbidden_current_tree_component`,
an AST-based scan). `historical_loader.py` reads the three historical
files' bytes from the detached worktree via `importlib` and never
writes to them. `adapter.py` calls the loaded historical
`ISCOClassifier.classify()` unmodified, with `job_title`, `context=""`,
`top_k=5` — exactly its historical call signature.

## 4. Isolated-worktree, isolated-Qdrant, and cache/dependency evidence

**Detached worktree**: `git worktree add --detach C:/task39_legacy824_worktree 824fcf235ae2f8787706cf479a07620519c914de`
→ `HEAD is now at 824fcf2`. `git worktree list` confirms it remains
registered, detached, at that exact commit. Never edited — proven by
`eval/legacy824/test_historical_source_identity.py::test_worktree_source_byte_identical_to_git_show`
(`git diff --quiet LEGACY_SHA -- <path>`, run from inside the worktree,
exit 0 for all three historical files), run both before and after all
work in this task.

**Isolated Qdrant**: a new, separate Docker container
(`task39_isolated_qdrant_20260810`, image `qdrant/qdrant:latest`) bound
to host ports `17333`/`17334` (never `6333`/`6334`), backed by a new,
dedicated Docker volume (`task39_isolated_qdrant_data_20260810`) —
verified empty (`get_collections().collections == []`) immediately
after creation, distinct from the running production container
(`lfs_qdrant`, ports `6333`-`6334`). `eval/legacy824/isolation.py::validate_isolated_endpoint`
refuses port `6333` and the known `localhost:6333`/`127.0.0.1:6333`
endpoints unconditionally before any environment variable is set.

**Legacy-index build**: `eval/legacy824/build_legacy_collection.py`,
run once, produced:
```json
{
  "isolated_qdrant_endpoint": "localhost:17333",
  "collection_name": "isco_occupations",
  "embedding_model": "intfloat/multilingual-e5-large",
  "vector_dim": 1024,
  "point_count": 124,
  "worktree": "C:\\task39_legacy824_worktree"
}
```
Verified independently via direct client calls: isolated instance has
exactly 1 collection (`isco_occupations`, 124 points) after the build;
the production instance's `isco_occupations` collection was checked
before and after and remained at 124 points throughout, with its total
collection count unchanged at 10.

**Cache/dependency gate**: `intfloat/multilingual-e5-large` confirmed
already fully cached locally (`~/.cache/huggingface/hub/models--intfloat--multilingual-e5-large`,
a complete snapshot including `model.safetensors`, 2,239,611,368
bytes). Loaded successfully with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` set (network access to the Hub forcibly
disabled), producing 1024-dimension embeddings — matching
`VECTOR_DIM=1024` exactly. No model was downloaded at any point in
this task.

## 5. WISCO development-only access manifest; proof heldout was not read

`eval/export_benchmark_to_run_eval_csv.py --records eval/local_benchmarks/wisco_isco08_v2_group_split/records.json --split dev --out <root>/wisco_dev_input.csv`
(the same pre-existing, already-tested exporter tool used by every
prior WISCO task in this series) wrote exactly 2,013 rows.

| Check | Result |
|---|---|
| WISCO package hash | `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c` — matches `dataset_hash.txt` exactly |
| Total records | 20,760 |
| Development records | 2,013 |
| Heldout records | 18,747 |
| Development/heldout `case_id` overlap | 0 |
| Development gold-code validity | 0 malformed (all match `^\d{4}$`) |
| `wisco_dev_input.csv` SHA-256 | `b959656e793734c36e422a1300a73a136d9afb3a95435032f1c5ac63255e6f2c` |

`eval/legacy824/dataset_gate.py::assert_not_heldout_path`/`load_and_validate_dev_rows`
refuse any path whose name contains `"heldout"` and any row count
matching `18,747` unconditionally, before any row is read — proven by
`test_dataset_gate.py::test_refuses_path_containing_heldout_by_name`
and `test_refuses_row_count_matching_heldout_size`. **No heldout file
was opened at any point in this task** (verified: only
`wisco_dev_input.csv` and `records.json`, via the pre-existing export
tool, were touched; the export tool itself is not part of "the
adapter" per the task's own text, and was already used identically by
every prior WISCO task in this series to produce split-specific
exports).

## 6. Preflight-gate table and exact stop status

```text
LITERAL_LEGACY_LLM_WISCO_DEV_PREFLIGHT_READY: no
```

| # | Gate | Result |
|---|---|---|
| 1 | Git gate | Pass |
| 2 | Historical-source gate | Pass |
| 3 | Preservation gate | Pass (Section 12) |
| 4 | Dataset gate | Pass |
| 5 | Cache gate | Pass |
| 6 | Isolation gate | Pass |
| 7 | Legacy-index gate | Pass (collection built: 124 points, isolated only) |
| 8 | Adapter gate | Pass (43/43 hermetic tests; import-hygiene AST scan clean) |
| 9 | Full-test gate | Pass (2,228 passed, 1 deselected, 1 warning) |
| — | **Functional-runnability check** (beyond the 11 named gates, performed as prudent verification before ever attempting Phase B) | **FAIL — see Section 9** |
| 10 | API-readiness gate | Not reached |
| 11 | Live-confirmation gate | Not reached |

**Stop reason**: a real semantic search against the isolated legacy
collection (`store.search("software engineer", top_k=5)`, called
through the unmodified historical `VectorStore.search()`) raised
`AttributeError: 'QdrantClient' object has no attribute 'search'`.
This is not one of the 11 explicitly named Phase A gates, but it is an
unavoidable, load-bearing blocker: `ISCOClassifier.classify()`
unconditionally calls `self._store.search(...)` as its very first
operation on every single row, so Phase B could not process even one
development row without hitting this exact failure. Stopping here
before attempting Phase B avoided burning any of the "maximum possible
calls: 2,013" budget on a run that was certain to fail on row one.

## 7. Exact test commands and outputs

Focused (new Task 39 tests only):
```bash
python -m pytest eval/legacy824/ -q
```
Result: `43 passed in 31.22s` (after one test-fixture fix — see
Section 11).

Full suite:
```bash
python -m pytest backend/tests eval/ -q
```
Result: `2228 passed, 1 deselected, 1 warning in 327.82s` (2,185 Task
38 baseline + 43 new). Zero regressions, zero skipped, zero xfailed.

## 8. Operator's explicit decision on how to proceed

Phase B was never reached, so the task's own "explicit live-call
confirmation" prompt was never shown. A **separate, earlier** decision
point was required when the qdrant-client incompatibility was
discovered: whether to (a) stop Phase A and report `no`, or (b) add an
unauthorized-by-the-task-text compatibility shim at the qdrant-client
library level (not touching the historical source) and continue. The
operator was presented with both options explicitly, including the
tradeoffs of each, and **chose option (a): stop Phase A here and
report `READY: no`**. No shim was written; no workaround was
attempted; the historical source was never modified.

## 9. Exact model route, temperature, threshold, top-k, prompt/parser provenance, call count

No live call occurred (Phase B was never reached), so there is no
observed call count to report. The model route, temperature,
threshold, top-k, and prompt/parser provenance that **would** have
governed any such call are documented in full, with exact source
evidence, in Section 2 above.

## 10. Literal command for the single live adapter run

Not executed. The command that would have been used (documented for
the record, never run):
```bash
python -m eval.legacy824.run_live_dev_preflight \
    --worktree C:/task39_legacy824_worktree \
    --qdrant-host localhost --qdrant-port 17333 \
    --input-csv eval/local_runs/task39_literal_legacy_llm_wisco_dev_preflight_20260810T000837Z/wisco_dev_input.csv \
    --output-csv eval/local_runs/task39_literal_legacy_llm_wisco_dev_preflight_20260810T000837Z/dev_results.csv
```
(This exact CLI module was never written, since Phase B was never
reached — `adapter.py`'s `classify_dev_rows()`/`build_isolated_classifier()`
functions exist and are hermetically tested, but no top-level runner
script wraps them, since there was nothing to run.)

## 11. Development-run gate table and descriptive result

Not applicable — Phase B never started. No row was classified, no
accuracy, latency, or method-label distribution was computed. No
five-row audit sample exists.

One test-authoring correction made during Phase A (not a historical
classifier change): `eval/legacy824/test_historical_loader.py`'s
`test_loaded_classifier_registered_only_under_private_name` originally
asserted `"backend.agents.isco_classifier" not in sys.modules`
unconditionally. When run as part of the full suite (not in isolation),
this failed because the current tree's own `backend.agents.isco_classifier`
is legitimately already imported by many unrelated existing test
files before this test runs. Corrected to assert non-interference
(the loaded historical module is a distinct object, and the current
tree's binding — if present — is left exactly as it was), not absence
of the current tree's own name. Two other test fixtures
(`test_output_schema_has_sufficient_audit_fields`,
`test_descriptive_stats_only_meaningful_when_all_rows_succeeded`) were
corrected to properly simulate the LLM call-counter increment for a
mixed semantic/`llm_ranked` row sequence, which the adapter's own
call-count cross-check correctly caught as inconsistent in the
fixtures' original (incomplete) setup. Neither correction touches
`eval/legacy824/adapter.py`'s actual logic — both are test-fixture
fixes, verified by re-running the full suite green afterward (Section 7).

## 12. Hashes proving historical evidence and current collections remained unchanged

**146-file preservation snapshot** (Task 24-38 raw-output roots,
official catalogue, B1 frozen config, `full130` leakage guard/manifest,
WISCO dataset/records/split-manifest, all Task 37/37.1/38 documentation
and analyzer source, and every prior task's final report through Task
38) — hashed before branching and re-hashed at task completion:
**zero mismatches, zero missing files, all 146/146 byte-identical.**

**Qdrant point counts** — checked before and after all work, for both
the five official `ilo2021_v1` collections and every legacy collection:

| Collection | Before | After |
|---|---:|---:|
| `isco08_major_groups_ilo2021_v1` | 10 | 10 |
| `isco08_submajor_groups_ilo2021_v1` | 43 | 43 |
| `isco08_minor_groups_ilo2021_v1` | 130 | 130 |
| `isco08_unit_groups_ilo2021_v1` | 436 | 436 |
| `isco08_unit_groups_flat_ilo2021_v1` | 436 | 436 |
| `isco08_major_groups` | 10 | 10 |
| `isco08_submajor_groups` | 43 | 43 |
| `isco08_minor_groups` | 131 | 131 |
| `isco08_unit_groups` | 441 | 441 |
| `isco_occupations` (production) | 124 | 124 |

Exact match, before and after, on every collection.

## 13. Confirmation of no collection contact outside the isolated local service

Every live Qdrant operation performed by this task's own code
(`build_legacy_collection.py`, the historical `VectorStore`'s
`_ensure_collection`/`_ensure_populated`/`count()` calls) was directed
at `localhost:17333` only, via `QDRANT_HOST`/`QDRANT_PORT` environment
variables set explicitly after `validate_isolated_endpoint` passed.
`eval/legacy824/isolation.py` refuses port `6333` and the known
`localhost:6333`/`127.0.0.1:6333` endpoints unconditionally — proven
by `test_isolation.py` (5 tests). Independently confirmed by direct
inspection of the production instance before and after (Section 12):
10 collections, unchanged counts throughout.

## 14. Confirmation of no other prohibited activity

No manuscript/paper file was touched. No heldout evaluation was run
(Section 5). No hierarchy engine, official-profile evaluator, ISIC,
ISCED, or SRE component was invoked (Section 3's import-hygiene scan).
No model was downloaded (Section 4). No merge, rebase, reset, clean,
stash, pull, or force-push was performed. No pull request was opened.

## 15. Protected-branch and working-tree evidence

No protected or prior-task branch was touched. Only
`reviewer2-literal-legacy-llm-wisco-dev-preflight-20260810` was
created. `git status --porcelain` immediately before this task's
commit shows exactly:
```
?? eval/legacy824/
```
— the new adapter/test package and nothing else. Working tree was
clean immediately before branch creation and remains clean of any
other change.

**Left in place, not destroyed, per the task's explicit "destroy
neither the isolated output nor current collections" instruction**,
for a future authorized follow-up task to reuse or clean up:
- Detached worktree: `C:/task39_legacy824_worktree` (registered via
  `git worktree list`, outside this repository's own directory tree —
  contributes no changes to this repo's git status).
- Isolated Qdrant container: `task39_isolated_qdrant_20260810`
  (ports `17333`/`17334`).
- Isolated Qdrant volume: `task39_isolated_qdrant_data_20260810`
  (contains the 124-point `isco_occupations` collection built in
  Section 4).

## 16. Explicit boundary

This task stops here: an original-flat-legacy-arm observation was
never obtained, because a real dependency-compatibility blocker (not a
decision-policy question) prevented any live classification from
being attempted at all. **Any later hierarchy-era legacy policy, and
any retry of this literal flat-legacy arm (e.g. under an explicitly
authorized qdrant-client compatibility shim, or a separately
provisioned historically-compatible environment), requires a new,
independent task** — this report does not authorize either, and no
such follow-up was started.
