# Task 42 — Flat-Retrieval-to-Decision-Policy Dev Integration Preflight: Final Report

## 1. Branch, SHAs

- **Branch:** `reviewer2-flat-retrieval-decision-policy-dev-integration-preflight-20260810`
- **Base:** `reviewer2-wisco-gold-label-ambiguity-audit-20260810` @ `c45efbb685508015adad7051b5720b016058db0d` (verified via `git rev-parse` and clean-tree check before any file was written)
- **Final commit SHA:** see `git log -1` on this branch after push — self-referential, cannot be embedded exactly (a commit's own resulting hash cannot be a fixed point of itself; see the same disclosure in the Task 41 report).

## 2. Scope

This is a **Flat-Retrieval-to-Decision-Policy Dev Integration Preflight** — not a WISCO accuracy measurement, not a live-LLM run, not an extension of the Task 36-38 evidence chain. It wires Task 41's decision-policy component (`policy.py`, used unmodified) to the current maintained flat retrieval (the same collection Task 36 measured against), and runs it against the WISCO v2 **dev split only** (2,013 cases). No LLM was called anywhere in this task.

## 3. Preserved vs. different

| Component | Historical / Task 41 assumption | This integration |
|---|---|---|
| `policy.py` | — | Used completely unmodified; zero source change. |
| Retrieval | `_flat_search()`/`search_flat_only()` (top-3 exposed) | This task's own `fetch_five_official_flat_candidates()` calls `HierarchicalISCOStore._embed_query()`/`._query()` directly with `limit=5`, reusing the exact embedding/query mechanics Task 36 measured, without modifying or depending on `_flat_search()`'s top-3 truncation. |
| `label_en`/`label_ar` payload keys | Assumed populated | **Disclosed pre-existing gap:** `_flat_search()` reads `payload.get("label_en"/"label_ar", "")`, but the official flat collection's payload (`build_official_isco08_collections.py::_build_payload()`) only ever writes `title_en` — no `label_en`/`label_ar` key exists. This means `HierarchicalResult.label_en`/`label_ar` are always blank for the official profile today; harmless for Task 36 (exact-code-match scoring only), not fixed here (would touch Task 36's frozen evaluation path). This task's adapter reads `title_en` directly and correctly instead. |
| `title_ar` | Always present | **Not available anywhere in the official catalogue.** Every `PolicyCandidate.title_ar` from this adapter is an honest `""`, never fabricated. |
| `description` | Always present | **Not available** (`embedding_text` is just `f"{code} {title}"`, not a real description). Every `PolicyCandidate.description` is `""`. |
| `level` | Int 1-4 | Always `4` — the flat collection is 4-digit unit-groups only, by construction. |
| Embedding model | `multilingual-e5-large` | `multilingual-e5-small` (unchanged from every prior official-profile task) — runs **locally** via `sentence-transformers`, zero external API cost. |
| Reranker | Real CrewAI → Claude 3.5 Sonnet | **Never invoked.** Below-threshold rows get outcome `PENDING_RERANK_NO_LLM_CALLED` instead. |

## 4. Implementation

New: `eval/legacy_decision_policy41/flat_retrieval_adapter.py`
- `fetch_five_official_flat_candidates(store, query_text)` — 5 raw hits → 5 `PolicyCandidate`s, reading `title_en` correctly (never the empty `label_en`), `title_ar`/`description` left honestly blank, `level=4`. Raises `InsufficientCandidatesError` if fewer than 5 hits come back.
- `run_dev_preflight(store, dev_rows, limit=None, run_manifest=None)` — for each row, fetches candidates and calls `classify_with_policy(..., lang=row.input_language, reranker=None)`. Catches exactly the one `ValueError` Task 41's `policy.py` raises when a reranker is required but absent, translating it to `PENDING_RERANK_NO_LLM_CALLED`; **any other exception propagates immediately and stops the run** (fail-closed, mirrors Task 39's `DevelopmentRunStopped` pattern — no silent per-row skipping).
- `build_run_manifest(store, dev_csv_path, repo_root)` — records UTC timestamp, git commit, embedding model, collection/profile, and the dev CSV's own SHA-256, attached to every report for future traceability.
- `load_dev_rows(csv_path)` — plain CSV reader.

No file under `backend/rag/`, `backend/agents/`, or `eval/legacy_decision_policy41/policy.py` was touched (confirmed by `git status` showing neither as modified).

## 5. Hermetic tests (13, all against a fake store — no real Qdrant/LLM)

```text
$ python -m pytest eval/legacy_decision_policy41/test_flat_retrieval_adapter.py -v
...
13 passed in 0.11s
```

| # | Requirement | Test |
|---|---|---|
| 1 | exactly 5 candidates, in order | `test_fetch_returns_exactly_five_candidates_in_order` |
| 2 | `title_ar`/`description` always blank | `test_title_ar_and_description_always_blank` |
| 3 | `level` always 4 | `test_level_always_four` |
| 4 | reads `title_en`, not `label_en` | `test_code_title_en_confidence_map_correctly_from_title_en_not_label_en` |
| 5 | <5 hits raises | `test_fewer_than_five_hits_raises` |
| 6 | fast path, zero LLM calls | `test_fast_path_row_produces_semantic_outcome_zero_llm_calls` |
| 7 | pending-rerank, no raise | `test_below_threshold_row_produces_pending_rerank_outcome_no_raise` |
| — | (explicit no-reranker-object-exists proof) | `test_below_threshold_row_never_calls_a_reranker` |
| 8 | aggregate counts sum correctly | `test_aggregate_counts_sum_correctly` |
| 9 | exact-match only over fast-path rows | `test_exact_match_only_computed_over_fast_path_rows` |
| 10, 11 | no LLM/Qdrant-client import/construction | `test_adapter_module_has_no_llm_or_network_import` |
| 12 | `lang` passed through from `input_language` | `test_input_language_passed_through_to_classify_with_policy` |
| 13 | unexpected exception stops the run | `test_unexpected_exception_mid_loop_stops_the_run` |

## 6. Live-Qdrant dev validation (real, read-only, zero LLM calls)

`lfs_qdrant` confirmed `Up ... (healthy)` before starting; never started/stopped/restarted. Generated `dev_run_eval_format.csv` via the exact command in the brief — 2,013 rows, confirmed to be an exact match (not just a subset) of `records.json`'s `split="dev"` `benchmark_id`s; the heldout file was never read anywhere in this task.

Smoke run (`limit=20`) completed cleanly with sane codes/confidences (0.80-0.87 range) before the full run.

Full run (all 2,013 dev cases):

```text
n_total: 2013
n_semantic_fast_path: 0
n_pending_rerank: 2013
n_semantic_fast_path_exact_match: 0 (of 0 fast-path rows -- undefined rate, not "0% accuracy")
```

**Confidence distribution across all 2,013 rows:** min `0.7795`, median `0.8524`, mean `0.8517`, **max `0.9148`**. Zero rows reached the 0.92 threshold; 6 rows reached ≥0.90; 1,109 reached ≥0.85.

Run manifest (attached to the output):

```json
{
  "utc_timestamp": "2026-08-10T15:57:33.944866+00:00",
  "git_commit": "c45efbb685508015adad7051b5720b016058db0d",
  "embedding_model": "intfloat/multilingual-e5-small",
  "collection": "isco08_unit_groups_flat_ilo2021_v1",
  "profile": "official_ilo2021_v1",
  "dev_csv_sha256": "b959656e793734c36e422a1300a73a136d9afb3a95435032f1c5ac63255e6f2c"
}
```

Output written to `eval/local_runs/task42_flat_retrieval_dev_preflight_20260810T155854Z/dev_preflight_report.json` (git-ignored, not committed).

### What this means — the honest headline finding

**Under the current maintained embedding model (`multilingual-e5-small`) against the official 436-code catalogue, the historical 0.92 fast-path threshold never fires on WISCO dev data.** Every single one of the 2,013 dev cases would require a reranker call. This is not a bug in this task's adapter (the smoke-test and full-run confidence values are consistent, plausible cosine similarities, and the maximum observed value, 0.9148, sits just under the threshold rather than being anomalously low) — it is a genuine, disclosable finding about how the historical threshold, tuned for a larger embedding model (`multilingual-e5-large`) and a differently-structured legacy catalogue, interacts with the current smaller model and the official catalogue. It is directionally consistent with Task 36's already-published low flat accuracy (21.19%): a retrieval setup whose top match is this often only "quite good" rather than "unambiguous" is exactly the kind of setup where an LLM reranking step would historically have been relied on for nearly every case, not just a minority.

## 7. Future-reranker-task cost estimate

If a future task were to add a real LLM reranker call using this same retrieval setup, it should budget for **essentially all cases**, not a fraction: 2,013 of 2,013 dev cases, and — if the same confidence pattern holds on heldout, which this task did not check (heldout was never read) — plausibly all or nearly all of the 18,747 heldout cases as well. Any such future task should independently confirm the heldout-side rate (still without spending LLM credit — e.g. reusing this same dry-run design against heldout) before committing to an actual reranking budget.

## 8. Preservation checks

`git status --porcelain` shows only 3 new files: this report, `flat_retrieval_adapter.py`, `test_flat_retrieval_adapter.py`, plus the task brief itself (`CLAUDE_TASK_42_...md`, authored earlier this session). Neither `policy.py`, `legacy_identity.py`, nor `hierarchical_store.py` appear as modified. SHA-256 spot checks, unchanged from before this task:

| File | SHA-256 |
|---|---|
| `eval/legacy_decision_policy41/policy.py` | `4abf9b228bf1c0a22798849ac5f20dd623b537807a89b6bdbee59dbeabf7730f` |
| `eval/legacy_decision_policy41/legacy_identity.py` | `048133196668a614f98b03bce445f4e6b1d9c5572c62479f911fe5bdae09d676` |
| `backend/rag/official_isco08_catalogue.py` | `02c1d0dae8d58fe7fb5bda6d1e0824b912a8f79779923e94c927f9f9741197b5` |
| `backend/rag/build_official_isco08_collections.py` | `63c061b4a663841cd629684cbc7e3fdc9dbc2542eb58d942b15e447e95e4480d` |
| `eval/configs/b1_frozen.json` | `f479ffef7e2e3b8342ffdbc11712df3cc2e78a405c20f09a940a2ec840a4967b` |
| `eval/local_benchmarks/wisco_isco08_v2_group_split/records.json` | `735b6241c7fad689ce167906dbf82d21cf81b790a220b992fa2a3e2fd1c5e368` |
| `eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv` | `41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c` (still matches the value cited in `OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`) |

## 9. Full test suite

```text
$ python -m pytest backend/tests eval/ -q
2307 passed, 1 deselected, 1 warning in 331.05s (0:05:31)
```

(2294 baseline + 13 new — zero regressions.)

## 10. Confirmation of zero forbidden activity

- Zero LLM/Anthropic/OpenAI call anywhere in this task (no `crewai`, `anthropic`, or `openai` import in the new module or its tests, verified by a source-scan test).
- Zero write to Qdrant of any kind — every call was a read-only `query_points` search against the already-populated, shared collection.
- Zero Docker command run.
- Zero heldout-split read (confirmed: the CSV's `case_id`s are an exact match to `records.json`'s `split="dev"` set).
- Embedding is local compute (`sentence-transformers`), not an API call — zero external credit spent by this task, for either embedding or (since none was called) any LLM.

## 11. Status

```text
FLAT_RETRIEVAL_DECISION_POLICY_DEV_INTEGRATION_PREFLIGHT_READY: yes
```

All conditions met: hermetic tests pass (13/13); full suite passes (2307/2307); the real 2,013-row dev run against live Qdrant completed with zero errors and zero LLM calls; preservation checks pass; `n_pending_rerank` was recorded (2,013 of 2,013).

Per this task's own boundary: **stopping here.** No LLM was called, no accuracy number beyond the (undefined, 0-of-0) fast-path exact-match rate was computed, the heldout split was never touched, and no "Task 43" is proposed or started. Given §7's finding — that a future reranking task would need budget for essentially the entire dev (and likely heldout) split, not a small fraction — that budget decision is explicitly left to the operator.

## 12. Clean-tree / protected-branch confirmation

`git status --porcelain` clean except this task's own new files, both before writing any file and immediately before commit. Base verified via `git rev-parse` = `c45efbb685508015adad7051b5720b016058db0d` exactly. No merge, rebase, reset, clean, stash, pull, or force-push was run. No PR was opened. Commit includes only task-specific files; push targets only `reviewer2-flat-retrieval-decision-policy-dev-integration-preflight-20260810`. `dev_run_eval_format.csv` and the `eval/local_runs/task42_.../` output stay git-ignored / not committed, consistent with every other WISCO split export in this project.

---

*Report ends. This task's own boundary now applies: stop here.*
