# Task 41 — Historical Decision-Policy Compatibility Preflight: Final Report

## 1. Branch, SHAs

- **Branch:** `reviewer2-legacy-decision-policy-compatibility-preflight-20260810`
- **Final commit SHA:** `e3a12f8f48dac589f99843843750e97efe122437`
- **Verified base SHA:** `reviewer2-legacy-runtime-provenance-recovery-20260810` @ `28ccd8d12eb28fe20e86c7d56231b6e358743c77` (confirmed via `git rev-parse origin/<base branch>` and `git merge-base HEAD 28ccd8d1...` before any file was written)
- **Historical policy source SHA:** `824fcf235ae2f8787706cf479a07620519c914de` ("Add two-stage ISCO-08 classifier agent"; parent `1d03e751a79af8ee6cc503bbc6706a44316616f9`)

## 2. Explicit non-literal label

**This is a Historical Decision-Policy Compatibility Study, not a literal old-runtime reproduction, not an exact reproduction, not literal old implementation execution, and not an old paper result reproduced.**

The strict literal-reproduction path was closed by Tasks 39/40/40.1:

```text
Legacy source: 824fcf235ae2f8787706cf479a07620519c914de
Recovered contemporaneous qdrant-client: 1.17.0
Result: QdrantClient.search() is unavailable in that recovered runtime
```

This task instead implements and hermetically tests, as a small additive component (`eval/legacy_decision_policy41/policy.py`), only the **decision logic** of the historical conditional LLM policy — independent of Qdrant, any embedding model, CrewAI, or any live LLM/provider call.

## 3. Preserved vs. different (matches the task document exactly)

### Preserved exactly

| Policy component | Required value/behavior | Preserved in `policy.py`? |
|---|---|---|
| Candidate count | 5 ordered candidates | Yes — `HISTORICAL_CANDIDATE_COUNT = 5`, enforced by `ValueError` otherwise |
| Confidence threshold | `0.92` | Yes — `HISTORICAL_THRESHOLD = 0.92`, not configurable/parameterized |
| High-confidence path | confidence `>= 0.92` returns semantic top candidate, no LLM invocation | Yes — `top.confidence >= HISTORICAL_THRESHOLD` fast path, `reranker_invocations=0` |
| Low-confidence path | confidence `< 0.92` invokes the reranker exactly once | Yes — exactly one `reranker(prompt_text)` call |
| Candidate constraint | LLM may select only a supplied candidate code | Yes — `selected_code in code_map` |
| Historical model route | `anthropic/claude-3-5-sonnet-20241022` | Yes — `HISTORICAL_MODEL` constant (documentation/identity-check only; no LLM client is constructed in this preflight) |
| Temperature | `0.0` | Yes — `HISTORICAL_TEMPERATURE` constant (same status as above) |
| Prompt fields | job title, language/context, candidate code, English/Arabic title, level, score, description | Yes — `build_prompt_text()` reproduces the historical `candidate_block`/`lang_note`/`context_line` construction verbatim |
| Response contract | JSON containing `selected_code` and `reasoning` | Yes — `parse_reranker_response()` |
| Invalid/malformed/out-of-candidate result | select semantic top candidate using historical fallback behavior | Yes — exact string `"Fallback to top semantic match (LLM response could not be parsed)."` |
| Method labels | `semantic` and `llm_ranked` | Yes — `ALLOWED_METHODS = {"semantic", "llm_ranked"}`, enforced by test |

### Context components that differ (disclosed, not hidden — not "only the transport changed")

| Component | Historical source | Task 41 compatibility study |
|---|---|---|
| Qdrant API | obsolete `QdrantClient.search()` | **not present at all** — this component makes no Qdrant call, no embedding-model call, of any kind |
| Flat collection | legacy mixed-granularity `isco_occupations` | **not present** — no collection, no retrieval, is queried by this task |
| Catalogue | legacy curated set | **not present** — no catalogue is loaded |
| Embedding/runtime | historical environment cannot be executed | **not applicable** — this task calls no embedding model and no LLM/provider |
| Dataset | original study data | **not present** — no WISCO row, no real job-title data of any kind is read; every test uses hand-constructed, fictitious candidates |
| Reranker invocation | real `CrewAI` `Crew`/`Agent` calling Claude 3.5 Sonnet | an injected, caller-supplied plain Python callable — no `CrewAI`, no `Agent`, no network call, no real model of any kind |
| Reranker exception handling | none in the historical code (`crew.kickoff()` was unguarded — an exception would propagate) | this component adds a single try/except around its one reranker call, folding an exception into the same semantic-top fallback used for a malformed response, so this required test scenario (an injected callable can legitimately raise) fails closed rather than crashing; this is a disclosed compatibility-layer addition, not a claim about historical `crew.kickoff()` failure behavior |

## 4. Direct legacy source evidence

Quoted verbatim from `git show 824fcf235ae2f8787706cf479a07620519c914de:backend/agents/isco_classifier.py`:

```python
def classify(
    self,
    job_title: str,
    context: str = "",
    top_k: int = 5,
) -> ISCOClassification:
```

```python
_HIGH_CONFIDENCE_THRESHOLD = 0.92
```

```python
        # Fast path: unambiguous top match — skip LLM
        if candidates[0].confidence >= _HIGH_CONFIDENCE_THRESHOLD:
            return ISCOClassification(
                query=job_title,
                ...
                method="semantic",
            )

        # Stage 2: LLM re-ranking
        primary, reasoning = self._llm_select(job_title, candidates, context, lang)
        return ISCOClassification(
            ...
            method="llm_ranked",
        )
```

```python
        candidate_block = "\n".join(
            f"{i + 1}. [{c.code}] {c.title_en} / {c.title_ar}\n"
            f"   Level {c.level} | Semantic score: {c.confidence:.2%}\n"
            f"   {c.description}"
            for i, c in enumerate(candidates)
        )

        lang_note = {
            "ar":    "The job title is written in Arabic.",
            "mixed": "The job title is code-switched (Arabic and English).",
        }.get(lang, "The job title is written in English.")

        context_line = f"\nAdditional context: {context}" if context.strip() else ""
```

```python
        if selected_code in code_map:
            return code_map[selected_code], reasoning or "Selected by LLM classifier."

        # Fallback: top semantic match
        return (
            candidates[0],
            "Fallback to top semantic match (LLM response could not be parsed).",
        )
```

Quoted verbatim from `git show 824fcf235ae2f8787706cf479a07620519c914de:backend/llm/llm_client.py`:

```python
class TaskType(str, Enum):
    GENERAL  = "general"
    CRITICAL = "critical"

MODEL_GENERAL  = "gpt-4o-mini"
MODEL_CRITICAL = "anthropic/claude-3-5-sonnet-20241022"

_TEMP_GENERAL  = 0.3
_TEMP_CRITICAL = 0.0
```

Every one of these snippets is asserted, byte-for-byte, by `eval/legacy_decision_policy41/legacy_identity.py::EXPECTED_SNIPPETS` and verified against a real, freshly-run `git show LEGACY_SHA:<path>` in `test_real_legacy_sources_pass_identity_check` (§6).

## 5. Changed files, and why none alter the detached historical source

Only new files were added — nothing in the repository was modified:

```text
eval/legacy_decision_policy41/__init__.py
eval/legacy_decision_policy41/policy.py
eval/legacy_decision_policy41/legacy_identity.py
eval/legacy_decision_policy41/test_policy.py
eval/legacy_decision_policy41/test_legacy_identity.py
Documentation/AI_HANDOFF/CLAUDE_TASK_41_HISTORICAL_DECISION_POLICY_COMPATIBILITY_PREFLIGHT_FINAL_REPORT.md
```

None of these files touch `C:/task39_legacy824_worktree` (Task 39's detached worktree, HEAD still at `824fcf2`) or any file inside `eval/legacy824/` — Task 41's `legacy_identity.py` only *reads* `backend/agents/isco_classifier.py`/`backend/llm/llm_client.py` at `LEGACY_SHA` via `git show` (a plain object read from the repository's own history, not a checkout, not a worktree edit) and never writes to any path under the historical worktree. `git status --porcelain` inside the worktree was not even required to check because no command in this task ever `cd`'d into or wrote to that path.

## 6. Focused and full-suite commands with literal output

```text
$ python -m pytest eval/legacy_decision_policy41/ -v
...
31 passed in 0.28s
```

(9 identity-check tests in `test_legacy_identity.py` + 22 policy tests in `test_policy.py`, including 5 parametrized wrong-candidate-count cases and 4 parametrized only-two-labels-possible cases — see §7 for the full 16-requirement mapping.)

```text
$ python -m pytest backend/tests eval/ -q
2285 passed, 1 deselected, 1 warning in 382.46s (0:06:22)
```

Task 40.1's baseline was `2254 passed, 1 deselected, 1 warning`; `2254 + 31 = 2285` — zero regressions. The single atexit `colorama`/`crewai` teardown exception printed after the summary line is the same pre-existing, harmless interpreter-shutdown noise documented in the Task 40.1 report (stream-close ordering in a third-party dependency at process exit, not a test failure) — the reported result line `2285 passed, 1 deselected, 1 warning` is authoritative.

## 7. Test coverage mapped to the required policy rules

| # | Required test | Test(s) |
|---|---|---|
| 1 | exactly five candidates required | `test_exactly_five_candidates_required_and_accepted` |
| 2 | fewer/more candidates fail closed | `test_wrong_candidate_count_fails_closed[0,1,4,6,10]` |
| 3 | confidence exactly 0.92 → semantic, zero reranker calls | `test_confidence_exactly_threshold_takes_semantic_path_zero_reranker_calls` |
| 4 | confidence above 0.92 → semantic, zero reranker calls | `test_confidence_above_threshold_takes_semantic_path_zero_reranker_calls` |
| 5 | confidence below 0.92 → reranker invoked exactly once | `test_confidence_below_threshold_invokes_reranker_exactly_once` |
| 6 | valid candidate-code JSON → `llm_ranked` | `test_valid_json_response_produces_llm_ranked_with_selected_candidate` |
| 7 | invalid JSON → semantic-top fallback | `test_invalid_json_falls_back_to_semantic_top` |
| 8 | missing `selected_code` → semantic-top fallback | `test_missing_selected_code_falls_back_to_semantic_top` |
| 9 | out-of-candidate code → semantic-top fallback | `test_out_of_candidate_code_falls_back_to_semantic_top` |
| 10 | reranker exception → semantic-top fallback | `test_reranker_exception_falls_back_to_semantic_top` |
| 11 | prompt includes every required field for all 5 candidates | `test_prompt_includes_all_required_fields_for_all_five_candidates` |
| 12 | candidate order unchanged in prompt and result | `test_candidate_order_preserved_in_prompt_and_result` |
| 13 | only `semantic`/`llm_ranked` emitted | `test_only_semantic_and_llm_ranked_labels_possible[4 scenarios]` |
| 14 | no prohibited import in the policy module | `test_policy_module_has_no_prohibited_imports` |
| 15 | static identity checks reject changed threshold/count/model/temperature/fallback | `test_identity_check_rejects_altered_threshold`, `..._candidate_count`, `..._model_route`, `..._temperature`, `..._fallback_condition`, plus `test_identity_check_reports_all_missing_snippets_not_just_first` |
| 16 | fixtures prove no network-capable reranker is used | `test_no_network_call_occurs_during_classification` (blocks `socket.socket` for the duration of both the fast- and slow-path calls) |

Additional: `test_identity_check_passes_on_matching_synthetic_source`, `test_real_legacy_sources_pass_identity_check` (real `git show` against `LEGACY_SHA`, proving the quoted evidence in §4 is what the repository actually contains today), `test_legacy_sha_constant_matches_task_specification`.

## 8. Before/after preservation evidence

`git status --porcelain` before and after this task's work: only `eval/legacy_decision_policy41/` and this report are new/untracked — no existing file modified. SHA-256 spot checks (unchanged from Task 40.1's own recorded values):

| File | SHA-256 |
|---|---|
| `eval/legacy824/historical_loader.py` | `fa34538b8eaa80182934437f11ab9b536947ef6b2a9a485f268ed2fdacc89bce` |
| `eval/legacy_runtime40/provenance.py` | `6b656249e1905ed07374a60efa452c3102328e32bddd0a3417099e24cdadc5e4` |
| `backend/rag/official_isco08_catalogue.py` | `02c1d0dae8d58fe7fb5bda6d1e0824b912a8f79779923e94c927f9f9741197b5` |
| `eval/configs/b1_frozen.json` | `f479ffef7e2e3b8342ffdbc11712df3cc2e78a405c20f09a940a2ec840a4967b` |
| `Documentation/AI_HANDOFF/CLAUDE_B2_INTEGRATION_REPORT.md` | `34cc0307cbc806a6cd25cb03719797386355e54d79f89e678c8556d7f75a6540` |

Task 39's detached worktree (`C:/task39_legacy824_worktree`, HEAD still detached at `824fcf2`) and isolated Qdrant container (`task39_isolated_qdrant_20260810`, `Up`) are confirmed present and untouched. Task 40.1's created-but-never-started Docker container (`cd76e766a588...`, still `Created`, never started) is confirmed present and untouched. Reports for Task 36 through 40.1 all remain present in `Documentation/AI_HANDOFF/` and `Documentation/Conference_I_Reviewer_2/`, unmodified.

## 9. Confirmation of zero WISCO/Qdrant/LLM/model/Docker activity

- No WISCO file (dev or heldout) was opened or read anywhere in this task.
- No Qdrant client was constructed and no Qdrant server was queried or connected to — `policy.py` has no Qdrant import of any kind (enforced by `test_policy_module_has_no_prohibited_imports`).
- No LLM/Anthropic/OpenAI call was made — every "reranker" in every test is a plain in-process Python function/lambda; `test_no_network_call_occurs_during_classification` actively blocks `socket.socket` for the duration of both classification paths and confirms neither the policy component nor the fake reranker opens one.
- No embedding model or any other model was downloaded or loaded.
- No Docker command of any kind was run in this task.
- The only git operations were `fetch`, `rev-parse`, `merge-base`, `cat-file -e`, `show` (read-only content reads), `ls-remote`, `switch --create`, `status`, `add`, `commit`, and (at the end) `push` — no merge, rebase, reset, clean, stash, pull, or force-push.

## 10. Status

```text
HISTORICAL_DECISION_POLICY_COMPATIBILITY_PREFLIGHT_READY: yes
```

All five `yes` conditions are met: historical policy source identity is verified (§4, §6); the compatibility policy preserves every required behavior (§3, §7); all 31 hermetic tests pass; the full suite passes (2285/2285, zero regressions); preservation checks pass (§8); and no forbidden live operation occurred (§9).

Per the task's own boundary, this preflight **stops here**. It does not run WISCO, integrate live retrieval, create an API credential request, call an LLM, make a paper update, or calculate an accuracy value. The next dev-only evaluation task (wiring the current maintained flat-retrieval caller to supply real candidates to this policy component, and running it against WISCO development data only) is a separate, not-yet-authorized task — this report does not authorize it; that is an operator decision for a future task document.

## 11. Clean-tree / protected-branch / no-forbidden-operation confirmation

- `git status --porcelain` clean except this task's own new files, both before writing any file and immediately before commit.
- Branch created via `git switch --create reviewer2-legacy-decision-policy-compatibility-preflight-20260810 origin/reviewer2-legacy-runtime-provenance-recovery-20260810`; base verified via `git rev-parse` = `28ccd8d12eb28fe20e86c7d56231b6e358743c77` exactly (the exact SHA specified).
- `master`, `reviewer2-legacy-runtime-provenance-recovery-20260810`, `reviewer2-legacy-runtime-reconstruction-20260810`, and `reviewer2-literal-legacy-llm-wisco-dev-preflight-20260810` all remain exactly as they were (confirmed via `git branch -a`) — none touched, none merged, none rebased.
- No merge, rebase, reset, clean, stash, pull, or force-push was ever run.
- No PR was opened.
- Commit includes only task-specific files (`eval/legacy_decision_policy41/*` and this report); push targets only `reviewer2-legacy-decision-policy-compatibility-preflight-20260810`.

---

*Report ends. This task's own boundary now applies: stop here.*
