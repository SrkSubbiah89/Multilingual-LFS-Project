# Task 43 — Live Reranker WISCO Run: Final Report

## Status

```text
LIVE_RERANKER_WISCO_RUN_READY: partial
reason: anthropic_account_had_zero_credit_for_the_entire_session -- no genuine
        Claude-3-5-Sonnet-reranked result exists anywhere in this task's output.
        A genuine result WAS obtained using a free local Ollama model instead
        (a disclosed, major deviation from the historical model identity) --
        see "Addendum: Ollama dev-split run" below. 17.19% (346/2,013), WISCO
        dev split, flat retrieval, llama3.2:latest.
```

**No real LLM-reranked accuracy number was obtained.** Every attempted run (dev split, full flat heldout, full hierarchical heldout) silently fell back to plain semantic-retrieval-top-1 for 100% of cases, because the configured Anthropic API key had insufficient credit for every single call made in this session. This was discovered, root-caused, and fixed (so it can never again happen silently) — but it means none of the numbers produced today are what they were reported as at the time. This report exists to correct that record and document the fix.

## 1. Branch, SHAs

- **Branch:** `reviewer2-live-reranker-wisco-dev-run-20260810`
- **Base:** `reviewer2-flat-retrieval-decision-policy-dev-integration-preflight-20260810` @ `54891e2fdf03a59c62fa3baa26c932f2d9570a05`

## 2. What was attempted

Building on Task 41 (`policy.py`) and Task 42 (`flat_retrieval_adapter.py`), this task added:
- `live_reranker.py`: a real Anthropic reranker (`make_anthropic_reranker`, direct API call, model `claude-3-5-sonnet-20241022`, temperature `0.0` — the same model/temperature Task 41 already locked in, CrewAI wrapper skipped as a disclosed transport-level difference) plus `run_dev_with_live_reranker`, a driver that scores every row (both fast-path and reranked) against gold labels.
- `hierarchical_retrieval_adapter.py`: mirrors the flat adapter but calls the real, unmodified `HierarchicalISCOStore.search(query, reranker_candidates=5)` (the existing 4-stage beam search, exercised exactly as Task 36 used it) instead of reimplementing retrieval, with the same disclosed `title_en`-vs-`label_en` payload-key workaround already found in Task 42.

Runs attempted, all real (live Qdrant + attempted live Anthropic calls):
1. WISCO dev split (2,013 cases), flat retrieval.
2. WISCO heldout split (18,747 cases), flat retrieval.
3. WISCO heldout split (18,747 cases), hierarchical retrieval.

## 3. What was actually discovered: a silent, total reranking failure

Every one of the three runs above reported `method="llm_ranked"` for effectively every row and produced plausible-looking accuracy figures (dev: 380/2,013 = 18.88%; flat heldout: 3,973/18,747 = 21.19%; hierarchical heldout: 1,616/18,747 = 8.62%). **All three are artifacts, not measurements.**

**How this was caught:** the flat-heldout run's predicted code matched Task 36's original *non-reranked* flat baseline prediction for **18,747 of 18,747 cases — exactly 100%**. A working reranker choosing among 5 real candidates would disagree with the plain top-1 semantic candidate at least some of the time; 100% agreement across nearly 19,000 cases is not statistically plausible for a functioning LLM call.

**Root cause, confirmed directly:** a manual, single-case trace of the real call chain (candidates → prompt → `client.messages.create(...)`) raised:

```text
anthropic.BadRequestError: Error code: 400 - {'type': 'error', 'error':
{'type': 'invalid_request_error', 'message': 'Your credit balance is too
low to access the Anthropic API. Please go to Plans & Billing to upgrade
or purchase credits.'}}
```

Task 41's `classify_with_policy` (used unmodified, correctly, per its own design) catches *any* exception from the injected `reranker` callable and folds it into the same semantic-top fallback used for a genuinely malformed LLM response — a deliberate, disclosed, and reasonable design for "the LLM answered but we couldn't parse it." It was never designed to distinguish that from "the call could not be made at all," and neither was this task's own driver code until now. The result: every reranked row silently became a semantic-top-1 prediction labeled `llm_ranked`.

**Confirmed this affected every run, from the very start:** a free, local-only re-check (re-fetching candidates via Qdrant only, zero LLM cost) on a 150-row sample of the dev run's own results showed the same 100% agreement with candidate rank 1. The credit was already exhausted at or before the very first live call of this entire session — including the small "smoke test" calls used earlier to validate the pipeline, which is why they appeared to succeed (a caught-and-silently-handled exception looks identical to a real answer at the call-site).

**Likely no real charge was incurred**: `BadRequestError` responses of this kind are pre-flight rejections (no completion is generated), which providers typically do not bill for. This could not be independently verified without billing-dashboard access, but no successful, billable completion occurred at any point in this session.

**Hierarchical heldout's 87.53%-agreement-with-baseline (not 100%)** is not evidence that hierarchical partially reranked successfully — it reflects that this task's hierarchical adapter's own top-1 candidate (from `search(reranker_candidates=5)`) does not always exactly match Task 36's original baseline call's top-1 prediction (different call configuration), independent of reranking. The same silent-fallback failure applies equally to both runs.

## 4. The fix

`live_reranker.py` now provides a `fatal_tracker` mechanism:
- `make_anthropic_reranker(client, fatal_tracker=None)`: on any non-retryable exception, checks `_is_fatal(exc)` — `True` for `AuthenticationError`, `PermissionDeniedError`, and `BadRequestError` whose message contains "credit balance is too low" — and if so, records it into the caller-supplied `fatal_tracker` dict before re-raising.
- `run_dev_with_live_reranker(..., fatal_tracker=None)`: after every row's `classify_with_policy` call, checks `fatal_tracker` and raises `FatalRerankerError` immediately if a fatal error was recorded — **before** that row's misleading fallback result is added to the report or the progress file. The run stops loudly, naming the exact case_id and how many genuinely-processed rows preceded it, instead of completing "successfully" with thousands of mislabeled results.
- Backward compatible: `fatal_tracker` is optional and defaults to `None`, so a caller that doesn't pass one sees the same behavior as before (documented explicitly by `test_run_without_fatal_tracker_keeps_prior_silent_fallback_behavior`) — this is a caller-opt-in fix, not a change to Task 41's `policy.py`, which remains completely untouched.

Verified against the exact regression: `test_run_aborts_immediately_on_fatal_error_without_recording_misleading_row` reproduces a 2-row run where row 1 succeeds genuinely and row 2 hits the fatal error, and asserts the run aborts with `FatalRerankerError` naming row 2, and that only row 1's genuine result — not row 2's misleading fallback — was ever written to the progress file.

**Verified this fix is not yet exercised against a real, working credit balance**: a follow-up minimal real call (single token, "reply OK") was attempted after implementing the fix and still returned the same `BadRequestError` — the account remains at zero credit as of this report. The fix is tested exhaustively at the hermetic level (mocked client) but has not yet been exercised end-to-end against a real success case in this task, since no credit was available to do so.

## 5. Tests

```text
$ python -m pytest eval/legacy_decision_policy41/ -v
...
67 passed in 9.08s
```

New in this task: 9 tests for `live_reranker.py`'s reranker/driver (model/temperature, retry-on-transient, no-retry-on-auth), 7 for `hierarchical_retrieval_adapter.py`, and 8 for the fatal-tracker fix (`_is_fatal` classification, tracker recording, full-run abort behavior, and the explicit backward-compatibility guarantee).

```text
$ python -m pytest backend/tests eval/ -q
2330 passed, 1 deselected, 1 warning in 350.38s (0:05:50)
```

Zero regressions.

## 6. Preservation checks

`git status --porcelain` shows only 4 new files (`hierarchical_retrieval_adapter.py`, `live_reranker.py`, and their two test files) — `policy.py`, `flat_retrieval_adapter.py` (Task 42's own file), `hierarchical_store.py`, `hierarchy_engine.py`, and `official_isco08_catalogue.py` are all confirmed unmodified (not listed). SHA-256 spot checks:

| File | SHA-256 |
|---|---|
| `eval/legacy_decision_policy41/policy.py` | `4abf9b228bf1c0a22798849ac5f20dd623b537807a89b6bdbee59dbeabf7730f` (unchanged from Task 42) |
| `backend/rag/hierarchical_store.py` | `805310e245d0152d28da937369d5bf6b5b6999df6486211d4dcf0fcbc022d0ee` (unchanged) |
| `eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv` | `41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c` (unchanged, still matches `OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`'s cited value — the heldout split was read-only in every run this task made) |
| `eval/local_benchmarks/wisco_isco08_v2_group_split/records.json` | `735b6241c7fad689ce167906dbf82d21cf81b790a220b992fa2a3e2fd1c5e368` (unchanged) |

All three attempted runs' raw output is preserved, git-ignored, at `eval/local_runs/task43_live_reranker_dev_run_20260810T171004Z/`, `eval/local_runs/task43_flat_heldout_live_reranker_20260810T180139Z/`, and `eval/local_runs/task43_hierarchical_heldout_live_reranker_20260810T180540Z/`, each with a `progress.jsonl` (per-row) and `final_report.json` (aggregate) — retained as the artifact record of the incident, not deleted, exactly labeled by this report as non-genuine reranking data.

## 7. What is actually needed to get real numbers

1. Add credit to the Anthropic account (console.anthropic.com → Plans & Billing).
2. Re-run using the now-fatal-tracker-protected `run_dev_with_live_reranker(..., fatal_tracker={})` — if credit runs out again mid-run, it will now stop immediately and say exactly where, instead of silently completing.
3. Given cost estimates from before this incident were themselves unverified against an actual successful call, budget conservatively and consider running the dev split (2,013 cases) alone first to get a real per-call cost reading before committing to either 18,747-case heldout run.

## 8. Confirmation

- No accuracy number in this report or in this task's commit should be cited as a real LLM-reranking result — every number produced today is semantic-retrieval-only, mislabeled, and superseded by this report's correction.
- `git status --porcelain` clean except this task's 4 new files, both before writing any file and immediately before commit.
- No merge, rebase, reset, clean, stash, pull, or force-push was run. No PR was opened.
- Commit includes only task-specific files; push targets only `reviewer2-live-reranker-wisco-dev-run-20260810`.

---

## Addendum: Ollama dev-split run (2026-08-11)

With no Anthropic credit available, the operator asked for a free local
alternative. `ollama_reranker.py` was added (see its own commit) — calls a
local Ollama server instead of Claude 3.5 Sonnet, **an explicit, major,
disclosed deviation from Task 41's historical-model identity lock**, used
only because no paid credit was available. Model: `llama3.2:latest` (this
project's own existing default general-task model, per
`backend/llm/llm_client.py`), temperature `0.0`. Everything else —
prompt, threshold, candidate count, JSON/fallback contract — is Task
41's `policy.py`, completely unmodified.

### Real result

```text
n_total: 2013
n_correct: 346
accuracy: 17.19% (346/2013)
n_semantic (fast path): 0
n_llm_ranked: 2013 (every case genuinely reranked -- 344 distinct codes
              used across the run, confirming real, varied model output,
              not a repeated fallback pattern)
```

WISCO dev split, flat retrieval (`isco08_unit_groups_flat_ilo2021_v1`).
Confirmed genuine (not a repeat of the Anthropic-credit incident): 344
distinct predicted codes were used across 2,013 rows, and individual
predictions were manually spot-checked against real, varied Ollama
output — this is real reranking, not a hidden fallback.

### Two operational incidents during this run, both disclosed

**1. A single long-running process is fragile on this hardware.** The
first full-run attempt (2,013 rows in one process) stopped after 318 rows
with a genuine Qdrant timeout (likely CPU contention with a concurrently-
running test suite) — correctly caught and reported by name via the
`fatal_tracker` mechanism (exactly as designed; see the main report
above). A resume attempt was launched for the remaining 1,695 rows. A
`tasklist` check appeared to show no Python process running, which was
**incorrectly** read as "the resume process died silently." Based on
that misread, the operator's own suggestion — split the remaining work
into small, independently-checkpointed batches rather than trust one
multi-hour run — was adopted, and a 150-row "batch 1" was started.

**2. This caused an accidental concurrent duplicate-write.** The
original resume process was, in fact, still alive and running the entire
time (11.6 hours total for the 1,695-row resume segment — far slower
than the ~12-19 sec/call measured in the earlier smoke test, likely
reflecting sustained load rather than a one-off cold-start cost) and
completed the full remaining set on its own. Because "batch 1" ran
concurrently against the same `progress.jsonl`, both processes
independently reprocessed the same 150 case_ids, producing 150
duplicated entries (2,163 lines for 2,013 distinct cases). Six of the
150 duplicate pairs (4%) disagreed on `predicted_code` despite identical
input and `temperature=0.0` — a real, disclosed finding about Ollama's
practical (not bit-exact) determinism under concurrent load, not
something papered over.

**Resolution:** deduplicated by keeping the first-written entry per
`case_id` (`progress_deduplicated.jsonl`, `deduplicated_final_report.json`,
both in the run's output directory). The 17.19% figure above is computed
from this deduplicated set of exactly 2,013 distinct cases. The batching
approach itself remains a sound idea for future long local-inference
runs — the actual failure here was a false "the process is dead" read,
not a flaw in batching as a strategy; batching should be paired with a
more reliable liveness check (e.g. polling the progress file's growth
rather than `tasklist`) next time.

### What this result means

17.19% is close to, but below, the earlier plain-semantic-top-1 dev
figure that the credit-exhaustion incident accidentally produced (18.88%
— itself just retrieval, mislabeled). A free, small local model
(`llama3.2:latest`, 3.2B parameters) reranking among 5 candidates with no
Arabic title or description text to work with (the same official-
catalogue data limitation disclosed in Task 42) does not improve on
plain retrieval here, and this data point on its own cannot separate
"reranking doesn't help for this benchmark" from "this specific small
local model isn't strong enough at this structured task" — both are
plausible; distinguishing them would need the real Claude 3.5 Sonnet
comparison once Anthropic credit is available.

### Files

`eval/legacy_decision_policy41/ollama_reranker.py` +
`test_ollama_reranker.py` (committed earlier, code + tests only). Raw
run output (git-ignored): `eval/local_runs/task43_ollama_flat_dev_run_20260810T222055Z/`
(`progress.jsonl` — the raw, duplicate-containing log, preserved as the
literal incident record; `progress_deduplicated.jsonl` and
`deduplicated_final_report.json` — the clean, authoritative result).

---

*Report ends.*
