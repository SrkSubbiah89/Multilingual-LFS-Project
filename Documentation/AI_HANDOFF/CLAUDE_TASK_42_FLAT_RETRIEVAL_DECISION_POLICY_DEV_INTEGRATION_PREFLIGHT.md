# Task 42: Flat-Retrieval-to-Decision-Policy Dev Integration Preflight

## Scope and honest label

Task 41 built and hermetically tested the historical decision-policy
compatibility component (`eval/legacy_decision_policy41/policy.py`) in
isolation, fed only by fake in-memory candidates, and explicitly named its
own next step as out of scope: *"wiring the current maintained flat-
retrieval caller to supply real candidates to this policy component, and
running it against WISCO development data only"*.

This task does exactly that integration, and nothing more. It is:

```text
Flat-Retrieval-to-Decision-Policy Dev Integration Preflight
```

It is **not** a WISCO accuracy measurement, **not** a live-LLM reranking
run, and **not** an extension of the Task 36-38 official Tier-1 evidence
chain. No accuracy number is computed or reported by this task.

## What is preserved and what is necessarily different

### Preserved exactly

- Task 41's `policy.py` is used **unmodified** — no threshold, candidate
  count, or fallback behavior is touched.
- Retrieval uses the exact same embedding + flat-collection query
  mechanics Task 36 already measured with:
  `HierarchicalISCOStore._embed_query()` (E5 query-prefix encoding,
  normalized) and `HierarchicalISCOStore._query()` (the same Qdrant
  client/collection Task 36's `search_flat_only()` calls), profile
  `official_ilo2021_v1`, collection `isco08_unit_groups_flat_ilo2021_v1`.
  This task calls these two existing methods directly with `limit=5`; it
  does **not** call `search_flat_only()` or `_flat_search()` themselves,
  and does **not** modify either — see "known gap" below for why.

### Known gap in the current maintained flat retrieval caller (disclosed, not fixed here)

`hierarchical_store.py::_flat_search()` reads `payload.get("label_en", "")`
/ `payload.get("label_ar", "")` from each Qdrant hit. The official flat
collection's actual payload schema (`backend/rag/
build_official_isco08_collections.py::_build_payload()`) only ever writes
`code`, `level`, `parent_code`, `title_en`, `profile`,
`source_catalogue_sha256`, `collection_role`, `embedding_text` — there is
no `label_en`/`label_ar` key at all. **This means `HierarchicalResult.
label_en`/`label_ar` and `UnitCandidate.label_en`/`label_ar` are always
blank strings for the official profile today.** This had zero effect on
Task 36's published numbers (scoring is exact-code-match only, never
label-text-based), so it is a pre-existing, harmless-for-Task-36, real gap
— not something this task fixes (fixing `_flat_search()` would touch
Task 36's frozen, already-measured evaluation path, which is out of
scope). This task's own new adapter code reads `title_en` directly from
the raw hit payload, correctly, and must never go through the buggy
`label_en`/`label_ar` fields.

### Context components that differ and must be disclosed

| Component | Historical policy assumption | This integration |
|---|---|---|
| `title_ar` (Arabic title) | Always present (`OccupationMatch.title_ar`) | **Not available.** The official catalogue has no Arabic title field anywhere. Every `PolicyCandidate.title_ar` built by this task's adapter is an honest empty string `""` — never fabricated/translated/copied from `title_en`. |
| `description` | Always present (`OccupationMatch.description`) | **Not available.** The official catalogue has no free-text description field (`embedding_text` is just `f"{code} {title}"`, not a real description). Every `PolicyCandidate.description` is an honest empty string `""`. |
| `level` | Int 1-4 depth | Official catalogue's `level` field is textual (`"unit"` for every flat-collection record, since the flat collection is unit-groups-only). Mapped to the historical int convention (`major=1, submajor=2, minor=3, unit=4`) — every candidate from this flat collection is therefore `level=4`, honestly, since the flat collection is 4-digit-only by design (matches `FLAT_BASELINE_COVERAGE_AUDIT.md`). |
| Embedding model | `intfloat/multilingual-e5-large` (historical) | `intfloat/multilingual-e5-small` (current maintained runtime) — already the case for every prior official-profile task; not new to this one. Runs **locally** via `sentence-transformers` (no API call, no external credit spent — only local CPU/GPU compute). |
| Reranker | Real CrewAI `Crew` calling Claude 3.5 Sonnet | **No LLM is called anywhere in this task** — see next section. |

Report language must state these differences plainly; never describe this
integration as "the same policy, just live now" without this table.

## The reranker boundary — the reason for the dry-run design

This task does **not** call any LLM, does **not** construct a CrewAI
agent, and does **not** spend any Anthropic/OpenAI credit or quota. For
every WISCO dev case whose top semantic candidate is below the 0.92
threshold, the integration records that a reranker call **would** be
required and stops there — it never invokes one. This uses a distinct,
unambiguous status that is never confused with Task 41's real
`"llm_ranked"` label (which means a reranker actually ran):

```text
PENDING_RERANK_NO_LLM_CALLED
```

The point of this design is to get a real, live-Qdrant-validated number
for how many of the 2,013 WISCO dev cases would actually need a reranker
call under the historical 0.92 threshold — directly useful for scoping
and budgeting any future task that does call a live LLM — without
spending any LLM credit to find out.

## Branch and source

Repository:

```text
https://github.com/SrkSubbiah89/Multilingual-LFS-Project.git
```

Create exactly one branch:

```text
reviewer2-flat-retrieval-decision-policy-dev-integration-preflight-20260810
```

Verified base:

```text
reviewer2-wisco-gold-label-ambiguity-audit-20260810
@ c45efbb685508015adad7051b5720b016058db0d
```

Use:

```bash
BASE_BRANCH="reviewer2-wisco-gold-label-ambiguity-audit-20260810"
BASE_SHA="c45efbb685508015adad7051b5720b016058db0d"
TASK_BRANCH="reviewer2-flat-retrieval-decision-policy-dev-integration-preflight-20260810"

git remote get-url origin
git status --porcelain
git rev-parse HEAD
git ls-remote --heads origin "$TASK_BRANCH"
git switch --create "$TASK_BRANCH"
git status --porcelain
```

Stop if the working tree is not already clean at `BASE_SHA`, or if the new
branch already exists remotely. Do not merge, rebase, reset, clean,
stash, pull, force-push, alter remote settings, touch protected branches,
or create a PR.

## Implementation

**New: `eval/legacy_decision_policy41/flat_retrieval_adapter.py`**

- `fetch_five_official_flat_candidates(store: HierarchicalISCOStore, query_text: str) -> list[PolicyCandidate]`:
  - Calls `store._embed_query(query_text)` then
    `store._query(collection=store._col_flat, query_vec=vec, limit=5, query_telemetry={})`
    directly (both already exist, both already exercised by Task 36's
    evaluation path via `search_flat_only`/`_flat_search` — this task
    reuses them at a lower level, it does not reimplement embedding or
    retrieval).
  - Raises a clear, typed error (never silently returns fewer than 5) if
    Qdrant returns fewer than 5 hits for a query — WISCO dev queries
    against a 436-entry collection should always return 5; a caller
    getting fewer than 5 is a signal worth surfacing, not swallowing.
  - Converts each raw hit to a `PolicyCandidate`: `code=hit.payload["code"]`,
    `title_en=hit.payload["title_en"]`, `title_ar=""`, `level=4`,
    `confidence=hit.score`, `description=""`.
  - Makes no LLM/CrewAI/Anthropic import or call.

- `run_dev_preflight(store, dev_rows, limit: int | None = None) -> DevPreflightReport`:
  - `dev_rows`: the WISCO v2 **dev** split only (`case_id`, `input_text`,
    `input_language`, `gold_isco_4digit`), exported via
    `python eval/export_benchmark_to_run_eval_csv.py --records eval/local_benchmarks/wisco_isco08_v2_group_split/records.json --split dev --out eval/local_benchmarks/wisco_isco08_v2_group_split/dev_run_eval_format.csv`
    (this file does not exist yet in this checkout and must be generated
    by this exact command before the run — the **heldout** file must
    never be read anywhere in this task).
  - For each row: fetch 5 candidates (real Qdrant query), call
    `policy.classify_with_policy(job_title=row.input_text, candidates=...,
    lang=row.input_language, reranker=None)` — **`lang` must be passed
    through from the row's own `input_language` column, never left at the
    `classify_with_policy` default.** It has no effect on the dry-run
    outcome today (language only feeds the prompt text, which is never
    built on the fast path, and the pending-rerank path raises before
    `build_prompt_text` is reached — see below), but a caller silently
    defaulting every row to `"en"` would be a landmine for whichever
    future task actually adds the reranker and starts using the prompt
    text this adapter's candidates would otherwise produce correctly.
  - `classify_with_policy` currently raises `ValueError` when `reranker is
    None` and the fast path is missed (Task 41's existing, correct
    fail-closed behavior for a *required* reranker). This task's adapter
    must catch exactly that one `ValueError` and translate it into a
    `PENDING_RERANK_NO_LLM_CALLED` row outcome — it must **not** change
    `policy.py` itself to add an optional-reranker code path (that would
    weaken a Task-41 fail-closed guarantee for every other caller); the
    translation happens only in this task's new adapter.
  - **Any other exception (Qdrant error/timeout, malformed row, an
    unexpected exception type from `classify_with_policy`, etc.) must stop
    the entire run immediately** — mirroring Task 39's
    `DevelopmentRunStopped` fail-closed pattern — not be caught, logged,
    and silently skipped so the loop can continue. A partial run that
    quietly drops failed rows would make `n_total`/the aggregate counts
    untrustworthy; a hard stop with the failing `case_id` and the real
    exception in the error message is the only acceptable behavior.
  - `DevPreflightReport`: per-row `case_id`, `top1_code`, `top1_confidence`,
    `outcome` (`"semantic"` or `"PENDING_RERANK_NO_LLM_CALLED"`), plus
    aggregate counts (`n_total`, `n_semantic_fast_path`,
    `n_pending_rerank`, `n_semantic_fast_path_exact_match` — exact-match
    against `gold_isco_4digit` is fine to report **only** for the fast-path
    rows, since those are the only ones with a real, complete prediction;
    never compute or imply an overall accuracy figure that includes the
    pending-rerank rows as if they had a prediction), and a `run_manifest`
    dict: UTC timestamp, git commit (`git rev-parse HEAD` at run time),
    embedding model name (`hierarchical_store.MODEL_NAME`), collection
    name/profile queried, and the dev CSV's own SHA-256 — the same
    provenance discipline every other `eval/local_runs/` output in this
    project already follows, so this run stays traceable later.
  - `limit` (optional, default `None` = all 2,013 dev rows): lets a smoke
    run use a small prefix first.

**No change to any file under `backend/rag/`, `backend/agents/`, or
`eval/legacy_decision_policy41/policy.py`.** This task is additive-only,
exactly like Tasks 39-41.

## Required hermetic tests (no live Qdrant, no network)

Using a fake/mock store object (a minimal stand-in exposing only
`_embed_query`, `_query`, `_col_flat` — never a real `HierarchicalISCOStore`
instance, never a real Qdrant client):

1. exactly 5 `PolicyCandidate`s are built from 5 fake hits, in order;
2. `title_ar` and `description` are always `""`, never fabricated;
3. `level` is always `4`;
4. `code`/`title_en`/`confidence` map correctly from the raw hit payload
   (using `title_en`, never `label_en`);
5. fewer than 5 hits raises a typed, clear error;
6. a fast-path dev row (confidence ≥ 0.92) produces `outcome="semantic"`
   with a populated `top1_code`, zero reranker/LLM calls;
7. a below-threshold dev row produces `outcome="PENDING_RERANK_NO_LLM_CALLED"`,
   zero reranker/LLM calls, and does not raise;
8. `DevPreflightReport` aggregate counts sum correctly
   (`n_semantic_fast_path + n_pending_rerank == n_total`);
9. `n_semantic_fast_path_exact_match` is only ever computed over fast-path
   rows;
10. no test in this file ever imports `crewai`, `anthropic`, `openai`, or
    constructs a real Qdrant client;
11. `fetch_five_official_flat_candidates` and `run_dev_preflight` both
    contain no LLM/network import at module scope (source-scan test,
    mirroring Task 41's `test_policy_module_has_no_prohibited_imports`);
12. a row's `input_language` is passed through to `classify_with_policy`'s
    `lang` parameter unchanged (assert on the call, e.g. via a spy/mock
    reranker path or by inspecting `prompt_text` on a forced below-
    threshold row with a real `reranker` supplied only in this one test);
13. an unexpected exception raised mid-loop (e.g. a fake store whose
    `_query` raises `RuntimeError` on row 3 of 5) stops `run_dev_preflight`
    immediately — the exception propagates, `DevPreflightReport` is never
    returned, and no later row is processed after the failing one.

## Required live-Qdrant dev validation (real Qdrant, zero LLM calls)

This is a **real** run against the already-populated, shared
`isco08_unit_groups_flat_ilo2021_v1` collection (the same collection
Task 36 measured against) — read-only search calls only, no collection
mutation, no write of any kind to Qdrant.

1. Confirm `docker ps` shows `lfs_qdrant` healthy before starting; do not
   start/stop/restart it.
2. Generate `dev_run_eval_format.csv` via the exact export command above;
   confirm it has exactly 2,013 rows and its `case_id`s are a subset of
   `records.json`'s `split="dev"` rows (never `heldout`).
3. Run `run_dev_preflight` against a small prefix first (`limit=20`),
   inspect a handful of rows by hand, confirm sane codes/confidences.
4. Run `run_dev_preflight` against the full 2,013-row dev split.
5. Write the raw per-row output and the aggregate `DevPreflightReport` to
   `eval/local_runs/task42_flat_retrieval_dev_preflight_<UTC-timestamp>/`
   (git-ignored, same convention as every prior `eval/local_runs/` output).
6. Report the real aggregate counts (`n_semantic_fast_path`,
   `n_pending_rerank`, and the fast-path-only exact-match count/rate) in
   the final report, labelled exactly as what they are: a **dev-split**,
   **fast-path-only** observation, never a WISCO accuracy claim, never
   compared to Task 36's heldout numbers.

## Preservation checks

Before and after this task, prove byte identity (SHA-256) for:

1. `eval/legacy_decision_policy41/policy.py`, `legacy_identity.py` (must be
   byte-identical — this task only adds a new adjacent file);
2. Task 36 raw official WISCO flat/hierarchical CSVs and Task 37/37.1/38
   evidence;
3. Task 39/40/40.1 reports and task-specific source/test files;
4. `eval/local_benchmarks/wisco_isco08_v2_group_split/records.json` and
   `heldout_run_eval_format.csv` (this task only ever reads these — never
   writes to them; it only ever *creates* the new `dev_run_eval_format.csv`
   sibling file, which does not exist yet);
5. official ILO catalogue and collection-builder code
   (`backend/rag/official_isco08_catalogue.py`,
   `backend/rag/build_official_isco08_collections.py`);
6. B1 frozen configuration and B2 safety files;
7. `hierarchical_store.py` itself (confirm this task made zero source
   change to it, despite calling two of its existing methods).

Zero write to Qdrant of any kind, zero LLM/Anthropic/OpenAI call, zero
model download, zero Docker command, zero heldout-split read anywhere in
this task.

## Stop status

Use exactly one:

```text
FLAT_RETRIEVAL_DECISION_POLICY_DEV_INTEGRATION_PREFLIGHT_READY: yes
FLAT_RETRIEVAL_DECISION_POLICY_DEV_INTEGRATION_PREFLIGHT_READY: no
```

Set `yes` only if: the adapter's hermetic tests all pass; the full suite
passes; the real 2,013-row dev run against live Qdrant completed with
zero errors and zero LLM calls; preservation checks pass; and the
aggregate `n_pending_rerank` count was recorded (not blocking `yes` by
itself — it is information for a future task's scoping, not a failure
condition here).

On `yes`, **stop**. Do not call an LLM, do not compute or claim an
accuracy number, do not touch the heldout split, do not propose or start
a "Task 43" without a separate, explicit operator go-ahead — especially
given that any future reranker-invoking task will need real LLM API
credit, which is exactly the kind of budget decision that must be an
explicit operator call, not an inferred one.

## Final report

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_42_FLAT_RETRIEVAL_DECISION_POLICY_DEV_INTEGRATION_PREFLIGHT_FINAL_REPORT.md
```

Include: branch/final SHA/base SHA; the preserved-vs-different table above
(reproduced, not just referenced); the disclosed `label_en`/`label_ar`
payload-key gap and how this task's adapter avoids it; hermetic test
list mapped to the 11 requirements; the live dev-run command(s) and
literal aggregate output (`n_total`, `n_semantic_fast_path`,
`n_pending_rerank`, fast-path exact-match count/rate); preservation
evidence; exact stop status; confirmation of zero LLM/Anthropic/OpenAI
calls and zero heldout reads; confirmation of clean tree, untouched
protected branches, no PR; and an explicit, separate note estimating what
a future live-reranker task would cost (number of dev cases needing a
call) so any future credit/budget decision is well-informed.

Commit only task-specific files: the new adapter and its tests, plus the
final report. `dev_run_eval_format.csv` (like every other file under
`eval/local_benchmarks/`, including the existing `heldout_run_eval_format.csv`
it sits beside) stays **git-ignored / local-only, not committed** — it is
mechanically regenerable at any time from the exact export command in
this brief, and committing it would be inconsistent with how every other
split export in this project has always been handled. Same for the
`eval/local_runs/task42_.../` run output. Push only:

```text
reviewer2-flat-retrieval-decision-policy-dev-integration-preflight-20260810
```

Do not open a pull request.

## Claude Code execution prompt

```text
Execute Task 42 exactly as specified in:
Documentation/AI_HANDOFF/CLAUDE_TASK_42_FLAT_RETRIEVAL_DECISION_POLICY_DEV_INTEGRATION_PREFLIGHT.md

Create only reviewer2-flat-retrieval-decision-policy-dev-integration-preflight-20260810 from reviewer2-wisco-gold-label-ambiguity-audit-20260810 @ c45efbb685508015adad7051b5720b016058db0d.

Wire Task 41's decision-policy component to the current maintained flat
retrieval caller (HierarchicalISCOStore._embed_query/_query against the
official isco08_unit_groups_flat_ilo2021_v1 collection, limit=5), and run
it against the WISCO v2 DEV split only (2,013 cases) -- never heldout.

Disclose the pre-existing label_en/label_ar payload-key gap in
_flat_search() and read title_en directly instead, without modifying
hierarchical_store.py. Leave title_ar and description honestly blank
(official catalogue has neither) rather than fabricating them. Pass each
row's own input_language through as lang -- never leave it at the
default. Any unexpected exception mid-run must stop the whole run
immediately (fail closed, no silent per-row skipping), matching Task 39's
DevelopmentRunStopped pattern. Record a run_manifest (git commit,
timestamp, embedding model, collection/profile, dev CSV SHA-256) on the
output. Do not commit dev_run_eval_format.csv or the local_runs output --
both stay git-ignored, same as every other eval/local_benchmarks/ split
export.

Do not call any LLM, construct any CrewAI agent, or spend any
Anthropic/OpenAI credit anywhere in this task. Below-threshold dev rows
must be recorded as PENDING_RERANK_NO_LLM_CALLED and stop there --
never invent or simulate a reranker response. Do not compute or report
any accuracy number beyond the fast-path-only exact-match count, clearly
labelled as such.

Write and run the required hermetic tests, then run the real live-Qdrant
dev-split validation (read-only queries against the already-populated
shared collection, zero writes), then the full test suite. Preserve all
prior evidence, report the exact stop status plus an explicit
future-reranker-task cost estimate, commit only task files/report, push
only the new branch, and do not open a PR.
```
