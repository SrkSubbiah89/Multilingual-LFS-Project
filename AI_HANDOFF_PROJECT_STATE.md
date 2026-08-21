# Project State — AI Handoff Overview

**Purpose of this file**: you are an AI assistant picking up work on this
repository without the conversation history that produced its current
state. This document tells you what the project is, what's actually true
right now (verified, not assumed), what's mid-flight, and — most
important — the **real risks and inconsistencies** a fresh session needs
to know about before touching anything. Read this fully before making
changes. If anything below conflicts with what you observe in the repo,
**trust the repo and flag the conflict** — this document can go stale the
moment someone else (human or AI) makes an uncoordinated change, which is
exactly the failure mode it exists to reduce.

Generated: 2026-08-12. Verified against the live repository at that time.
**This is a full rewrite, not an incremental edit** — the previous version
(dated 2026-08-10) described a much earlier state (pre-Task-27) and was
badly stale by the time this was written; do not trust cached knowledge
of its contents.

---

## 1. What this project is

A multilingual Labour Force Survey (LFS) conversational-AI system: a
FastAPI + CrewAI backend and Next.js 14 frontend that conducts employment
interviews in English, Arabic (MSA + Gulf dialect), Urdu, Hindi, and
Tagalog, classifies job titles to **ISCO-08**, industries to **ISIC
Rev.4**, and education to **ISCED 2011 / ISCED-F 2013**, and implements a
full UAE Labour Force Survey questionnaire (Sections A–K) with dynamic skip
logic. It is an M.Tech thesis project (IIIT Kottayam, supervisor Dr.
Goutam Mali) also being prepared as a Conference I paper submission.

**Read `README.md` at the repo root first** — updated 2026-08-21, its
banner line and test counts are now accurate (2,282 tests, 89 files).
Everything else in it about architecture/endpoints/DB schema/directory
structure was last spot-verified accurate as of that date.

## 2. Repository identity

- GitHub: `SrkSubbiah89/Multilingual-LFS-Project`
- Local clone: `c:\Multilingual_LFS_Project` (Windows, PowerShell/git-bash).

## 3. ⚠️ Git state — read this before doing anything with branches or commits

**This is not the same situation the previous version of this document
described (uncommitted working-tree changes on `master`). That has been
completely superseded.**

- **`master` is far behind.** `origin/master`'s tip is
  `5e0ff5d88c6c973f636b48cacc25e5885c11d41c` ("gitignore Software/ and
  *.exe..."), predating essentially everything in §5 below.
- **All real work since then lives on a long, linear chain of task-specific
  branches**, each created from the previous task's final commit, each
  fully committed and pushed to `origin`, **none merged back into
  `master`, none opened as a PR** — this has been the standing, explicit
  workflow for every task (Task 27 through the branch you're likely on
  now). There is nothing sitting uncommitted in a working tree waiting to
  be lost — check `git status --porcelain` yourself to confirm the tree
  is clean before you start; if it isn't, that's new since this was
  written and you should investigate before proceeding.
- **Current branch** (as of this writing):
  `reviewer2-live-reranker-wisco-dev-run-20260810`, final commit
  `066d792ff64698bd6bec8adbda42f33bde073cea`. Run `git branch
  --show-current` and `git log -1` yourself — a later session may have
  created a further branch on top of this one.
- **The branch chain, most recent first** (each entry is that task's own
  branch tip; run `git log --oneline -40` for the full commit-level
  history): live-reranker/Ollama work and documentation alignment →
  Task 43 (live Anthropic reranker attempt, blocked on zero account
  credit, root-caused and fixed; Ollama fallback run) → Task 42 (flat
  retrieval wired to Task 41's policy, WISCO dev) → WISCO gold-label audit
  → Task 41 (historical decision-policy compatibility study) → Task 40.1 /
  Task 40 (historical qdrant-client provenance investigation, Outcome C —
  no defensible historical version) → Task 39 (literal legacy
  reproduction attempt, blocked, correctly stopped rather than shimmed) →
  Tasks 27-38 (Qdrant resilience, precise client-side deadlines, the full
  official WISCO Tier-1 run and its analysis, Phase 1/Reviewer-2 doc
  alignment).
- **Nothing here has been merged, rebased, or force-pushed anywhere.**
  Every task's own instructions explicitly forbade PRs; that has held
  throughout. If your task requires reconciling this branch chain with
  `master`, that is a decision for the user to make explicitly — do not
  decide it yourself.
- **Do not commit, push, merge, rebase, or switch branches without the
  user's explicit, current-session approval** for that specific action —
  standing rule, still binding.
- The `conference1-b2-evaluation` branch mentioned in the previous version
  of this document was **not re-checked in this pass** — assume it may
  still be diverged from the current branch chain; verify before touching
  anything it also touches (`eval/dev_set_schema.md`,
  `eval/validate_dev_set.py`, `eval/dev_sweep.py`, `eval/test_run_eval_b2.py`,
  `requirements.txt`).

## 4. Test suite — current verified state

```
pytest backend/tests eval/ -q
```
→ **2,347 passed, 0 failed, 1 deselected, 1 warning** (~7.5 min wall clock).
Backend-only (`pytest backend/tests -q`): **1,522 passed** (~6 min).
`eval/`-only: **825 passed**. 88 test files total (44 + 44).

No known standing failures. If you see one, it's new — investigate it as
real, don't assume it's a previously-documented artifact (the one
formerly documented here, a mock-signature mismatch in
`test_isco_classifier_extended.py`, was checked directly in this pass and
now passes — it was fixed at some point without this doc being updated,
exactly the staleness failure mode this document exists to prevent).

> **Update, 2026-08-21**: the `2,347`/`88 test files` figures above are a
> historical record from when this section was last written and are left
> unedited. A later cleanup pass removed the confirmed-dead-code
> `backend/agents/survey_orchestrator.py` and its dependent tests. See
> `README.md`'s header for the current, maintained count (`2,282 passed,
> 89 test files`).

## 5. What's been built — chronological, real state only

### 5a. Qdrant resilience and precise deadlines (Tasks 27-35)

Bounded, opt-in Qdrant query retry with a strict exception allowlist;
per-request timeouts and strict stage-level deadline budgets; a
client-side deadline pool corrected from `ceil()` to `floor()` (Task 34.1,
a real precision bug: qdrant-client's own internal rounding could exceed
the caller's budget under `ceil()`). All integrated into the official WISCO
evidence line and live-preflighted before each full run.

### 5b. The official WISCO Tier-1 evaluation — RUN, ANALYZED, PUBLISHED

**This is the single most important correction to the previous version of
this document, which said "no accuracy number for WISCO exists anywhere
in this repo." That is no longer true and has not been true since Task 36.**

- **Task 36** ran the full official Tier-1 evaluation against the frozen
  18,747-case WISCO v2 heldout split (dataset: Zenodo DOI
  `10.5281/zenodo.8262593`, official ILO 2021 ISCO-08 catalogue profile,
  10/43/130/436 major/submajor/minor/unit groups), flat and strict
  hierarchical, both arms fully clean (zero errors) under the precise
  client-side deadline enforcement.
- **Task 37 / 37.1** analyzed it. Task 37 disclosed one out-of-scope
  read-only Qdrant call during its own preservation check; Task 37.1 is
  the clean, fully-offline reproduction of record and is the citable one.
- **Published, canonical result** (character-for-character source of
  truth: `Documentation/Conference_I_Reviewer_2/OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`):

  | Metric | Flat | Strict hierarchical |
  |---|---:|---:|
  | Exact 4-digit accuracy | **21.19%** (3,973/18,747) | **10.35%** (1,941/18,747) |
  | 95% Wilson CI | [20.61%, 21.78%] | [9.93%, 10.80%] |

  Paired difference (hierarchical − flat): **−10.84 percentage points**
  (McNemar exact two-sided p ≈ 1.86×10⁻³⁰¹, 18,747 pairs). **Hierarchical
  retrieval is substantially worse than flat in this configuration** — a
  genuine, disclosed negative finding, never to be described as
  hierarchical improving accuracy. Safe/unsafe manuscript phrasing is
  pre-drafted in `MANUSCRIPT_SAFE_WISCO_WORDING.md` — use it verbatim
  rather than re-deriving wording.
- **Task 38** aligned Phase 1 and Reviewer-2 documentation to these real
  numbers (replacing whatever earlier placeholder/target figures existed).
- **A gold-label audit** (separate from Tasks 36-38, done later) checked
  WISCO's own gold codes for correctness: they exactly match the official
  436-code catalogue with one narrow exception — two records share
  identical English text ("Veterinary assistant") with different gold
  codes, now flagged via `ambiguity_flag` in the benchmark schema, proven
  to have zero effect on the published numbers above (byte-identical
  re-export of the scored CSV, confirmed by hash).

### 5c. Historical/legacy reproduction investigation (Tasks 39-41) — closed, correctly

An attempt to literally re-run the earliest (Feb 2026) ISCO classifier
commit's exact conditional-LLM decision policy hit a real, permanent
blocker: `qdrant-client`'s `QdrantClient.search()` method (which that
historical code calls literally) no longer exists in any qdrant-client
version installable today. Tasks 40/40.1 exhaustively searched git
history, local archives, Python environments, and Docker image metadata
for a defensible historical version pin — found one via a Docker image
built one day after the legacy commit (`qdrant-client==1.17.0`), but that
recovered version is the *same* version already proven incompatible, so
recovery didn't unblock anything; it confirmed the incompatibility was
present essentially from day one. Per explicit operator decision, **no
shim/monkeypatch was ever applied** — Task 41 instead built a disclosed
**Historical Decision-Policy Compatibility Study**: the old policy's exact
decision logic (5 candidates, 0.92 threshold, exactly-once reranker call,
candidate-only selection, historical prompt fields, fallback behavior)
reimplemented as a small additive component
(`eval/legacy_decision_policy41/policy.py`), explicitly and repeatedly
labeled as *not* a literal reproduction.

### 5d. Live reranker integration (Tasks 42-43) — infrastructure works, real numbers still pending

- **Task 42** wired Task 41's policy component to the real, current flat
  retrieval (`eval/legacy_decision_policy41/flat_retrieval_adapter.py`),
  ran it against the WISCO dev split with **zero LLM calls** (a
  `PENDING_RERANK_NO_LLM_CALLED` dry-run design) to find out, for free, how
  many cases would need a reranker: **0 of 2,013 dev cases reached the
  0.92 fast-path threshold** — every case would need one.
- **Task 43** built the real Anthropic reranker
  (`eval/legacy_decision_policy41/live_reranker.py`) and a hierarchical
  retrieval adapter. **First attempt found a serious silent-failure bug**:
  the configured Anthropic account had zero credit for the entire
  session; every reranker call failed with `BadRequestError`, and because
  `classify_with_policy` (correctly, by design) treats any reranker
  exception the same as a malformed response, this silently produced
  thousands of `method="llm_ranked"` rows that were actually just
  semantic-top-1 predictions — caught only because the flat-heldout run's
  predictions matched Task 36's non-reranked baseline for **100% of
  18,747 cases**, which is not statistically plausible for a working
  reranker. **Fixed**: a `fatal_tracker` mechanism now aborts the whole
  run immediately, by name, on a fatal (non-retryable, non-parsing)
  reranker failure, instead of silently completing with mislabeled rows.
  No real Claude 3.5 Sonnet reranked accuracy number was ever obtained —
  **Anthropic credit is still required** for that; do not cite any number
  from this task as a real Claude result.
- **A free Ollama (`llama3.2:latest`) fallback was run instead** (full
  2,013-case WISCO dev, flat retrieval): **17.19% (346/2,013)** — real,
  genuine (344 distinct codes used, confirmed not another silent-fallback
  incident), but a different, weaker, free local model, not Claude —
  label results accordingly. One operational incident during this run
  (a `tasklist`-based liveness check gave a false "process is dead"
  reading, leading to an accidental concurrent duplicate 150-row batch)
  was caught and cleanly deduplicated; full account in
  `Documentation/AI_HANDOFF/CLAUDE_TASK_43_LIVE_RERANKER_WISCO_RUN_FINAL_REPORT.md`.

### 5e. Real bugs found and fixed live (2026-08-11/12)

- **`label_en`/`label_ar` payload-key gap**: the official-profile Qdrant
  collections write `title_en`, never `label_en`, and have no Arabic
  title at all; every raw-payload read across `hierarchical_store.py`/
  `hierarchy_engine.py` checked only `label_en`, so official-profile
  search results always had a blank English label (harmless to Task 36's
  exact-code scoring, real for any UI/display use). Fixed via
  `hierarchy_engine.extract_label_en()`.
- **Subsistence-farming unit-code mis-numbering**: the legacy knowledge
  base (`backend/rag/load_full_isco.py`) had codes 6161-6164 filed under a
  sub-major group with no defined parent at all, when the correct parent
  (sub-major 63) already existed correctly. Renumbered to 6310-6340.
  **This fix required a live database action to actually take effect** —
  `start.bat`'s own logic skips reloading Qdrant collections that already
  exist, so the source fix alone did nothing until
  `python -m backend.rag.load_full_isco --recreate` was run. Verified live
  end-to-end via `/debug/isco/subsistence crop farmer` → `6310`. **15**
  further non-standard codes and **10** missing-official codes from the
  same original finding remain unresolved (need a full official-standard
  cross-check, not a guess — see
  `Documentation/Phase_2/Week_1/module_a_week1_report.md` §5).
- **A real, currently-was-broken DB migration gap**: `models.py` already
  referenced `SurveyResponse.supersedes_id`, but the live database was one
  migration behind (`a3f9c1d2e4b6_add_survey_response_supersedes_id`) —
  any code path using that column would have failed. Applied; DB now at
  head.

### 5f. Phase 2 (thesis) work — real, partial, and reviewed against an external plan

`Documentation/Phase_2/Week_1/` and `Week_2/` contain real, dated Module A
(WISCO external validation) work: the canonical WISCO file/DOI verified
directly (`8262593`, not `7598568` — an externally-supplied "corrected"
plan actually had this backwards), 4,232 clean parsed occupation records,
9/9 integrity checks, and the root cause of the subsistence-farming bug
above. A later externally-supplied "Phase II Comprehensive Plan" document
was checked against this real work and against the Task 36 evidence
above; found to contain several factual errors (wrong WISCO DOI, a
Phase-I baseline of "85.3% ISCO accuracy / hierarchical +6.4pp" that
doesn't match the real measured 21.19%/10.35%/−10.84pp, a "13-agent"
architecture claim that doesn't match the actual 12-agent codebase) — full
itemized corrections in `Documentation/Phase_2/PHASE_II_PLAN_CORRECTIONS.md`.
A subsequent module-by-module code review (same file, §8) found Modules
B/C (ISIC/ISCED-F hierarchical retrieval) and Module D's LOW-severity SRE
gap already have real, tested infrastructure the external plan didn't know
about — remaining gaps are data-coverage (134/419 ISIC classes, 63/~80
ISCED-F fields) and a live-Qdrant deployment decision (`--execute`, never
run), not missing code. Module H's "delegation correctness" framing
doesn't apply to this codebase at all (zero CrewAI hierarchical-delegation
usage anywhere, confirmed exhaustively; needs rescoping to "orchestration
correctness"). Module E (pilot, n=30) has the longest lead time of
anything in the plan and, as of the last check, still has no ethics
application submitted — the single most time-critical open item across
the whole project, independent of any code work.

### 5g. `start.bat` end-to-end verification (2026-08-11/12)

Every step actually run, not just read: Docker/Postgres/Redis/Qdrant/
Ollama health, pip install, Alembic migration (found and fixed the gap in
§5e), Qdrant collection reload (found and fixed the staleness in §5e),
backend startup (`/ready`, `/health` both green — DB/Redis/Qdrant/Ollama
all connected), frontend startup (all 4 pages 200), and a real functional
test through the actual running API: OTP request → verify → JWT → create
survey session. **The backend (port 8000) and frontend (port 3000) may
still be running** from this test — check before assuming they aren't,
and check with the user before stopping them if they might be using it.

## 6. What has NOT happened yet

- **A real Claude 3.5 Sonnet reranked WISCO accuracy number.** Blocked
  purely on Anthropic account credit (§5d). The code is ready; adding
  credit and re-running (dev split first, ~$3-8) is the immediate next
  step once available.
- **No real, permissioned LFS respondent dataset has ever been obtained.**
  `approved_real_lfs_validation` has never been used for anything but test
  fixtures proving the governance gate works.
- **Module E (pilot, n=30)**: not started. No ethics application
  submitted as of the last check — this is the highest-lead-time item in
  the entire project.
- **ISIC/ISCED-F hierarchical retrieval**: infrastructure built and
  tested (§5f), but the live Qdrant collections have never been populated
  (`--execute` never run) and data coverage is well short of the 419/~80
  targets. Do not describe ISIC/ISCED as "deployed hierarchical RAG" —
  only ISCO-08's is live.
- **The SRE crosswalk tables** (`_ISCO_MAJOR_TO_ISIC` etc. in
  `semantic_relation.py`) are confirmed still hand-built, not yet rebuilt
  from the official ILO ISCO-08 Volume I / UNESCO ISCED 2011 Operational
  Manual Table 7 correspondence tables.
- **The remaining 15/10 legacy-knowledge-base code discrepancies** (§5e) —
  only the 4-code subsistence-farming cluster was fixed; the rest needs a
  full official ISCO-08 cross-check, not a guess.
- **The `conference1-b2-evaluation` branch** — not re-checked in this
  pass; assume still diverged (§3).

## 7. Conventions you must follow

Established deliberately, tested extensively, violating them will look
like regression to anyone continuing this work:

- **Null + reason, never a placeholder.** Any metric that can't be
  measured is `None`/`null` with a sibling `<field>_unavailable_reason`
  string. Never 0, never `"N/A"` alone, never silently omitted.
- **`dataset_label`** is a closed 3-value enum
  (`eval/dataset_card_schema.py::DATASET_LABELS`). A controlled benchmark
  (WISCO, any future one) can **never** be `approved_real_lfs_validation`,
  unconditionally — it's reference/dictionary data, not LFS respondent data.
- **`evaluation_status`** is orthogonal to `dataset_label` — one describes
  data realness, the other whether metrics were actually measured.
- **`manuscript_eligible`** is computed automatically, never
  caller-supplied.
- **Gitignored local-output convention**: `eval/local_runs/` and
  `eval/local_benchmarks/` are both gitignored unconditionally. Regenerate
  from tracked source + script; never try to commit their contents.
- **No paid LLM/API call without explicit, separate pre-approval** naming
  model/provider, estimated call count, cost, and reason. (Task 43 learned
  this the hard way in the other direction too: verify credit is actually
  present with a trivial real call before trusting a "smoke test" that
  only checks the call *completed*, not that it *succeeded* — a caught
  exception can look identical to a real answer at the call site.)
- **Any reranker/external-call wrapper used to drive many rows must have a
  fatal-vs-retryable distinction that can abort the whole run**, not just
  retry-then-silently-fall-back — the exact lesson from §5d.
- **For long local-inference runs, use short checkpointed batches with a
  liveness check based on output-file growth, not `tasklist`/process
  listing** — `tasklist` gave a false negative twice in this session
  (§5d, §5g) for processes that were, in fact, still running.
- **Never commit/push/merge/rebase/PR without explicit approval in the
  current session** for that specific action.
- **Branch-per-task workflow**: verify the exact base branch/SHA the task
  specifies before creating a new branch; commit only task-specific files;
  push only the new branch; never touch protected/prior branches; no PR
  ever, unless a task explicitly says otherwise.

## 8. Suggested next steps (for the user to prioritize)

1. **Ethics submission for Module E (pilot)** — the single most
   time-critical item in the project, independent of all code work.
2. **Add Anthropic credit and get the real Claude 3.5 Sonnet reranked
   WISCO number** — infrastructure ready, dev split first (~$3-8) before
   committing to the full heldout.
3. Decide whether to deploy the ISIC/ISCED-F hierarchical Qdrant
   collections (`--execute`) and/or prioritize their data-coverage
   expansion.
4. Decide whether/when to rebuild the SRE crosswalk tables from real
   official ILO/UNESCO source documents.
5. Decide whether to pursue the remaining 15/10 legacy-knowledge-base code
   discrepancies, and whether/when to reconcile `conference1-b2-evaluation`.
6. Stop or keep running the backend/frontend test instances from §5g, per
   the user's actual intent.
