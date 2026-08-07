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

Generated: 2026-08-10. Verified against the live repository at that time.

---

## 1. What this project is

A multilingual Labour Force Survey (LFS) conversational-AI system: a
FastAPI + CrewAI backend and Next.js 14 frontend that conducts employment
interviews in English, Arabic (MSA + Gulf dialect), Urdu, Hindi, and
Tagalog, classifies job titles to **ISCO-08**, industries to **ISIC
Rev.4**, and education to **ISCED 2011 / ISCED-F 2013**, and implements a
full UAE Labour Force Survey questionnaire (Sections A–K) with dynamic skip
logic. It is an M.Tech thesis project (IIIT Kottayam) also being prepared
as a Conference I paper submission.

**Read `README.md` at the repo root first** — it's a substantial (1,183-
line), largely accurate architecture reference: system diagram, tech
stack, the Semantic Relation Engine (a real thesis contribution — a
three-way ISCO/ISIC/ISCED coherence crosswalk), the full questionnaire
field reference, API reference, DB schema, directory structure. Treat its
banner line ("1,178 tests passing") as **stale** — see §4 for the current
number. Everything else in it about architecture/endpoints/DB schema was
last verified accurate as of its "Last Updated: June 2026" note; re-verify
before relying on specifics if it's been a while.

## 2. Repository identity

- GitHub: `SrkSubbiah89/Multilingual-LFS-Project`
- You are almost certainly working in a local clone at
  `c:\Multilingual_LFS_Project` (Windows).

## 3. ⚠️ Git state — read this before doing anything with branches or commits

This is the single most important section in this document.

- **Current branch: `master`**, up to date with `origin/master`.
- **A second remote branch exists: `conference1-b2-evaluation`**, which is
  **3 commits ahead of `master`** and contains substantial, unmerged
  "B2 evaluation hardening" work: `eval/pre_run_check.py`,
  `eval/full130_access_guard.py` (both **do not exist on `master` at all**),
  plus much more extensive versions of `eval/dev_sweep.py`,
  `eval/validate_dev_set.py`, `eval/dev_set_schema.md`,
  `eval/test_validate_dev_set.py`, `eval/test_dev_set_v1_readiness_report.md`,
  and a `scikit-learn==1.6.1` addition to `requirements.txt`. Diff size:
  ~5,100 lines across 20 files.
- **These two branches have diverged and were never reconciled.** All of
  the "Conference I Reviewer #2 response" work described in §5 below was
  built entirely on top of `master`'s (older, less-hardened) version of
  `eval/dev_set_schema.md`, `eval/validate_dev_set.py`,
  `eval/test_run_eval_b2.py`, and `requirements.txt` — all four of which
  **also independently exist and differ on `conference1-b2-evaluation`**.
  Merging the two branches today would very likely produce real conflicts
  in those four files, and `pre_run_check.py`/`full130_access_guard.py`
  would need to be evaluated for whether they should now depend on/be
  updated for the newer governance/manifest machinery built in Steps 3-6
  below (they predate it).
- **Every file described in §5 is currently UNCOMMITTED** — 53 changed/new
  files sitting in the working tree on top of `master`, never committed,
  never pushed, never merged anywhere. If you `git stash`, `git checkout .`,
  `git reset --hard`, switch branches, or do anything else that discards
  working-tree changes, **all of Steps 1 through 7A described below are
  permanently lost** unless you've verified a backup exists. There is no
  safety net beyond the working directory right now.
- **Do not commit, push, merge, rebase, or switch branches without the
  user's explicit, current-session approval** — this has been the standing
  rule throughout the work described below, and remains binding on you.
  If your task requires resolving the branch divergence, that is a
  decision for the user to make (which branch is authoritative, whether to
  cherry-pick the B2-hardening work forward, whether `pre_run_check.py`/
  `full130_access_guard.py` are still wanted) — do not decide it yourself.

**Action for you, if your task touches any of**: `eval/dev_set_schema.md`,
`eval/validate_dev_set.py`, `eval/dev_sweep.py`, `eval/test_run_eval_b2.py`,
or `requirements.txt` — stop and confirm with the user which branch's
version is authoritative before editing. Editing blind risks silently
discarding the other branch's work when someone eventually reconciles them.

## 4. Test suite — current verified state

```
pytest backend/tests eval/ -q
```
→ **1656 passed, 1 known pre-existing failure, 1 deselected, 1 warning**
(last run: this session, ~22-25 min wall clock — has been getting slower
as the suite grows; budget accordingly, don't assume the old ~4 min figure).

The one known failure:
`backend/tests/test_isco_classifier_extended.py::TestHierarchicalStages::test_llm_used_for_low_similarity`
— a pre-existing mock-signature mismatch (`clf.<locals>.<lambda>() got an
unexpected keyword argument 'temperature'`), unrelated to any Reviewer #2
work, documented since the first QA baseline in this project. Treat it as
expected/ignorable; do not "fix" it without being asked, and do not treat
its presence as a sign something you did broke something.

**`Documentation/Conference_I_Reviewer_2/FINAL_QA_BASELINE.md` says "1501
passed"** — that is Step 1's historical snapshot, not current. Always
re-run the command above rather than trusting any written-down test count,
including this one.

## 5. What's been built: the "Conference I Reviewer #2 response" work

A reviewer gave 8 criticisms of the manuscript (Springer formatting,
unclear novelty, no real-LFS validation, missing computational analysis,
an abstract overclaiming ISIC coverage, unclear multi-LLM division of
labour, low-res figures, untraceable claims). The response has been an
extended, multi-session effort to build **defensible, non-fabricated
evidence and instrumentation** — never manuscript text itself — governed by
one hard rule that has held across every step: **an unmeasured/unmeasurable
value is `null` + an explicit reason string, never a placeholder or
fabricated number.** All work lives under `Documentation/Conference_I_Reviewer_2/`
(docs) and `eval/` (tooling), plus a few `backend/` additions.

**Start here**: `Documentation/Conference_I_Reviewer_2/REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md`
is the authoritative index — one row per reviewer comment (5 allowed status
labels only), plus a dated section per numbered Step summarizing what that
step did. Read it before reading anything else in that directory.

### Step-by-step summary (chronological)

1. **Initial pass (Sections A-J)**: classifier method registry
   (`backend/agents/method_registry.py`), a generic hierarchical-retrieval
   engine extracted from the ISCO-specific one
   (`backend/rag/hierarchy_engine.py`), coverage-audit tooling
   (`eval/coverage_audit.py`), evaluation/reproducibility manifests
   (`eval/manifest.py`, `eval/analyze.py`), ablation infrastructure
   (`eval/ablation_runner.py`), real-LFS-intake governance schema
   (`eval/dataset_card_schema.py`, `eval/validate_real_lfs_governance.py`),
   Semantic Relation Engine evidence tracking, figure-data exports.
2. **Step 2** — official classification-standard provenance: distinguishes
   a sourced-but-unverified official count from a verified one (only ever
   produced by importing a real catalogue file via
   `eval/catalogue_importer.py`); `coverage_percentage` is `null` unless
   verified.
3. **Step 3** — hardened the real-LFS governance gate: expanded
   `DatasetCard` to ~45 fields, closed 3-value `dataset_label` vocabulary
   (`synthetic_or_operationally_realistic` / `approved_real_lfs_validation`
   / `invalid_incomplete_governance`), fail-closed validator with a
   synthetic-marker content safeguard and an in-repo-path safeguard.
4. **Step 4** — added a genuine `--dry-run` mode to `eval/run_eval.py` and
   `eval/ablation_runner.py`: validates everything, writes a full manifest,
   makes **zero** model/Qdrant/LLM/network calls. New
   `evaluation_status` field (`"measured"` / `"dry_run_not_measured"`),
   orthogonal to `dataset_label`.
5. **Step 5 / 5.1** — ran an actual measured (non-dry-run) evaluation
   against a tiny (n=5) synthetic fixture, purely as a pipeline-integration
   check — new `evaluation_status="measured_synthetic_fixture_only"` value
   and a `manuscript_eligible` manifest field (computed automatically,
   never caller-supplied; only `True` for `measured` + `approved_real_lfs_validation`).
   **This run found and Step 5.1 fixed a real bug**: `--sre off` was
   accidentally also disabling ISIC/ISCED classification (shared
   conditional in `eval/run_eval.py`), not just the SRE coherence check —
   see `SRE_COUPLING_BUGFIX.md`.
6. **Step 6** — audited every candidate benchmark dataset already in the
   repo for label provenance. Finding: the existing "130-case ISCO set"
   (`eval/test_set_full130.csv`) has **no documented label source** and is
   not benchmark-defensible. **But a much stronger, previously-dormant
   asset was found**: `backend/evaluation/wisco/` — a real, externally
   published (Zenodo DOI `10.5281/zenodo.8262593`, CC-BY-4.0), multilingual
   (61 languages, 5 of which are this project's target set)
   occupation-title-to-ISCO-08-code dataset ("WISCO"), downloaded in an
   earlier project phase for an unrelated purpose and never used as an
   evaluation benchmark. New schema (`eval/controlled_benchmark_schema.py`)
   and validator (`eval/validate_controlled_benchmark.py`) built; WISCO
   converted into a 20,760-record benchmark package
   (`eval/build_wisco_isco_benchmark.py` →
   `eval/local_benchmarks/wisco_isco08_v1/`, **gitignored**, regenerate
   from tracked source + script).
7. **Step 7A** — strict leakage audit of that v1 package
   (`eval/audit_wisco_benchmark_leakage.py`, deterministic, zero
   model/network calls). **Found real leakage**: 4 cross-split
   exact-duplicate-title groups (different WISCO entries, same text, same
   code, split apart by chance). Built a corrected, group-aware **v2**
   (`eval/build_wisco_isco_benchmark_v2_group_split.py`, fixed seed 42 →
   `eval/local_benchmarks/wisco_isco08_v2_group_split/`, **gitignored**)
   — re-audited clean. v1 retained unaltered as the audit record. A
   detailed, cost/time-aware Step 7B execution plan was prepared (deterministic
   stratified 500-record reranking subset, full-heldout no-LLM tier, exact
   commands, wall-clock/Ollama-call estimates) but **not executed**.

### What has NOT happened yet

- **Step 7B — the actual measured WISCO benchmark run — has not been run.**
  Nothing in `eval/local_benchmarks/wisco_isco08_v2_group_split/` has been
  fed through `eval/run_eval.py` or `eval/ablation_runner.py`. No accuracy
  number for WISCO exists anywhere in this repo. The exact commands and a
  full resource/time estimate are in
  `Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`
  — running the reranking-subset tier (~5-6.5 hours of local LLM inference)
  explicitly requires user approval before it happens.
- No real, permissioned LFS respondent dataset has ever been obtained —
  `approved_real_lfs_validation` has never been used for anything but test
  fixtures proving the governance gate works.
- ISIC Rev.4 and ISCED-F 2013 **hierarchical retrieval** (new Qdrant
  collections, live multi-stage search) was never built — only
  method-label stubs. Do not describe ISIC/ISCED as "hierarchical RAG" in
  any manuscript-facing text.
- The B2-hardening work on `conference1-b2-evaluation` (§3) has not been
  merged or reconciled with any of the above.

## 6. Conventions you must follow if you touch `eval/` or governance code

These were established deliberately, tested extensively, and violating
them will look like regression to anyone continuing this work:

- **Null + reason, never a placeholder.** Any metric that can't be
  measured is `None`/`null` with a sibling `<field>_unavailable_reason`
  string. Never 0, never `"N/A"` alone, never silently omitted.
- **`dataset_label`** is a closed 3-value enum
  (`eval/dataset_card_schema.py::DATASET_LABELS`) — never a free string.
  A controlled benchmark (WISCO, any future one) can **never** be
  `approved_real_lfs_validation`, by design, unconditionally, regardless of
  label quality — it's reference/dictionary data, not LFS respondent data.
- **`evaluation_status`** (`eval/manifest.py::EVALUATION_STATUSES`) is
  orthogonal to `dataset_label` — one describes data realness, the other
  describes whether metrics were actually measured.
- **`manuscript_eligible`** on every manifest is computed automatically
  inside `build_manifest()`, never caller-supplied, and is `True` only for
  `evaluation_status="measured"` + `dataset_label="approved_real_lfs_validation"`.
- **Gitignored local-output convention**: `eval/local_runs/` (dry-run and
  measured-run artifacts) and `eval/local_benchmarks/` (generated benchmark
  packages) are both gitignored unconditionally. Regenerate from tracked
  source + script; never try to commit their contents. Never write to the
  tracked `eval/results/` directory without an explicit reason — check
  `git status` after any `run_eval.py`/`ablation_runner.py` invocation that
  doesn't pass `--output-root`.
- **No paid LLM/API call without a separate, explicit pre-approval** naming
  the model/provider, estimated call count, cost method, and reason —
  every run so far has used local Ollama (`llama3.2:1b`), which is free.
- **Never commit/push/PR/delete without explicit approval in the current
  session** — a standing rule across every step above, not just a
  one-time instruction.

## 7. Other issues found during this review (2026-08-10)

- `Documentation/Conference_I_Reviewer_2/README.md`'s document index was
  stale (missing Steps 4-7A's docs) — **fixed** as part of this review.
- `.git` is 1.2 GB locally. A 1.2 GB installer (`Software/OllamaSetup.exe`)
  was previously committed by accident and stripped from history
  (2026-08-05, see `.gitignore`'s comment) — the strip may not have
  reclaimed local `.git` size (a `git gc --aggressive` would need explicit
  approval since it rewrites packfiles; not done here).
- No leftover `TODO`/`FIXME`/`XXX` markers in any `eval/*.py` file added
  during this work — checked, clean.
- `requirements.txt` currently only differs from committed `HEAD` by one
  line (`+pyyaml`, a genuine dependency of `eval/coverage_audit.py`/
  `catalogue_importer.py`) — legitimate, not an issue on its own; the
  `conference1-b2-evaluation` branch's additional `scikit-learn==1.6.1` is
  not needed by anything on `master` today (verified: no current file
  imports `sklearn`), only relevant if that branch's `dev_sweep.py`
  extensions are later merged in.

## 8. Suggested next steps (not started, for the user to prioritize)

1. Decide what to do about the branch divergence (§3) before any further
   `eval/` work — this is the highest-risk open item.
2. If continuing the Reviewer #2 response: Step 7B (execute the prepared
   WISCO evaluation plan) is the natural next step, gated on user approval
   for the reranking tier's runtime.
3. Commit the current working-tree state (53 files) to *something* soon —
   it has no backup beyond the local working directory right now.
