# Claude Merge-Impact Analysis — Reviewer #2 Snapshot vs. `conference1-b2-evaluation`

Produced in response to
`Documentation/AI_HANDOFF/PERPLEXITY_TO_CLAUDE_02_MERGE_IMPACT_ANALYSIS.md`
(task ID `02-merge-impact-analysis`). **This is analysis only. No merge,
rebase, reset, clean, stash, pull, checkout/switch to another branch,
benchmark run, or source-code/test/dependency/config modification was
performed.** `master`, `conference1-b2-evaluation`, and
`reviewer2-enhancement` were not touched. `git merge-tree --write-tree` was
used twice below to empirically verify predicted conflicts — this command
writes a loose tree object to `.git/objects` but does **not** modify the
working tree, the index, HEAD, or any branch/ref; both invocations were
followed by an explicit `git status --short` / `git rev-parse HEAD` check
confirming no observable state changed.

## 1. Executive summary

The literal git-level overlap between the protected Reviewer #2 snapshot
and `conference1-b2-evaluation` is **much smaller than the task's starting
assumption implied**: only **2 files** (`eval/test_run_eval_b2.py`,
`requirements.txt`) were changed by *both* branches relative to `master`.
Empirically simulating the merge (`git merge-tree`) confirms exactly one
real textual conflict — `eval/test_run_eval_b2.py` — and one clean
auto-merge — `requirements.txt`.

The three files named in the task's "shared divergence" list
(`eval/dev_set_schema.md`, `eval/validate_dev_set.py`, `eval/dev_sweep.py`)
are **not actually touched by the Reviewer #2 snapshot at all** — the
snapshot only *reads/references* them in documentation (Step 6's benchmark
audit cites `eval/dev_set_schema.md` and `eval/validate_dev_set.py` as
pre-existing infrastructure, never edits them). A git merge would silently
take `conference1-b2-evaluation`'s versions of these three files with zero
conflict markers. The real risk here is **not textual, it's semantic
staleness**: several Reviewer #2 docs describe the *master* version of
these files' behaviour (e.g. `eval/dev_set_v1_template.csv` being empty),
and after a merge those docs would be describing an already-superseded
reality without any git conflict ever flagging that a re-read is needed.

Two capability areas are genuinely complementary and worth deliberate
integration, not just conflict-avoidance: B2's `full130_access_guard.py`
(a *runtime* file-open guard) has no counterpart in the Reviewer #2
snapshot, which only has *static/audit-based* leakage detection
(`eval/audit_wisco_benchmark_leakage.py`); and B2's `eval/configs/
b1_frozen.json` records a real, already-measured accuracy figure (54/130,
41.5% top-1 on `eval/test_set_full130.csv`, verified 2026-08-05) that sits
in tension with Step 6's finding that `full130` lacks documented label
provenance (`not_eligible_unknown_provenance`) — both facts are true
simultaneously and must be reconciled in any manuscript-facing integration,
not silently dropped.

**No merge is recommended in this task or the near term without deliberate,
manual reconciliation of `eval/test_run_eval_b2.py` and a documented
decision on how the B1/B2 baseline (54/130 on `full130`) relates to Step
6's `full130` provenance finding.**

## 2. Verified commit graph and branch state

```
git status --short                                   → (empty; clean)
git rev-parse --abbrev-ref HEAD                       → reviewer2-wip-snapshot-20260807
git rev-parse HEAD                                    → 92a84d7578bd35a5aa13e2293613eb0c4754c3ee
git fetch origin                                      → (up to date, no new refs beyond what's used below)
git rev-parse origin/master                           → 5e0ff5d88c6c973f636b48cacc25e5885c11d41c
git rev-parse origin/conference1-b2-evaluation         → 675121bbf656dad3e611f3280d95606b5b123e92
git rev-parse origin/reviewer2-wip-snapshot-20260807   → 92a84d7578bd35a5aa13e2293613eb0c4754c3ee
git merge-base origin/conference1-b2-evaluation HEAD   → 5e0ff5d88c6c973f636b48cacc25e5885c11d41c
```

`git merge-base` confirms both branches share `master`'s tip
(`5e0ff5d`) as their common ancestor — a simple two-way divergence, no
prior partial merges to account for.

**Correction to the previous audit report**
(`Documentation/AI_HANDOFF/CLAUDE_PROJECT_AUDIT.md`): its §1 "Confirmed
remote branch" line reported `git ls-remote`'s output at the time it was
run, which reflected the branch tip *as of the first (snapshot) commit*
being described in that section — but the audit report's own §1 headline
SHA (`b188202ea843b69dcdaf9fc6bb40dce97f709d72`) was in fact already the
**second** commit (the audit report itself), correctly labelled "Second
audit-report commit SHA" in that report's final response. That detail was
accurate. What has changed since is that **the branch tip has moved again**:
a third commit, `92a84d7578bd35a5aa13e2293613eb0c4754c3ee` ("Add B2 and
Reviewer 2 merge impact task"), was added on top by this task's own
handoff-file delivery (arrived via `git pull --ff-only` in this session,
before this analysis task began). Ancestry check performed:

```
git merge-base --is-ancestor b188202ea843b69dcdaf9fc6bb40dce97f709d72 HEAD
→ true (confirmed: b188202 remains an ancestor of the current branch tip)
```

**Protected snapshot baseline** (content-freeze point, per task 01):
`b188202ea843b69dcdaf9fc6bb40dce97f709d72` — 114 files changed vs. `master`
(113 from the Step-1-through-7A snapshot commit + 1 audit-report file),
**2 commits** ahead of `master`.

**Current branch tip** (after this task-file commit and, once this report
is committed, one more): `92a84d7578bd35a5aa13e2293613eb0c4754c3ee` before
this report; **115 files changed** vs. `master`, **3 commits** ahead. The
task file itself (`PERPLEXITY_TO_CLAUDE_02_MERGE_IMPACT_ANALYSIS.md`,
documentation-only) is the delta between the protected baseline and the
current tip. This report will be the 4th commit.

Full commit list, `master..reviewer2-wip-snapshot-20260807`:
```
92a84d7 Add B2 and Reviewer 2 merge impact task
b188202 Add Reviewer 2 worktree recovery audit
e20af39 WIP snapshot: Conference I Reviewer 2 work through Step 7A
```

Full commit list, `master..origin/conference1-b2-evaluation`:
```
675121b Add missing scikit-learn dependency to requirements.txt
66bac86 Fix final three B2 validator weaknesses: robust full130 guard, header validation, fail-closed catalogue
658c4c2 Add B2 reproducibility gates and dev-set validation
```

## 3. Change inventory table

**`master` → `reviewer2-wip-snapshot-20260807`**: 115 files (113 from the
content commit, 1 audit report, 1 task-handoff doc). Full list omitted here
for brevity (available via `git diff --name-status master..HEAD`) — see
`Documentation/AI_HANDOFF/CLAUDE_PROJECT_AUDIT.md` §3 for the prior
enumeration of the content commit's 113 files.

**`master` → `origin/conference1-b2-evaluation`**: 20 files.

| Path | Change | Notes |
|---|---|---|
| `eval/PRE_RUN_B2_CHECKLIST.md` | A | New doc |
| `eval/build_full130_leakage_manifest.py` | A | New script |
| `eval/configs/b1_frozen.json` | A | New config — records a real measured B1 baseline (see §5) |
| `eval/configs/full130_leakage_manifest.json` | A | New config — hash-only manifest |
| `eval/dev_set_schema.md` | M | 68 → 386 lines |
| `eval/dev_set_v1.csv` | A | Header-only, 0 data rows (same state as `master`'s separately-named `eval/dev_set_v1_template.csv`) |
| `eval/dev_set_v1_data_dictionary.md` | A | New doc |
| `eval/dev_set_v1_provenance.md` | A | New doc |
| `eval/dev_set_v1_readiness_report.md` | A | New doc |
| `eval/dev_sweep.py` | M | 680 → ~1,225 lines |
| `eval/full130_access_guard.py` | A | New script |
| `eval/pre_run_check.py` | A | New script |
| `eval/test_dev_sweep.py` | M | |
| `eval/test_full130_access_guard.py` | A | New tests |
| `eval/test_pre_run_check.py` | A | New tests |
| `eval/test_run_eval_b2.py` | M | **Overlaps with snapshot — see §4/§5** |
| `eval/test_validate_dev_set.py` | M | |
| `eval/validate_dev_set.py` | M | 281 → 487 lines |
| `requirements-dev.txt` | A | New, B2-test-only dependency lock |
| `requirements.txt` | M | **Overlaps with snapshot — see §4/§5** |

## 4. Overlap and conflict-risk table

**Exact path intersection** (`comm -12` on sorted path lists from both
diffs against `master`): **2 paths**.

| Path | Category | Empirical `git merge-tree` result |
|---|---|---|
| `eval/test_run_eval_b2.py` | **Likely direct textual conflict** | `CONFLICT (content): Merge conflict in eval/test_run_eval_b2.py` |
| `requirements.txt` | Independently additive | Clean auto-merge (different insertion points: snapshot adds `pyyaml` after `openpyxl`; B2 adds `scikit-learn==1.6.1` after `sentence-transformers`) |

**Non-overlapping paths with real coexistence risk** (touched by only one
branch, but semantically entangled with the other's assumptions):

| Path | Category | Why it's flagged despite no git conflict |
|---|---|---|
| `eval/dev_set_schema.md` | Likely semantic conflict (no textual conflict) | Snapshot docs (`CONTROLLED_BENCHMARK_AUDIT.md`, `AI_HANDOFF_PROJECT_STATE.md`) describe the *current/master* 68-line version's simpler schema (no `full130_access_guard`, no ISCO-catalogue validation, no normalization fingerprinting) as "existing infrastructure" — post-merge those descriptions would be stale, though no file would show a conflict marker |
| `eval/validate_dev_set.py` | Likely semantic conflict | `validate_dev_set()`'s parameter signature changes shape (see §5) — any *future* snapshot code that calls it positionally would silently break, though today nothing in the snapshot calls it at all (checked: zero references) |
| `eval/dev_sweep.py` | Potentially compatible implementation (mixed) | Core dataclasses (`KSweepResult`) are byte-identical; `main()`'s CLI contract is a breaking rewrite (see §5) |
| B2-only files (`pre_run_check.py`, `full130_access_guard.py`, etc.) | Independently additive | No path collision possible (files don't exist in snapshot) — categorized fully in §6 |

**For B2-only files**: none has an equivalent/superseding module in the
Reviewer #2 snapshot under the same name or for the same exact dataset
(`full130`) — the snapshot's closest conceptual analogues target a
*different* dataset (WISCO) with a *different* mechanism (static audit
vs. runtime guard). **Recommendation: retain all B2-only files unchanged**
in any future integration; none should be deleted or superseded outright.
See §6 for the detailed capability mapping.

## 5. Detailed semantic-conflict analysis

### 5.1 `eval/dev_set_schema.md`

- **Snapshot/master version** (68 lines): documents a 9-column
  `dev_set_v1.csv` schema (`case_id`, `language`, `respondent_text`,
  `gold_isco_code`, `gold_label_source`, `coder_or_adjudicator`,
  `major_group`, `difficulty_level`, `notes`), leakage rules checked via
  direct comparison against `test_set_smoke20.csv`/`test_set_full130.csv`
  (both opened directly), and a simple case-id/text-duplicate check.
- **B2 version** (386 lines): documents a **7-column** canonical schema
  (`case_id`, `language`, `respondent_text`, `gold_isco_code`,
  `gold_label_source`, `annotator_or_adjudication_reference`,
  `dataset_split`) — genuinely different column set/names, not just an
  extension. Adds: semantic ISCO-08 code validation against a
  classifier-supported catalogue (441 codes, extracted from
  `backend/rag/load_full_isco.py`'s `_UNIT` literal via regex, not an
  import); a **documented, unresolved discrepancy** that this 441 count
  does not match `load_full_isco.py`'s own docstring claim of 436
  (official ILO count) and flags this as requiring "a dedicated
  data-quality audit... before any conference-paper claim about complete
  or correct official ISCO-08 coverage"; a hash-only leakage-manifest
  design (`eval/test_set_full130.csv` is *never* opened by the validator —
  only `eval/configs/full130_leakage_manifest.json`'s pre-built hashes are
  compared) enforced by `eval/full130_access_guard.py`; and normalization-
  fingerprint integrity checking (a manifest is untrusted if
  `normalize_text()`'s source has changed since the manifest was built).
- **Cross-reference already present in this repo**: the 436-vs-441
  discrepancy B2 flags as "not yet independently audited" **was already
  independently audited** by Step 6's Reviewer #2 work — see
  `Documentation/Phase_2/Week_1/module_a_week1_report.md` §1 (pre-dating
  both branches' divergence, so on `master` and both descendant branches
  identically): "cross-referencing WISCO's 436 ISCO-08 unit groups...
  against our own system's 441-entry knowledge base shows our system
  carries 19 non-standard codes... and is missing 14 real ISCO-08 unit
  groups." **A future integration should point B2's "required follow-up"
  note directly at this existing report rather than re-commissioning the
  audit** — this is a genuine, actionable finding, not a conflict.
- **Governance/leakage overlap with Reviewer #2 machinery**: none direct.
  The snapshot's `eval/dataset_card_schema.py`/`eval/validate_real_lfs_governance.py`
  govern *real-LFS-respondent* intake; B2's schema governs a *K-selection
  dev set* sourced from (per its own schema) human/authoritative-coded
  cases, never respondent free text at survey scale. Different data
  categories, no direct schema clash — but both independently reinvent a
  "hash-only comparison so raw text never needs re-exposure" pattern
  (B2: `full130_leakage_manifest.json`; snapshot:
  `eval/split_manifest_schema.py` + `eval/audit_wisco_benchmark_leakage.py`'s
  dataset/record hashing). Worth unifying under one pattern in a future
  pass, not urgent.

### 5.2 `eval/validate_dev_set.py`

- **Snapshot/master**: `validate_dev_set(dev_rows, other_case_ids,
  other_normalized_texts, other_major_groups=None)` — `other_normalized_texts`
  is a **required positional** parameter carrying raw normalized text from
  both `smoke20` and `full130` (both opened directly by callers).
- **B2**: `validate_dev_set(dev_rows, other_case_ids,
  other_normalized_texts=None, other_text_hashes=None,
  other_major_groups=None, valid_isco_codes=None)` — `other_normalized_texts`
  becomes **optional**, a **new parameter `other_text_hashes` is inserted
  before `other_major_groups`**, and a new `valid_isco_codes` parameter is
  appended.
- **Input/output contract change**: this is a breaking positional-parameter
  reshape. Any code calling `validate_dev_set(rows, ids, texts, groups)`
  positionally (4 positional args) would, after a naive merge favouring
  B2's signature, silently pass `groups` into the new `other_text_hashes`
  slot instead of `other_major_groups`. **Checked: nothing in the current
  Reviewer #2 snapshot calls `validate_dev_set()` at all** (grep-verified,
  zero references outside `eval/validate_dev_set.py`/`eval/test_validate_dev_set.py`
  themselves), so this is a **latent** risk, not an active break — but it
  means any *future* snapshot-side code must be written against whichever
  signature ends up authoritative, not assumed compatible with today's.
- **Error behaviour**: B2 adds fail-closed behaviour absent from
  snapshot/master — an empty/unparsable ISCO catalogue is FATAL at the CLI
  level (`main()` exits 1), consistent in spirit with the Reviewer #2
  snapshot's own fail-closed conventions (`eval/validate_real_lfs_governance.py`,
  `eval/validate_controlled_benchmark.py`) even though built independently.
- **Overlap with Reviewer #2 governance validation**: `eval/
  validate_evaluation_discipline.py` (Step 4) and B2's `validate_dev_set()`
  both enforce "dev and held-out must not overlap" as a hard rule, using
  different mechanisms (B2: case-id/hash comparison against a specific
  manifest file; snapshot: `split_id` distinctness within a generic
  `SplitManifest` schema applicable to any benchmark). No direct code
  overlap; philosophically aligned, mechanically incompatible without
  adaptation work.

### 5.3 `eval/dev_sweep.py`

- **Parameter-selection discipline**: both versions share the same
  founding constraint (documented identically in the top docstring, byte
  for byte, on both branches) — B2 "varies ONLY K", `branch_collapse`
  hardcoded `False`, never touches `full130`.
- **Held-out protection**: master/snapshot's version never reads `full130`
  either (confirmed via the module docstring), but B2 adds the *enforced*
  version via `full130_access_guard.py` wrapping the whole checklist —
  master/snapshot relies on the docstring promise alone, B2 makes it a
  checked runtime property.
- **Generated artifacts**: `KSweepResult`, `SelectionDecision`,
  `compute_metrics()`, `check_eligibility()`, `select_k()`,
  `render_markdown_report()` are **byte-identical** between the two
  versions (diffed directly, zero differences) — the core sweep/selection
  logic is unchanged.
- **Breaking CLI-contract change**: B2's `main()` replaces direct
  `--reranker-model`/`--beam`/`--stage1-mode`/`--disable-keyword-map` flags
  with a **mandatory `--baseline-config`** JSON file (`eval/configs/
  b1_frozen.json`) from which those values are now read and asserted
  against, plus new `--seed`, `--confirm-inferred-beam`, and
  `--allow-dirty-tree` flags, and a large (~546-line) prepended block of
  new functions: baseline-config validation, composite implementation
  fingerprinting, Ollama model/version resolution, git-tree-clean
  enforcement, deterministic seeded shuffling, environment-provenance
  resolution. None of this exists on master/snapshot's version at all.
- **Compatibility with current manifests/ablation configuration**: B2's
  `dev_sweep.py` is entirely orthogonal to the Reviewer #2 snapshot's
  `eval/manifest.py`/`eval/ablation_runner.py` — different CLI, different
  purpose (K-selection vs. named-config ablation), no shared function
  calls in either direction (grep-verified). They could run side by side
  today with zero interference; the only reason this file is high-risk is
  the CLI-contract break for anyone who has memorized the master-era
  invocation.

### 5.4 `eval/test_run_eval_b2.py`

- Both branches append **exactly 122 lines** at the **exact same anchor
  point** (immediately after `test_retry_count_always_zero_for_standard_reranker`,
  line 193 in both diffs), with **zero deletions** on either side, but
  **entirely different content**: the snapshot adds `sre_enabled`-related
  tests (`_kwargs_with_sre_inputs`, `test_sre_enabled_default_true_runs_isic_isced_and_sre`,
  etc. — later superseded/extended further by
  `eval/test_sre_isic_isced_coupling_fix.py`, a Step 5.1 file not in this
  overlap); B2 adds an "exact classifier-input regression test" proving
  `respondent_text`/`input_text` reaches `ISCOClassifier.classify()`
  completely unmodified (no normalization/truncation/rewriting).
- **Empirically confirmed conflict**: `git merge-tree --write-tree HEAD
  origin/conference1-b2-evaluation` reports `CONFLICT (content): Merge
  conflict in eval/test_run_eval_b2.py` — this is not a prediction, it is
  git's own merge algorithm's output.
- **Test assumptions that must remain true after integration**: both
  additions are pure new test functions with no shared fixtures modified
  and no assertions about each other's subject matter — the actual
  resolution is almost certainly "keep both blocks, in either order" (a
  manual concatenation, not a redesign). The base 193 lines both branches
  share must remain unchanged by either side (confirmed: neither diff
  touches lines 1-193).

### 5.5 `requirements.txt`

- **Dependency changes**: snapshot adds `pyyaml` (used by
  `eval/coverage_audit.py`/`eval/catalogue_importer.py`, confirmed via
  live import check this session). B2 adds `scikit-learn==1.6.1` (used
  somewhere in B2's extended `dev_sweep.py`/`validate_dev_set.py` —
  not exhaustively traced in this task, out of scope per "do not edit...
  dependencies", but confirmed via prior-session grep that **nothing in
  the current snapshot working tree imports `sklearn`**).
- **Version conflicts**: none — the two additions are different packages
  at different insertion points; `git merge-tree` confirms a clean
  auto-merge.
- **Security/reproducibility implications**: none identified; both are
  well-established, actively-maintained packages with no known
  overlapping-version pin conflict (snapshot pins nothing new by version;
  B2 pins `scikit-learn==1.6.1` exactly).
- **`requirements-dev.txt` separation**: B2 introduces this as a
  B2-test-only dependency lock, explicitly scoped ("Locked development
  dependencies for collecting and running the B2 non-inference test suite
  ONLY — not the full application stack"). The Reviewer #2 snapshot has no
  equivalent split (all its new dependencies, just `pyyaml`, went into the
  single `requirements.txt`). **Recommendation: retain B2's
  `requirements-dev.txt` split unchanged** in any future integration —
  it's a reasonable pattern the snapshot's single new dependency didn't
  need to establish, not a competing design.

### 5.6 B2 `pre_run_check.py` vs. snapshot governance/discipline validators

`eval/pre_run_check.py` is a standalone pre-flight checklist for B2's
K-sweep: asserts baseline-config integrity, beam-provenance evidence,
Ollama model identity, composite implementation fingerprint, git-tree
cleanliness, **and** dev-set leakage safety — all before `eval/dev_sweep.py`
is invoked for real. It makes zero classification/retrieval/LLM-inference
calls (only metadata-only Ollama/Qdrant GET requests, confirmed from its
own docstring). This is conceptually parallel to, but does not literally
overlap with, three Reviewer #2 snapshot modules:

- `eval/validate_evaluation_discipline.py` (Step 4) — checks split-manifest
  distinctness, dataset-hash presence, "measured" status requiring
  non-null metrics. Different subject (any benchmark's split/manifest
  discipline vs. B2's specific K-sweep baseline discipline).
- `eval/generate_dry_run_readiness_report.py` (Step 4) — a "confirm
  everything is ready before spending compute" script, same *spirit* as
  `pre_run_check.py`, different scope (dry-run validation of the named
  5-config ablation pipeline vs. B2's K-sweep baseline-fingerprint gate).
- `eval/validate_controlled_benchmark.py` (Step 6) — rejects
  self-generated/non-independent labels, absent hashes, split overlap;
  `pre_run_check.py`'s leakage check is a narrower, full130-specific
  instance of the same idea (manifest-hash comparison instead of a general
  schema check).

No function names collide; no import-time interaction is possible today
(the snapshot never imports `pre_run_check`). **These three snapshot
modules and `pre_run_check.py` are complementary "is this run trustworthy
before it starts" tools for different pipelines — a deliberate future
integration might route B2's K-sweep discipline through the same
`ExperimentRunManifest`/`evaluation_status` vocabulary the snapshot
established, but that is a design decision, not a conflict to resolve.**

### 5.7 B2 `full130_access_guard.py` vs. snapshot benchmark-governance/leakage machinery

`full130_access_guard.py` is a **runtime enforcement** mechanism: a context
manager that patches five independent Python file-reading entry points
(`builtins.open`, `io.open`, `pathlib.Path.open`/`.read_text()`/
`.read_bytes()`) to raise `Full130AccessBlocked` immediately if anything
tries to open a path containing `test_set_full130`. This has **no
counterpart in the Reviewer #2 snapshot**. The closest analogues are:

- `eval/validate_real_lfs_governance.py::_path_points_inside_repo()` — a
  **static** safeguard: inspects `DatasetCard` string fields for
  path-shaped tokens that resolve inside the repo. It never touches actual
  file-open calls and cannot catch a script that reads a forbidden file
  through a code path the safeguard doesn't inspect.
- `eval/audit_wisco_benchmark_leakage.py` — a **post-hoc audit**: reads
  already-built benchmark records and reports whether dev/heldout splits
  overlap. It runs *after* data has already been assembled, not as a
  preventive gate during assembly.
- `eval/split_manifest_schema.py`'s `leakage_check_performed`/
  `leakage_check_method` fields — a **self-reported attestation**, not an
  enforced guarantee.

**This is a genuine capability gap, not a conflict**: the Reviewer #2
snapshot's WISCO/real-LFS pipeline has no equivalent to B2's "physically
cannot open the forbidden file" guarantee. A future integration should
consider generalizing `full130_access_guard.py` into a reusable guard
(parameterized on which path substring is forbidden) usable by both B2's
`full130` protection and a future real-LFS-respondent-data protection path
— this is additive, low-risk, high-value, and does not require reconciling
any conflicting logic, only extraction/generalization of B2's existing,
well-tested implementation.

## 6. B2-to-Reviewer-#2 capability mapping

| B2 capability | Reviewer #2 snapshot equivalent | Relationship |
|---|---|---|
| `eval/pre_run_check.py` (pre-flight gate for K-sweep) | `eval/validate_evaluation_discipline.py`, `eval/generate_dry_run_readiness_report.py` | Complementary, different scope — retain both |
| `eval/full130_access_guard.py` (runtime file-open guard) | *(none — static/audit-based only)* | Capability gap on snapshot side; retain B2's, consider generalizing |
| `eval/build_full130_leakage_manifest.py` (hash-only manifest builder) | `eval/build_wisco_isco_benchmark.py`/`_v2_group_split.py` (hash-recording benchmark builders) | Same pattern, different dataset — retain both |
| `eval/dev_set_v1_provenance.md`/`_data_dictionary.md`/`_readiness_report.md` (narrative docs) | `eval/dataset_card_schema.py` + `*_DATASET_CARD_TEMPLATE.md` (structured, machine-validated cards) | Snapshot's is strictly more rigorous (machine-checkable); B2's docs could eventually be upgraded to a `DatasetCard`/`BenchmarkRecord`-style structured record, but that is new work, not a conflict |
| `eval/configs/b1_frozen.json` (frozen B1 baseline + real 54/130 accuracy claim) | *(no equivalent — Reviewer #2 has not measured anything on `full130`)* | Must remain frozen — see §7 |
| `requirements-dev.txt` (B2-test-only dependency split) | *(none — single `requirements.txt`)* | Retain B2's pattern unchanged |
| `eval/dev_set_schema.md`'s ISCO-catalogue validation (441 vs. 436 discrepancy) | `Documentation/Phase_2/Week_1/module_a_week1_report.md` (already audited this exact discrepancy via WISCO) | **Already resolved by existing evidence** — point B2's "required follow-up" at this report |

## 7. Recommended future integration plan

**1. Recommended future integration base branch**: `reviewer2-wip-snapshot-20260807`
(or its eventual successor once Step 7B lands) should be the integration
target, with `conference1-b2-evaluation`'s B2-only work merged *into* it —
not the reverse — because the snapshot's governance/manifest/benchmark
schema (Steps 3-6) is the more general, more recently validated framework,
and B2's work is narrowly scoped to one dataset (`full130`) that Step 6
already found ineligible as manuscript evidence on its own.

**2. Recommended order for resolving each file group**:
1. `requirements.txt` / `requirements-dev.txt` first (trivial, no conflict).
2. `eval/test_run_eval_b2.py` next (the one real textual conflict) —
   manually concatenate both test blocks; confirm the shared 193-line base
   is untouched by either side before merging.
3. B2-only new files (`pre_run_check.py`, `full130_access_guard.py`,
   `build_full130_leakage_manifest.py`, `configs/*.json`,
   `dev_set_v1*.md`) — pure additions, no conflict, bring in as-is.
4. `eval/dev_set_schema.md`, `eval/validate_dev_set.py`, `eval/dev_sweep.py`
   last, and only after a human (not an automated merge) has confirmed
   which of the two competing schemas (9-column master/snapshot vs.
   7-column B2) is authoritative going forward — this is a design decision
   the git merge itself cannot make safely, since it will silently take
   whichever side's version by default with **no conflict marker at all**.

**3. Files that must be manually reconciled**: `eval/test_run_eval_b2.py`
(confirmed git conflict) and, as a *design* reconciliation rather than a
textual one, `eval/dev_set_schema.md` + `eval/validate_dev_set.py` +
`eval/dev_sweep.py` together as one unit (the schema and its two consuming
scripts must agree with each other, so partial adoption of just one file
risks internal inconsistency).

**4. Files that can probably be retained unchanged from each branch**:
- From B2, unchanged: `eval/full130_access_guard.py`,
  `eval/build_full130_leakage_manifest.py`,
  `eval/configs/full130_leakage_manifest.json`,
  `eval/configs/b1_frozen.json`, `eval/PRE_RUN_B2_CHECKLIST.md`,
  `eval/dev_set_v1_provenance.md`, `eval/dev_set_v1_data_dictionary.md`,
  `eval/dev_set_v1_readiness_report.md`, `requirements-dev.txt`.
- From the snapshot, unchanged: everything with no B2 equivalent at all
  (the large majority of the 113-115 file set — `eval/manifest.py`,
  `eval/dataset_card_schema.py`, `eval/validate_real_lfs_governance.py`,
  `eval/validate_controlled_benchmark.py`, `eval/ablation_runner.py`,
  `eval/analyze.py`, all WISCO-related modules, all `Documentation/
  Conference_I_Reviewer_2/` docs).

**5. Tests that must run after a future integration**:
```
pytest backend/tests eval/ -q
pytest eval/test_run_eval_b2.py -q            # after manual reconciliation
pytest eval/test_validate_dev_set.py eval/test_dev_sweep.py \
       eval/test_pre_run_check.py eval/test_full130_access_guard.py -q
pytest eval/test_validate_real_lfs_governance.py eval/test_validate_evaluation_discipline.py \
       eval/test_validate_controlled_benchmark.py eval/test_wisco_leakage_audit.py -q
pytest eval/test_docs_consistency.py -q
```
Current baseline (this snapshot, unmodified): 1656 passed, 1 known
pre-existing failure, 1 deselected, 1 warning
(`test_isco_classifier_extended.py::test_llm_used_for_low_similarity`,
documented, unrelated). Any post-integration run must be compared against
this exact baseline, not assumed clean.

**6. Rollback strategy**: perform the integration on a **new** branch cut
from `reviewer2-wip-snapshot-20260807` (never on the snapshot branch
itself, which must remain a frozen, protected reference point per task 01)
— e.g. `reviewer2-b2-integration-<date>`. If the integration proves
unworkable, the new branch can be deleted with zero impact on either
source branch; both `reviewer2-wip-snapshot-20260807` and
`conference1-b2-evaluation` remain untouched and independently checkable
at any time via their recorded SHAs in this report and
`CLAUDE_PROJECT_AUDIT.md`.

**7. Claims, benchmarks, and source files that must remain frozen during integration**:
- **`eval/configs/b1_frozen.json`'s 54/130 (41.5%) top-1 accuracy claim**
  on `eval/test_set_full130.csv`, verified 2026-08-05, sourced from
  `eval/results/raw_runs/20260805T055741Z_full130_leafvote_beam3_llama3b_pooled.csv`
  — a real, already-measured, already-committed-to-history number. It must
  not be silently recomputed, overwritten, or reinterpreted during
  integration.
- **Simultaneously**, Step 6's `CONTROLLED_BENCHMARK_AUDIT.md` finding that
  `eval/test_set_full130.csv` itself is `not_eligible_unknown_provenance`
  (no documented label source, no coder identity, no double-coding/
  adjudication evidence) must also remain frozen/unaltered. **These two
  facts coexist and must both be preserved and cross-referenced** — the
  54/130 figure is a real measurement of *this system's* behaviour on
  *this specific file*, but that file is not manuscript-defensible
  benchmark evidence on its own. Neither fact should be used to
  retroactively suppress or reinterpret the other.
- WISCO's Zenodo provenance record (`Documentation/Phase_2/Week_1/PROVENANCE.md`,
  DOI `10.5281/zenodo.8262593`) and the 436-vs-441 ISCO coverage
  discrepancy finding (`module_a_week1_report.md`) — both already
  referenced by B2's own (independently-written) "required follow-up"
  note, confirming they are shared, uncontested ground truth.
- The WISCO v1/v2 leakage-audit finding and both benchmark packages'
  dataset hashes (`aad7f99e...` for v1, `a3b3c1a3...` for v2) — per Step
  7A's own rule, v1 must remain unaltered as the audit record regardless
  of any future integration work.

## 8. No-merge conclusion and evidence boundaries

**No merge, rebase, or conflict resolution was performed in this task.**
The only git-state-changing actions taken were: writing this report file
to the working tree, and (per the task's completion rules, executed
immediately after this report was finalized) staging, committing, and
pushing that single file to `reviewer2-wip-snapshot-20260807`. `master`,
`conference1-b2-evaluation`, and `reviewer2-enhancement` were fetched
(read-only) but never checked out, merged, or modified. `git merge-tree`
was used twice, strictly as a read-only simulation tool, to convert
predicted conflicts into empirically verified ones — both invocations were
immediately followed by a clean `git status --short` confirming no
persistent state change.

Every claim above about file content, function signatures, and diff
behaviour was verified directly against the actual blob content on both
branches (`git show <ref>:<path>`) or actual diff/merge-tree output in this
session — none is inferred from commit messages or file names alone.
