OFFICIAL_TIER1_ANALYSIS_CLEAN_REPRODUCTION_COMPLETED: yes

# Task 37.1 Final Report — Clean Offline Reproduction of Official Tier-1 Analysis

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_37_1_CLEAN_OFFLINE_ANALYSIS_REPRODUCTION.md`.
This task independently reproduces Task 37's official Tier-1 analysis
in a strictly offline record, with no Qdrant connection or other live
operation performed anywhere in the reproduction itself.

## 1. Status, branch, SHA, push, working tree

```text
OFFICIAL_TIER1_ANALYSIS_CLEAN_REPRODUCTION_COMPLETED: yes
```

| | |
|---|---|
| Base branch | `reviewer2-wisco-official-tier1-precise-deadline-analysis-20260810` |
| Required/verified base SHA | `85fcda8e7907b5dbd56780766d4642be9578e872` (confirmed against both the local branch and `origin` before branching, re-confirmed identical throughout) |
| New branch | `reviewer2-wisco-official-tier1-analysis-clean-reproduction-20260810` |
| Final commit SHA | reported in Claude's end-of-turn response, not inside this file |

Working tree was clean before branching, and `git status --porcelain`
was empty immediately before this report's own commit — every derived
artifact lives under the Git-ignored
`eval/local_runs/wisco_official_tier1_precise_deadline_full_20260809T190821Z/analysis_task37_1_clean_reproduction/`
directory. No source, test, dataset, catalogue, or manuscript file was
touched (Section 10).

## 2. Task 37's disclosed scope breach: permanent, not repeated

Task 37's final report permanently records one read-only Qdrant
`get_collection()` point-count call made outside its own declared
scope, during an operator-side preservation sanity-check (not by the
analyzer script itself). That disclosure is **not revised, concealed,
or reinterpreted** by this task. Task 37.1 does not repeat it, excuse
it, or rely on it for any result here: every value in this report was
produced with zero Qdrant connection anywhere in this task's own
execution (Section 4).

## 3. Exact commands proving no live operations

**Direct static scan** of `eval/analyze_official_tier1.py`'s own
source for prohibited live-operation tokens:
```bash
grep -niE "qdrant|requests|httpx|urllib|socket|sentence_transformers|ollama|openai|anthropic|crewai|torch|tensorflow" eval/analyze_official_tier1.py
```
Result: the only two matches are in a code comment explaining why
keyword-anchored hierarchical rows have no stage-1 Qdrant query
telemetry (documentation of an absence, not a reference in executable
code). Zero prohibited imports.

**Direct import list** of `eval/analyze_official_tier1.py` and its
three directly-referenced modules (`eval/analyze.py`,
`eval/analyze_wisco_tier1.py`, `backend/rag/official_isco08_catalogue.py`):
```bash
grep -n "^import\|^from" eval/analyze_official_tier1.py eval/analyze.py eval/analyze_wisco_tier1.py backend/rag/official_isco08_catalogue.py
```
Result: standard library (`csv`, `json`, `math`, `statistics`, `hashlib`,
`re`, `argparse`, `sys`, `pathlib`, `dataclasses`, `datetime`, `typing`)
plus `yaml`, plus the three intra-project imports named above. Zero
Qdrant, `requests`, `httpx`, `urllib`, `socket`, model, or LLM/reranker
library referenced directly by any of these four files.

**Additional dynamic scan** (a more rigorous check than the task
strictly required, run for completeness): importing
`analyze_official_tier1` in a fresh Python process was checked against
`sys.modules` before/after. This revealed that `qdrant_client`,
`sentence_transformers`, `torch`, and `tensorflow` DO end up loaded —
traced to `backend/rag/__init__.py`'s own top-level
`from .vector_store import ...` (a pre-existing package-init side
effect, unrelated to and unmodified by Task 37/37.1, which pulls in
`backend/rag/vector_store.py`'s own `QdrantClient`/`SentenceTransformer`
imports the moment ANYTHING under `backend.rag.*` — including the
unrelated, file-only `official_isco08_catalogue.py` submodule this
analyzer needs — is imported). Direct proof this is transitive
package-init noise, not something `official_isco08_catalogue.py` or
`analyze_official_tier1.py` itself references or calls:
```bash
grep -n "QdrantClient\|SentenceTransformer\|\.search(\|\.query_points(\|\.get_collection(" backend/rag/official_isco08_catalogue.py
```
Result: zero matches — confirmed by full source read in Task 37 that
`official_isco08_catalogue.py`'s functions (`load_official_catalogue`,
`load_verified_metadata`, `records_by_level`) are 100% local file I/O
(`Path.read_bytes()`, `csv.DictReader`, `yaml.safe_load()`), never
instantiate or call `QdrantClient`/`SentenceTransformer` anywhere.
**Conclusion**: running `eval/analyze_official_tier1.py` is
behaviorally incapable of a live Qdrant/model/network operation —
proven both by its own and its direct dependencies' source containing
no such call, and by this task's own execution (Section 4) completing
using only local file reads. The transitive module-loading fact is
disclosed here in full because it is real and inspectable, exactly
like Task 37's own Qdrant disclosure — it is not something Task 37.1
has any authority to fix, since modifying `backend/rag/__init__.py`
would be a source-code change this task explicitly prohibits.

**Confirmation no live operation was performed by this task**: the
eligibility gate re-run and clean-reproduction run (Section 4) and the
independent computation (Section 5) each completed using only local
CSV/YAML file reads; no network socket was opened, no Qdrant server
was contacted, and no model was loaded anywhere in this task's actual
execution.

## 4. Input and preservation hashes — before and after

**Task 36 raw-output identities**, re-verified before any computation:

| Input | SHA-256 | Match |
|---|---|---|
| Flat CSV | `d0692b4a87db11945dbd046ead79a32dcc36d715fbaa66fcc4e4853102be5f02` | yes |
| Hierarchical CSV | `b72193c8411b076df827abf2fa8bc6c2c72ac60f86799e435b94ff5eb237a8a4` | yes |
| Heldout export | `41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c` | yes |

**134-file preservation snapshot** — every Task 24–36 raw-output-root
file, the official catalogue, B1 frozen config, `full130` leakage
guard/manifest, WISCO dataset/records/split-manifest, Task 37's own
derived analysis artifacts (`analysis_task37/`), Task 37's source/test
changes (`eval/analyze.py`, `eval/analyze_official_tier1.py`,
`eval/test_analyze.py`, `eval/test_analyze_official_tier1.py`), and
every prior task's final report through Task 37 — hashed before
branching and re-hashed at task completion: **zero mismatches, zero
missing files, all 134/134 byte-identical.**

Task 36's final report specifically re-confirmed byte-identical via
eligibility condition 10 (`git diff --quiet`) — see Section 5.

## 5. Eligibility gate — all ten conditions, re-run against the required inputs

```bash
python eval/analyze_official_tier1.py \
  --flat-csv .../flat/20260809T191133Z_..._flat.csv \
  --hierarchical-csv .../hierarchical/20260809T192256Z_..._hierarchical.csv \
  --heldout-csv .../heldout_export_fresh.csv \
  --catalogue-csv eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv \
  --task36-report Documentation/AI_HANDOFF/CLAUDE_TASK_36_FINAL_REPORT.md \
  --task36-base-sha 514bc31f78cd5d66b018714c9f99ba591df0fdfe \
  --expected-flat-sha256 d0692b4a87db11945dbd046ead79a32dcc36d715fbaa66fcc4e4853102be5f02 \
  --expected-hierarchical-sha256 b72193c8411b076df827abf2fa8bc6c2c72ac60f86799e435b94ff5eb237a8a4 \
  --expected-heldout-sha256 41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c \
  --expected-n 18747 --max-stage-latency-ms 30000 \
  --out .../analysis_task37_1_clean_reproduction
```

Exact CLI as used in Task 37 — no flag, source, or configuration
changed.

| # | Check | Result |
|---|---|---|
| 1 | Input SHA-256 match (flat, hierarchical, heldout) | Pass |
| 2 | Row counts (18,747 each), unique case IDs, case-ID set == heldout | Pass |
| 3 | Flat `pred_method` exactly `flat_isco08_official_ilo2021_v1`, all rows | Pass |
| 4 | Hierarchical `pred_method` exactly `hierarchical_isco08_official_ilo2021_v1`, all rows | Pass |
| 5 | Every prediction a valid 4-digit code in the verified official catalogue | Pass |
| 6 | Zero non-blank row-level error, both files | Pass |
| 7 | Complete hierarchical stage 1–4 evidence, parseable telemetry, zero exception, zero budget exhaustion, zero over-cap latency | Pass |
| 8 | Reranking off, zero tokens/cost, blank ISIC/ISCED, `sre_status=not_applicable` | Pass |
| 9 | Heldout gold codes valid + in catalogue; gold ISIC/ISCED blank | Pass |
| 10 | Task 36 report unchanged since base commit; prior evidence unchanged | Pass |

`All eligibility gates passed. Wrote official Tier-1 analysis bundle
to .../analysis_task37_1_clean_reproduction` — exit 0, no traceback.

## 6. Derived-artifact hashes

| File | SHA-256 |
|---|---|
| `.../analysis_task37_1_clean_reproduction/analysis_manifest.json` | `c861f11b726a294cf0ddeefb9fd667f05ed11d7be871e022a328954cae6cdc0f` |
| `.../analysis_task37_1_clean_reproduction/official_tier1_analysis.json` | `368a96ff98ffcf14a1301f79d54aa4e5258f83b5c3f785525a6b497e9125a7b2` |
| `.../analysis_task37_1_clean_reproduction/official_tier1_analysis.md` | `20aa99ec2e4b572d86f1bf49c28a9fe9ef83eaffe0412d282b56b128250060d1` |

(Distinct from Task 37's own `analysis_task37/` artifact hashes —
timestamps differ inside the JSON/manifest — but every metric value
inside is identical; see Section 8.)

## 7. Independent computation — method, values, tolerance, comparison

A standalone Python script (session scratchpad only, never committed,
matching the task's "no source-code modification" rule) read the
three raw CSVs and the official catalogue CSV **directly via
`csv.DictReader`** — it imports nothing from
`eval/analyze.py`/`eval/analyze_wisco_tier1.py`/`eval/analyze_official_tier1.py`.
Wilson intervals used a from-scratch closed-form implementation; the
McNemar p-value used a from-scratch log-space binomial-tail sum
(always log-space, not merely re-invoking `eval/analyze.py`'s
overflow-fallback branch — a genuinely separate numerical path).

| Metric | Independent computation | Task 37 recorded | Task 37.1 JSON | Match |
|---|---:|---:|---:|---|
| Catalogue unit-code count | 436 | 436 | 436 | exact |
| Bad flat predicted codes | 0 | 0 | 0 | exact |
| Bad hierarchical predicted codes | 0 | 0 | 0 | exact |
| Flat correct | 3,973 / 18,747 | 3,973 / 18,747 | 3,973 / 18,747 | exact |
| Flat accuracy | 0.21192724169200405 | 0.2119272... | 0.21192724169200405 | exact |
| Hierarchical correct | 1,941 / 18,747 | 1,941 / 18,747 | 1,941 / 18,747 | exact |
| Hierarchical accuracy | 0.10353656585053608 | 0.1035366... | 0.10353656585053608 | exact |
| Both correct | 1,341 | 1,341 | 1,341 | exact |
| Flat-only correct | 2,632 | 2,632 | 2,632 | exact |
| Hierarchical-only correct | 600 | 600 | 600 | exact |
| Both incorrect | 14,174 | 14,174 | 14,174 | exact |
| Diff (hierarchical − flat), pp | −10.839067584146797 | −10.8391 (displayed) | −10.839067584146797 | exact (matches to displayed rounding) |
| McNemar statistic (min(b,c)) | 600.0 | 600.0 | 600.0 | exact |
| McNemar p-value (two-sided exact) | 1.8573559951149046e-301 | 1.8573559951149046e-301 | 1.8573559951149046e-301 | exact |

**Tolerance**: every integer count matched exactly (0 tolerance
needed — these are exact case counts, not floating computations).
Every floating value (accuracy, Wilson bounds, McNemar p-value)
matched the analyzer's own output to full double-precision (compared
via Python `==` on the raw floats, not merely `pytest.approx`) — this
is expected, not merely "close enough," because both the independent
script and the analyzer implement the identical closed-form Wilson
formula and the identical (for this n, non-overflowing) log-space
McNemar computation over the identical input data; any residual
float-ordering difference would be bounded by IEEE-754 double
precision (~1e-15 relative), and none was observed. **All required
expected-result targets from the task file matched exactly — nothing
was adjusted to fit them.**

## 8. Full subgroup and operational tables — presence and identity confirmed

`official_tier1_analysis.json` from Task 37 and from this task's
`analysis_task37_1_clean_reproduction/` were compared key-by-key in
Python (`==` on the parsed JSON structures):

- `headline.flat`, `headline.hierarchical`: identical
- `paired`: identical
- `by_language` (`ar`, `en`, `hi`, `tl`, `ur` — all 5 present, each with `flat`/`hierarchical` n/correct/accuracy/Wilson CI and paired discordant counts): identical
- `by_major_group` (`0`–`9` — all 10 present, same structure): identical
- `operational` (flat query duration mean/median/p95/p99/max, all four hierarchical stage-latency distributions, total stage-query count, flat outcome distribution, retry/exception/budget-exhaustion counts, `stage1_source` retrieval-path distribution): identical

## 9. Test command and output

```bash
python -m pytest backend/tests eval/ -q
```
Result: `2185 passed, 1 deselected, 1 warning in 322.10s` — matches
the required `2185 passed, 1 deselected, 1 warning` exactly. No code
or test file was modified, so this result was expected to be
byte-for-byte the same as Task 37's own closing run, and it was.

## 10. Confirmation of no code/test/data/manuscript change

`git status --porcelain` was empty both before branching and
immediately before this report's own commit. No file under
`backend/`, `eval/` (other than the new Git-ignored
`analysis_task37_1_clean_reproduction/` output directory), any
dataset/catalogue path, `Documentation/` (other than this new report),
any figure, README, or reviewer-response file was created, modified,
or deleted by this task.

## 11. Strict limitations

- This is a controlled multilingual WISCO ISCO-08 benchmark, not real
  Labour Force Survey validation.
- It supports no ISIC, ISCED, SRE, cost, coverage, generalization, or
  production-performance claim.
- The observed latency figures remain local-run descriptive evidence
  only, not a production SLA.
- B1 remains stale/quarantined; nothing in this task touches or
  revalidates it.
- No manuscript, figure, README, or reviewer-response work was begun
  in this task, per its own explicit stop instruction.
