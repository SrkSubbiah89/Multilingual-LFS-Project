# Module I — Computational Efficiency & Scalability (Prompt 6 of 9)

**Status:** Real measurements obtained for every metric that could be
measured on the available hardware. One category (real Claude
3.5 Sonnet reranking cost/latency) remains genuinely unmeasured — not
because it wasn't attempted, but because a prior attempt (Task 43) was
found to be a silent-failure artifact and is explicitly not usable; see
Phase 0.

All raw evidence: `backend/evaluation/embedding_timing_benchmark.json`,
`backend/evaluation/qdrant_collection_memory_audit.json`,
`backend/evaluation/load_test_results.json` (last load-test run only —
per-run summaries below are transcribed directly from each run's console
output, captured at measurement time).

---

## Phase 0 — What Module A's existing runs already answer

Module A's canonical run (Task 36: 18,747-case WISCO heldout, flat +
strict-hierarchical, official ILO 2021 profile — the same run
`OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md` reports 21.19%/10.35% from)
**does** contain real, logged per-call latency data. Verified by
re-hashing the raw CSVs directly: SHA-256 of both files matches the values
already documented in that canonical results doc, byte-for-byte, so this
is confirmed to be the same, un-modified data.

| Metric | Flat | Hierarchical (4-stage) |
|---|---:|---:|
| End-to-end latency, mean | 31.1 ms | 133.9 ms |
| End-to-end latency, median | 30.1 ms | 137.4 ms |
| End-to-end latency, P95 | 42.7 ms | 182.6 ms |
| End-to-end latency, P99 | 55.9 ms | 217.0 ms |
| n | 18,747 | 18,747 |

Hierarchical stage breakdown (real, mean): stage1 8.6ms, stage2 15.3ms,
stage3 31.3ms, stage4 54.3ms (sums to ~109ms of the 133.9ms end-to-end
mean; remainder is call/serialization overhead).

**This directly replaces Week 2's "~150ms/query, not yet measured"
estimate** — real measured hierarchical RAG latency (134ms mean) is close
to, and confirms the order of magnitude of, that estimate.

**What this run does NOT answer:** `reranker_fired` is `False` for all
18,747×2 rows. This is not an accident or a missing measurement — 99.98%
of flat-run cases scored below the 0.92 LLM-rerank trigger threshold, yet
none reranked, confirming the controlled Tier-1 benchmark was run with
the LLM reranking tier deliberately excluded (a separate, dedicated
~500-case stratified reranking subset exists in
`eval/select_wisco_reranking_subset.py` for that purpose instead). So
Week 2's "LLM re-ranking trigger rate" and "Claude cost/latency per call"
rows are **not** answered by Task 36.

### A second attempt exists — and was found to be invalid, not used

`eval/local_runs/task43_flat_heldout_live_reranker_20260810T180139Z/` and
the matching hierarchical directory contain what looks like a complete,
real 18,747-case Claude-3.5-Sonnet-reranked run (real elapsed times:
7,691.6s flat / 9,889.6s hierarchical; real-looking accuracy figures).
This was **not used**: cross-checked directly against
`Documentation/AI_HANDOFF/CLAUDE_TASK_43_LIVE_RERANKER_WISCO_RUN_FINAL_REPORT.md`,
which documents that every one of these runs silently fell back to
plain semantic-top-1 for effectively 100% of cases because the Anthropic
account had zero credit for the entire session — caught because the
flat-heldout run's predictions matched the non-reranked baseline for
exactly 18,747/18,747 cases (not statistically plausible for a working
reranker). **No real Claude-reranked number exists anywhere in this
repository.** This confirms — independently, on this pass — the same
conclusion `CLAUDE.md` already states.

There is one genuine (but explicitly non-Claude) data point: a real
Ollama (`llama3.2:latest`) reranking run against the 2,013-case dev split
completed cleanly (17.19%, 346/2,013, confirmed genuine via 344 distinct
predicted codes across the run) — with disclosed real per-call latency of
**~12–19 sec/call** (smoke test) rising to **~24.6 sec/call average**
under sustained load (11.6 hours / 1,695 rows). This is real, but is a
free 3B-parameter local model, not Claude, and must not be substituted
for a Claude cost/latency figure.

### A correction to this prompt's own claim

Prompt 6 asked to extend "the existing $84 target (mentioned in earlier
project docs)". A full-repo search (`grep -rn "\$84"` across all tracked
Markdown/Python/JSON/text files) found **no such figure anywhere in this
repository.** Per this project's standing discipline, this is disclosed
rather than silently invented or dropped — no $84 target exists to
extend; Phase 1 point 3 below builds cost estimates from scratch instead.

---

## Phase 1 — Filling in what was genuinely missing

### 1. Qdrant index size / memory — real, not estimated

`eval/qdrant_collection_memory_audit.py` (new, committed) queries Qdrant's
real collection-info + `/metrics` endpoints directly.

| Collection | Points | Vector dim | On-disk size (real, `du`) |
|---|---:|---:|---:|
| `isco08_major_groups_ilo2021_v1` | 10 | 384 | 428 KB |
| `isco08_submajor_groups_ilo2021_v1` | 43 | 384 | 588 KB |
| `isco08_minor_groups_ilo2021_v1` | 130 | 384 | 1,012 KB |
| `isco08_unit_groups_ilo2021_v1` (hierarchical) | 436 | 384 | 2.3 MB |
| `isco08_unit_groups_flat_ilo2021_v1` (flat) | 436 | 384 | 2.3 MB |

**Real total, official ILO 2021 profile (the 4 hierarchical collections
actually used for the published Tier-1 result): ~4.3 MB on disk.**

Real, measured whole-process RSS (covers all 10 ISCO collections
combined on this dev instance — Qdrant's API does not expose a
per-collection memory breakdown): **94.9 MB** (`memory_resident_bytes`,
via `/metrics`).

**Correction to this prompt's own assumption:** Prompt 6 assumed 1024-dim
vectors for the analytical-estimate fallback. The real, live vector
dimension — confirmed directly from Qdrant's own collection config — is
**384**, matching `intfloat/multilingual-e5-small`. No analytical
estimate was needed at all: real on-disk sizes were obtained directly via
`du` inside the Qdrant container, and all 10 collections are below
Qdrant's 10,000-point `indexing_threshold`, so `indexed_vectors_count=0`
for every one of them — there is no HNSW graph overhead to estimate;
these collections use flat (brute-force) search.

### 2. Embedding computation cost per turn — real, measured

`eval/embedding_timing_benchmark.py` (new, committed), 120 calls,
batch=1, exact production call pattern (`"query: " + text`,
`normalize_embeddings=True`) from `hierarchical_store.py._embed_query`.

**Explicit hardware disclosure:** this sandbox is an 8GB-RAM, 12-logical-core
CPU-only Windows machine running under significant concurrent memory
pressure from other processes (VS Code, Docker Desktop, this session
itself) — not the target deployment host. Real numbers, this hardware:

| Metric | Value |
|---|---:|
| Mean | 19.1 ms |
| Median | 18.3 ms |
| P95 | 25.6 ms |
| P99 | 31.2 ms |
| Vector dim | 384 |
| Model load time (one-time, cold start) | 6.4 s |

### 3. Cost per interview (USD) — genuinely not measurable; stays an estimate, now with a clean $0 floor established

Checked directly for real Claude API token counts from actual sessions,
per Prompt 6's "if available" instruction:

- `AgentDecisionLog`'s real schema (`backend/database/models.py`) has
  **no token-count or cost field at all** — only `confidence`,
  `reasoning`, `duration_ms`. Not just unpopulated: not capturable in
  the current schema.
- Queried the live, running `lfs_postgres` container directly:
  `SELECT count(*) FROM agent_decision_logs` → **0 rows.** No historical
  usage data exists to extract from, confirming (again, independently)
  that this dev environment has never made a billable Claude call.

**Cost per interview therefore remains a disclosed estimate**, not a
measurement — same status as Week 2's original table, now with the
$84 figure explicitly identified as not traceable to any document in
this repo (see Phase 0). No new cost-per-interview point figure is
asserted here; Week 2's own low/mid/high scenario table (§4 below)
remains the citable estimate, corrected only where Phase 0/1/2 provide
real substitutions.

---

## Phase 2 — Real load test results

### Environment disclosure (read before the numbers below)

This sandbox has 8.28 GB total RAM, and free RAM fluctuated between
**0.23 GB and 1.07 GB** throughout this task, competing with several
other processes on the same host. `docker-compose.yml` sets no memory or
CPU limits on any service. The backend was started as a single Uvicorn
process (no `--reload`, no worker pool) — this is a single-instance,
resource-constrained measurement, explicitly not representative of a
properly provisioned deployment host. All numbers below are real
measurements on this actual constrained hardware, not simulated or
adjusted.

### A real, dominant finding: a fixed ~2-second-per-turn floor

Every load-test run below shows a per-turn latency floor of almost
exactly **2.0–2.1 seconds on every single turn**, regardless of
concurrency (5, 10, 25, or 35 users all show virtually identical P95
latency). This does **not** come from the RAG/classification pipeline
itself — Phase 0 and Phase 1 above measured that at 20–150ms, two orders
of magnitude smaller. Root-caused (measurement only, no code changed, per
this task's own ground rules) to `backend/llm/llm_client.py`:

```python
_OLLAMA_HEALTH_TIMEOUT = 2   # seconds
```

Ollama is not running in this sandbox (`http://localhost:11434`
unreachable), so a 2-second health-check timeout fires before every
LLM-touching code path falls through to (failing, zero-credit) Claude.
This is a genuine, disclosed measurement finding, not a fix — per this
task's scope, `ISCOClassifier` and agent logic were not touched. **In a
deployment where Ollama is actually reachable, this 2-second floor would
not exist** — real per-turn latency there would likely be dominated by
the 20–150ms RAG/embedding costs measured in Phase 0/1 instead, not this
sandbox artifact. This distinction matters for interpreting every number
below: they represent this sandbox's real behavior, not a lower bound on
the system's real achievable latency.

### 3 runs at 10 concurrent users (chosen tier)

| Run | Success rate | Mean turn latency | P95 | P99 |
|---|---:|---:|---:|---:|
| 1 | 100% (10/10) | 2,138 ms | 2,458 ms | 2,787 ms |
| 2 | 100% (10/10) | 2,136 ms | 2,412 ms | 2,709 ms |
| 3 | 100% (10/10) | 2,132 ms | 2,419 ms | 2,643 ms |

**Range across 3 runs: 100% success (no variance); P95 range 2,412–2,458 ms
(46ms spread — tight and reproducible on this hardware).**

Why 10, not 20-30: a 5-user smoke test was run first to validate the
harness end-to-end against the live backend before committing to a
larger 3x measurement; 10 concurrent users was chosen as the primary
tier balancing "meaningful concurrency" against this sandbox's real,
observed RAM fragility (free RAM dropped as low as 0.23 GB earlier in
this session).

### Stepping up to find the real breaking point

| Concurrency | Success rate | Mean | P95 | P99 |
|---:|---:|---:|---:|---:|
| 10 (avg of 3 runs) | 100% | 2,135 ms | 2,430 ms | 2,713 ms |
| 25 | 100% (25/25) | 2,133 ms | 2,445 ms | 2,550 ms |
| 35 | 100% (35/35) | 2,134 ms | 2,627 ms | 2,846 ms |
| 42 | 100% (42/42) | 2,176 ms | 2,715 ms | 3,295 ms |
| **50** | **66.0% (33/50)** | **3,392 ms** | **5,382 ms** | **31,034 ms** |

**Breaking point found: between 42 and 50 concurrent users on this
sandbox.** At 50, both stated thresholds are breached simultaneously —
P95 (5,382ms) well past 3,000ms, and success rate (66.0%) well below 80%.
The 17 failures at 50 users were all client-side timeouts (`HTTP 0`,
`load_test.py`'s 30-second per-request timeout) on later turns (14–18),
consistent with genuine server-side saturation, not a harness bug — max
observed latency (31,834ms) sits just past the 30s timeout that
triggered the failure.

Not narrowed further than [42, 50) given this sandbox's demonstrated RAM
fragility during this task (free RAM ranged 0.23–1.07GB across the whole
session) — further probing risked crashing the host rather than refining
a number whose precision is already secondary to the environment
disclosure above.

### What was explicitly not tested

**Multi-instance horizontal scaling (multiple backend replicas behind a
load balancer) was not tested.** No load balancer or multi-replica
configuration exists in this repository's `docker-compose.yml`, and
setting one up was out of scope for a measurement-only task. The
single-instance breaking point above (between 42 and 50 concurrent users,
**on this specific resource-constrained sandbox**) is the only real data
point available; any horizontal-scaling capacity estimate built from it
would be a labeled extrapolation, not a measurement, and is not asserted
here.

---

## 4. Updated Week 2 estimate table

Original table (`Documentation/Phase_2/Week_2/week2_brief.md` §4),
corrected in place — see that file for the live version. Real
substitutions from this task:

| Component | Week 2's original assumption | Real number from this task |
|---|---|---|
| Hierarchical RAG stage latency | ~150ms/query, "not yet measured" | **Measured: 133.9ms mean / 137.4ms median** (Task 36, 18,747 real queries) — confirms the estimate's order of magnitude |
| Flat RAG stage latency | not stated | **Measured: 31.1ms mean / 30.1ms median** (Task 36, 18,747 real queries) |
| Embedding compute per call | not stated | **Measured: 19.1ms mean / 18.3ms median** (batch=1, this sandbox) |
| LLM re-ranking trigger rate | 30%–70%, "unmeasured... a guess" | **Still unmeasured** — Task 36 deliberately excluded the LLM tier; Task 43's attempt is a confirmed invalid artifact (Phase 0) |
| Claude 3.5 Sonnet cost/latency per call | ~$0.006 / ~1.5–2.5s, "not yet measured" | **Still unmeasured** — no valid real Claude reranking run exists anywhere in this repository (Phase 0) |
| Cost per interview ("$84 target") | assumed to exist in "earlier project docs" | **Not found anywhere in this repository** — disclosed as unverifiable, not extended |

Recommendation carried forward unchanged: the stratified 300–500-case
pilot Week 2 already recommends remains the right next step to get a
*real* LLM re-ranking trigger rate and Claude cost/latency — this task
did not attempt that pilot (out of scope; requires Anthropic credit,
which remains at zero as of this task).

---

## 5. Full test suite

Unaffected by this task (measurement-only; no production code changed).
Last full confirmed run this session: **2,362 passed, 1 deselected, 0
failed** (264s).
