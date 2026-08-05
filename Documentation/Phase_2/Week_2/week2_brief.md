# Module A — Week 2 Brief (revised)

> Location: `Documentation/Phase_2/Week_2/week2_brief.md`
> Produced: 2026-08-02, as the direct output of the Week 1 corrected pack
> (`Documentation/Phase_2/Week_1/module_a_week1_report.md`). This is a **brief**, not a report —
> it exists to hand Week 2's implementer a correct starting point, informed by what Week 1
> actually found rather than what the original plan assumed. Three things below are direct
> consequences of Week 1 corrections and would have been discovered the hard way in Week 8/9 if
> not carried forward now.

## 1. What Week 2 is for

Build the per-language test sets and the evaluation harness from `wisco_raw_parsed.json`
(4,232 records, 5 languages), ahead of the first external ISCO classification run in Week 3.

## 2. Three plan changes carried in from Week 1 (read before starting)

### 2.1 Module D: build against both industry sources, not one

`wisco_industry_crosswalk.json` (produced in Week 1, not the original plan) has two
separately-tagged fields per occupation:
- `nace2_0_rev2` — single value, NACE Rev.2, maps cleanly to ISIC Rev.4 at class level, but only
  ~17% of occupations have one.
- `nace2004_rev1_1` — list of values, NACE Rev.1.1, 100% occupation coverage, but needs an
  ISIC-Rev.3.1-era hop before it's comparable to ISIC Rev.4.

**Do not pick one in Week 2.** Whatever Module D's evaluation harness looks like (Week 8), it
needs to be able to report accuracy separately for "occupations with a `nace2_0_rev2` value"
(small n, high-precision source) and "occupations with only `nace2004_rev1_1`" (large n,
needs-conversion source) — collapsing them into one number now would hide exactly the precision
trade-off Week 1 found. If Week 2's harness design doesn't need to touch industry data at all,
no action needed yet — just don't let a later step quietly standardise on one source without
this context.

### 2.2 Arabic test set: usable for accuracy, not for the dialect experiment

Week 1 confirmed WISCO's Arabic data (`ar_AE`, 4,167 titles) is byte-for-byte identical Modern
Standard Arabic across all 22 country-locale columns — there is no dialectal variation in this
dataset. This does **not** block Week 2 or Week 3: the Arabic test set is fine for a standard
ISCO classification accuracy measurement, exactly like the other 4 languages.

**It does block Week 9 as currently scoped.** The planned Gulf-dialect-normalisation A/B test
needs input text that actually varies by dialect; WISCO's Arabic titles don't. This needs a
decision before Week 9, not during it:
- Option A: source or synthesise genuine Gulf-dialect job-title variants for that specific test.
- Option B: use real respondent free-text captured during the Module E pilot (if the pilot's
  timeline allows — see the ethics-submission schedule risk in `week1_status_note.md`).
- Option C: redefine what Week 9 measures if neither source materialises in time.

This is flagged here so it's a known open item carried through Weeks 2–8, not a surprise in
Week 9. No action needed in Week 2 itself.

### 2.3 Query-time normalisation: lowercase must happen in the harness

`parse_wisco.py` deliberately preserves original title casing (matches the corpus, not the
query path). Production's `ISCOClassifier.classify()` lowercases queries at call time
(`backend/agents/isco_classifier.py:372`, `job_title.lower().strip()`). **Week 2's evaluation
harness must replicate this** — call `.lower().strip()` on each WISCO title before passing it to
the classifier, or the harness will be testing different input normalisation than production
actually uses, and any accuracy gap won't be attributable to anything real.

## 3. Week 2 inputs checklist (updated)

- [x] `wisco_raw_parsed.json` — 4,232 records, 9/9 integrity checks passed
- [x] `wisco_industry_crosswalk.json` — both industry sources, separately tagged (new — not in the original plan)
- [x] `language_mapping_note.md` — all 61 base codes dispositioned, Arabic caveat documented
- [x] Per-language retained counts, verified per-unit-group coverage
- [ ] **New:** harness applies `.lower().strip()` to every title before querying the classifier (§2.3)
- [ ] **New:** harness design keeps `nace2_0_rev2` and `nace2004_rev1_1` reportable separately if it touches industry data at all (§2.1)

## 4. Week 3 compute estimate (requested — was previously unestimated)

Week 3 runs the first external ISCO classification pass: 4,232 titles × 5 languages = **21,160
classification calls** through the 4-stage hierarchical RAG pipeline, with conditional Claude
3.5 Sonnet re-ranking when the top hierarchical match's confidence is below 0.92
(`_HIGH_CONFIDENCE_THRESHOLD` in `isco_classifier.py`).

**This is an estimate with disclosed assumptions, not a measured figure** — the actual LLM
re-ranking trigger rate on WISCO's specific title phrasing is not yet known and is itself part
of what Week 3 will measure.

| Component | Assumption | Basis |
|---|---|---|
| Hierarchical RAG stage (all 21,160 queries) | ~150ms/query, local Qdrant + embedding | Typical local vector-search latency; not yet measured against this exact deployment |
| LLM re-ranking trigger rate | 30%–70% of queries | Unmeasured — genuinely a range, not a point estimate. Short titles with clear standard phrasing (WISCO's style) plausibly skew toward the low end vs. real respondent free text, but this is a guess, not a measurement. |
| Claude 3.5 Sonnet cost per re-ranking call | ~$0.006 (≈1,000 input + 200 output tokens) | Anthropic's published per-token pricing; actual prompt size depends on candidate-list length, not yet measured for this specific prompt template |
| Claude 3.5 Sonnet latency per call | ~1.5–2.5s | Typical API latency; not yet measured for this workload |

| Scenario | LLM calls | Estimated cost | Sequential wall-clock (LLM calls only) |
|---|---|---|---|
| Low (30% trigger) | ~6,350 | ~$38 | ~2.6–4.4 hours |
| Mid (50% trigger) | ~10,580 | ~$63 | ~4.4–7.3 hours |
| High (70% trigger) | ~14,810 | ~$89 | ~6.2–10.3 hours |

Sequential wall-clock for the non-LLM RAG-only stage adds roughly another 53 minutes on top
regardless of scenario. All wall-clock figures assume **sequential** execution; the classifier
has no built-in batching, so real wall-clock depends heavily on how many concurrent workers
Week 3's harness runs (a 5–10x reduction is plausible with straightforward parallelisation, at
the cost of Anthropic API rate limits becoming the new bottleneck).

**Recommendation: do not run the full 21,160-query set blind.** Run a stratified pilot subsample
first — e.g. 300–500 titles, stratified across all 10 ISCO major groups and all 5 languages —
to measure the *real* re-ranking trigger rate, real per-call latency, and real cost against this
specific workload. That pilot converts the range above into an actual number before committing
API budget and a full week of wall-clock to the complete run, and directly informs whether Week
3 needs to run on a subsample instead of the full 21,160.

## 5. Open items not resolved by this brief

- The Week 3 pilot-subsample decision above (§4) — needs an explicit go/no-go before Week 3 starts.
- Week 9's Arabic-dialect data source (§2.2) — needs a decision before Week 9, not during it.
- The `load_full_isco.py` subsistence-farming code-shift bug
  (`module_a_week1_report.md` §5.3) — not a Week 2 blocker, but should land before Chapter 3
  cites a unit-group count.
