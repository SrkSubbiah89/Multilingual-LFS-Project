"""
eval/run_eval.py

Evaluation harness for ISCO-08 classification. Reads a labelled test-set
CSV, runs every case through one of three systems, and writes one row per
case with full per-stage instrumentation to
results/raw_runs/<timestamp>_<config>.csv. All three write the identical
output schema so their CSVs stack for downstream analysis (eval/analyze.py).

Systems (--system flag)
------------------------
hierarchical  4-stage hierarchical Qdrant pipeline (backend.agents.isco_classifier)
flat          Single-stage dense retrieval, SAME embedding model and SAME
              LLM reranking rule as hierarchical (ISCOClassifier(force_flat=True)
              -- this reuses _classify_flat()'s real reranking call, not a
              separately-reimplemented baseline that could silently drift).
bm25          Sparse lexical retrieval (rank-bm25), refactored from the
              existing backend.evaluation.evaluate.BM25Baseline rather than
              rewritten -- see the "refactor note" below. No reranking (the
              session brief only requires reranking parity for Flat RAG).

For flat and bm25, stage1-3 columns are the JSON literal `null` (not an
empty list, and not omitted) -- they have no stages 1-3 at all, which is a
different fact from "stage ran and returned zero candidates" (empty list,
which can happen for hierarchical on a pathological query).

Refactor note (flat RAG)
--------------------------
backend/evaluation/evaluate.py already has a FlatVectorBaseline class from
earlier ad hoc runs. It does NOT rerank -- pure top-k dense retrieval, no
LLM step at all. The session brief requires Flat RAG to use "the same
reranking rule" as hierarchical, which FlatVectorBaseline does not do. Two
options: bolt reranking onto a copy of that class, or reuse
ISCOClassifier._classify_flat(), which already shares the exact same
_llm_select_from_candidates() call (same 0.92 threshold, same prompt
construction) as the hierarchical path, since that's what it's for in
production (fallback when hierarchical collections aren't populated). The
harness uses the latter (`--system flat` = force_flat=True) -- zero
duplicated reranking logic, and genuine behavioural parity, not "close
enough." FlatVectorBaseline itself is unused by this harness; it's still
imported by the existing backend/evaluation/evaluate.py script, untouched.

This session is instrumentation only. No classification/pipeline logic was
changed to build this harness -- backend/rag/hierarchical_store.py and
backend/agents/isco_classifier.py gained purely additive, opt-in parameters
(`trace=`, `llm_temperature=`, `force_flat=`) that are None/False/unused by
every existing caller and therefore change nothing about current production
behaviour. See the module docstrings in those two files for the exact diff
rationale.

Test-set CSV columns
---------------------
Required : case_id, input_text, input_language, gold_isco_4digit
Optional : gold_isic (1-letter ISIC section, e.g. "Q") -- scores the ISIC
             classifier's own accuracy; NEVER fed to the SRE (see below).
           gold_isced (integer 0-8 ISCED 2011 level) -- same, for ISCED.
           industry_text -- free-text industry description, as a
             respondent would answer a separate "what industry do you
             work in" survey question. NOT input_text/job_title. Required
             (together with education_text) for SRE coherence to be
             computed at all; if either is blank the row's
             sre_coherence_score is left empty rather than fabricated.
           education_text -- free-text education/qualification
             description, analogous to industry_text.

SRE coverage note
-------------------
The SRE (backend.agents.semantic_relation) checks whether a PREDICTED
ISCO code is jointly plausible with a PREDICTED ISIC section and a
PREDICTED ISCED level -- three independent classifier outputs, mirroring
three independently-answered survey questions. This harness runs
ISICClassifier.classify(industry_text) and ISCEDClassifier.classify(
education_text) and feeds THOSE predictions to sre.analyse(), never
gold_isic/gold_isced (feeding gold would make the ISIC/ISCED side correct
by construction and understate incoherence -- exactly the failure mode
that produces an unpublishable complementarity result that looks
publishable). Existing test sets (test_set_smoke20.csv,
test_set_full130.csv, test_set_sys_compare50.csv) predate this and only
test_set_smoke20.csv has industry_text/education_text populated as of
this revision -- SRE columns will be empty for every row of a CSV that
lacks them.

Reranker pinning (--reranker-model) and retrieval-only mode (--use-llm-reranker off)
--------------------------------------------------------------------------------------
--reranker-model is required for --system hierarchical/flat ONLY when
--use-llm-reranker is 'on' (the default). Passed to
ISCOClassifier(reranker_model=...), which resolves it via
backend.llm.get_llm_strict() -- health-checked once at startup, hard
failure (run aborts, no cases run) if the pinned model is unavailable,
and NEVER silently substituted for a different model mid-run. This
replaces the previous behaviour of building the reranker through
get_llm(TaskType.GENERAL), which silently falls back to Claude when
Ollama is unreachable -- correct for production conversational agents,
wrong for an eval run where every case must be answered by the same,
known model. 'ollama/llama3.2:1b' is a free/local smoke-test choice, not
the paper's reranker -- the paper attributes reranking to GPT-4o and
Claude 3.5 Sonnet, so any number destined for the manuscript must be run
with --reranker-model anthropic/claude-3-5-sonnet-20241022 (requires
ANTHROPIC_API_KEY with available credit).

Task 09: --use-llm-reranker off makes a hierarchical/flat run genuinely
retrieval-only, not merely a run that skips calling the reranker.
ISCOClassifier is constructed with enable_llm=False, which skips Stage 2
of __init__ entirely -- no get_llm_strict(), get_llm(), or
_build_reranker_agent() call, no LLM/agent object exists, and
reranker_model_resolved reports the explicit string "none (reranking
disabled)". --reranker-model is neither required nor consulted in this
mode. This is NOT a reranking comparison (there is nothing to compare --
no reranking runs at all) and it supports ISCO-08 classification only;
ISICClassifier/ISCEDClassifier/SemanticRelationEngine are separately
gated on whether the loaded test-set rows actually carry paired
industry_text/education_text (see "Test-set CSV columns" above) --
independent of --use-llm-reranker, but for a WISCO-style ISCO-only CSV
neither condition holds, so a retrieval-only WISCO run constructs no LLM
and no ISIC/ISCED/SRE component at all.

degraded column
------------------
True when the reranker fired but failed (trace["reranker_error"] set) --
the row still has a prediction (the pre-rerank top candidate), but that
prediction did not come from the system under test. analyze.py should
report degraded-case counts separately rather than silently folding them
into "clean" accuracy.

Usage
-----
    python eval/run_eval.py --test-set path/to/test_set.csv --system hierarchical --reranker-model ollama/llama3.2:1b
    python eval/run_eval.py --test-set path/to/test_set.csv --system flat --reranker-model ollama/llama3.2:1b
    python eval/run_eval.py --test-set path/to/test_set.csv --system bm25
    python eval/run_eval.py --test-set path/to/test_set.csv --system hierarchical --reranker-model anthropic/claude-3-5-sonnet-20241022 --limit 20

    # Retrieval-only (Task 09) -- no --reranker-model, no LLM constructed:
    python eval/run_eval.py --test-set path/to/test_set.csv --system hierarchical --use-llm-reranker off
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agents.isco_classifier import ISCOClassifier  # noqa: E402
from backend.agents.isic_classifier import ISICClassifier  # noqa: E402
from backend.agents.isced_classifier import ISCEDClassifier  # noqa: E402
from backend.agents.semantic_relation import SemanticRelationEngine  # noqa: E402
from backend.rag.hierarchical_store import MODEL_NAME as EMBEDDING_MODEL_NAME  # noqa: E402
from backend.rag.hierarchical_store import LEGACY_PROFILE, OFFICIAL_PROFILE_ILO2021_V1  # noqa: E402
from backend.evaluation.evaluate import BM25Baseline  # noqa: E402

_logger = logging.getLogger("eval.run_eval")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "raw_runs"

# Fixed seed for anything sampling-related in this harness (e.g. the
# stratified-QC-sample escalation reason). Not used for the classification
# pipeline itself, which is already deterministic at temperature=0.
RANDOM_SEED = 42

# One case in every N is flagged for the stratified QC sample, independent
# of confidence/SRE -- deterministic on case index, not on any RNG draw, so
# the exact same cases are flagged on every re-run regardless of seed
# plumbing elsewhere.
STRATIFIED_SAMPLE_EVERY_N = 20

HITL_CONFIDENCE_THRESHOLD = ISCOClassifier.HITL_THRESHOLD  # 0.70, imported not hardcoded

# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens). Hardcoded and documented here rather than
# resolved through litellm's cost DB, which does not recognise this repo's
# pinned model strings ("anthropic/claude-3-5-sonnet-20241022",
# "ollama/llama3.2:1b") out of the box. Update this table if pricing changes
# or a new model is added -- every cost figure in the paper traces back to
# exactly these two numbers.
# ---------------------------------------------------------------------------
_PRICING_PER_MTOK = {
    # model substring -> (input $/MTok, output $/MTok)
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
}


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return 0.0 for local Ollama models; look up Anthropic pricing by
    substring match on the pinned model string; log and return 0.0 (not a
    silent wrong number) for anything unrecognised."""
    if not model or model.startswith("ollama/"):
        return 0.0
    for key, (in_price, out_price) in _PRICING_PER_MTOK.items():
        if key in model:
            return round(
                prompt_tokens / 1_000_000 * in_price
                + completion_tokens / 1_000_000 * out_price,
                6,
            )
    _logger.warning("No pricing entry for model=%r; cost recorded as 0.0, not estimated.", model)
    return 0.0


def _digits(code: str, n: int) -> str:
    """First n characters of an ISCO code, or '' if code is shorter/empty."""
    code = (code or "").strip()
    return code[:n] if len(code) >= n else ""


def _max_severity(violations: list) -> str:
    order = {"HIGH": 3, "MODERATE": 2, "LOW": 1}
    if not violations:
        return "NONE"
    return max(violations, key=lambda v: order.get(v.severity, 0)).severity


class _BM25Adapter:
    """Wraps the existing backend.evaluation.evaluate.BM25Baseline (unmodified,
    reused not reimplemented) behind the subset of ISCOClassifier.classify()'s
    interface run_one_case() needs, so the harness doesn't need a separate
    code path per system. No reranking for BM25 -- see module docstring."""

    HITL_THRESHOLD = ISCOClassifier.HITL_THRESHOLD

    def __init__(self) -> None:
        self._bm25 = BM25Baseline()
        # No reranking stage exists for BM25 -- recorded explicitly (not
        # left blank) so the config hash and CSV always say what actually
        # ran rather than implying an LLM was involved.
        self.reranker_model_resolved = "none (bm25 has no reranking stage)"

    def classify(self, job_title: str, language: str = "", top_k: int = 5,
                 use_llm: bool = True, trace: Optional[dict] = None,
                 max_stage_latency_ms: Optional[float] = None):
        # max_stage_latency_ms (Task 31): accepted for call-site signature
        # compatibility with ISCOClassifier.classify() -- unused here.
        # bm25 has no hierarchical stages/Qdrant queries to budget, and
        # --require-genuine-hierarchical is already rejected outright for
        # --system bm25 (see argument validation in main()).
        pred = self._bm25.predict(job_title, top_k=top_k)
        if trace is not None:
            trace["stage1"], trace["stage2"], trace["stage3"] = None, None, None
            trace["stage1_latency_ms"], trace["stage2_latency_ms"], trace["stage3_latency_ms"] = None, None, None
            trace["stage1_source"] = "not_applicable (bm25 has no stage1)"
            trace["stage4"] = [{"code": c, "label_en": "", "score": None} for c in pred.top3_codes]
            trace["stage4_latency_ms"] = pred.latency_ms
            trace["reranker_fired"] = False

        from types import SimpleNamespace
        primary = SimpleNamespace(code=pred.predicted_code, title_en="", title_ar="", confidence=pred.confidence)
        return SimpleNamespace(
            primary=primary,
            method="bm25",
            hitl_required=pred.confidence < self.HITL_THRESHOLD,
            reasoning="BM25 top-1 lexical match (no reranking; rank-bm25 over ISCO-08 title+description text).",
        )


def build_system(system: str, reranker_model: Optional[str] = None, disable_keyword_map: bool = False,
                  beam: int = 2, stage1_mode: str = "description",
                  reranker_candidates: int = 5, branch_collapse: bool = False,
                  capture_pool_metadata: bool = False, use_llm_reranker: bool = True,
                  isco_catalogue_profile: str = LEGACY_PROFILE):
    """Return a classifier object exposing .classify(...) for the requested
    --system value. hierarchical and flat share ISCOClassifier (same class,
    force_flat toggles the retrieval path); bm25 uses the adapter above.

    reranker_model is required (non-None) for hierarchical/flat WHEN
    use_llm_reranker=True -- see ISCOClassifier's reranker_model parameter:
    it pins the reranking LLM via get_llm_strict() (hard-fail if
    unavailable, no silent substitution) and the exception from an
    unavailable pinned model propagates out of this call, aborting the run
    before any case is classified. Ignored for bm25, which has no
    reranking stage at all. beam/stage1_mode/reranker_candidates/
    branch_collapse are likewise ignored for bm25 (no stages) and unused by
    flat (single-stage retrieval, no branches to pool across).

    use_llm_reranker=False (Task 09): for hierarchical/flat, constructs
    ISCOClassifier(enable_llm=False, reranker_model=None) -- no LLM/agent
    is initialised at all (not just skipped per-call), reranker_model is
    ignored entirely (never passed to get_llm_strict), and
    reranker_model_resolved reports the explicit "none (reranking
    disabled)" string rather than any model identity. This is what makes
    a --use-llm-reranker off run genuinely retrieval-only, not merely
    reranker-call-skipping.

    capture_pool_metadata (B2, default False): opt-in, observational only --
    see HierarchicalISCOStore.search()'s capture_pool_metadata docstring.
    Ignored for bm25 (no stage4 pool to capture).

    isco_catalogue_profile (Task 21, default "legacy"): passed straight
    through to ISCOClassifier(isco_catalogue_profile=...). "legacy" is
    byte-identical to every prior version of this function -- existing
    callers that omit this see zero behavioural change. A non-legacy
    profile (e.g. "official_ilo2021_v1") makes both --system hierarchical
    and --system flat select the versioned official collections instead
    (flat becomes the genuine, direct, unfiltered official four-digit
    comparator via ISCOClassifier's force_flat_only wiring -- not the
    legacy mixed-granularity isco_occupations collection). Ignored for
    bm25 (no ISCOClassifier is constructed for that system)."""
    common_kwargs = dict(
        llm_temperature=0.0, disable_keyword_map=disable_keyword_map,
        beam=beam, stage1_mode=stage1_mode,
        reranker_candidates=reranker_candidates,
        branch_collapse=branch_collapse,
        capture_pool_metadata=capture_pool_metadata,
        isco_catalogue_profile=isco_catalogue_profile,
    )
    if system == "hierarchical":
        if use_llm_reranker:
            return ISCOClassifier(reranker_model=reranker_model, **common_kwargs)
        return ISCOClassifier(reranker_model=None, enable_llm=False, **common_kwargs)
    if system == "flat":
        if use_llm_reranker:
            return ISCOClassifier(force_flat=True, reranker_model=reranker_model, **common_kwargs)
        return ISCOClassifier(force_flat=True, reranker_model=None, enable_llm=False, **common_kwargs)
    if system == "bm25":
        return _BM25Adapter()
    raise ValueError(f"Unknown --system {system!r}; expected hierarchical, flat, or bm25")


def build_dry_run_case_result(
    row_index: int,
    case_id: str,
    input_text: str,
    input_language: str,
    gold_isco_4digit: str,
    gold_isic: str,
    gold_isced: str,
    config_hash: str,
    run_id: str = "",
    git_commit: str = "",
    seed: Optional[int] = None,
) -> "CaseResult":
    """
    Conference I Reviewer #2, Step 4 (evaluation-readiness dry-run pass).

    The --dry-run code path takes THIS instead of run_one_case(). Echoes
    the case/gold columns straight from the loaded test-set CSV -- proving
    CSV parsing, per-row iteration, and CSV/JSONL output serialization all
    work end-to-end on real row counts -- but calls no classifier, builds
    no ISCOClassifier/ISICClassifier/ISCEDClassifier, and makes no Qdrant/
    LLM/network call of any kind. Every pred_* field is left at its
    dataclass default (blank string / None), and evaluation_status is
    explicitly "dry_run_not_measured" so no downstream consumer can mistake
    this row for a completed classification result.
    """
    return CaseResult(
        case_id=case_id,
        input_text=input_text,
        input_language=input_language,
        gold_isco_4digit=gold_isco_4digit,
        gold_isco_1digit=_digits(gold_isco_4digit, 1),
        gold_isco_2digit=_digits(gold_isco_4digit, 2),
        gold_isco_3digit=_digits(gold_isco_4digit, 3),
        gold_isic=gold_isic or "",
        gold_isced=gold_isced or "",
        config_hash=config_hash,
        run_id=run_id,
        git_commit=git_commit,
        seed=seed,
        input_order_position=row_index,
        evaluation_status="dry_run_not_measured",
    )


@dataclass
class CaseResult:
    """One output row. Field order here is the CSV column order."""
    case_id: str
    input_text: str
    input_language: str

    gold_isco_1digit: str = ""
    gold_isco_2digit: str = ""
    gold_isco_3digit: str = ""
    gold_isco_4digit: str = ""
    gold_isic: str = ""
    gold_isced: str = ""

    pred_isco_1digit: str = ""
    pred_isco_2digit: str = ""
    pred_isco_3digit: str = ""
    pred_isco_4digit: str = ""
    pred_title_en: str = ""
    pred_confidence: Optional[float] = None
    pred_method: str = ""
    pred_hitl_required: Optional[bool] = None
    pred_reasoning: str = ""

    keyword_map_enabled: bool = True  # run-level flag: was --disable-keyword-map passed for THIS run
    stage1_source: str = ""  # "keyword_map" | "semantic_retrieval" | "not_applicable (...)"
    stage1_candidates: str = "[]"   # JSON
    stage2_candidates: str = "[]"
    stage3_candidates: str = "[]"
    stage4_candidates: str = "[]"
    stage1_latency_ms: Optional[float] = None
    stage2_latency_ms: Optional[float] = None
    stage3_latency_ms: Optional[float] = None
    stage4_latency_ms: Optional[float] = None

    # Task 25: additive flat-only-retrieval query telemetry (backend/rag/
    # hierarchical_store.py::_flat_search(), the official-profile flat
    # comparator's Qdrant call). Distinguishes a genuine successful
    # zero-hit Qdrant response from a swallowed query exception (e.g. a
    # timeout) -- both previously produced an identical empty result with
    # no telemetry. Populated only when the flat query path actually ran
    # (official-profile --system flat and any future non-legacy-profile
    # flat run); blank/None for every hierarchical run and for the
    # legacy-profile flat path (backend/agents/isco_classifier.py's
    # _classify_flat(), which uses the separate legacy VectorStore and is
    # untouched by this task). flat_query_outcome is "success" | "exception"
    # | "" (flat query path not used for this row). exception_type/message
    # are populated ONLY when outcome == "exception"; exception_message is
    # sanitized (bounded length, single line, never raw query text, a
    # stack trace, or credentials -- see hierarchy_engine._sanitize_exception_message).
    flat_query_outcome: str = ""
    flat_query_duration_ms: Optional[float] = None
    flat_query_exception_type: str = ""
    flat_query_exception_message: str = ""

    # Task 27: additive bounded-retry telemetry. flat_query_outcome may
    # now additionally take the values "success_after_retry" and
    # "retry_exhausted" -- ONLY when a caller has explicitly opted into
    # max_query_attempts > 1 (QDRANT_QUERY_MAX_ATTEMPTS env var or an
    # explicit HierarchicalISCOStore(max_query_attempts=...) argument).
    # Under the unchanged default (max_query_attempts=1, no retry), only
    # "success"/"exception" ever appear and flat_query_attempts is always
    # 1 -- fully backward-compatible with every Task 25 consumer.
    # flat_query_attempts: total attempts made (None when the flat query
    # path did not run for this row, same convention as flat_query_outcome).
    # flat_query_attempt_durations_ms: JSON list, one entry per attempt.
    flat_query_attempts: Optional[int] = None
    flat_query_attempt_durations_ms: str = "[]"

    # Task 27: separate, clearly-named hierarchical-stage retry/exception
    # telemetry -- distinct from the flat_query_* fields above (never
    # overloaded/misused as stage evidence; check_strict_hierarchical()
    # never inspects this field). JSON object keyed "stage1".."stage4",
    # present only for stages that actually issued a live Qdrant query
    # (absent for a keyword-anchor-seeded or leaf_vote-overridden stage
    # 1, which never calls _query() at all). Each stage's value is
    # {"queries": int, "any_retry": bool, "any_exception": bool,
    # "max_attempts_used": int, "exception_types": [str, ...]} --
    # aggregated across every beam-branch query issued at that stage, so
    # a single genuinely-retried or genuinely-failed query anywhere in
    # that stage is never hidden by an average or a last-write-wins
    # overwrite. Default "{}" for every row where the hierarchical beam
    # search did not run (flat/bm25 systems) or trace was not requested.
    hier_stage_query_telemetry: str = "{}"

    reranker_fired: Optional[bool] = None
    reranker_input_candidates: str = "[]"  # JSON
    reranker_output: str = "{}"            # JSON: {"code", "reasoning"}
    reranker_model: str = ""
    reranker_latency_ms: Optional[float] = None

    branch_collapse_enabled: bool = False  # run-level flag: was --branch-collapse passed for THIS run
    reranker_candidate_pool_size: Optional[int] = None  # unique codes pooled across all explored branches
    reranker_candidate_branches: Optional[int] = None   # how many branches contributed stage-4 hits
    stage4_pool: str = "[]"        # JSON, full pooled+sorted list (branch_collapse=False runs only)
    gold_rank_in_pool: Optional[int] = None  # 1-based rank of gold_isco_4digit in stage4_pool, empty if absent

    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0

    pred_isic_section: str = ""       # predicted (not gold) -- fed to SRE
    pred_isic_confidence: Optional[float] = None
    pred_isced_level: str = ""        # predicted (not gold) -- fed to SRE
    pred_isced_confidence: Optional[float] = None

    # Additive (Conference I Reviewer #2, Section D): full-depth ISIC/ISCED-F
    # predictions, alongside the section-only/level-only fields above, so
    # eval/analyze.py can compute division/group/class and broad/narrow/
    # detailed accuracy rather than just the single top-level digit. Existing
    # columns/consumers above are unaffected -- these are new, harmless-
    # default fields only, same pattern as every other additive column in
    # this dataclass.
    pred_isic_division: str = ""
    pred_isic_group: str = ""
    pred_isic_class: str = ""
    pred_isced_broad: str = ""
    pred_isced_narrow: str = ""
    pred_isced_detailed: str = ""

    sre_coherence_score: Optional[float] = None
    sre_severity: str = ""

    # Conference I Reviewer #2, Step 5.1 (SRE-to-ISIC/ISCED coupling bugfix).
    # Authoritative status for whether the SRE actually evaluated this row --
    # sre_severity/sre_coherence_score alone cannot distinguish "SRE disabled
    # by --sre off" from "no industry/education text on this row" from
    # "evaluated, zero violations" (all three left sre_severity=="" or
    # ambiguous pre-fix). One of: "evaluated" (sre.analyse() ran; see
    # sre_severity/sre_coherence_score for the result), "disabled_by_
    # configuration" (--sre off; ISCO/ISIC/ISCED classification still ran
    # normally), "not_applicable" (no industry_text/education_text on this
    # row -- SRE was never in scope regardless of --sre), "error" (ISIC/
    # ISCED classification or the SRE call raised; see sre_status_reason).
    sre_status: str = "not_applicable"
    sre_status_reason: str = ""

    escalation_triggered: Optional[bool] = None
    escalation_reason: str = ""  # "confidence" | "sre_severity" | "stratified_sample" | "" | combined w/ ";"

    degraded: bool = False
    # True when the reranker was invoked but failed (trace["reranker_error"]
    # set) -- the case still has a prediction (pre-rerank top candidate),
    # but that prediction was not produced by the system under test, and
    # analyze.py must be able to exclude/report on these separately rather
    # than silently averaging them into "ok" accuracy.

    retrieval_latency_ms: Optional[float] = None
    end_to_end_latency_ms: Optional[float] = None

    embedding_model_version: str = EMBEDDING_MODEL_NAME
    reranker_model_version: str = ""  # filled post-hoc from reranker_model for a stable column name
    config_hash: str = ""

    error: str = ""

    # ── B2 candidate-capacity experiment fields ─────────────────────────────
    # All observational / run-bookkeeping only -- populated in addition to,
    # never in place of, any field above. Empty/default for every B0/B1 run
    # (capture_pool_metadata defaults to False, --run-id/--seed/--jsonl-output
    # are new opt-in flags with no effect unless passed). See
    # backend/rag/candidate_pool.py and hierarchical_store.py's
    # capture_pool_metadata parameter docstrings for what stage4_pool_enriched
    # actually contains; it is NOT used for selection/ordering in B2 -- B2
    # still selects/orders via the untouched B1 stage4_pool path.
    capture_pool_metadata_enabled: bool = False  # run-level flag: was --capture-pool-metadata passed
    stage4_pool_enriched: str = "[]"  # JSON; per-candidate branch_id/source_rank/path/raw+normalized score
    run_id: str = ""
    git_commit: str = ""
    seed: Optional[int] = None
    input_order_position: Optional[int] = None  # 0-based row index in the test-set CSV as loaded
    retry_count: int = 0  # always 0 for B0/B1/B2 (standard reranker has no retry loop); non-zero only
    # once a future B3-Reliability run wires backend.agents.isco_reranker_strict.StrictReranker through
    invalid_output_flag: bool = False  # LLM responded but JSON could not be parsed / code not recognised
    timed_out_flag: bool = False       # reranker_error signature matched a timeout (see run_one_case())
    peak_memory_mb: Optional[float] = None

    # Task 13: keyword-anchor recovery metadata (backend/rag/hierarchical_store.py
    # ::_hierarchical_search()). False/blank for every case that didn't use a
    # keyword major_hint at all, or whose keyword-anchored search succeeded on
    # the first attempt -- unchanged from prior behaviour for those cases.
    # True only when the keyword anchor reached no complete hierarchical path
    # and the store retried once, unseeded, before this row's final
    # pred_method/stage evidence were recorded. See stage1_source for what the
    # FINAL winning attempt actually was ("semantic_retrieval" after a retry,
    # "keyword_map" when the anchor succeeded outright) -- this field is
    # deliberately separate so a retry is never conflated with stage1_source.
    keyword_anchor_retry_used: bool = False
    keyword_anchor_original_hint: str = ""

    # Conference I Reviewer #2, Step 4 (evaluation-readiness dry-run pass).
    # "measured" for every real run (default, unchanged for every existing
    # caller). --dry-run sets this to "dry_run_not_measured" on every row it
    # writes, so no dry-run CSV can be mistaken for a completed result by a
    # downstream consumer that only looks at the prediction columns (which
    # are correctly blank/None either way, but this field makes the REASON
    # for that blankness explicit and machine-checkable). See
    # eval.manifest.ExperimentRunManifest.evaluation_status for the
    # run-level counterpart of this same field.
    evaluation_status: str = "measured"


def run_one_case(
    clf,  # ISCOClassifier (hierarchical/flat) or _BM25Adapter -- duck-typed on .classify()
    sre: Optional[SemanticRelationEngine],
    isic_clf: Optional[ISICClassifier],
    isced_clf: Optional[ISCEDClassifier],
    row_index: int,
    case_id: str,
    input_text: str,
    input_language: str,
    gold_isco_4digit: str,
    gold_isic: str,
    gold_isced: str,
    config_hash: str,
    system: str = "hierarchical",
    industry_text: str = "",
    education_text: str = "",
    keyword_map_enabled: bool = True,
    branch_collapse_enabled: bool = False,
    capture_pool_metadata_enabled: bool = False,
    run_id: str = "",
    git_commit: str = "",
    seed: Optional[int] = None,
    input_order_position: Optional[int] = None,
    sre_enabled: bool = True,
    use_llm_reranker: bool = True,
    max_stage_latency_ms: Optional[float] = None,
) -> CaseResult:
    """sre/isic_clf/isced_clf may all be None (Task 09) -- main() only
    constructs them when at least one selected row has both non-blank
    industry_text and education_text; when None, every row's ISIC/ISCED/SRE
    block below takes the "not_applicable" branch, matching the guard's
    own per-row industry_text/education_text check.

    max_stage_latency_ms (Task 31): opt-in strict stage deadline budget,
    passed straight through to clf.classify(). Callers should pass this
    only when BOTH --system hierarchical and --require-genuine-hierarchical
    are active (see main()) -- matching --max-stage-latency-ms's existing
    documented relationship to --require-genuine-hierarchical. Default
    None reproduces prior behaviour exactly for every other case."""
    result = CaseResult(
        case_id=case_id,
        input_text=input_text,
        input_language=input_language,
        gold_isco_4digit=gold_isco_4digit,
        gold_isco_1digit=_digits(gold_isco_4digit, 1),
        gold_isco_2digit=_digits(gold_isco_4digit, 2),
        gold_isco_3digit=_digits(gold_isco_4digit, 3),
        gold_isic=gold_isic or "",
        gold_isced=gold_isced or "",
        config_hash=config_hash,
        keyword_map_enabled=keyword_map_enabled,
        branch_collapse_enabled=branch_collapse_enabled,
        embedding_model_version=(
            "none (bm25: rank-bm25 sparse lexical retrieval, no embedding model)"
            if system == "bm25" else EMBEDDING_MODEL_NAME
        ),
        capture_pool_metadata_enabled=capture_pool_metadata_enabled,
        run_id=run_id,
        git_commit=git_commit,
        seed=seed,
        input_order_position=input_order_position,
    )

    trace: dict = {}
    t_start = time.perf_counter()
    try:
        clf_result = clf.classify(
            job_title=input_text,
            language=input_language if input_language in ("en", "ar", "mixed") else "",
            top_k=5,
            use_llm=use_llm_reranker,
            trace=trace,
            max_stage_latency_ms=max_stage_latency_ms,
        )
    except Exception as exc:  # noqa: BLE001 - a failed case must not crash the run
        result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"
        result.end_to_end_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        _logger.error("case_id=%s failed: %s", case_id, exc)
        return result
    t_end = time.perf_counter()

    result.pred_isco_4digit = clf_result.primary.code
    result.pred_isco_1digit = _digits(clf_result.primary.code, 1)
    result.pred_isco_2digit = _digits(clf_result.primary.code, 2)
    result.pred_isco_3digit = _digits(clf_result.primary.code, 3)
    result.pred_title_en = clf_result.primary.title_en
    result.pred_confidence = clf_result.primary.confidence
    result.pred_method = clf_result.method
    result.pred_hitl_required = clf_result.hitl_required
    result.pred_reasoning = clf_result.reasoning

    result.stage1_source = trace.get("stage1_source", "")
    result.keyword_anchor_retry_used = bool(trace.get("keyword_anchor_retry", False))
    result.keyword_anchor_original_hint = trace.get("keyword_anchor_original_hint", "")
    result.stage1_candidates = json.dumps(trace.get("stage1", []), ensure_ascii=False)
    result.stage2_candidates = json.dumps(trace.get("stage2", []), ensure_ascii=False)
    result.stage3_candidates = json.dumps(trace.get("stage3", []), ensure_ascii=False)
    result.stage4_candidates = json.dumps(trace.get("stage4", []), ensure_ascii=False)

    result.reranker_candidate_pool_size = trace.get("reranker_candidate_pool_size")
    result.reranker_candidate_branches = trace.get("reranker_candidate_branches")
    stage4_pool = trace.get("stage4_pool")
    if stage4_pool is not None:
        result.stage4_pool = json.dumps(stage4_pool, ensure_ascii=False)
        pool_codes = [c.get("code", "") for c in stage4_pool]
        if gold_isco_4digit in pool_codes:
            result.gold_rank_in_pool = pool_codes.index(gold_isco_4digit) + 1  # 1-based

    # B2 instrumentation only -- present only when capture_pool_metadata_enabled
    # was passed through to clf.classify(); never read for selection/ordering,
    # gold_rank_in_pool above (used by every metric) still comes from the
    # untouched stage4_pool (B1) line, not from this enriched pool.
    stage4_pool_enriched = trace.get("stage4_pool_enriched")
    if stage4_pool_enriched is not None:
        result.stage4_pool_enriched = json.dumps(stage4_pool_enriched, ensure_ascii=False)

    # trace["stageN_latency_ms"] is explicitly None (not 0.0, not absent) for
    # systems with no stage N at all (flat/bm25, stages 1-3) -- .get()'s
    # default only applies when the key is *missing*, so round() would still
    # be called on None here without this guard (confirmed: this crashed the
    # first --system bm25 run before this fix).
    def _safe_round(v, ndigits=2):
        return round(v, ndigits) if v is not None else None
    result.stage1_latency_ms = _safe_round(trace.get("stage1_latency_ms", 0.0))
    result.stage2_latency_ms = _safe_round(trace.get("stage2_latency_ms", 0.0))
    result.stage3_latency_ms = _safe_round(trace.get("stage3_latency_ms", 0.0))
    result.stage4_latency_ms = _safe_round(trace.get("stage4_latency_ms", 0.0))

    # Task 25: flat-only-retrieval query telemetry -- absent (blank/None)
    # for every row that didn't go through _flat_search() (hierarchical
    # runs, legacy-profile flat runs).
    result.flat_query_outcome = trace.get("flat_query_outcome", "")
    result.flat_query_duration_ms = _safe_round(trace.get("flat_query_duration_ms"), 3)
    result.flat_query_exception_type = trace.get("flat_query_exception_type", "")
    result.flat_query_exception_message = trace.get("flat_query_exception_message", "")
    result.flat_query_attempts = trace.get("flat_query_attempts")
    result.flat_query_attempt_durations_ms = json.dumps(trace.get("flat_query_attempt_durations_ms", []))

    # Task 27: aggregate every stage's "stageN_query_telemetry" trace
    # bucket (written by hierarchy_engine.HierarchyBeamSearchEngine.search()'s
    # _timed_query()) into one JSON object, keyed by stage name -- kept
    # separate from flat_query_* by construction (this key never appears
    # in a flat-only run's trace).
    _hier_stage_telemetry = {
        f"stage{i}": trace[f"stage{i}_query_telemetry"]
        for i in range(1, 5)
        if f"stage{i}_query_telemetry" in trace
    }
    if _hier_stage_telemetry:
        result.hier_stage_query_telemetry = json.dumps(_hier_stage_telemetry, ensure_ascii=False)

    result.retrieval_latency_ms = round(
        sum(v for i in range(1, 5) if (v := trace.get(f"stage{i}_latency_ms")) is not None),
        2,
    )

    result.reranker_fired = bool(trace.get("reranker_fired", False))
    result.reranker_input_candidates = json.dumps(
        trace.get("reranker_input_candidates", []), ensure_ascii=False
    )
    result.reranker_output = json.dumps(trace.get("reranker_output", {}), ensure_ascii=False)
    result.reranker_model = trace.get("reranker_model", "")
    result.reranker_model_version = result.reranker_model
    result.prompt_tokens = int(trace.get("reranker_prompt_tokens", 0))
    result.completion_tokens = int(trace.get("reranker_completion_tokens", 0))
    result.estimated_cost_usd = _estimate_cost_usd(
        result.reranker_model, result.prompt_tokens, result.completion_tokens
    )
    if "reranker_error" in trace:
        result.error = f"reranker_error: {trace['reranker_error']}"  # non-fatal; case still has a result
        result.degraded = True  # prediction is pre-rerank top candidate, not the system under test
        result.timed_out_flag = "timeout" in result.error.lower() or "timed out" in result.error.lower()

    # B2 bookkeeping only -- the standard reranker (_llm_select_from_candidates
    # -> _parse_llm_response) has no retry loop, so retry_count stays 0 and
    # this only ever detects the single-attempt outcome. A response that
    # arrived (no reranker_error) but could not be parsed into a known code
    # falls back to the top candidate silently, recorded via this exact
    # reasoning string set in _parse_llm_response() -- see backend/agents/
    # isco_classifier.py. Distinct from timed_out_flag (kickoff() itself
    # raised, e.g. litellm.Timeout) which is set above.
    result.invalid_output_flag = "could not be parsed" in (result.pred_reasoning or "")

    result.end_to_end_latency_ms = round((t_end - t_start) * 1000, 2)
    result.reranker_latency_ms = (
        round(result.end_to_end_latency_ms - result.retrieval_latency_ms, 2)
        if result.reranker_fired else 0.0
    )

    # ── ISIC / ISCED base classification ────────────────────────────────────
    # Runs the ISIC and ISCED classifiers on their own free-text inputs
    # (industry_text / education_text -- these are NOT the job title; ISIC
    # needs an industry description, ISCED an education/qualification
    # description, matching what a respondent would answer as separate
    # survey questions). gold_isic / gold_isced remain available on the row
    # for scoring the ISIC/ISCED classifiers' own accuracy separately.
    #
    # Conference I Reviewer #2, Step 5.1 (SRE-to-ISIC/ISCED coupling
    # bugfix). ALWAYS runs when industry_text/education_text are present,
    # independent of sre_enabled -- see Documentation/Conference_I_
    # Reviewer_2/SRE_COUPLING_BUGFIX.md. Prior to this fix, ISIC/ISCED
    # classification was incorrectly gated on sre_enabled too, so --sre off
    # silently produced blank ISIC/ISCED predictions for every case.
    #
    # Task 09: isic_clf/isced_clf are None whenever main() determined no
    # selected row has both texts -- guarded explicitly here (not just
    # inferred from the text check) so this block can never be reached
    # with a None classifier even if a future caller's row population
    # diverges from the one main() inspected.
    if isic_clf is not None and isced_clf is not None and industry_text.strip() and education_text.strip():
        try:
            isic_result = isic_clf.classify(industry_text)
            isced_result = isced_clf.classify(education_text)
            result.pred_isic_section = isic_result.section
            result.pred_isic_confidence = isic_result.confidence
            result.pred_isic_division = isic_result.division_code
            result.pred_isic_group = isic_result.group_code
            result.pred_isic_class = isic_result.class_code
            result.pred_isced_level = str(isced_result.level)
            result.pred_isced_confidence = isced_result.confidence
            result.pred_isced_broad = isced_result.broad_code
            result.pred_isced_narrow = isced_result.narrow_code
            result.pred_isced_detailed = isced_result.detailed_code

            # ── SRE coherence: only after base ISIC/ISCED outputs exist,
            # and only when sre_enabled. The SRE's premise is catching
            # *predicted* ISCO/ISIC/ISCED that are jointly implausible
            # despite each looking individually confident -- disabling it
            # must never affect the base classifications above.
            if sre_enabled and sre is not None:
                coherence = sre.analyse(
                    isco_code=result.pred_isco_4digit,
                    isic_section=result.pred_isic_section,
                    isced_level=isced_result.level,
                    job_title=input_text,
                    language="ar" if input_language == "ar" else "en",
                )
                result.sre_coherence_score = coherence.score
                result.sre_severity = _max_severity(coherence.violations)
                result.sre_status = "evaluated"
            else:
                # Explicit disabled status -- never a bare "" or 0.0 that
                # could be misread as "evaluated, no violation found."
                result.sre_status = "disabled_by_configuration"
                result.sre_status_reason = "semantic_relation_engine_disabled_by_configuration"
        except Exception as exc:  # noqa: BLE001
            _logger.warning("case_id=%s ISIC/ISCED/SRE analysis failed: %s", case_id, exc)
            result.sre_status = "error"
            result.sre_status_reason = f"{type(exc).__name__}: {exc}"
    else:
        # No industry_text/education_text on this row -- ISIC/ISCED/SRE are
        # all skipped rather than fed job_title as a stand-in for either
        # (that would be fabricating input the respondent never gave).
        # Current test sets that lack these two columns will show this
        # status for every row; see the module docstring's "SRE coverage"
        # note. Independent of sre_enabled -- this row was never in scope
        # for either ISIC/ISCED classification or SRE, regardless of the
        # --sre flag.
        result.sre_status = "not_applicable"
        result.sre_status_reason = "industry_text and/or education_text not supplied on this row"

    # ── Escalation ───────────────────────────────────────────────────────────
    reasons = []
    if result.pred_confidence is not None and result.pred_confidence < HITL_CONFIDENCE_THRESHOLD:
        reasons.append("confidence")
    if result.sre_severity == "HIGH":
        reasons.append("sre_severity")
    if row_index % STRATIFIED_SAMPLE_EVERY_N == 0:
        reasons.append("stratified_sample")
    result.escalation_triggered = bool(reasons)
    result.escalation_reason = ";".join(reasons)

    return result


# Genuine hierarchical method labels ISCOClassifier ever returns for
# --system hierarchical (see backend/agents/isco_classifier.py's
# method_prefix = "flat" if h.fallback_used else "hierarchical"). Anything
# else recorded in pred_method for a hierarchical-system row means the
# flat fallback fired for that case.
_GENUINE_HIERARCHICAL_METHOD_PREFIX = "hierarchical_"


def check_strict_hierarchical(result: CaseResult, max_stage_latency_ms: Optional[float]) -> Optional[str]:
    """Task 13 --require-genuine-hierarchical guard. Purely evaluative --
    never mutates *result* or reinterprets its fields. Returns None when
    the case is genuine, non-fallback, complete 4-stage hierarchical
    retrieval (within the optional latency bound); otherwise returns a
    human-readable reason naming exactly what failed, for the caller to
    abort the run on."""
    if not result.pred_method.startswith(_GENUINE_HIERARCHICAL_METHOD_PREFIX):
        return (
            f"case_id={result.case_id}: pred_method={result.pred_method!r} is not a "
            f"genuine hierarchical method -- the flat fallback fired for this case"
        )

    for stage_num in (1, 2, 3, 4):
        raw = getattr(result, f"stage{stage_num}_candidates", "")
        if not raw or raw == "null":
            return f"case_id={result.case_id}: stage{stage_num}_candidates is missing/empty"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return f"case_id={result.case_id}: stage{stage_num}_candidates is not valid JSON"
        if not isinstance(parsed, list) or not parsed:
            return f"case_id={result.case_id}: stage{stage_num}_candidates has no entries"

    if max_stage_latency_ms is not None:
        for stage_num in (1, 2, 3, 4):
            latency = getattr(result, f"stage{stage_num}_latency_ms", None)
            if latency is not None and latency > max_stage_latency_ms:
                return (
                    f"case_id={result.case_id}: stage{stage_num}_latency_ms={latency} "
                    f"exceeds --max-stage-latency-ms={max_stage_latency_ms}"
                )

    return None


def _config_hash(args: argparse.Namespace, resolved_reranker_model: str, keyword_map_enabled: bool) -> str:
    """Hash built ONLY from values actually passed into the pipeline for
    this run. A hash that describes a configuration different from the
    one that ran is worse than no hash. resolved_reranker_model must be
    the post-init resolved model string (ISCOClassifier.reranker_model_
    resolved / _BM25Adapter's fixed value), not a config-time guess.
    "beam" and "stage1_mode" are now genuinely threaded through to
    HierarchicalISCOStore.search() (args.beam / args.stage1_mode) as of
    the retrieval-fix session -- previously "beam": 2 was recorded here
    despite no parameter of classify() ever setting it; that was fixed by
    actually wiring the parameter through, not by removing the field.

    Task 09: "use_llm" used to be hardcoded True regardless of
    --use-llm-reranker -- a retrieval-only run's hash still claimed
    use_llm=True even though no LLM was constructed. Now derived from the
    actual flag, so a retrieval-only run's hash (together with
    resolved_reranker_model reporting "none (reranking disabled)" instead
    of a model string) cannot be mistaken for a reranked run's.

    Task 21: "isco_catalogue_profile" makes legacy vs. official-catalogue
    runs unambiguous in the manifest/config hash itself -- two runs that
    differ only in --isco-catalogue-profile always produce different
    hashes, so a legacy run's raw output can never be mistaken for an
    official-profile run's by anyone reading only the config hash."""
    payload = json.dumps(
        {
            "system": args.system,
            "top_k": 5,
            "use_llm": args.use_llm_reranker == "on",
            "llm_temperature": 0.0,
            "beam": args.beam,
            "stage1_mode": args.stage1_mode,
            "reranker_candidates": args.reranker_candidates,
            "branch_collapse": args.branch_collapse,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_model_resolved": resolved_reranker_model,
            "keyword_map_enabled": keyword_map_enabled,
            "hitl_threshold": HITL_CONFIDENCE_THRESHOLD,
            "config_label": args.config,
            "sre": args.sre,
            "use_llm_reranker": args.use_llm_reranker,
            "isco_catalogue_profile": args.isco_catalogue_profile,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", required=True, type=Path, help="Path to the test-set CSV")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases (smoke test)")
    parser.add_argument("--system", choices=["hierarchical", "flat", "bm25"], default="hierarchical")
    parser.add_argument(
        "--reranker-model", type=str, default=None,
        help=(
            "LiteLLM model string for the reranker, e.g. 'ollama/llama3.2:1b' "
            "(free, local -- NOT the model the paper attributes reranking to) "
            "or 'anthropic/claude-3-5-sonnet-20241022' (matches the paper, "
            "costs money, requires ANTHROPIC_API_KEY with available credit). "
            "Required for --system hierarchical/flat ONLY when "
            "--use-llm-reranker is 'on' (the default) -- with 'off', this "
            "flag is ignored entirely and no LLM is constructed at all "
            "(a genuinely retrieval-only run, not just a skipped call). "
            "Health-checked before any case runs; the run aborts (does NOT "
            "silently substitute a different model) if the pinned model is "
            "unavailable. Ignored for --system bm25 (no reranking stage)."
        ),
    )
    parser.add_argument(
        "--beam", type=int, default=2,
        help=(
            "Beam width at hierarchical stages 1-3 (default 2, the previous "
            "hardcoded value -- no prior caller ever overrode it). Recorded "
            "in the config hash. Ignored for --system bm25/flat."
        ),
    )
    parser.add_argument(
        "--stage1-mode", choices=["description", "leaf_vote"], default="description",
        help=(
            "Stage-1 major-group selection method for --system hierarchical, "
            "when the keyword map doesn't fire (major_hint always takes "
            "priority regardless of this flag). 'description' (default): "
            "unchanged prior behaviour, searches isco08_major_groups by its "
            "short official label text. 'leaf_vote': bottom-up alternative, "
            "retrieves top-20 isco08_unit_groups leaves directly and "
            "aggregates by major group via summed cosine. See "
            "HierarchicalISCOStore.search()'s docstring for the full "
            "rationale (Step 1 audit finding: major-group labels embed too "
            "abstractly against specific job titles)."
        ),
    )
    parser.add_argument(
        "--reranker-candidates", type=int, default=5,
        help=(
            "How many unit-group candidates reach the reranker for --system "
            "hierarchical, after pooling across all explored beam branches "
            "(default 5). See --branch-collapse for the pre-fix alternative. "
            "Recorded in the config hash."
        ),
    )
    parser.add_argument(
        "--branch-collapse", action="store_true",
        help=(
            "Reproduce the PRE-FIX candidate-assembly behaviour: only the "
            "winning beam branch's own top-N children are shown to the "
            "reranker, discarding every candidate found in every other "
            "explored branch. This was the only behaviour that existed "
            "before the candidate-pooling fix, and is a quantified defect "
            "(49/82 errors on the 130-case full set were cases where the "
            "correct code was found in a losing branch and silently "
            "discarded). Opt-in only, purely for the before/after ablation "
            "comparison -- default (omitted) is the fixed, pooled behaviour."
        ),
    )
    parser.add_argument(
        "--disable-keyword-map", action="store_true",
        help=(
            "Bypass _keyword_major_hint() entirely -- stage 1 always uses "
            "semantic retrieval, regardless of _MAJOR_KEYWORD_MAP contents. "
            "Nothing else changes (same beam widths, top_k, reranker, "
            "threshold). For the two-arm keyword-map measurement: run once "
            "with this flag omitted (dictionary active, current behaviour) "
            "and once with it passed (dictionary bypassed), same test set, "
            "to measure semantic-only stage-1 accuracy on more than the "
            "handful of cases that happen not to match any entry."
        ),
    )
    parser.add_argument("--config", type=str, default=None,
                         help="Config tag for the output filename (default: same as --system)")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--capture-pool-metadata", action="store_true",
        help=(
            "B2 instrumentation: additionally capture per-candidate branch_id/"
            "source_rank/path/raw+normalized score (backend/rag/candidate_pool.py) "
            "into the new stage4_pool_enriched column. Observational only -- does "
            "NOT change candidate ordering, selection, or any other B0/B1 column. "
            "Default (omitted) leaves stage4_pool_enriched as the default '[]', "
            "identical to every B0/B1 run. Ignored for --system bm25."
        ),
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help=(
            "Identifier stamped into every row's run_id column, for grouping "
            "rows from this invocation across the CSV and --jsonl-output. "
            "Default: the same UTC timestamp used in the output filename."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=(
            "Recorded in every row's seed column for reproducibility bookkeeping "
            "(default: the harness's fixed RANDOM_SEED=42). Not consumed by the "
            "classification pipeline itself, which is deterministic at "
            "temperature=0 -- see RANDOM_SEED's module-level docstring."
        ),
    )
    parser.add_argument(
        "--use-llm-reranker", choices=["on", "off"], default="on",
        help=(
            "Whether the LLM reranking stage runs at all (default: on, "
            "unchanged prior behaviour -- previously hardcoded True with no "
            "flag). 'off' (Task 09: genuinely retrieval-only, not merely "
            "reranker-call-skipping) builds ISCOClassifier with "
            "enable_llm=False -- no LLM/agent is constructed in __init__ at "
            "all, --reranker-model is neither required nor consulted, and "
            "the top pre-rerank candidate is returned directly (method "
            "suffix '_semantic' instead of '_llm'/'hierarchical_llm'). This "
            "is a retrieval-only run, not a reranking comparison, and "
            "supports ISCO-08 classification only -- used by "
            "eval/ablation_runner.py's hierarchical-no-rerank vs "
            "hierarchical-with-rerank configs (Conference I Reviewer #2 "
            "response, Section E). Ignored for --system bm25 (no reranking "
            "stage exists there regardless)."
        ),
    )
    parser.add_argument(
        "--isco-catalogue-profile", choices=[LEGACY_PROFILE, OFFICIAL_PROFILE_ILO2021_V1], default=LEGACY_PROFILE,
        help=(
            "Task 21: which ISCO-08 catalogue/collection identity --system "
            "hierarchical/flat use (ignored for --system bm25). Default "
            "'legacy' is byte-identical to every prior run -- omitting this "
            "flag changes nothing. 'official_ilo2021_v1' selects the "
            "versioned, primary-ILO-sourced collections instead (Task 20/21) "
            "-- --system flat becomes the genuine, direct, unfiltered "
            "official four-digit unit-group comparator rather than the "
            "legacy mixed-granularity isco_occupations collection, and both "
            "systems' pred_method values are prefixed distinctly "
            "(hierarchical_isco08_official_ilo2021_v1 / "
            "flat_isco08_official_ilo2021_v1), never flat_semantic. Legacy "
            "vs. official is always unambiguous in the config hash (see "
            "_config_hash()) and in every row's pred_method -- this run's "
            "raw output can never be mistaken for the other profile's. "
            "Rejected immediately (fail closed) if the requested official "
            "catalogue/collection is unavailable -- never silently falls "
            "back to legacy."
        ),
    )
    parser.add_argument(
        "--sre", choices=["on", "off"], default="on",
        help=(
            "Whether to run the SemanticRelationEngine coherence check for "
            "each case (default: on, unchanged prior behaviour). 'off' "
            "skips ONLY the coherence check itself -- ISCO/ISIC/ISCED "
            "classification still run normally either way (Step 5.1 "
            "bugfix; see Documentation/Conference_I_Reviewer_2/"
            "SRE_COUPLING_BUGFIX.md). With 'off', sre_status is "
            "'disabled_by_configuration', sre_coherence_score/sre_severity "
            "stay empty, and escalation_reason can never include "
            "'sre_severity' -- used by eval/ablation_runner.py's no-SRE vs "
            "with-SRE ablation configs (Conference I Reviewer #2 response, "
            "Section E)."
        ),
    )
    parser.add_argument(
        "--jsonl-output", type=Path, default=None,
        help=(
            "B2: in addition to the standard CSV, write one JSON object per "
            "case to this path (parent directories created as needed). Opt-in "
            "only -- omitted by default, so B0/B1 invocations write no JSONL "
            "at all, exactly as before this flag existed."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Conference I Reviewer #2, Step 4 (evaluation-readiness pass). "
            "Validate every CLI argument and the test-set CSV exactly as a "
            "real run would, write a CaseResult CSV with the same schema and "
            "row count, but call NO classifier -- no ISCOClassifier/"
            "ISICClassifier/ISCEDClassifier is constructed, no Qdrant/LLM/"
            "network call of any kind is made. Every row's evaluation_status "
            "is 'dry_run_not_measured' and every pred_* field is blank/None. "
            "--reranker-model is NOT required in this mode (nothing pings "
            "it). Use this to verify the pipeline is wired correctly before "
            "a real, compute/API-consuming run."
        ),
    )
    parser.add_argument(
        "--require-genuine-hierarchical", action="store_true",
        help=(
            "Task 13: strict evaluation-only guard. Valid only with "
            "--system hierarchical (rejected otherwise). Checks every case "
            "immediately after classification -- a non-hierarchical "
            "pred_method (fallback_used), missing/empty stage 1-4 evidence, "
            "or (with --max-stage-latency-ms) an exceeded stage-latency "
            "threshold aborts the run at once, non-zero exit, with NO "
            "result CSV written -- so a run containing even one fallback or "
            "stalled case can never be mistaken for complete, valid "
            "hierarchical benchmark evidence. Never hides, retries, or "
            "reinterprets a fallback as a successful hierarchical "
            "prediction. Omitted by default -- ordinary runs are completely "
            "unaffected and keep writing their CSV with explicit fallback "
            "labelling as before."
        ),
    )
    parser.add_argument(
        "--max-stage-latency-ms", type=float, default=None,
        help=(
            "Task 13: opt-in: only checked when --require-genuine-hierarchical "
            "is also passed; has no effect otherwise. Must be a positive "
            "number (rejected if zero/negative). If any case's recorded "
            "stage1_latency_ms..stage4_latency_ms exceeds this value, the "
            "strict guard aborts the run naming the exact stage and observed "
            "value. This is a guard against unrepresentative stalled timing "
            "data (see Task 12's report) -- it does not prove what caused "
            "any particular stall."
        ),
    )
    args = parser.parse_args()
    if args.config is None:
        args.config = args.system
    if (
        args.system in ("hierarchical", "flat")
        and not args.reranker_model
        and not args.dry_run
        and args.use_llm_reranker == "on"
    ):
        parser.error(
            "--reranker-model is required for --system hierarchical/flat "
            "when --use-llm-reranker is 'on' (the default) -- no default "
            "reranker (the choice changes what the run's numbers mean); "
            "pass e.g. 'ollama/llama3.2:1b' for a smoke test or "
            "'anthropic/claude-3-5-sonnet-20241022' to match the paper. "
            "Pass --use-llm-reranker off for a genuinely retrieval-only "
            "run that needs no reranker model at all."
        )

    if args.require_genuine_hierarchical and args.system != "hierarchical":
        parser.error(
            "--require-genuine-hierarchical is valid only with --system hierarchical "
            f"(got --system {args.system!r})."
        )
    if args.max_stage_latency_ms is not None and args.max_stage_latency_ms <= 0:
        parser.error(
            f"--max-stage-latency-ms must be a positive number (got {args.max_stage_latency_ms!r})."
        )

    if not args.test_set.exists():
        parser.error(f"Test set not found: {args.test_set}")

    with open(args.test_set, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    for required_col in ("case_id", "input_text", "input_language", "gold_isco_4digit"):
        if rows and required_col not in rows[0]:
            parser.error(f"Test set is missing required column: {required_col}")

    print(f"Loaded {len(rows)} case(s) from {args.test_set}  |  system={args.system}")

    # Moved earlier than the original post-loop computation so --run-id can
    # default to it and so it's available for both the CSV and --jsonl-output
    # filenames -- the value and the CSV filename it produces are unchanged
    # from before this move (same format, same point in wall-clock time
    # relative to the run), so B0/B1 output is identical either way.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or timestamp

    git_commit = ""
    try:
        import subprocess
        git_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - best-effort bookkeeping, never fatal to the run
        _logger.debug("Could not resolve git_commit: %s", exc)

    if args.dry_run:
        # No ISCOClassifier/ISICClassifier/ISCEDClassifier construction at
        # all -- ISCOClassifier connects to Qdrant and health-checks the
        # reranker LLM in __init__; ISICClassifier calls get_llm() (which
        # probes Ollama) in __init__. Neither is safe to construct in a
        # network-free dry run, so neither is constructed. resolved_
        # reranker_model / keyword_map_enabled still get real values (pure
        # string/bool logic, no I/O) so the config hash and printed summary
        # are identical in shape to a real run.
        resolved_reranker_model = args.reranker_model or "(not pinned -- dry run)"
        keyword_map_enabled = not args.disable_keyword_map
        cfg_hash = _config_hash(args, resolved_reranker_model, keyword_map_enabled)
        print(f"[DRY RUN] config_hash={cfg_hash}  reranker_model={resolved_reranker_model}  "
              f"keyword_map_enabled={keyword_map_enabled}  run_id={run_id}")
        print("[DRY RUN] no classifier constructed, no Qdrant/LLM/network call will be made")

        results: list[CaseResult] = []
        t_run_start = time.perf_counter()
        for i, row in enumerate(rows):
            r = build_dry_run_case_result(
                row_index=i,
                case_id=row["case_id"],
                input_text=row["input_text"],
                input_language=row.get("input_language", ""),
                gold_isco_4digit=row.get("gold_isco_4digit", ""),
                gold_isic=row.get("gold_isic", ""),
                gold_isced=row.get("gold_isced", ""),
                config_hash=cfg_hash,
                run_id=run_id,
                git_commit=git_commit,
                seed=args.seed,
            )
            results.append(r)
            print(f"[DRY RUN] [{i + 1}/{len(rows)}] case_id={r.case_id} gold={r.gold_isco_4digit} "
                  f"status=dry_run_not_measured")

        total_s = time.perf_counter() - t_run_start
        print(f"\n[DRY RUN] Completed {len(results)} case(s) (validation only) in {total_s:.2f}s")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / f"{timestamp}_{args.config}.csv"
        fieldnames = list(CaseResult.__dataclass_fields__.keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.__dict__)
        # Matches the exact "Wrote N row(s) to <path.csv>" shape a real run
        # prints -- eval/ablation_runner.py's _find_written_csv() parses
        # this line first (falling back to a directory glob only if it
        # doesn't match), so dry-run output must remain parseable the same
        # way. The dry-run/evaluation_status disclosure is a separate line.
        print(f"\nWrote {len(results)} row(s) to {out_path}")
        print("[DRY RUN] evaluation_status=dry_run_not_measured for all rows -- no classification was performed")

        if args.jsonl_output:
            args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.jsonl_output, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
            print(f"[DRY RUN] Wrote {len(results)} row(s) to {args.jsonl_output}")
        return

    clf = build_system(args.system, reranker_model=args.reranker_model,
                        disable_keyword_map=args.disable_keyword_map,
                        beam=args.beam, stage1_mode=args.stage1_mode,
                        reranker_candidates=args.reranker_candidates,
                        branch_collapse=args.branch_collapse,
                        capture_pool_metadata=args.capture_pool_metadata,
                        use_llm_reranker=(args.use_llm_reranker == "on"),
                        isco_catalogue_profile=args.isco_catalogue_profile)

    # Task 21: fail closed BEFORE running any case if an official profile
    # was explicitly requested but its collections are unavailable --
    # never silently proceed row-by-row into a RuntimeError per case, and
    # never silently fall back to a legacy result.
    if (
        args.isco_catalogue_profile != LEGACY_PROFILE
        and args.system in ("hierarchical", "flat")
        and getattr(clf, "_hierarchical_store", None) is None
    ):
        print(
            f"ERROR: --isco-catalogue-profile={args.isco_catalogue_profile!r} was requested "
            "but its collections are unavailable -- refusing to run (no legacy fallback "
            "for an explicitly-requested official profile). No result CSV was written.",
            file=sys.stderr,
        )
        sys.exit(1)

    resolved_reranker_model = getattr(clf, "reranker_model_resolved", "")
    keyword_map_enabled = not args.disable_keyword_map

    # Task 09: ISICClassifier/ISCEDClassifier/SemanticRelationEngine are
    # only constructed when at least one selected row (post --limit, same
    # population the loop below iterates) has both non-blank industry_text
    # and education_text -- otherwise every row would hit run_one_case()'s
    # "not_applicable" branch anyway, so constructing them (which, for
    # ISICClassifier, initialises an LLM -- see get_llm(TaskType.GENERAL)
    # in its __init__) would be pure unused overhead for a WISCO-style,
    # ISCO-only test set. Both are set to the *same* condition; see
    # run_one_case()'s own None-guard for the corresponding per-row check.
    any_row_has_paired_industry_education = any(
        row.get("industry_text", "").strip() and row.get("education_text", "").strip()
        for row in rows
    )
    if any_row_has_paired_industry_education:
        isic_clf = ISICClassifier()
        isced_clf = ISCEDClassifier()
        sre = SemanticRelationEngine(use_llm=False)  # deterministic crosswalk only, no LLM disambiguation
    else:
        isic_clf = None
        isced_clf = None
        sre = None
        print("No row has both industry_text and education_text -- ISICClassifier/"
              "ISCEDClassifier/SemanticRelationEngine will not be constructed; "
              "this is an ISCO-08-only retrieval run.")

    cfg_hash = _config_hash(args, resolved_reranker_model, keyword_map_enabled)
    print(f"config_hash={cfg_hash}  reranker_model={resolved_reranker_model or '(none)'}  "
          f"keyword_map_enabled={keyword_map_enabled}  run_id={run_id}")
    if args.capture_pool_metadata:
        print(f"B2: capture_pool_metadata enabled  seed={args.seed}"
              + (f"  jsonl_output={args.jsonl_output}" if args.jsonl_output else ""))

    # Task 31: the strict stage-level deadline budget is threaded into
    # live retrieval ONLY when both --system hierarchical and
    # --require-genuine-hierarchical are active -- the exact same
    # documented relationship --max-stage-latency-ms already has to
    # --require-genuine-hierarchical for the (still-unchanged) post-hoc
    # check_strict_hierarchical() guard below. Every other invocation
    # (including a plain --system hierarchical run without the strict
    # flag) passes None and sees zero behavioural change.
    effective_stage_budget_ms = (
        args.max_stage_latency_ms
        if (args.system == "hierarchical" and args.require_genuine_hierarchical)
        else None
    )

    results: list[CaseResult] = []
    t_run_start = time.perf_counter()
    for i, row in enumerate(rows):
        r = run_one_case(
            clf, sre, isic_clf, isced_clf, row_index=i,
            case_id=row["case_id"],
            input_text=row["input_text"],
            input_language=row.get("input_language", ""),
            gold_isco_4digit=row.get("gold_isco_4digit", ""),
            gold_isic=row.get("gold_isic", ""),
            gold_isced=row.get("gold_isced", ""),
            config_hash=cfg_hash,
            system=args.system,
            industry_text=row.get("industry_text", ""),
            education_text=row.get("education_text", ""),
            keyword_map_enabled=keyword_map_enabled,
            branch_collapse_enabled=args.branch_collapse,
            capture_pool_metadata_enabled=args.capture_pool_metadata,
            run_id=run_id,
            git_commit=git_commit,
            seed=args.seed,
            input_order_position=i,
            sre_enabled=(args.sre == "on"),
            use_llm_reranker=(args.use_llm_reranker == "on"),
            max_stage_latency_ms=effective_stage_budget_ms,
        )

        if args.require_genuine_hierarchical:
            violation = check_strict_hierarchical(r, args.max_stage_latency_ms)
            if violation is not None:
                print(
                    f"\nSTRICT GUARD FAILURE (--require-genuine-hierarchical): {violation}",
                    file=sys.stderr,
                )
                print(
                    "Aborting immediately -- no result CSV written. This run cannot "
                    "be used as genuine hierarchical benchmark evidence.",
                    file=sys.stderr,
                )
                sys.exit(1)

        results.append(r)
        status = "ERROR" if r.error and not r.pred_isco_4digit else "ok"
        print(
            f"[{i + 1}/{len(rows)}] case_id={r.case_id} gold={r.gold_isco_4digit} "
            f"pred={r.pred_isco_4digit} conf={r.pred_confidence} status={status}"
        )

    total_s = time.perf_counter() - t_run_start
    print(f"\nCompleted {len(results)} case(s) in {total_s:.1f}s "
          f"({total_s / max(len(results), 1):.2f}s/case)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{timestamp}_{args.config}.csv"

    fieldnames = list(CaseResult.__dataclass_fields__.keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)

    n_errors = sum(1 for r in results if r.error and not r.pred_isco_4digit)
    print(f"\nWrote {len(results)} row(s) to {out_path}")
    if n_errors:
        print(f"WARNING: {n_errors} case(s) failed outright (see 'error' column) — run was not lost, just those rows.")

    # B2: opt-in JSONL, in addition to (never instead of) the CSV above.
    # Omitted --jsonl-output => this block does not run => zero difference
    # from pre-B2 behaviour, for B0/B1 or any B2 run that doesn't ask for it.
    if args.jsonl_output:
        args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.jsonl_output, "w", encoding="utf-8") as f:
            for r in results:
                record = dict(r.__dict__)
                for json_field in ("stage1_candidates", "stage2_candidates", "stage3_candidates",
                                    "stage4_candidates", "reranker_input_candidates", "reranker_output",
                                    "stage4_pool", "stage4_pool_enriched"):
                    try:
                        record[json_field] = json.loads(record[json_field])
                    except (json.JSONDecodeError, TypeError):
                        pass  # leave as the raw string if it wasn't valid JSON for some reason
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(results)} row(s) to {args.jsonl_output}")


if __name__ == "__main__":
    main()
