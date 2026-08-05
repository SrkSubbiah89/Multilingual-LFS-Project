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

Reranker pinning (--reranker-model)
--------------------------------------
Required for --system hierarchical/flat. Passed to
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
                 use_llm: bool = True, trace: Optional[dict] = None):
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
                  capture_pool_metadata: bool = False):
    """Return a classifier object exposing .classify(...) for the requested
    --system value. hierarchical and flat share ISCOClassifier (same class,
    force_flat toggles the retrieval path); bm25 uses the adapter above.

    reranker_model is required (non-None) for hierarchical/flat -- see
    ISCOClassifier's reranker_model parameter: it pins the reranking LLM via
    get_llm_strict() (hard-fail if unavailable, no silent substitution) and
    the exception from an unavailable pinned model propagates out of this
    call, aborting the run before any case is classified. Ignored for bm25,
    which has no reranking stage at all. beam/stage1_mode/reranker_candidates/
    branch_collapse are likewise ignored for bm25 (no stages) and unused by
    flat (single-stage retrieval, no branches to pool across).

    capture_pool_metadata (B2, default False): opt-in, observational only --
    see HierarchicalISCOStore.search()'s capture_pool_metadata docstring.
    Ignored for bm25 (no stage4 pool to capture)."""
    if system == "hierarchical":
        return ISCOClassifier(llm_temperature=0.0, reranker_model=reranker_model,
                               disable_keyword_map=disable_keyword_map,
                               beam=beam, stage1_mode=stage1_mode,
                               reranker_candidates=reranker_candidates,
                               branch_collapse=branch_collapse,
                               capture_pool_metadata=capture_pool_metadata)
    if system == "flat":
        return ISCOClassifier(llm_temperature=0.0, force_flat=True, reranker_model=reranker_model,
                               disable_keyword_map=disable_keyword_map,
                               beam=beam, stage1_mode=stage1_mode,
                               reranker_candidates=reranker_candidates,
                               branch_collapse=branch_collapse,
                               capture_pool_metadata=capture_pool_metadata)
    if system == "bm25":
        return _BM25Adapter()
    raise ValueError(f"Unknown --system {system!r}; expected hierarchical, flat, or bm25")


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

    sre_coherence_score: Optional[float] = None
    sre_severity: str = ""

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


def run_one_case(
    clf,  # ISCOClassifier (hierarchical/flat) or _BM25Adapter -- duck-typed on .classify()
    sre: SemanticRelationEngine,
    isic_clf: ISICClassifier,
    isced_clf: ISCEDClassifier,
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
) -> CaseResult:
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
            use_llm=True,
            trace=trace,
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

    # ── SRE coherence ────────────────────────────────────────────────────────
    # The SRE's premise is catching *predicted* ISCO/ISIC/ISCED that are
    # jointly implausible despite each looking individually confident.
    # Feeding it gold ISIC/ISCED (as this used to do) makes the ISIC/ISCED
    # side correct by construction and systematically understates
    # incoherence -- every complementarity number downstream would be
    # built on a system that never actually ran ISIC/ISCED classification.
    # So: run the ISIC and ISCED classifiers on their own free-text inputs
    # (industry_text / education_text -- these are NOT the job title; ISIC
    # needs an industry description, ISCED an education/qualification
    # description, matching what a respondent would answer as separate
    # survey questions) and feed *those* predictions to the SRE. gold_isic
    # / gold_isced remain available on the row for scoring the ISIC/ISCED
    # classifiers' own accuracy separately -- they are never passed to SRE.
    if industry_text.strip() and education_text.strip():
        try:
            isic_result = isic_clf.classify(industry_text)
            isced_result = isced_clf.classify(education_text)
            result.pred_isic_section = isic_result.section
            result.pred_isic_confidence = isic_result.confidence
            result.pred_isced_level = str(isced_result.level)
            result.pred_isced_confidence = isced_result.confidence

            coherence = sre.analyse(
                isco_code=result.pred_isco_4digit,
                isic_section=result.pred_isic_section,
                isced_level=isced_result.level,
                job_title=input_text,
                language="ar" if input_language == "ar" else "en",
            )
            result.sre_coherence_score = coherence.score
            result.sre_severity = _max_severity(coherence.violations)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("case_id=%s SRE analysis failed: %s", case_id, exc)
    # else: no industry_text/education_text on this row -- SRE is skipped
    # entirely rather than fed job_title as a stand-in for either (that
    # would be fabricating input the respondent never gave). Current test
    # sets that lack these two columns will show sre_coherence_score as
    # empty for every row; see the module docstring's "SRE coverage" note.

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
    actually wiring the parameter through, not by removing the field."""
    payload = json.dumps(
        {
            "system": args.system,
            "top_k": 5,
            "use_llm": True,
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
            "Required for --system hierarchical/flat. Health-checked before "
            "any case runs; the run aborts (does NOT silently substitute a "
            "different model) if the pinned model is unavailable. Ignored "
            "for --system bm25 (no reranking stage)."
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
        "--jsonl-output", type=Path, default=None,
        help=(
            "B2: in addition to the standard CSV, write one JSON object per "
            "case to this path (parent directories created as needed). Opt-in "
            "only -- omitted by default, so B0/B1 invocations write no JSONL "
            "at all, exactly as before this flag existed."
        ),
    )
    args = parser.parse_args()
    if args.config is None:
        args.config = args.system
    if args.system in ("hierarchical", "flat") and not args.reranker_model:
        parser.error(
            "--reranker-model is required for --system hierarchical/flat "
            "(no default -- the choice of reranker changes what the run's "
            "numbers mean; pass e.g. 'ollama/llama3.2:1b' for a smoke test "
            "or 'anthropic/claude-3-5-sonnet-20241022' to match the paper)."
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

    clf = build_system(args.system, reranker_model=args.reranker_model,
                        disable_keyword_map=args.disable_keyword_map,
                        beam=args.beam, stage1_mode=args.stage1_mode,
                        reranker_candidates=args.reranker_candidates,
                        branch_collapse=args.branch_collapse,
                        capture_pool_metadata=args.capture_pool_metadata)
    resolved_reranker_model = getattr(clf, "reranker_model_resolved", "")
    keyword_map_enabled = not args.disable_keyword_map
    isic_clf = ISICClassifier()
    isced_clf = ISCEDClassifier()
    sre = SemanticRelationEngine(use_llm=False)  # deterministic crosswalk only, no LLM disambiguation
    cfg_hash = _config_hash(args, resolved_reranker_model, keyword_map_enabled)
    print(f"config_hash={cfg_hash}  reranker_model={resolved_reranker_model or '(none)'}  "
          f"keyword_map_enabled={keyword_map_enabled}  run_id={run_id}")
    if args.capture_pool_metadata:
        print(f"B2: capture_pool_metadata enabled  seed={args.seed}"
              + (f"  jsonl_output={args.jsonl_output}" if args.jsonl_output else ""))

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
        )
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
