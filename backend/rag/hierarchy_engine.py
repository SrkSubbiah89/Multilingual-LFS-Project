"""
backend/rag/hierarchy_engine.py

Generic, stage-count-agnostic multi-stage beam-search retrieval engine,
extracted from the ISCO-08-specific hand-unrolled 4-stage loop that used to
live entirely inside ``HierarchicalISCOStore._hierarchical_search``
(backend/rag/hierarchical_store.py).

This module contains no ISCO-, ISIC-, or ISCED-specific logic. A caller
configures it with an ordered list of ``StageConfig`` (collection name +
confidence weight per level) and gets back a stage-count-agnostic
``EngineResult``. ``HierarchicalISCOStore`` now builds one instance of this
engine with ISCO's 4 stages and translates between ``EngineResult`` and its
own public ``HierarchicalResult`` dataclass -- see hierarchical_store.py's
``_hierarchical_search`` for the translation layer. That refactor is
behavior-preserving: every documented knob of the original ISCO pipeline
(major_hint / stage1_mode="leaf_vote" / branch_collapse /
capture_pool_metadata / trace) has a direct generic equivalent here (seed /
stage_overrides / branch_collapse / capture_pool_metadata / trace), and the
trace dict's key names ("stage1".."stageN", "stageN_latency_ms",
"stageN_pool", "stageN_pool_enriched") are produced generically as
f"stage{i+1}", which reproduces ISCO's exact "stage1".."stage4" names for a
4-stage configuration.

Requires at least 2 stages (stage 0 is always an unfiltered/seeded/overridden
"root" query; stages 1..N-1 are parent-filtered expansions, with the last
stage also serving as the pooled/reranked candidate stage). A single-stage
config is out of scope -- every stage after the first needs a parent to
filter on, which a length-1 stage list cannot provide.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import FieldCondition, Filter, MatchValue

from backend.rag.candidate_pool import pool_and_rank_candidates

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task 27: bounded, opt-in Qdrant query retry configuration
# ---------------------------------------------------------------------------
#
# Task 24 and Task 26 each independently observed exactly one transient
# Qdrant query failure during an otherwise-clean ~18,700-case sustained
# run (Task 24: the official flat collection; Task 26: the official
# major-groups collection at hierarchical stage 1). Both were logged by
# HierarchyBeamSearchEngine._query()'s existing except-block with message
# text ending "failed: timed out". Neither incident's server/network root
# cause is known or claimed here -- this module only makes the *class* of
# failure retryable, narrowly and transparently, never masks it.
#
# Default is UNCHANGED behaviour: exactly one attempt, no retry, no
# backoff -- byte-for-byte identical to every prior caller that does not
# opt in.
DEFAULT_MAX_QUERY_ATTEMPTS = 1

# Hard upper bound enforced in code (see HierarchyBeamSearchEngine.__init__
# and _resolve_max_query_attempts()). Justified in
# Documentation/AI_HANDOFF/CLAUDE_TASK_27_FINAL_REPORT.md: 1 initial
# attempt + at most 2 retries. A transient client-side timeout is either
# gone within a couple of attempts or reflects a real outage a bounded
# retry cannot paper over; a benchmark run over ~18,700 cases must not
# risk an unbounded or large per-case multiplier if timeouts turn out to
# be correlated (e.g. sustained server-side load) rather than independent.
MAX_QUERY_ATTEMPTS_HARD_CAP = 3

# Default is UNCHANGED behaviour: zero backoff (only relevant once
# max_query_attempts > 1, which is itself opt-in).
DEFAULT_RETRY_BACKOFF_SECONDS = 0.0

# Hard upper bound enforced in code. A bounded, small, fixed backoff --
# not exponential, not unbounded -- so a worst-case retried case adds at
# most (MAX_QUERY_ATTEMPTS_HARD_CAP - 1) * MAX_RETRY_BACKOFF_SECONDS to
# its own latency, never to the whole run's structure.
MAX_RETRY_BACKOFF_SECONDS_HARD_CAP = 2.0


def _is_retryable_exception(exc: BaseException) -> bool:
    """
    Task 27: strict, type-based (never message-substring-based) retry
    eligibility check for the exact exception taxonomy the installed
    qdrant-client (1.17.0, REST/httpx transport -- no grpc client is
    constructed anywhere in this codebase, confirmed by inspection) can
    raise from a read-only query call:

    - ``httpx.TimeoutException`` (and its subclasses ConnectTimeout /
      ReadTimeout / WriteTimeout / PoolTimeout) directly, if a future
      call path ever surfaces one unwrapped.
    - ``qdrant_client.http.exceptions.ResponseHandlingException``, but
      ONLY when its wrapped ``.source`` cause (qdrant_client's own
      ``ApiClient.send_inner()`` wraps *any* exception the underlying
      httpx client raises, including a successfully-parsed-but-
      schema-invalid response's ``pydantic.ValidationError`` -- see
      Task 27's final report for the full audited call path) is itself
      an ``httpx.TimeoutException``. A ``ResponseHandlingException``
      wrapping anything else (a parse/validation error, a connection
      reset, etc.) is deliberately NOT retried -- retrying a malformed-
      response or non-timeout transport failure would not plausibly
      help and could mask a real data/schema defect.

    Explicitly NOT retryable (fails closed, single attempt, existing
    behaviour): ``qdrant_client.http.exceptions.UnexpectedResponse``
    (any non-2xx HTTP status -- programmer/validation/auth errors),
    ``ResourceExhaustedResponse`` (HTTP 429 rate-limiting -- a different
    signal than a timeout, out of this task's narrow scope), any other
    ``ApiException`` subclass, and any exception type not named above,
    including bare ``Exception``.
    """
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, ResponseHandlingException):
        source = getattr(exc, "source", None)
        if isinstance(source, httpx.TimeoutException):
            return True
    return False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    """One level of the hierarchy: its Qdrant collection and the weight its
    score contributes to the final confidence."""
    name: str
    collection: str
    weight: float
    parent_field: str = "parent_code"


@dataclass
class SeedSpec:
    """Bypasses stage 0's Qdrant query with a fixed candidate list (e.g.
    ISCO's ``major_hint`` keyword-map lookup)."""
    candidates: list[tuple[str, float]]
    source_label: str = "seed"
    candidate_label: str = "(seed candidate, search skipped)"


@dataclass
class StageOverride:
    """Replaces a stage's normal parent-filtered Qdrant query with a custom
    callable (e.g. ISCO's bottom-up ``leaf_vote`` stage-1 strategy)."""
    fn: Callable[[list[float], int], list[tuple[str, float]]]
    source_label: str = "override"
    candidate_label: str = "(override aggregate)"


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class EngineCandidate:
    code: str
    label_en: str
    label_ar: str
    score: float


@dataclass
class EngineResult:
    code: str
    label_en: str
    label_ar: str
    confidence: float
    stage_confidences: dict
    hierarchy_path: list = field(default_factory=list)
    top_candidates: list = field(default_factory=list)
    hitl_required: bool = False
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class HierarchyBeamSearchEngine:
    """Generic N-stage beam-search retrieval over a chain of Qdrant
    collections, each level filtered by the previous level's chosen code."""

    def __init__(
        self,
        client: QdrantClient,
        stages: list[StageConfig],
        hitl_threshold: float,
        max_query_attempts: int = DEFAULT_MAX_QUERY_ATTEMPTS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        query_timeout_seconds: Optional[float] = None,
    ) -> None:
        """
        max_query_attempts : int, default 1 (Task 27)
            Total attempts (including the first) for a single Qdrant
            query call before it is treated as failed. 1 (the default)
            reproduces prior behaviour exactly -- no retry loop runs at
            all. Must be an int in [1, MAX_QUERY_ATTEMPTS_HARD_CAP]
            (currently 3); any other value raises ValueError immediately
            at construction -- fail closed on explicit misuse, mirroring
            UnknownISCOCatalogueProfileError's precedent in
            hierarchical_store.py. (Ambient environment misconfiguration
            -- QDRANT_QUERY_MAX_ATTEMPTS -- is handled separately by
            hierarchical_store._resolve_max_query_attempts(), which
            fails SAFE to the default instead of raising, matching
            _resolve_qdrant_timeout_seconds()'s existing precedent.)
        retry_backoff_seconds : float, default 0.0 (Task 27)
            Fixed (never exponential, never unbounded) delay between
            attempts, only ever consulted when max_query_attempts > 1.
            Must be a number in [0, MAX_RETRY_BACKOFF_SECONDS_HARD_CAP]
            (currently 2.0); any other value raises ValueError
            immediately at construction.
        query_timeout_seconds : float, optional (Task 31)
            The per-request timeout value passed explicitly to every
            ``client.query_points(..., timeout=...)`` call (see
            ``_query()``'s docstring and Task 31's final report for the
            full audited explanation of what this Qdrant REST parameter
            actually controls -- a server-side operation-timeout hint,
            not a client-side socket timeout). Default ``None`` means no
            explicit ``timeout=`` is passed at all, reproducing every
            prior caller's exact behaviour (the client's own
            constructor-level default, if any, still applies). Must be a
            positive number or ``None``; any other value raises
            ``ValueError`` immediately at construction. When a stage
            budget is active for a given call (see ``search()``'s
            ``max_stage_latency_ms``), the actual per-attempt timeout
            used is the smaller of this value and the remaining stage
            budget -- see ``_query()``.
        """
        if len(stages) < 2:
            raise ValueError(
                "HierarchyBeamSearchEngine requires at least 2 stages "
                f"(got {len(stages)}); every stage after the first needs a "
                "parent to filter on."
            )
        if not isinstance(max_query_attempts, int) or isinstance(max_query_attempts, bool) or not (
            1 <= max_query_attempts <= MAX_QUERY_ATTEMPTS_HARD_CAP
        ):
            raise ValueError(
                f"max_query_attempts must be an int in [1, {MAX_QUERY_ATTEMPTS_HARD_CAP}]; "
                f"got {max_query_attempts!r}"
            )
        if not isinstance(retry_backoff_seconds, (int, float)) or isinstance(retry_backoff_seconds, bool) or not (
            0 <= retry_backoff_seconds <= MAX_RETRY_BACKOFF_SECONDS_HARD_CAP
        ):
            raise ValueError(
                f"retry_backoff_seconds must be a number in [0, {MAX_RETRY_BACKOFF_SECONDS_HARD_CAP}]; "
                f"got {retry_backoff_seconds!r}"
            )
        if query_timeout_seconds is not None and (
            not isinstance(query_timeout_seconds, (int, float))
            or isinstance(query_timeout_seconds, bool)
            or query_timeout_seconds <= 0
        ):
            raise ValueError(
                f"query_timeout_seconds must be a positive number or None; "
                f"got {query_timeout_seconds!r}"
            )
        self._client = client
        self.stages = stages
        self.hitl_threshold = hitl_threshold
        self.max_query_attempts = max_query_attempts
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.query_timeout_seconds = (
            float(query_timeout_seconds) if query_timeout_seconds is not None else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query_vec: list[float],
        top_k: int,
        beam: int = 2,
        seed: Optional[SeedSpec] = None,
        stage_overrides: Optional[dict[int, StageOverride]] = None,
        reranker_candidates: int = 5,
        branch_collapse: bool = False,
        capture_pool_metadata: bool = False,
        trace: Optional[dict] = None,
        max_stage_latency_ms: Optional[float] = None,
    ) -> Optional[EngineResult]:
        """
        Run the beam search. Returns ``None`` only if every explored branch
        returns zero hits at the final stage (caller decides the fallback).

        max_stage_latency_ms : float, optional (Task 31)
            Opt-in, clearly-named strict stage deadline budget in
            milliseconds. Default ``None`` preserves prior behaviour
            exactly -- no deadline is ever established, and this method
            behaves byte-for-byte as before this parameter existed. When
            provided, a single fresh monotonic deadline
            (``time.monotonic() + max_stage_latency_ms / 1000``) is
            established once at the START of each stage's processing
            (stage 0's own query, and each of stages 1..N-1's parent-
            filtered beam expansion) and shared across EVERY beam-branch
            query, retry, and retry backoff issued for that stage --
            never per-query, never reset mid-stage. Before every branch
            query and before every retry/backoff sleep, the remaining
            budget is computed; a query is never started and a backoff
            is never slept once the budget is exhausted (see
            ``_query()``). On exhaustion, the affected query returns an
            empty hit list with ``query_telemetry["outcome"] ==
            "stage_budget_exhausted"`` -- never a fabricated candidate --
            which naturally propagates through this method's existing
            zero-hits handling (``if not hits: continue`` /
            ``if hits0 is empty: return None``) into the SAME existing
            flat-fallback path already used for a genuine zero-hit
            response, an exception, or retry exhaustion. No new
            fallback-detection logic is required: the caller's existing
            ``pred_method`` prefix check (unmodified since Task 13)
            already rejects any state that is not genuine, complete
            hierarchical evidence, regardless of which of these causes
            produced it.
        """
        stage_overrides = stage_overrides or {}
        n = len(self.stages)

        def _trace_hits(stage_idx: int, hits) -> None:
            if trace is None:
                return
            key = f"stage{stage_idx + 1}"
            bucket = trace.setdefault(key, [])
            seen = {c["code"] for c in bucket}
            for hit in hits:
                code = hit.payload.get("code", "")
                if code in seen:
                    continue
                seen.add(code)
                bucket.append({
                    "code": code,
                    "label_en": hit.payload.get("label_en", ""),
                    "score": round(float(hit.score), 4),
                })

        def _stage_deadline_for(stage_idx: int) -> Optional[float]:
            # Task 31: one fresh deadline established at the start of
            # THIS stage's processing, shared by every branch query,
            # retry, and backoff issued for this stage_idx -- computed
            # once per call site (stage 0's single query, or once per
            # stage_idx before its branch loop begins), never per-query.
            if max_stage_latency_ms is None:
                return None
            return time.monotonic() + (max_stage_latency_ms / 1000.0)

        def _timed_query(stage_idx: int, stage_deadline_monotonic: Optional[float] = None, **kwargs):
            key = f"stage{stage_idx + 1}"
            if trace is None:
                return self._query(stage_deadline_monotonic=stage_deadline_monotonic, **kwargs)
            t0 = time.perf_counter()
            query_telemetry: dict = {}
            result = self._query(
                query_telemetry=query_telemetry,
                stage_deadline_monotonic=stage_deadline_monotonic,
                **kwargs,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            lk = f"{key}_latency_ms"
            # Task 13: this already includes the full duration of every
            # attempt and any Task 27 retry backoff, since _query()'s own
            # internal retry loop is inside the single self._query() call
            # timed here -- --max-stage-latency-ms is never blind to
            # retry/backoff time. Task 31: also includes any time spent
            # before a budget-exhaustion short-circuit -- elapsed_ms is
            # measured around the whole _query() call regardless of outcome.
            trace[lk] = trace.get(lk, 0.0) + elapsed_ms
            # Task 27/31: separate, clearly-named per-stage retry/exception/
            # budget telemetry -- never written into the flat_query_* fields
            # (those are exclusively HierarchicalISCOStore._flat_search()'s).
            # A stage may be queried once per explored beam branch, so
            # this accumulates (never overwrites) across every query call
            # at this stage.
            tk = f"{key}_query_telemetry"
            summary = trace.setdefault(tk, {
                "queries": 0, "any_retry": False, "any_exception": False,
                "max_attempts_used": 0, "exception_types": [],
                "stage_budget_exhausted": False,
                "configured_query_timeout_seconds": self.query_timeout_seconds,
                "initial_stage_budget_ms": max_stage_latency_ms,
                "queries_detail": [],
            })
            summary["queries"] += 1
            summary["max_attempts_used"] = max(summary["max_attempts_used"], query_telemetry.get("attempts", 0))
            outcome = query_telemetry.get("outcome")
            if outcome == "success_after_retry":
                summary["any_retry"] = True
            if outcome in ("exception", "retry_exhausted"):
                summary["any_exception"] = True
                exc_type = query_telemetry.get("exception_type")
                if exc_type and exc_type not in summary["exception_types"]:
                    summary["exception_types"].append(exc_type)
            if outcome == "stage_budget_exhausted":
                summary["stage_budget_exhausted"] = True
            summary["queries_detail"].append({
                "outcome": outcome,
                "attempts": query_telemetry.get("attempts", 0),
                "attempt_durations_ms": query_telemetry.get("attempt_durations_ms", []),
                "remaining_stage_budget_ms_at_entry": query_telemetry.get("remaining_stage_budget_ms_at_entry"),
            })
            return result

        # ── Stage 0 ──────────────────────────────────────────────────
        override0 = stage_overrides.get(0)
        if seed is not None:
            s0_candidates = list(seed.candidates)
            if not s0_candidates:
                return None
            if trace is not None:
                trace.setdefault("stage1", []).extend([
                    {"code": code, "label_en": seed.candidate_label, "score": round(score, 4)}
                    for code, score in s0_candidates
                ])
                trace["stage1_latency_ms"] = trace.get("stage1_latency_ms", 0.0)
                trace["stage1_source"] = seed.source_label
        elif override0 is not None:
            t0 = time.perf_counter()
            s0_candidates = override0.fn(query_vec, beam)
            if trace is not None:
                trace["stage1_latency_ms"] = trace.get("stage1_latency_ms", 0.0) + (time.perf_counter() - t0) * 1000
                trace.setdefault("stage1", []).extend([
                    {"code": code, "label_en": override0.candidate_label, "score": round(score, 4)}
                    for code, score in s0_candidates
                ])
                trace["stage1_source"] = override0.source_label
            if not s0_candidates:
                return None
        else:
            hits0 = _timed_query(
                0, collection=self.stages[0].collection, query_vec=query_vec, limit=beam,
                stage_deadline_monotonic=_stage_deadline_for(0),
            )
            if not hits0:
                return None
            _trace_hits(0, hits0)
            s0_candidates = [(h.payload.get("code", ""), float(h.score)) for h in hits0]
            if trace is not None:
                trace["stage1_source"] = "semantic_retrieval"

        branches = [{"codes": [code], "scores": [score]} for code, score in s0_candidates]

        final_pool: dict[str, object] = {}
        pool_branches = 0
        branch_hits_plain: Optional[list[list[dict]]] = [] if capture_pool_metadata else None

        best: Optional[dict] = None
        best_final_score = -1.0

        # ── Stages 1..N-1: parent-filtered beam expansion ──────────────
        frontier = branches
        for stage_idx in range(1, n):
            stage = self.stages[stage_idx]
            is_final = stage_idx == n - 1
            next_frontier: list[dict] = []
            # Task 31: ONE deadline for this stage_idx, shared by every
            # branch below (not recomputed per branch) -- see
            # _stage_deadline_for()'s docstring reference in search().
            stage_deadline = _stage_deadline_for(stage_idx)

            for br in frontier:
                parent_code = br["codes"][-1]
                limit = top_k if is_final else beam
                hits = _timed_query(
                    stage_idx, collection=stage.collection, query_vec=query_vec,
                    limit=limit, parent_code=parent_code,
                    stage_deadline_monotonic=stage_deadline,
                )
                if not hits:
                    continue
                _trace_hits(stage_idx, hits)

                if is_final:
                    pool_branches += 1
                    for hit in hits:
                        code = hit.payload.get("code", "")
                        if not code:
                            continue
                        existing = final_pool.get(code)
                        if existing is None or float(hit.score) > float(existing.score):
                            final_pool[code] = hit

                    if capture_pool_metadata:
                        branch_id = "/".join(br["codes"])
                        branch_hits_plain.append([
                            {
                                "code": hit.payload.get("code", ""),
                                "label_en": hit.payload.get("label_en", ""),
                                "label_ar": hit.payload.get("label_ar", ""),
                                "score": float(hit.score),
                                "branch_id": branch_id,
                                "source_rank": rank,
                                "path": br["codes"] + [hit.payload.get("code", "")],
                            }
                            for rank, hit in enumerate(hits, start=1)
                        ])

                    top_score = float(hits[0].score)
                    if top_score > best_final_score:
                        best_final_score = top_score
                        best = {
                            "codes": br["codes"] + [hits[0].payload.get("code", "")],
                            "scores": br["scores"] + [top_score],
                            "final_hits": hits,
                        }
                else:
                    for hit in hits:
                        code = hit.payload.get("code", "")
                        score = float(hit.score)
                        next_frontier.append({"codes": br["codes"] + [code], "scores": br["scores"] + [score]})

            if not is_final:
                frontier = next_frontier

        if best is None:
            return None

        # ── Build result from the best beam path ────────────────────
        stage_confidences = {
            f"stage{i + 1}": round(best["scores"][i], 4)
            for i in range(n)
        }
        confidence = round(
            sum(self.stages[i].weight * best["scores"][i] for i in range(n)), 4,
        )
        confidence = max(0.0, min(1.0, confidence))

        hierarchy_path = list(best["codes"])
        final_hits = best["final_hits"]
        final_top = final_hits[0]

        if branch_collapse:
            pool_hits = final_hits[:reranker_candidates]
        else:
            pool_sorted = sorted(final_pool.values(), key=lambda h: float(h.score), reverse=True)
            pool_hits = pool_sorted[:reranker_candidates]

        if trace is not None:
            trace["reranker_candidate_pool_size"] = len(final_pool)
            trace["reranker_candidate_branches"] = pool_branches
            final_key = f"stage{n}"
            if not branch_collapse:
                trace[f"{final_key}_pool"] = [
                    {"code": h.payload.get("code", ""), "label_en": h.payload.get("label_en", ""),
                     "score": round(float(h.score), 4)}
                    for h in pool_sorted
                ]
            if capture_pool_metadata and branch_hits_plain is not None:
                trace[f"{final_key}_pool_enriched"] = pool_and_rank_candidates(
                    branch_hits_plain, sort_mode="raw_score", k=None,
                )

        top_candidates = [
            EngineCandidate(
                code=hit.payload.get("code", ""),
                label_en=hit.payload.get("label_en", ""),
                label_ar=hit.payload.get("label_ar", ""),
                score=round(float(hit.score), 4),
            )
            for hit in pool_hits
        ]

        return EngineResult(
            code=final_top.payload.get("code", ""),
            label_en=final_top.payload.get("label_en", ""),
            label_ar=final_top.payload.get("label_ar", ""),
            confidence=confidence,
            stage_confidences=stage_confidences,
            hierarchy_path=hierarchy_path,
            top_candidates=top_candidates,
            hitl_required=confidence < self.hitl_threshold,
            fallback_used=False,
        )

    # ------------------------------------------------------------------
    # Qdrant helper (moved verbatim from HierarchicalISCOStore._query)
    # ------------------------------------------------------------------

    def _query(
        self,
        collection: str,
        query_vec: list[float],
        limit: int,
        parent_code: Optional[str] = None,
        query_telemetry: Optional[dict] = None,
        stage_deadline_monotonic: Optional[float] = None,
    ):
        """
        Wrap ``client.query_points()`` with an optional parent_code filter.

        Returns a list of ``ScoredPoint`` objects (empty list on any error
        or on stage-budget exhaustion -- never a fabricated candidate).

        query_telemetry : dict, optional
            Task 25/27/31: when provided, populated (as a side effect,
            mirroring the existing ``trace`` dict convention used
            throughout this module) with this query's outcome so a
            caller can tell a genuine successful zero-hit response apart
            from a swallowed exception, a first-attempt success apart
            from a retried success or a retry-exhausted failure, and
            (Task 31) any of those apart from stage-budget exhaustion.
            Always sets ``"outcome"`` (one of ``"success"``,
            ``"success_after_retry"``, ``"exception"`` (non-retryable, or
            retryable-but-``max_query_attempts``-is-1), ``"retry_exhausted"``
            (retryable exception type, every attempt failed),
            ``"stage_budget_exhausted"`` (Task 31: the remaining stage
            budget reached zero before this attempt could be started, or
            before a retry/backoff could proceed -- never a fabricated
            result)), ``"attempts"`` (int, how many attempts were
            actually made; may be 0 if the budget was already exhausted
            before the first attempt), ``"duration_ms"`` (float, TOTAL
            wall time across every attempt and any backoff -- for the
            default ``max_query_attempts=1`` and no active stage budget
            this is byte-identical to Task 25's original single-attempt
            semantics), ``"attempt_durations_ms"`` (list[float], one
            entry per attempt actually made), and (Task 31)
            ``"remaining_stage_budget_ms_at_entry"`` (float or None: the
            remaining stage budget, in ms, at the moment this call
            began -- None when no stage budget is active). Only on
            ``"exception"``/``"retry_exhausted"`` also sets
            ``"exception_type"`` (the LAST attempt's exception class
            name), ``"exception_message"`` (the last attempt's
            ``str(exc)``, sanitized via ``_sanitize_exception_message`` --
            bounded length, single line, never a stack trace, never the
            query vector/text), and ``"retryable"`` (bool, whether that
            last exception was in the Task 27 retry allowlist -- see
            ``_is_retryable_exception``). Omitted (None, the default)
            reproduces prior behaviour exactly for every existing caller.

        stage_deadline_monotonic : float, optional (Task 31)
            An absolute ``time.monotonic()``-based deadline shared by
            every branch query/retry/backoff within one stage (see
            ``search()``'s ``max_stage_latency_ms``). Default ``None``
            reproduces prior behaviour exactly -- no budget is ever
            checked or applied.
        """
        query_filter = None
        if parent_code is not None:
            query_filter = Filter(
                must=[FieldCondition(key="parent_code", match=MatchValue(value=parent_code))]
            )

        t_total_start = time.perf_counter()
        attempt_durations_ms: list[float] = []
        last_exc: Optional[BaseException] = None

        remaining_at_entry_ms: Optional[float] = None
        if stage_deadline_monotonic is not None:
            remaining_at_entry_ms = round((stage_deadline_monotonic - time.monotonic()) * 1000, 3)

        def _fill_budget_exhausted_telemetry(attempts_made: int) -> None:
            if query_telemetry is None:
                return
            query_telemetry["outcome"] = "stage_budget_exhausted"
            query_telemetry["attempts"] = attempts_made
            query_telemetry["duration_ms"] = round((time.perf_counter() - t_total_start) * 1000, 3)
            query_telemetry["attempt_durations_ms"] = list(attempt_durations_ms)
            query_telemetry["remaining_stage_budget_ms_at_entry"] = remaining_at_entry_ms

        for attempt_num in range(1, self.max_query_attempts + 1):
            # Task 31: never start a query once the shared stage budget
            # (if active) has run out -- checked fresh before EVERY
            # attempt, including the first.
            effective_timeout_seconds = self.query_timeout_seconds
            if stage_deadline_monotonic is not None:
                remaining_seconds = stage_deadline_monotonic - time.monotonic()
                if remaining_seconds <= 0:
                    _logger.warning(
                        "HierarchyBeamSearchEngine: stage budget exhausted before "
                        "querying '%s' (attempt %d/%d) -- not starting a query.",
                        collection, attempt_num, self.max_query_attempts,
                    )
                    _fill_budget_exhausted_telemetry(attempt_num - 1)
                    return []
                effective_timeout_seconds = (
                    remaining_seconds if effective_timeout_seconds is None
                    else min(effective_timeout_seconds, remaining_seconds)
                )

            t_attempt_start = time.perf_counter()
            try:
                query_points_kwargs = dict(
                    collection_name=collection,
                    query=query_vec,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
                if effective_timeout_seconds is not None:
                    # Task 31: explicit, finite, per-request timeout,
                    # derived from the configured timeout value and
                    # capped by any remaining stage budget -- see this
                    # method's docstring and Task 31's final report for
                    # what qdrant-client's own `timeout` parameter here
                    # actually controls (a server-side operation-timeout
                    # hint delivered via the request's query string, per
                    # the audited qdrant-client 1.17.0 source).
                    query_points_kwargs["timeout"] = max(1, math.ceil(effective_timeout_seconds))
                response = self._client.query_points(**query_points_kwargs)
            except Exception as exc:  # noqa: BLE001 - classified below, never silently swallowed
                attempt_durations_ms.append(round((time.perf_counter() - t_attempt_start) * 1000, 3))
                last_exc = exc
                retryable = _is_retryable_exception(exc)
                more_attempts_remain = attempt_num < self.max_query_attempts
                budget_remains_for_retry = True
                backoff = self.retry_backoff_seconds
                if stage_deadline_monotonic is not None:
                    remaining_for_backoff = stage_deadline_monotonic - time.monotonic()
                    budget_remains_for_retry = remaining_for_backoff > 0
                    if backoff > 0:
                        backoff = max(0.0, min(backoff, remaining_for_backoff))
                if retryable and more_attempts_remain and not budget_remains_for_retry:
                    # Would have retried, but the shared stage budget is
                    # exhausted -- never sleep, never start another attempt.
                    _logger.warning(
                        "HierarchyBeamSearchEngine: stage budget exhausted after a "
                        "retryable failure on '%s' (attempt %d/%d) -- not retrying.",
                        collection, attempt_num, self.max_query_attempts,
                    )
                    _fill_budget_exhausted_telemetry(attempt_num)
                    return []
                if retryable and more_attempts_remain:
                    _logger.warning(
                        "HierarchyBeamSearchEngine: query on '%s' failed on attempt %d/%d "
                        "(retryable): %s -- retrying.",
                        collection, attempt_num, self.max_query_attempts, exc,
                    )
                    if backoff > 0:
                        time.sleep(backoff)
                    continue
                # Non-retryable, or retryable but attempts exhausted: fail closed.
                _logger.warning(
                    "HierarchyBeamSearchEngine: query on '%s' failed (attempt %d/%d, "
                    "retryable=%s): %s",
                    collection, attempt_num, self.max_query_attempts, retryable, exc,
                )
                if query_telemetry is not None:
                    # "retry_exhausted" is reported only when retry was
                    # actually configured/possible (max_query_attempts > 1)
                    # AND the exception was retryable -- under the default
                    # max_query_attempts=1, no retry was ever attempted or
                    # even possible, so the outcome stays plain "exception",
                    # byte-for-byte identical to Task 25's original
                    # semantics regardless of whether the single exception
                    # happened to be a retryable-shaped one.
                    is_retry_exhausted = retryable and self.max_query_attempts > 1
                    query_telemetry["outcome"] = "retry_exhausted" if is_retry_exhausted else "exception"
                    query_telemetry["attempts"] = attempt_num
                    query_telemetry["duration_ms"] = round((time.perf_counter() - t_total_start) * 1000, 3)
                    query_telemetry["attempt_durations_ms"] = attempt_durations_ms
                    query_telemetry["exception_type"] = type(exc).__name__
                    query_telemetry["exception_message"] = _sanitize_exception_message(str(exc))
                    query_telemetry["retryable"] = retryable
                    query_telemetry["remaining_stage_budget_ms_at_entry"] = remaining_at_entry_ms
                return []
            else:
                attempt_durations_ms.append(round((time.perf_counter() - t_attempt_start) * 1000, 3))
                if query_telemetry is not None:
                    query_telemetry["outcome"] = "success" if attempt_num == 1 else "success_after_retry"
                    query_telemetry["attempts"] = attempt_num
                    query_telemetry["duration_ms"] = round((time.perf_counter() - t_total_start) * 1000, 3)
                    query_telemetry["attempt_durations_ms"] = attempt_durations_ms
                    query_telemetry["remaining_stage_budget_ms_at_entry"] = remaining_at_entry_ms
                return response.points

        # Unreachable: the loop always returns or continues; a final
        # non-retryable/exhausted failure returns inside the except
        # branch above. Kept only as a fail-closed guard against a
        # future refactor accidentally falling through.
        if query_telemetry is not None and last_exc is not None:  # pragma: no cover
            query_telemetry["outcome"] = "exception"
            query_telemetry["attempts"] = self.max_query_attempts
            query_telemetry["exception_type"] = type(last_exc).__name__
            query_telemetry["exception_message"] = _sanitize_exception_message(str(last_exc))
        return []


def _sanitize_exception_message(message: str) -> str:
    """Task 25: bound and flatten a raw ``str(exc)`` before it is ever
    written to an evaluation trace/CSV. Single line, length-capped, and
    never derived from the query vector/text, a stack trace, or any
    request payload -- only from the exception's own ``str()``, which for
    the qdrant-client transport exceptions this wraps (timeouts, connection
    errors) never includes request/response bodies or credentials."""
    flat = " ".join(message.split())
    _MAX_LEN = 300
    if len(flat) > _MAX_LEN:
        return flat[:_MAX_LEN] + "...(truncated)"
    return flat
