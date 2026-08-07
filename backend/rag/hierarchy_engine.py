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
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from backend.rag.candidate_pool import pool_and_rank_candidates

_logger = logging.getLogger(__name__)


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
    ) -> None:
        if len(stages) < 2:
            raise ValueError(
                "HierarchyBeamSearchEngine requires at least 2 stages "
                f"(got {len(stages)}); every stage after the first needs a "
                "parent to filter on."
            )
        self._client = client
        self.stages = stages
        self.hitl_threshold = hitl_threshold

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
    ) -> Optional[EngineResult]:
        """
        Run the beam search. Returns ``None`` only if every explored branch
        returns zero hits at the final stage (caller decides the fallback).
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

        def _timed_query(stage_idx: int, **kwargs):
            key = f"stage{stage_idx + 1}"
            if trace is None:
                return self._query(**kwargs)
            t0 = time.perf_counter()
            result = self._query(**kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            lk = f"{key}_latency_ms"
            trace[lk] = trace.get(lk, 0.0) + elapsed_ms
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
            hits0 = _timed_query(0, collection=self.stages[0].collection, query_vec=query_vec, limit=beam)
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

            for br in frontier:
                parent_code = br["codes"][-1]
                limit = top_k if is_final else beam
                hits = _timed_query(
                    stage_idx, collection=stage.collection, query_vec=query_vec,
                    limit=limit, parent_code=parent_code,
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
    ):
        """
        Wrap ``client.query_points()`` with an optional parent_code filter.

        Returns a list of ``ScoredPoint`` objects (empty list on any error).
        """
        try:
            query_filter = None
            if parent_code is not None:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="parent_code",
                            match=MatchValue(value=parent_code),
                        )
                    ]
                )

            response = self._client.query_points(
                collection_name=collection,
                query=query_vec,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            return response.points

        except Exception as exc:
            _logger.warning(
                "HierarchyBeamSearchEngine: query on '%s' failed: %s",
                collection,
                exc,
            )
            return []
