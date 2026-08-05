"""
backend/rag/hierarchical_store.py

4-stage hierarchical ISCO-08 retrieval using Qdrant.

Architecture
------------
Uses four Qdrant collections, one per ISCO-08 hierarchy level:

  isco08_major_groups    – 10 major groups   (1-digit code)
  isco08_submajor_groups – ~40 sub-major groups (2-digit code)
  isco08_minor_groups    – ~130 minor groups  (3-digit code)
  isco08_unit_groups     – ~430 unit groups   (4-digit code)

Each entry carries a ``parent_code`` payload field so Qdrant
field-condition filters can prune the search at every stage.

Pipeline
--------
1. Search ``isco08_major_groups`` (10 entries, no filter) → top-1 major code.
2. Search ``isco08_submajor_groups`` filtered by ``parent_code == major_code`` → top-1.
3. Search ``isco08_minor_groups`` filtered by ``parent_code == submajor_code`` → top-1.
4. Search ``isco08_unit_groups`` filtered by ``parent_code == minor_code`` → top-5.

Returns ``HierarchicalResult`` with the best unit-group code, confidence,
hierarchy path, top candidates, and a HITL flag.

Fallback
--------
If *any* hierarchical collection is absent or returns 0 results the store
falls back to searching the original flat ``isco_occupations`` collection
so the system keeps working while the hierarchical collections are being
populated.

Embedding
---------
Same model as vector_store.py: ``intfloat/multilingual-e5-small`` (384-dim).
E5 prefix convention:  "query: " for queries, "passage: " for indexing.

Usage
-----
from backend.rag.hierarchical_store import get_hierarchical_store

hs     = get_hierarchical_store()
result = hs.search("software engineer")
result = hs.search("مهندس برمجيات")
print(result.code, result.label_en, result.confidence)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams
from sentence_transformers import SentenceTransformer

from backend.rag.candidate_pool import pool_and_rank_candidates

load_dotenv()

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME  = "intfloat/multilingual-e5-small"
VECTOR_DIM  = 384
_BATCH_SIZE = 64

HITL_THRESHOLD = 0.70  # below this → human-in-the-loop review required

# Confidence weights for the four stages
_W1, _W2, _W3, _W4 = 0.10, 0.20, 0.20, 0.50

# Collection names
_COL_MAJOR    = "isco08_major_groups"
_COL_SUBMAJOR = "isco08_submajor_groups"
_COL_MINOR    = "isco08_minor_groups"
_COL_UNIT     = "isco08_unit_groups"
_COL_FLAT     = "isco_occupations"  # fallback


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class UnitCandidate:
    """One of the top-N unit-group candidates returned from stage 4."""
    code: str
    label_en: str
    label_ar: str
    score: float


@dataclass
class HierarchicalResult:
    """
    Structured output from ``HierarchicalISCOStore.search()``.

    Attributes
    ----------
    code : str
        Best-matching 4-digit ISCO-08 unit-group code (or major/minor code
        when the search terminates early due to collection absence).
    label_en : str
        English label of the selected occupation.
    label_ar : str
        Arabic label of the selected occupation.
    confidence : float
        Weighted mean of stage confidences:
        stage4×0.5 + stage3×0.2 + stage2×0.2 + stage1×0.1
    stage_confidences : dict
        Raw cosine-similarity scores at each stage
        {stage1, stage2, stage3, stage4}.
    hierarchy_path : list[str]
        Ordered list of ISCO codes traversed, e.g. ["2", "25", "251", "2512"].
    top_candidates : list[UnitCandidate]
        Top-3 unit-group candidates from stage 4.
    hitl_required : bool
        True when ``confidence < HITL_THRESHOLD`` (0.70).
    fallback_used : bool
        True when the flat ``isco_occupations`` collection was used instead
        of the hierarchical pipeline.
    """
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
# HierarchicalISCOStore
# ---------------------------------------------------------------------------

class HierarchicalISCOStore:
    """
    4-stage hierarchical ISCO-08 retrieval backed by Qdrant.

    Instantiation probes Qdrant and loads the SentenceTransformer model.
    All four hierarchical collections must already exist and be populated;
    this class does *not* create or populate them (use the companion
    ``build_hierarchical_collections.py`` script for that).

    When collections are absent the instance degrades gracefully to
    flat search on ``isco_occupations``.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        _host = host or os.getenv("QDRANT_HOST", "localhost")
        _port = int(port or os.getenv("QDRANT_PORT", 6333))

        self._client = QdrantClient(host=_host, port=_port)
        self._model  = SentenceTransformer(MODEL_NAME)

        # Pre-check which collections exist so we know upfront whether to
        # run the hierarchical pipeline or fall back immediately.
        existing = {c.name for c in self._client.get_collections().collections}
        self._hierarchical_ready = all(
            col in existing
            for col in (_COL_MAJOR, _COL_SUBMAJOR, _COL_MINOR, _COL_UNIT)
        )
        self._flat_ready = _COL_FLAT in existing

        if not self._hierarchical_ready:
            _logger.warning(
                "HierarchicalISCOStore: one or more hierarchical Qdrant collections "
                "are missing (%s). Will fall back to flat search on '%s'.",
                ", ".join(
                    c for c in (_COL_MAJOR, _COL_SUBMAJOR, _COL_MINOR, _COL_UNIT)
                    if c not in existing
                ),
                _COL_FLAT,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        major_hint: str = "",
        beam: int = 2,
        stage1_mode: str = "description",
        reranker_candidates: int = 5,
        branch_collapse: bool = False,
        capture_pool_metadata: bool = False,
        trace: Optional[dict] = None,
    ) -> HierarchicalResult:
        """
        Run 4-stage hierarchical ISCO-08 retrieval for *query*.

        Falls back to the flat ``isco_occupations`` collection if the
        hierarchical collections are unavailable or any stage yields
        zero results.

        Parameters
        ----------
        query : str
            Free-text job title, description, or occupation-related text
            (English, Arabic, or code-switched).
        top_k : int
            Number of unit-group candidates to retrieve per branch in
            stage 4 (i.e. the Qdrant query limit for each explored
            minor-group parent). See reranker_candidates for how many of
            the union across branches end up in
            ``HierarchicalResult.top_candidates``.
        major_hint : str
            Optional 1-digit ISCO major group code (e.g. "5" for Service
            workers).  When provided, stage 1 is skipped and this code is
            used directly, preventing semantic drift at the top of the
            hierarchy (e.g. "chef" embedding to "Professionals" instead of
            "Service and Sales Workers"). Takes priority over stage1_mode
            unconditionally -- unaffected by this parameter's value.
        beam : int
            Beam width at stages 1–3 (default 2).  Set to 1 for strict
            greedy top-1 selection; used by the ablation study to measure
            the accuracy gain attributable to beam search.
        stage1_mode : str, default "description"
            "description" (default): unchanged prior behaviour -- stage 1
            searches isco08_major_groups directly (each major group
            embedded from its short official label only, e.g.
            "Professionals").
            "leaf_vote": bottom-up alternative -- retrieves the top-20
            isco08_unit_groups leaves directly (no major-group anchoring),
            aggregates by major group via summed cosine of that group's
            leaf hits, and takes the top-`beam` majors forward. Added
            after the Step 1 audit found major-group description text too
            abstract to embed near specific job titles (e.g. "primary
            school teacher" never placed major group 2 in its stage-1
            candidates under "description" mode). Only consulted when
            major_hint is empty -- the keyword map, when it fires, still
            skips stage 1 entirely regardless of this parameter.
        reranker_candidates : int, default 5
            How many unit-group candidates end up in
            ``HierarchicalResult.top_candidates`` (which is exactly what
            gets shown to the re-ranking LLM -- see isco_classifier.py's
            _classify_hierarchical()). Default behaviour as of the
            candidate-pooling fix: candidates are pooled across every
            explored beam branch (not just the winning one), deduplicated
            by code keeping the highest score seen, sorted globally, and
            the top `reranker_candidates` are kept. See branch_collapse
            for why this changed.
        branch_collapse : bool, default False
            When True, reproduces the PRE-FIX behaviour: only the winning
            branch's own top-N children (N = reranker_candidates) are
            shown to the reranker, discarding every candidate found in
            every other explored branch. This was the only behaviour that
            existed before the fix, and is a real, quantified defect --
            on the 130-case full set, 49/82 errors were cases where the
            correct code was found in a losing branch (competitive score,
            e.g. 0.792 vs the winning branch's 0.800) and silently
            discarded before the reranker ever ran, regardless of which
            model executed it. Kept as an explicit opt-in (not the
            default) purely so the before/after comparison can still be
            run and reported -- not because the old behaviour is
            considered acceptable for production use.
        capture_pool_metadata : bool, default False
            B2 instrumentation, OBSERVATIONAL ONLY. When True, populates
            trace["stage4_pool_enriched"] with the full pooled candidate
            list (all codes, not truncated to reranker_candidates) each
            carrying branch_id, source_rank (its rank within its own
            branch), path (major/submajor/minor/unit codes for that
            branch), raw_score, and normalized_score. Computed via
            backend.rag.candidate_pool.pool_and_rank_candidates() using
            sort_mode="raw_score" -- i.e. a richer LOGGING view of exactly
            the same B1 pool/order, never a different selection. Does not
            alter pool_hits/top_candidates/the reranker's actual input in
            any way, and defaults to False so B0/B1 callers (which never
            pass this) see zero behavioural change and pay zero extra
            cost. See candidate_pool.py's module docstring for why this
            logic lives in a separate, independently unit-tested module
            rather than being added inline here.
        trace : dict, optional
            Instrumentation-only. When a dict is passed, it is populated
            in-place with the full candidate list considered at each stage
            (not just the winning beam path) under keys "stage1".."stage4",
            each a list of {"code", "label_en", "score"}. Does not affect
            the return value or any selection/branching decision — this is
            purely additive logging for the evaluation harness (see
            eval/run_eval.py). Existing callers that omit this parameter
            see zero behavioural change.

        Returns
        -------
        HierarchicalResult
        """
        query = query.strip()
        if not query:
            return self._empty_result()

        query_vec = self._embed_query(query)

        if self._hierarchical_ready:
            result = self._hierarchical_search(
                query_vec, top_k=max(top_k, 5), major_hint=major_hint.strip(),
                beam=max(1, int(beam)), stage1_mode=stage1_mode,
                reranker_candidates=max(1, int(reranker_candidates)),
                branch_collapse=branch_collapse,
                capture_pool_metadata=capture_pool_metadata, trace=trace,
            )
            if result is not None:
                return result
            # At least one stage returned 0 results; fall through to flat.
            _logger.info(
                "HierarchicalISCOStore: hierarchical pipeline returned no results "
                "for query=%r; falling back to flat search.",
                query,
            )

        return self._flat_search(query_vec, top_k=top_k, trace=trace)

    # ------------------------------------------------------------------
    # Hierarchical pipeline
    # ------------------------------------------------------------------

    def _hierarchical_search(
        self,
        query_vec: list[float],
        top_k: int,
        major_hint: str = "",
        beam: int = 2,
        stage1_mode: str = "description",
        reranker_candidates: int = 5,
        branch_collapse: bool = False,
        capture_pool_metadata: bool = False,
        trace: Optional[dict] = None,
    ) -> Optional[HierarchicalResult]:
        """
        Execute a beam-search hierarchical pipeline across 4 stages.

        Instead of strict greedy top-1 at every stage (which cascades a
        single wrong choice into a completely wrong result), we keep up to
        ``beam`` candidates at stages 1–3 and select the path whose
        stage-4 unit-group score is highest.

        Returns ``None`` only if every beam path returns 0 unit-group hits
        (caller falls back to flat search in that case).
        """
        _BEAM = beam  # configurable; default 2, set 1 for greedy ablation

        def _trace_hits(stage_key: str, hits) -> None:
            # Instrumentation-only: record every distinct candidate considered
            # at this stage across the whole beam exploration (not just the
            # winning path). No effect on selection/branching.
            if trace is None:
                return
            bucket = trace.setdefault(stage_key, [])
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

        def _timed_query(stage_key: str, **kwargs):
            # Instrumentation-only: accumulates wall time spent in this
            # stage's Qdrant queries (summed across beam branches — a real,
            # additive measurement, not a per-branch estimate) into
            # trace["stageN_latency_ms"]. No effect on the query itself.
            if trace is None:
                return self._query(**kwargs)
            t0 = time.perf_counter()
            result = self._query(**kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            key = f"{stage_key}_latency_ms"
            trace[key] = trace.get(key, 0.0) + elapsed_ms
            return result

        # ── Stage 1: major group ───────────────────────────────────────────
        # When a keyword hint is supplied (e.g. "chef" → "5") we trust it
        # and skip semantic search entirely for this stage.
        if major_hint:
            s1_candidates = [(major_hint, 1.0)]
            _logger.debug(
                "HierarchicalISCOStore: keyword major_hint=%r, stage 1 skipped.",
                major_hint,
            )
            if trace is not None:
                trace.setdefault("stage1", []).append({
                    "code": major_hint, "label_en": "(keyword hint, search skipped)", "score": 1.0,
                })
                trace["stage1_latency_ms"] = trace.get("stage1_latency_ms", 0.0)  # 0ms: search skipped
                # Disclosed explicitly rather than left implicit in a 1.0
                # score: when this fires, stage 1 is a dictionary lookup,
                # not hierarchical semantic retrieval -- a run's reported
                # accuracy/latency is a blend of both mechanisms unless a
                # reader can see which cases took which path.
                trace["stage1_source"] = "keyword_map"
        elif stage1_mode == "leaf_vote":
            t0 = time.perf_counter()
            s1_candidates = self._leaf_vote_stage1(query_vec, beam=_BEAM)
            if trace is not None:
                trace["stage1_latency_ms"] = trace.get("stage1_latency_ms", 0.0) + (time.perf_counter() - t0) * 1000
                trace.setdefault("stage1", []).extend([
                    {"code": code, "label_en": "(leaf-vote aggregate over top-20 unit groups)", "score": round(score, 4)}
                    for code, score in s1_candidates
                ])
                trace["stage1_source"] = "leaf_vote"
            if not s1_candidates:
                return None
        else:
            s1_hits = _timed_query(
                "stage1",
                collection=_COL_MAJOR,
                query_vec=query_vec,
                limit=_BEAM,
            )
            if not s1_hits:
                return None
            _trace_hits("stage1", s1_hits)
            s1_candidates = [
                (hit.payload.get("code", ""), float(hit.score))
                for hit in s1_hits
            ]
            if trace is not None:
                trace["stage1_source"] = "semantic_retrieval"

        # ── Stages 2-4: beam expansion ────────────────────────────────────
        # Each path is (s1_score, s1_code, s2_score, s2_code,
        #               s3_score, s3_code, s4_hit_list)
        best_result: Optional[dict] = None
        best_s4_score = -1.0

        # Candidate-pooling fix: accumulate every stage-4 hit seen across
        # ALL explored branches (not just the eventual winner), keyed by
        # code so a code found in more than one branch keeps its highest
        # score. Without this, a candidate found in a losing branch with a
        # competitive-but-not-quite-best score (e.g. 0.792 vs the winning
        # branch's 0.800) is silently discarded before the reranker ever
        # sees it -- confirmed as the majority failure mode (49/82 errors
        # on the 130-case full set) once this was diagnosed.
        s4_pool: dict[str, object] = {}
        s4_pool_branches = 0

        # B2 instrumentation ONLY (see candidate_pool.py's module docstring):
        # plain-dict copies of every branch's stage-4 hits, used solely to
        # compute OBSERVATIONAL enriched metadata (branch id, source rank,
        # path, normalized score) via candidate_pool.pool_and_rank_candidates()
        # after the loop. Never read by the B0/B1 selection logic above --
        # only populated at all when capture_pool_metadata=True, so B0/B1
        # (which never pass that flag) pay zero cost and see zero behaviour
        # change from this list's existence.
        _branch_hits_plain: list[list[dict]] = [] if capture_pool_metadata else None  # type: ignore[assignment]

        for major_code, s1_score in s1_candidates:
            # Stage 2: sub-major groups within this major
            s2_hits = _timed_query(
                "stage2",
                collection=_COL_SUBMAJOR,
                query_vec=query_vec,
                limit=_BEAM,
                parent_code=major_code,
            )
            if not s2_hits:
                continue  # try next major branch
            _trace_hits("stage2", s2_hits)

            for s2_hit in s2_hits:
                submajor_code = s2_hit.payload.get("code", "")
                s2_score      = float(s2_hit.score)

                # Stage 3: minor groups within this sub-major
                s3_hits = _timed_query(
                    "stage3",
                    collection=_COL_MINOR,
                    query_vec=query_vec,
                    limit=_BEAM,
                    parent_code=submajor_code,
                )
                if not s3_hits:
                    continue
                _trace_hits("stage3", s3_hits)

                for s3_hit in s3_hits:
                    minor_code = s3_hit.payload.get("code", "")
                    s3_score   = float(s3_hit.score)

                    # Stage 4: unit groups within this minor
                    s4_hits = _timed_query(
                        "stage4",
                        collection=_COL_UNIT,
                        query_vec=query_vec,
                        limit=top_k,
                        parent_code=minor_code,
                    )
                    if not s4_hits:
                        continue
                    _trace_hits("stage4", s4_hits)

                    s4_pool_branches += 1
                    for hit in s4_hits:
                        code = hit.payload.get("code", "")
                        if not code:
                            continue
                        existing = s4_pool.get(code)
                        if existing is None or float(hit.score) > float(existing.score):
                            s4_pool[code] = hit

                    if capture_pool_metadata:
                        branch_id = f"{major_code}/{submajor_code}/{minor_code}"
                        _branch_hits_plain.append([
                            {
                                "code": hit.payload.get("code", ""),
                                "label_en": hit.payload.get("label_en", ""),
                                "label_ar": hit.payload.get("label_ar", ""),
                                "score": float(hit.score),
                                "branch_id": branch_id,
                                "source_rank": rank,
                                "path": [major_code, submajor_code, minor_code, hit.payload.get("code", "")],
                            }
                            for rank, hit in enumerate(s4_hits, start=1)
                        ])

                    s4_top_score = float(s4_hits[0].score)
                    if s4_top_score > best_s4_score:
                        best_s4_score = s4_top_score
                        best_result = {
                            "major_code":    major_code,
                            "s1_score":      s1_score,
                            "submajor_code": submajor_code,
                            "s2_score":      s2_score,
                            "minor_code":    minor_code,
                            "s3_score":      s3_score,
                            "s4_hits":       s4_hits,
                            "s4_score":      s4_top_score,
                        }

        if best_result is None:
            return None

        # ── Build result from best beam path ──────────────────────────────
        r          = best_result
        s4_hits    = r["s4_hits"]
        s4_top     = s4_hits[0]

        confidence = round(
            _W1 * r["s1_score"]
            + _W2 * r["s2_score"]
            + _W3 * r["s3_score"]
            + _W4 * r["s4_score"],
            4,
        )
        confidence = max(0.0, min(1.0, confidence))

        stage_confidences = {
            "stage1": round(r["s1_score"], 4),
            "stage2": round(r["s2_score"], 4),
            "stage3": round(r["s3_score"], 4),
            "stage4": round(r["s4_score"], 4),
        }

        hierarchy_path = [
            r["major_code"], r["submajor_code"],
            r["minor_code"], s4_top.payload.get("code", ""),
        ]

        if branch_collapse:
            # Pre-fix behaviour, kept only as an explicit opt-in for the
            # before/after ablation: winning branch's own top-N children,
            # discarding every candidate found in every other branch.
            pool_hits = s4_hits[:reranker_candidates]
        else:
            # Fixed (default) behaviour: global top-N across every branch
            # explored, deduplicated by code. The winning branch's own #1
            # is always the pool's #1 by construction (best_s4_score is
            # the max of every branch's own top score), so it's already
            # included -- no special-casing needed.
            pool_sorted = sorted(s4_pool.values(), key=lambda h: float(h.score), reverse=True)
            pool_hits = pool_sorted[:reranker_candidates]

        if trace is not None:
            trace["reranker_candidate_pool_size"] = len(s4_pool)
            trace["reranker_candidate_branches"] = s4_pool_branches
            if not branch_collapse:
                trace["stage4_pool"] = [
                    {"code": h.payload.get("code", ""), "label_en": h.payload.get("label_en", ""),
                     "score": round(float(h.score), 4)}
                    for h in pool_sorted
                ]
            # B2 instrumentation ONLY -- observational, does not feed back
            # into pool_hits/top_candidates above in any way. sort_mode is
            # hardcoded to "raw_score" here deliberately: this is a LOGGING
            # view of the same B1 ordering, not an activation of the
            # separate B3-Sort "deterministic_3key" mode (that mode is
            # never used by B0/B1/B2 -- see candidate_pool.py).
            if capture_pool_metadata and _branch_hits_plain is not None:
                trace["stage4_pool_enriched"] = pool_and_rank_candidates(
                    _branch_hits_plain, sort_mode="raw_score", k=None,
                )

        top_candidates = [
            UnitCandidate(
                code=hit.payload.get("code", ""),
                label_en=hit.payload.get("label_en", ""),
                label_ar=hit.payload.get("label_ar", ""),
                score=round(float(hit.score), 4),
            )
            for hit in pool_hits
        ]

        return HierarchicalResult(
            code=s4_top.payload.get("code", ""),
            label_en=s4_top.payload.get("label_en", ""),
            label_ar=s4_top.payload.get("label_ar", ""),
            confidence=confidence,
            stage_confidences=stage_confidences,
            hierarchy_path=hierarchy_path,
            top_candidates=top_candidates,
            hitl_required=confidence < HITL_THRESHOLD,
            fallback_used=False,
        )

    # ------------------------------------------------------------------
    # Flat fallback
    # ------------------------------------------------------------------

    def _flat_search(
        self,
        query_vec: list[float],
        top_k: int,
        trace: Optional[dict] = None,
    ) -> HierarchicalResult:
        """
        Search the flat ``isco_occupations`` collection and wrap the result
        as a ``HierarchicalResult`` so callers get a consistent return type.
        """
        if not self._flat_ready:
            _logger.error(
                "HierarchicalISCOStore: neither hierarchical collections nor "
                "flat '%s' collection is available.",
                _COL_FLAT,
            )
            return self._empty_result()

        hits = self._query(collection=_COL_FLAT, query_vec=query_vec, limit=top_k)
        if not hits:
            return self._empty_result()
        if trace is not None:
            # Flat fallback has no real stage 1-3; record the same candidate
            # set under all four keys so callers can see this was a flat, not
            # hierarchical, retrieval (HierarchicalResult.fallback_used=True
            # is the authoritative flag; this just keeps the trace non-empty).
            flat_bucket = [
                {"code": h.payload.get("code", ""), "label_en": h.payload.get("label_en", ""),
                 "score": round(float(h.score), 4)}
                for h in hits
            ]
            for key in ("stage1", "stage2", "stage3", "stage4"):
                trace[key] = flat_bucket

        best    = hits[0]
        score   = round(float(best.score), 4)
        p       = best.payload

        top_candidates = [
            UnitCandidate(
                code=hit.payload.get("code", ""),
                label_en=hit.payload.get("label_en", ""),
                label_ar=hit.payload.get("label_ar", ""),
                score=round(float(hit.score), 4),
            )
            for hit in hits[:3]
        ]

        code = p.get("code", "")
        hierarchy_path = _infer_path(code)

        return HierarchicalResult(
            code=code,
            label_en=p.get("label_en", ""),
            label_ar=p.get("label_ar", ""),
            confidence=score,
            stage_confidences={
                "stage1": score,
                "stage2": score,
                "stage3": score,
                "stage4": score,
            },
            hierarchy_path=hierarchy_path,
            top_candidates=top_candidates,
            hitl_required=score < HITL_THRESHOLD,
            fallback_used=True,
        )

    # ------------------------------------------------------------------
    # Bottom-up stage 1 (leaf-anchored ancestor voting)
    # ------------------------------------------------------------------

    def _leaf_vote_stage1(
        self, query_vec: list[float], top_n: int = 20, beam: int = 2,
    ) -> list[tuple[str, float]]:
        """
        Retrieve the top-`top_n` isco08_unit_groups leaves directly
        (isco08_unit_groups is a dedicated 436-entry, 4-digit-only
        collection -- no mixed-granularity filtering needed, unlike the
        separate flat isco_occupations fallback collection), then rank
        major groups by the SUMMED cosine score of their leaf hits within
        that top-n set (more supporting leaves near the query outweighs a
        single strong hit -- a genuine vote, not a max-pool).

        Returns up to `beam` (major_code, score) pairs, sorted by that
        summed-cosine ranking. The returned score is each major group's
        MEAN (not summed) cosine among its hits in the top-n, so it stays
        in the same ~[0,1] range as a single cosine score and is safe to
        feed into the existing stage-confidence weighted-average formula
        (_W1 * s1_score + ...) without separate rescaling -- summing
        would blow past 1.0 for any major group with several hits.
        """
        hits = self._query(collection=_COL_UNIT, query_vec=query_vec, limit=top_n)
        if not hits:
            return []

        score_sum: dict[str, float] = {}
        score_count: dict[str, int] = {}
        for hit in hits:
            code = hit.payload.get("code", "")
            if not code:
                continue
            major = code[0]
            score_sum[major] = score_sum.get(major, 0.0) + float(hit.score)
            score_count[major] = score_count.get(major, 0) + 1

        ranked = sorted(score_sum.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (major, score_sum[major] / score_count[major])
            for major, _ in ranked[:beam]
        ]

    # ------------------------------------------------------------------
    # Qdrant helpers
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
                "HierarchicalISCOStore: query on '%s' failed: %s",
                collection,
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_query(self, text: str) -> list[float]:
        """Encode *text* with the E5 query prefix, returns a normalised vector."""
        prefixed = f"query: {text.strip()}"
        vec = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=1,
        )
        return vec[0].tolist()

    # ------------------------------------------------------------------
    # Sentinel
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> HierarchicalResult:
        return HierarchicalResult(
            code="",
            label_en="Unknown",
            label_ar="غير معروف",
            confidence=0.0,
            stage_confidences={"stage1": 0.0, "stage2": 0.0,
                               "stage3": 0.0, "stage4": 0.0},
            hierarchy_path=[],
            top_candidates=[],
            hitl_required=True,
            fallback_used=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_path(code: str) -> list[str]:
    """
    Derive the hierarchy path from an ISCO code by string prefix.

    Examples
    --------
    "2512" → ["2", "25", "251", "2512"]
    "25"   → ["2", "25"]
    "2"    → ["2"]
    ""     → []
    """
    if not code:
        return []
    path: list[str] = []
    for length in range(1, len(code) + 1):
        path.append(code[:length])
    return path


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_hierarchical_store: Optional[HierarchicalISCOStore] = None


def get_hierarchical_store() -> HierarchicalISCOStore:
    """
    Return the module-level ``HierarchicalISCOStore`` singleton.

    The first call creates the instance (loads the SentenceTransformer model
    and connects to Qdrant).  Subsequent calls return the cached instance.

    Raises
    ------
    Exception
        Propagated from ``HierarchicalISCOStore.__init__()`` if Qdrant is
        unreachable or the model cannot be loaded.
    """
    global _hierarchical_store
    if _hierarchical_store is None:
        _hierarchical_store = HierarchicalISCOStore()
    return _hierarchical_store
