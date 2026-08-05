"""
backend/rag/candidate_pool.py

Pure, side-effect-free candidate pooling for the B2 candidate-capacity
experiment and future B3-Sort ablation.

This module is deliberately NOT wired into the B0/B1 code path in
hierarchical_store.py (HierarchicalISCOStore._hierarchical_search()'s
existing branch-collapse / raw-score-pooling logic is untouched and does
not call anything here). It exists so that:

  1. B2's *observational* enriched metadata (branch id, source rank, path
     evidence, normalized score) can be computed without touching B0/B1's
     inline pooling code at all.
  2. The pooling/sort/truncation/recall logic is unit-testable in
     isolation, with no Qdrant or Ollama dependency -- these functions
     operate on plain dicts, not live qdrant_client ScoredPoint objects.
  3. The future B3-Sort ablation (deterministic 3-key sort) has a home
     that is opt-in and separate from B1's frozen raw-score-only sort.

Nothing in this module is called by the B0 or B1 code path. It is used by
hierarchical_store.py only when capture_pool_metadata=True (B2
instrumentation) and will be used by the future B3-Sort experiment.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class CandidateHit(TypedDict, total=False):
    """One stage-4 hit from one branch, as a plain dict (no Qdrant object
    dependency -- this is what makes the module unit-testable without a
    live Qdrant instance)."""
    code: str
    label_en: str
    label_ar: str
    score: float           # raw retrieval (cosine) score
    branch_id: str         # e.g. "major/submajor/minor" path identifier
    source_rank: int       # 1-based rank of this hit within its OWN branch's stage-4 result list
    path: list             # [major_code, submajor_code, minor_code, unit_code] for this branch


class PooledCandidate(TypedDict, total=False):
    code: str
    label_en: str
    label_ar: str
    raw_score: float
    normalized_score: float
    branch_id: str
    source_rank: int
    path: list
    pool_rank: int          # 1-based rank in the final sorted, deduplicated pool


def dedupe_by_code(branch_hits: list[list[CandidateHit]]) -> dict[str, CandidateHit]:
    """
    Flatten hits from every branch and deduplicate by ISCO code, keeping
    the hit with the highest raw score for each code (matches B1's
    existing dedup rule: "if one ISCO code occurs in multiple branches,
    retain the strongest score"). Ties (identical score) keep the FIRST
    hit encountered, for determinism given a fixed branch-iteration order.

    Returns a dict keyed by code -> the winning CandidateHit (unmodified,
    still carrying its own branch_id/source_rank/path/score).
    """
    winners: dict[str, CandidateHit] = {}
    for branch in branch_hits:
        for hit in branch:
            code = hit.get("code", "")
            if not code:
                continue
            existing = winners.get(code)
            if existing is None or float(hit.get("score", 0.0)) > float(existing.get("score", 0.0)):
                winners[code] = hit
    return winners


def normalize_scores(deduped: dict[str, CandidateHit]) -> dict[str, float]:
    """
    Min-max normalize raw scores across the deduplicated pool into [0, 1].
    Purely observational (see module docstring) -- never used to change
    B0/B1 ordering. If every candidate has the same score (or the pool is
    empty/singleton), every normalized score is 1.0 (avoids a 0/0 divide
    and avoids arbitrarily zeroing out a single-candidate pool).
    """
    if not deduped:
        return {}
    scores = [float(h.get("score", 0.0)) for h in deduped.values()]
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return {code: 1.0 for code in deduped}
    return {
        code: (float(hit.get("score", 0.0)) - lo) / (hi - lo)
        for code, hit in deduped.items()
    }


def sort_pool(
    deduped: dict[str, CandidateHit],
    normalized: dict[str, float],
    sort_mode: str = "raw_score",
) -> list[str]:
    """
    Return codes sorted according to `sort_mode`.

    "raw_score" (default): single key, raw score descending. This is B1's
        EXACT existing sort (see hierarchical_store.py's
        `sorted(s4_pool.values(), key=lambda h: float(h.score), reverse=True)`)
        -- reproduced here only so it can be unit tested and used as the
        baseline comparison in tests, NOT because B1 calls this function.
    "deterministic_3key" (B3-Sort, not activated for B2): normalized score
        descending, then raw score descending, then ISCO code ascending.
        Guarantees a fully deterministic order even when scores tie
        exactly, which "raw_score" mode does not (Python's sort is stable,
        so raw_score ties resolve by insertion/dict order, not by a
        documented rule).
    """
    if sort_mode == "raw_score":
        return sorted(deduped.keys(), key=lambda c: float(deduped[c].get("score", 0.0)), reverse=True)
    if sort_mode == "deterministic_3key":
        return sorted(
            deduped.keys(),
            key=lambda c: (-normalized.get(c, 0.0), -float(deduped[c].get("score", 0.0)), c),
        )
    raise ValueError(f"Unknown sort_mode {sort_mode!r}; expected 'raw_score' or 'deterministic_3key'")


def pool_and_rank_candidates(
    branch_hits: list[list[CandidateHit]],
    sort_mode: str = "raw_score",
    k: Optional[int] = None,
) -> list[PooledCandidate]:
    """
    Full pipeline: dedupe -> normalize -> sort -> (optionally truncate to
    top-k). Returns the FULL sorted pool (all unique codes) when k is
    None, or the top-k when k is given -- callers that need both "full
    pool for recall metrics" and "top-K shown to reranker" should call
    with k=None and slice the result themselves (result[:k]), which is
    exactly what the K-truncation unit tests exercise.
    """
    deduped = dedupe_by_code(branch_hits)
    normalized = normalize_scores(deduped)
    ordered_codes = sort_pool(deduped, normalized, sort_mode=sort_mode)

    pooled: list[PooledCandidate] = []
    for rank, code in enumerate(ordered_codes, start=1):
        hit = deduped[code]
        pooled.append({
            "code": code,
            "label_en": hit.get("label_en", ""),
            "label_ar": hit.get("label_ar", ""),
            "raw_score": float(hit.get("score", 0.0)),
            "normalized_score": round(normalized.get(code, 0.0), 6),
            "branch_id": hit.get("branch_id", ""),
            "source_rank": hit.get("source_rank"),
            "path": hit.get("path", []),
            "pool_rank": rank,
        })

    if k is not None:
        return pooled[:k]
    return pooled


def gold_rank_in_pool(pooled: list[PooledCandidate], gold_code: str) -> Optional[int]:
    """1-based rank of gold_code in `pooled` (as returned by
    pool_and_rank_candidates with k=None, i.e. the FULL pool), or None if
    gold_code is not present anywhere in the pool."""
    for c in pooled:
        if c["code"] == gold_code:
            return c["pool_rank"]
    return None


def gold_visible_at_k(pooled_full: list[PooledCandidate], gold_code: str, k: int) -> bool:
    """True if gold_code's rank in the full pool is <= k (i.e. it would
    have been among the top-K actually shown to the reranker)."""
    rank = gold_rank_in_pool(pooled_full, gold_code)
    return rank is not None and rank <= k


def candidate_recall_at_k(
    cases: list[tuple[list[PooledCandidate], str]],
    k: int,
) -> float:
    """
    Candidate Recall@K across a set of cases.

    cases: list of (full_pool, gold_code) pairs, one per evaluated case.
    Returns the fraction of cases where gold_visible_at_k(...) is True.
    Returns 0.0 for an empty case list (explicit, not a ZeroDivisionError
    -- callers should treat 0.0 on an empty set as "no data", not "no
    recall", but this matches the convention used elsewhere in this
    codebase's eval harness for empty-denominator guards).
    """
    if not cases:
        return 0.0
    hits = sum(1 for pool, gold in cases if gold_visible_at_k(pool, gold, k))
    return hits / len(cases)
