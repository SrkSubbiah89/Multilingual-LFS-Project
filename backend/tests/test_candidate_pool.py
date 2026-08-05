"""
Tests for backend/rag/candidate_pool.py -- the B2 pooling/sort/recall logic.

Pure functions, no Qdrant/Ollama dependency. This module is NOT called by
the B0/B1 code path (see candidate_pool.py's module docstring); these
tests exist to validate it in isolation before it's used for B2's
observational metadata and the future B3-Sort ablation.
"""

import pytest

from backend.rag.candidate_pool import (
    candidate_recall_at_k,
    dedupe_by_code,
    gold_rank_in_pool,
    gold_visible_at_k,
    normalize_scores,
    pool_and_rank_candidates,
    sort_pool,
)


def hit(code, score, branch_id="b1", source_rank=1, path=None, label_en=""):
    return {
        "code": code,
        "label_en": label_en or code,
        "label_ar": "",
        "score": score,
        "branch_id": branch_id,
        "source_rank": source_rank,
        "path": path or [],
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_dedupe_by_code_removes_duplicates_across_branches():
    branches = [
        [hit("2512", 0.80, branch_id="b1"), hit("2513", 0.75, branch_id="b1")],
        [hit("2512", 0.70, branch_id="b2")],  # duplicate code, lower score
    ]
    deduped = dedupe_by_code(branches)
    assert set(deduped.keys()) == {"2512", "2513"}


def test_dedupe_by_code_retains_highest_scoring_duplicate():
    branches = [
        [hit("2512", 0.70, branch_id="b1")],
        [hit("2512", 0.95, branch_id="b2")],  # higher score, different branch
        [hit("2512", 0.60, branch_id="b3")],
    ]
    deduped = dedupe_by_code(branches)
    assert deduped["2512"]["score"] == 0.95
    assert deduped["2512"]["branch_id"] == "b2"


def test_dedupe_by_code_skips_hits_with_no_code():
    branches = [[hit("", 0.9), hit("2512", 0.5)]]
    deduped = dedupe_by_code(branches)
    assert list(deduped.keys()) == ["2512"]


def test_dedupe_by_code_empty_input():
    assert dedupe_by_code([]) == {}
    assert dedupe_by_code([[]]) == {}


# ---------------------------------------------------------------------------
# Normalization (observational only)
# ---------------------------------------------------------------------------

def test_normalize_scores_min_max_range():
    deduped = {"a": hit("a", 0.5), "b": hit("b", 1.0), "c": hit("c", 0.0)}
    norm = normalize_scores(deduped)
    assert norm["b"] == pytest.approx(1.0)
    assert norm["c"] == pytest.approx(0.0)
    assert norm["a"] == pytest.approx(0.5)


def test_normalize_scores_all_equal_defaults_to_one():
    deduped = {"a": hit("a", 0.8), "b": hit("b", 0.8)}
    norm = normalize_scores(deduped)
    assert norm == {"a": 1.0, "b": 1.0}


def test_normalize_scores_empty():
    assert normalize_scores({}) == {}


# ---------------------------------------------------------------------------
# Deterministic sorting
# ---------------------------------------------------------------------------

def test_sort_pool_raw_score_mode_descending():
    deduped = {"a": hit("a", 0.5), "b": hit("b", 0.9), "c": hit("c", 0.7)}
    norm = normalize_scores(deduped)
    order = sort_pool(deduped, norm, sort_mode="raw_score")
    assert order == ["b", "c", "a"]


def test_sort_pool_deterministic_3key_breaks_exact_ties_by_code():
    # Identical scores -> normalize_scores gives everyone 1.0 -> tie must
    # break on ISCO code ascending, not insertion order.
    deduped = {"9999": hit("9999", 0.8), "1111": hit("1111", 0.8), "5555": hit("5555", 0.8)}
    norm = normalize_scores(deduped)
    order = sort_pool(deduped, norm, sort_mode="deterministic_3key")
    assert order == ["1111", "5555", "9999"]


def test_sort_pool_deterministic_3key_primary_key_is_normalized_score():
    deduped = {
        "a": hit("a", 0.9),   # highest raw score
        "b": hit("b", 0.5),
    }
    norm = normalize_scores(deduped)  # a=1.0, b=0.0
    order = sort_pool(deduped, norm, sort_mode="deterministic_3key")
    assert order == ["a", "b"]


def test_sort_pool_unknown_mode_raises():
    with pytest.raises(ValueError):
        sort_pool({}, {}, sort_mode="not_a_real_mode")


# ---------------------------------------------------------------------------
# Full pipeline + K truncation
# ---------------------------------------------------------------------------

def test_pool_and_rank_candidates_full_pool_when_k_none():
    branches = [[hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]]
    pooled = pool_and_rank_candidates(branches, k=None)
    assert [c["code"] for c in pooled] == ["a", "b", "c"]
    assert [c["pool_rank"] for c in pooled] == [1, 2, 3]


def test_pool_and_rank_candidates_k_truncation():
    branches = [[hit(str(i), 1.0 - i * 0.01) for i in range(10)]]
    pooled_k5 = pool_and_rank_candidates(branches, k=5)
    assert len(pooled_k5) == 5
    assert [c["code"] for c in pooled_k5] == ["0", "1", "2", "3", "4"]

    pooled_full = pool_and_rank_candidates(branches, k=None)
    assert len(pooled_full) == 10  # truncation only happens when k is given


def test_pool_and_rank_candidates_k_larger_than_pool_returns_whole_pool():
    branches = [[hit("a", 0.9), hit("b", 0.8)]]
    pooled = pool_and_rank_candidates(branches, k=20)
    assert len(pooled) == 2


def test_pool_and_rank_candidates_carries_branch_metadata():
    branches = [[hit("a", 0.9, branch_id="1/12/121", source_rank=1, path=["1", "12", "121", "a"])]]
    pooled = pool_and_rank_candidates(branches, k=None)
    assert pooled[0]["branch_id"] == "1/12/121"
    assert pooled[0]["source_rank"] == 1
    assert pooled[0]["path"] == ["1", "12", "121", "a"]


# ---------------------------------------------------------------------------
# Gold visibility / rank
# ---------------------------------------------------------------------------

def test_gold_rank_in_pool_found():
    branches = [[hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]]
    pooled = pool_and_rank_candidates(branches, k=None)
    assert gold_rank_in_pool(pooled, "b") == 2


def test_gold_rank_in_pool_absent_returns_none():
    branches = [[hit("a", 0.9)]]
    pooled = pool_and_rank_candidates(branches, k=None)
    assert gold_rank_in_pool(pooled, "zzzz") is None


def test_gold_visible_at_k_true_when_rank_within_k():
    branches = [[hit(str(i), 1.0 - i * 0.01) for i in range(10)]]
    pooled_full = pool_and_rank_candidates(branches, k=None)
    assert gold_visible_at_k(pooled_full, "4", k=5) is True   # rank 5
    assert gold_visible_at_k(pooled_full, "5", k=5) is False  # rank 6


def test_gold_visible_at_k_false_when_absent():
    branches = [[hit("a", 0.9)]]
    pooled_full = pool_and_rank_candidates(branches, k=None)
    assert gold_visible_at_k(pooled_full, "not_there", k=5) is False


# ---------------------------------------------------------------------------
# Candidate Recall@K
# ---------------------------------------------------------------------------

def test_candidate_recall_at_k_basic():
    branches = [[hit(str(i), 1.0 - i * 0.01) for i in range(10)]]
    pool_full = pool_and_rank_candidates(branches, k=None)
    # gold at rank 3 (visible at k=5) and gold at rank 8 (not visible at k=5)
    cases = [(pool_full, "2"), (pool_full, "7")]
    recall = candidate_recall_at_k(cases, k=5)
    assert recall == pytest.approx(0.5)


def test_candidate_recall_at_k_all_visible():
    branches = [[hit("a", 0.9), hit("b", 0.8)]]
    pool_full = pool_and_rank_candidates(branches, k=None)
    cases = [(pool_full, "a"), (pool_full, "b")]
    assert candidate_recall_at_k(cases, k=5) == pytest.approx(1.0)


def test_candidate_recall_at_k_empty_cases_returns_zero_not_error():
    assert candidate_recall_at_k([], k=5) == 0.0
