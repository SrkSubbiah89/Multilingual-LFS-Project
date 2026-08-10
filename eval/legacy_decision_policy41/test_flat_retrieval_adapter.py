"""
Tests for eval/legacy_decision_policy41/flat_retrieval_adapter.py (Task 42).

Fully hermetic: every "store" here is a fake object exposing only
`_embed_query`/`_query`/`_col_flat`/`profile` -- never a real
`HierarchicalISCOStore`, never a real Qdrant client, never a real LLM.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy_decision_policy41 import flat_retrieval_adapter as fra  # noqa: E402
from eval.legacy_decision_policy41.policy import HISTORICAL_THRESHOLD  # noqa: E402


@dataclass
class FakeHit:
    payload: dict
    score: float


class FakeStore:
    """Minimal stand-in for HierarchicalISCOStore. `hits_by_text` maps a
    query string to the list of FakeHit objects `_query` should return for
    it (independent of the fake embedding vector, which is never used for
    anything except being passed through)."""

    def __init__(self, hits_by_text: dict[str, list[FakeHit]], col_flat="fake_flat", profile="official_ilo2021_v1"):
        self._hits_by_text = hits_by_text
        self._col_flat = col_flat
        self.profile = profile
        self.embed_calls = []
        self.query_calls = []

    def _embed_query(self, text: str):
        self.embed_calls.append(text)
        return [0.0, 0.0, 0.0, 0.0]

    def _query(self, collection, query_vec, limit):
        self.query_calls.append((collection, limit))
        # The real text isn't threaded through _query (it takes a vector),
        # so tests key hits_by_text by the *last embedded text* instead.
        text = self.embed_calls[-1]
        return self._hits_by_text.get(text, [])[:limit]


def _hit(code, title_en, score):
    return FakeHit(payload={"code": code, "title_en": title_en, "level": "unit"}, score=score)


def _five_hits(base_score=0.5):
    return [_hit(f"{1000+i}", f"Title {i}", base_score - i * 0.01) for i in range(5)]


# ---------------------------------------------------------------------------
# fetch_five_official_flat_candidates
# ---------------------------------------------------------------------------

def test_fetch_returns_exactly_five_candidates_in_order():
    hits = _five_hits()
    store = FakeStore({"plumber": hits})
    candidates = fra.fetch_five_official_flat_candidates(store, "plumber")
    assert len(candidates) == 5
    assert [c.code for c in candidates] == ["1000", "1001", "1002", "1003", "1004"]


def test_title_ar_and_description_always_blank():
    store = FakeStore({"x": _five_hits()})
    candidates = fra.fetch_five_official_flat_candidates(store, "x")
    assert all(c.title_ar == "" for c in candidates)
    assert all(c.description == "" for c in candidates)


def test_level_always_four():
    store = FakeStore({"x": _five_hits()})
    candidates = fra.fetch_five_official_flat_candidates(store, "x")
    assert all(c.level == 4 for c in candidates)


def test_code_title_en_confidence_map_correctly_from_title_en_not_label_en():
    hits = [FakeHit(payload={"code": "2512", "title_en": "Real Title", "label_en": "Wrong Title"}, score=0.87)]
    store = FakeStore({"x": hits * 5})
    candidates = fra.fetch_five_official_flat_candidates(store, "x")
    assert candidates[0].code == "2512"
    assert candidates[0].title_en == "Real Title"
    assert candidates[0].confidence == pytest.approx(0.87)


def test_fewer_than_five_hits_raises():
    store = FakeStore({"x": _five_hits()[:3]})
    with pytest.raises(fra.InsufficientCandidatesError):
        fra.fetch_five_official_flat_candidates(store, "x")


# ---------------------------------------------------------------------------
# run_dev_preflight -- fast path / pending-rerank path
# ---------------------------------------------------------------------------

def _dev_row(case_id="C1", text="plumber", lang="en", gold="1000"):
    return fra.DevRow(case_id=case_id, input_text=text, input_language=lang, gold_isco_4digit=gold)


def test_fast_path_row_produces_semantic_outcome_zero_llm_calls():
    hits = _five_hits(base_score=0.99)  # top candidate well above threshold
    store = FakeStore({"plumber": hits})
    report = fra.run_dev_preflight(store, [_dev_row()])
    assert report.n_total == 1
    assert report.n_semantic_fast_path == 1
    assert report.n_pending_rerank == 0
    assert report.rows[0].outcome == fra.OUTCOME_SEMANTIC
    assert report.rows[0].top1_code == "1000"


def test_below_threshold_row_produces_pending_rerank_outcome_no_raise():
    hits = _five_hits(base_score=0.5)  # top candidate well below threshold
    store = FakeStore({"plumber": hits})
    report = fra.run_dev_preflight(store, [_dev_row()])
    assert report.n_pending_rerank == 1
    assert report.n_semantic_fast_path == 0
    assert report.rows[0].outcome == fra.OUTCOME_PENDING_RERANK
    assert report.rows[0].top1_code == "1000"  # still recorded, even though no full prediction was made


def test_below_threshold_row_never_calls_a_reranker():
    """FakeStore has no reranker/LLM concept at all -- the fact this test
    can run to completion with only _embed_query/_query defined proves no
    reranker callable was ever invoked."""
    hits = _five_hits(base_score=0.1)
    store = FakeStore({"plumber": hits})
    report = fra.run_dev_preflight(store, [_dev_row()])
    assert report.n_pending_rerank == 1  # completed without needing a reranker


# ---------------------------------------------------------------------------
# Aggregate counts
# ---------------------------------------------------------------------------

def test_aggregate_counts_sum_correctly():
    fast_hits = _five_hits(base_score=0.99)
    slow_hits = _five_hits(base_score=0.5)
    store = FakeStore({"fast": fast_hits, "slow": slow_hits})
    rows = [
        _dev_row(case_id="C1", text="fast"),
        _dev_row(case_id="C2", text="slow"),
        _dev_row(case_id="C3", text="fast"),
    ]
    report = fra.run_dev_preflight(store, rows)
    assert report.n_total == 3
    assert report.n_semantic_fast_path + report.n_pending_rerank == report.n_total
    assert report.n_semantic_fast_path == 2
    assert report.n_pending_rerank == 1


def test_exact_match_only_computed_over_fast_path_rows():
    fast_hits = _five_hits(base_score=0.99)  # top code "1000"
    slow_hits = _five_hits(base_score=0.5)
    store = FakeStore({"fast": fast_hits, "slow": slow_hits})
    rows = [
        _dev_row(case_id="C1", text="fast", gold="1000"),   # fast path, correct
        _dev_row(case_id="C2", text="fast", gold="9999"),   # fast path, incorrect
        _dev_row(case_id="C3", text="slow", gold="1000"),   # pending -- must NOT count even though top1_code matches
    ]
    report = fra.run_dev_preflight(store, rows)
    assert report.n_semantic_fast_path == 2
    assert report.n_semantic_fast_path_exact_match == 1


# ---------------------------------------------------------------------------
# lang pass-through
# ---------------------------------------------------------------------------

def test_input_language_passed_through_to_classify_with_policy(monkeypatch):
    captured = {}
    real_classify = fra.classify_with_policy

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_classify(*args, **kwargs)

    monkeypatch.setattr(fra, "classify_with_policy", spy)
    hits = _five_hits(base_score=0.5)  # below threshold, so reranker=None path is exercised
    store = FakeStore({"plumber": hits})
    fra.run_dev_preflight(store, [_dev_row(text="plumber", lang="ar")])
    assert captured.get("lang") == "ar"


# ---------------------------------------------------------------------------
# Fail-closed on unexpected exceptions
# ---------------------------------------------------------------------------

def test_unexpected_exception_mid_loop_stops_the_run():
    class FlakyStore(FakeStore):
        def _query(self, collection, query_vec, limit):
            if len(self.query_calls) == 2:  # fail on the 3rd row
                raise RuntimeError("simulated Qdrant failure")
            return super()._query(collection, query_vec, limit)

    hits = _five_hits(base_score=0.99)
    store = FlakyStore({"row1": hits, "row2": hits, "row3": hits, "row4": hits, "row5": hits})
    rows = [_dev_row(case_id=f"C{i}", text=f"row{i}") for i in range(1, 6)]
    with pytest.raises(RuntimeError, match="simulated Qdrant failure"):
        fra.run_dev_preflight(store, rows)


# ---------------------------------------------------------------------------
# No prohibited imports anywhere in this module
# ---------------------------------------------------------------------------

_PROHIBITED_IMPORT_TOKENS = ("crewai", "anthropic", "openai", "qdrant_client")


def test_adapter_module_has_no_llm_or_network_import():
    source = Path(__file__).resolve().parent.joinpath("flat_retrieval_adapter.py").read_text(encoding="utf-8")
    import_lines = [
        line.strip() for line in source.splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    lowered = "\n".join(import_lines).lower()
    for token in _PROHIBITED_IMPORT_TOKENS:
        assert token not in lowered, f"prohibited import token {token!r} found"
