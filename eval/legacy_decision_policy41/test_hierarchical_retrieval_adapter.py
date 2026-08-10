"""
Tests for eval/legacy_decision_policy41/hierarchical_retrieval_adapter.py
(Task 43, hierarchical extension).

Fully hermetic: fake store, fake catalogue records -- no real Qdrant, no
real hierarchical_store.py instance.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy_decision_policy41 import hierarchical_retrieval_adapter as hra  # noqa: E402
from eval.legacy_decision_policy41.flat_retrieval_adapter import InsufficientCandidatesError  # noqa: E402


@dataclass
class FakeUnitCandidate:
    code: str
    label_en: str
    label_ar: str
    score: float


@dataclass
class FakeHierResult:
    top_candidates: list


class FakeHierStore:
    def __init__(self, candidates_by_text):
        self._candidates_by_text = candidates_by_text
        self.search_calls = []

    def search(self, query_text, reranker_candidates=5):
        self.search_calls.append((query_text, reranker_candidates))
        return FakeHierResult(top_candidates=self._candidates_by_text.get(query_text, [])[:reranker_candidates])


@dataclass
class FakeCatalogueRecord:
    code: str
    level: str
    title_en: str


def _five_unit_candidates(base_score=0.5):
    # label_en left blank on purpose -- mirrors the real, disclosed gap
    return [FakeUnitCandidate(code=f"{2000+i}", label_en="", label_ar="", score=base_score - i * 0.01) for i in range(5)]


def test_build_unit_code_to_title_en_only_includes_unit_level():
    records = [
        FakeCatalogueRecord(code="2", level="major", title_en="Professionals"),
        FakeCatalogueRecord(code="2512", level="unit", title_en="Software Developers"),
        FakeCatalogueRecord(code="2513", level="unit", title_en="Web Developers"),
    ]
    mapping = hra.build_unit_code_to_title_en(records)
    assert mapping == {"2512": "Software Developers", "2513": "Web Developers"}


def test_fetch_returns_five_candidates_with_looked_up_titles_not_blank_label_en():
    cands = _five_unit_candidates()
    store = FakeHierStore({"plumber": cands})
    titles = {f"{2000+i}": f"Title {i}" for i in range(5)}
    result = hra.fetch_five_official_hierarchical_candidates(store, "plumber", titles)
    assert len(result) == 5
    assert [c.code for c in result] == ["2000", "2001", "2002", "2003", "2004"]
    assert [c.title_en for c in result] == ["Title 0", "Title 1", "Title 2", "Title 3", "Title 4"]


def test_missing_title_lookup_yields_empty_string_not_fabricated():
    cands = _five_unit_candidates()
    store = FakeHierStore({"x": cands})
    result = hra.fetch_five_official_hierarchical_candidates(store, "x", {})  # empty lookup
    assert all(c.title_en == "" for c in result)


def test_title_ar_description_blank_level_always_four():
    cands = _five_unit_candidates()
    store = FakeHierStore({"x": cands})
    result = hra.fetch_five_official_hierarchical_candidates(store, "x", {})
    assert all(c.title_ar == "" for c in result)
    assert all(c.description == "" for c in result)
    assert all(c.level == 4 for c in result)


def test_fewer_than_five_candidates_raises():
    store = FakeHierStore({"x": _five_unit_candidates()[:2]})
    with pytest.raises(InsufficientCandidatesError):
        hra.fetch_five_official_hierarchical_candidates(store, "x", {})


def test_search_called_with_reranker_candidates_five():
    cands = _five_unit_candidates()
    store = FakeHierStore({"x": cands})
    hra.fetch_five_official_hierarchical_candidates(store, "x", {})
    assert store.search_calls == [("x", 5)]


def test_make_hierarchical_candidate_fetcher_matches_flat_fetcher_signature():
    cands = _five_unit_candidates()
    store = FakeHierStore({"x": cands})
    fetcher = hra.make_hierarchical_candidate_fetcher({"2000": "Title 0", "2001": "T1", "2002": "T2", "2003": "T3", "2004": "T4"})
    result = fetcher(store, "x")  # same (store, query_text) shape as fetch_five_official_flat_candidates
    assert len(result) == 5
    assert result[0].title_en == "Title 0"
