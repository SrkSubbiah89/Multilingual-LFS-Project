"""
Tests for ISCOClassifier's Task 21 isco_catalogue_profile wiring.

Hermetic: HierarchicalISCOStore is monkeypatched at the isco_classifier
module level to a controllable fake; no live Qdrant, embedding model, or
LLM/CrewAI construction occurs (Agent/Crew/Task/get_llm* are asserted
never called for enable_llm=False cases, matching Task 09's existing
pattern in test_isco_classifier.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.agents.isco_classifier import ISCOClassifier
from backend.rag.hierarchical_store import HierarchicalResult, LEGACY_PROFILE, UnitCandidate

_OFFICIAL_PROFILE = "official_ilo2021_v1"


def _hierarchical_result(code="2512", confidence=0.95, fallback_used=False):
    return HierarchicalResult(
        code=code, label_en="Software Developers", label_ar="",
        confidence=confidence,
        stage_confidences={"stage1": confidence, "stage2": confidence, "stage3": confidence, "stage4": confidence},
        hierarchy_path=["2", "25", "251", code] if code else [],
        top_candidates=[UnitCandidate(code=code, label_en="Software Developers", label_ar="", score=confidence)] if code else [],
        hitl_required=confidence < 0.70,
        fallback_used=fallback_used,
    )


def _empty_hierarchical_result():
    return HierarchicalResult(
        code="", label_en="Unknown", label_ar="غير معروف", confidence=0.0,
        stage_confidences={"stage1": 0.0, "stage2": 0.0, "stage3": 0.0, "stage4": 0.0},
        hierarchy_path=[], top_candidates=[], hitl_required=True, fallback_used=True,
    )


class FakeOfficialStore:
    """Records constructor args and lets tests control search()/search_flat_only()."""

    last_constructed_kwargs: dict = {}

    def __init__(self, **kwargs):
        FakeOfficialStore.last_constructed_kwargs = kwargs
        self.search_result = _hierarchical_result()
        self.flat_only_result = _hierarchical_result(fallback_used=True)

    def search(self, *args, **kwargs):
        return self.search_result

    def search_flat_only(self, *args, **kwargs):
        return self.flat_only_result


@pytest.fixture
def fake_store_factory(monkeypatch):
    factory = MagicMock(side_effect=lambda **kw: FakeOfficialStore(**kw))
    monkeypatch.setattr("backend.agents.isco_classifier.HierarchicalISCOStore", factory)
    return factory


# ---------------------------------------------------------------------------
# 16. Official flat method label is distinct from flat_semantic
# ---------------------------------------------------------------------------

def test_official_hierarchical_method_label_distinct(fake_store_factory):
    clf = ISCOClassifier(isco_catalogue_profile=_OFFICIAL_PROFILE, enable_llm=False)
    result = clf.classify("software developer")
    assert result.method == f"hierarchical_isco08_{_OFFICIAL_PROFILE}"
    assert result.method != "hierarchical_semantic"
    assert result.method != "flat_semantic"


def test_official_flat_only_method_label_distinct(fake_store_factory):
    clf = ISCOClassifier(isco_catalogue_profile=_OFFICIAL_PROFILE, force_flat=True, enable_llm=False)
    result = clf.classify("software developer")
    assert result.method == f"flat_isco08_{_OFFICIAL_PROFILE}"
    assert result.method != "flat_semantic"
    # Constructor was actually asked to build the official-profile store
    # in force_flat_only mode -- not the legacy VectorStore/flat path.
    assert FakeOfficialStore.last_constructed_kwargs.get("profile") == _OFFICIAL_PROFILE
    assert clf._force_flat_only is True


def test_official_flat_only_uses_search_flat_only_not_search(fake_store_factory):
    clf = ISCOClassifier(isco_catalogue_profile=_OFFICIAL_PROFILE, force_flat=True, enable_llm=False)
    store = clf._hierarchical_store
    store.search = MagicMock(side_effect=AssertionError("search() must not be called in force_flat_only mode"))
    store.search_flat_only = MagicMock(return_value=_hierarchical_result(fallback_used=True))
    result = clf.classify("software developer")
    assert result.method == f"flat_isco08_{_OFFICIAL_PROFILE}"
    store.search_flat_only.assert_called_once()


# ---------------------------------------------------------------------------
# 17. Official profile unavailable/error is explicit, never silently legacy
# ---------------------------------------------------------------------------

def test_official_profile_unavailable_result_is_explicit(fake_store_factory):
    clf = ISCOClassifier(isco_catalogue_profile=_OFFICIAL_PROFILE, enable_llm=False)
    clf._hierarchical_store.search = MagicMock(return_value=_empty_hierarchical_result())
    result = clf.classify("some unclassifiable text")
    assert result.method == f"unavailable_isco08_{_OFFICIAL_PROFILE}"
    assert result.method not in ("flat_semantic", "hierarchical_semantic")
    assert result.primary.code == ""


def test_official_profile_construction_failure_never_falls_back_to_legacy(monkeypatch):
    monkeypatch.setattr(
        "backend.agents.isco_classifier.HierarchicalISCOStore",
        MagicMock(side_effect=RuntimeError("official collections absent")),
    )
    legacy_flat_spy = MagicMock(side_effect=AssertionError("legacy get_vector_store() must never be called for an official profile"))
    monkeypatch.setattr("backend.agents.isco_classifier.get_vector_store", legacy_flat_spy)
    clf = ISCOClassifier(isco_catalogue_profile=_OFFICIAL_PROFILE, enable_llm=False)
    assert clf._hierarchical_store is None
    assert clf._flat_store is None
    with pytest.raises(RuntimeError, match="no vector store initialised"):
        clf.classify("software developer")


# ---------------------------------------------------------------------------
# 18. Legacy profile / default compatibility remains covered
# ---------------------------------------------------------------------------

def test_default_profile_is_legacy_and_uses_singleton_getter(monkeypatch):
    singleton = MagicMock()
    singleton.search.return_value = _hierarchical_result()
    get_singleton = MagicMock(return_value=singleton)
    monkeypatch.setattr("backend.agents.isco_classifier.get_hierarchical_store", get_singleton)
    hierarchical_store_class_spy = MagicMock(side_effect=AssertionError("HierarchicalISCOStore(profile=...) must not be constructed for the default legacy profile"))
    monkeypatch.setattr("backend.agents.isco_classifier.HierarchicalISCOStore", hierarchical_store_class_spy)

    clf = ISCOClassifier(enable_llm=False)  # profile omitted -> legacy default
    assert clf._isco_catalogue_profile == LEGACY_PROFILE
    assert clf._force_flat_only is False
    result = clf.classify("software developer")
    assert result.method in ("hierarchical_semantic", "hierarchical_llm")
    get_singleton.assert_called_once()
