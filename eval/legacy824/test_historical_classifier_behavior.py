"""
Tests for the historical ISCOClassifier's own decision logic, loaded
verbatim from the detached LEGACY_SHA worktree via
eval/legacy824/historical_loader.py (Task 39, scenarios 3-8).

Fully hermetic: `get_vector_store` is monkeypatched to return a small
in-memory fake store with scripted OccupationMatch confidences, and
`Crew` is monkeypatched to a scripted stand-in that never makes a
network call. No real Qdrant, embedding model, or Anthropic call
happens anywhere in this file. `ANTHROPIC_API_KEY` is monkeypatched to
a syntactically-valid but fake value only so the historical
`get_llm(TaskType.CRITICAL)` factory's own environment check
(unmodified, at LEGACY_SHA) does not raise -- the real crewai.LLM
object it constructs is never actually called (Crew is scripted).

Skipped gracefully if the Task 39 detached worktree is not present in
the current environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy824.historical_loader import load_historical_isco_classifier  # noqa: E402

WORKTREE_PATH = Path("C:/task39_legacy824_worktree")

pytestmark = pytest.mark.skipif(
    not WORKTREE_PATH.exists(), reason="Task 39 detached worktree not present in this environment"
)


@pytest.fixture
def loaded():
    return load_historical_isco_classifier(WORKTREE_PATH)


class _FakeStore:
    """Scripted stand-in for the historical VectorStore.search()."""

    def __init__(self, matches):
        self._matches = matches

    def search(self, query, top_k=5):
        return self._matches[:top_k]


class _ScriptedCrew:
    """Scripted stand-in for crewai.Crew -- never makes a network call."""

    calls: list = []
    script: list = ["{}"]

    def __init__(self, agents, tasks, verbose=False):
        self._tasks = tasks

    def kickoff(self):
        idx = min(len(type(self).calls), len(type(self).script) - 1)
        type(self).calls.append(1)
        resp = type(self).script[idx]
        if isinstance(resp, Exception):
            raise resp
        return resp


def _make_matches(vector_store_mod, confidences_with_codes):
    OccupationMatch = vector_store_mod.OccupationMatch
    return [
        OccupationMatch(
            code=code, title_en=f"Title {code}", title_ar=f"عنوان {code}",
            level=4, description=f"desc {code}", confidence=conf,
        )
        for code, conf in confidences_with_codes
    ]


def _build_classifier(loaded, monkeypatch, matches, crew_script):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(loaded.isco_classifier, "get_vector_store", lambda: _FakeStore(matches))
    _ScriptedCrew.calls = []
    _ScriptedCrew.script = crew_script
    monkeypatch.setattr(loaded.isco_classifier, "Crew", _ScriptedCrew)
    return loaded.isco_classifier.ISCOClassifier()


_FIVE_CANDIDATES_LOW = [
    ("2512", 0.85), ("2511", 0.80), ("2519", 0.75), ("2521", 0.70), ("2522", 0.65),
]


def test_confidence_exactly_0_92_does_not_call_llm(loaded, monkeypatch):
    matches = _make_matches(loaded.vector_store, [("2512", 0.92), ("2511", 0.80), ("2519", 0.75), ("2521", 0.70), ("2522", 0.65)])
    clf = _build_classifier(loaded, monkeypatch, matches, crew_script=[Exception("must not be called")])
    result = clf.classify("software developer")
    assert result.method == "semantic"
    assert result.primary.code == "2512"
    assert len(_ScriptedCrew.calls) == 0


def test_confidence_above_0_92_does_not_call_llm(loaded, monkeypatch):
    matches = _make_matches(loaded.vector_store, [("2512", 0.97), ("2511", 0.80), ("2519", 0.75), ("2521", 0.70), ("2522", 0.65)])
    clf = _build_classifier(loaded, monkeypatch, matches, crew_script=[Exception("must not be called")])
    result = clf.classify("software developer")
    assert result.method == "semantic"
    assert len(_ScriptedCrew.calls) == 0


def test_confidence_below_0_92_calls_llm_exactly_once(loaded, monkeypatch):
    matches = _make_matches(loaded.vector_store, _FIVE_CANDIDATES_LOW)
    clf = _build_classifier(
        loaded, monkeypatch, matches,
        crew_script=['{"selected_code": "2511", "reasoning": "closer match"}'],
    )
    result = clf.classify("codes stuff")
    assert result.method == "llm_ranked"
    assert len(_ScriptedCrew.calls) == 1


def test_llm_receives_exactly_five_ordered_candidates_and_chooses_only_among_them(loaded, monkeypatch):
    matches = _make_matches(loaded.vector_store, _FIVE_CANDIDATES_LOW)
    captured_tasks = []

    class _CapturingCrew(_ScriptedCrew):
        def __init__(self, agents, tasks, verbose=False):
            captured_tasks.extend(tasks)
            super().__init__(agents, tasks, verbose)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(loaded.isco_classifier, "get_vector_store", lambda: _FakeStore(matches))
    _CapturingCrew.calls = []
    _CapturingCrew.script = ['{"selected_code": "2521", "reasoning": "best of the five"}']
    monkeypatch.setattr(loaded.isco_classifier, "Crew", _CapturingCrew)
    clf = loaded.isco_classifier.ISCOClassifier()

    result = clf.classify("codes stuff")
    assert result.method == "llm_ranked"
    assert result.primary.code == "2521"
    assert [c.code for c in result.candidates] == ["2512", "2511", "2519", "2521", "2522"]
    task_description = captured_tasks[0].description
    for code in ["2512", "2511", "2519", "2521", "2522"]:
        assert f"[{code}]" in task_description


def test_valid_selected_code_preserves_llm_ranked_behavior(loaded, monkeypatch):
    matches = _make_matches(loaded.vector_store, _FIVE_CANDIDATES_LOW)
    clf = _build_classifier(
        loaded, monkeypatch, matches,
        crew_script=['{"selected_code": "2519", "reasoning": "most specific match"}'],
    )
    result = clf.classify("codes stuff")
    assert result.method == "llm_ranked"
    assert result.primary.code == "2519"
    assert result.reasoning == "most specific match"


def test_invalid_json_preserves_historical_fallback(loaded, monkeypatch):
    matches = _make_matches(loaded.vector_store, _FIVE_CANDIDATES_LOW)
    clf = _build_classifier(loaded, monkeypatch, matches, crew_script=["not valid json at all {{{"])
    result = clf.classify("codes stuff")
    assert result.method == "llm_ranked"  # classify() always labels the low-confidence branch this way
    assert result.primary.code == "2512"  # falls back to top semantic candidate
    assert result.reasoning == "Fallback to top semantic match (LLM response could not be parsed)."


def test_out_of_candidate_code_preserves_historical_fallback(loaded, monkeypatch):
    matches = _make_matches(loaded.vector_store, _FIVE_CANDIDATES_LOW)
    clf = _build_classifier(
        loaded, monkeypatch, matches,
        crew_script=['{"selected_code": "9999", "reasoning": "not a real candidate"}'],
    )
    result = clf.classify("codes stuff")
    assert result.primary.code == "2512"
    assert result.reasoning == "Fallback to top semantic match (LLM response could not be parsed)."


def test_llm_exception_propagates_to_caller(loaded, monkeypatch):
    """The historical classify() does not catch a Crew.kickoff() exception
    -- it propagates to the caller (the development-run adapter is
    responsible for converting this into a stopped preflight result;
    see test_adapter.py)."""
    matches = _make_matches(loaded.vector_store, _FIVE_CANDIDATES_LOW)
    clf = _build_classifier(loaded, monkeypatch, matches, crew_script=[RuntimeError("simulated API failure")])
    with pytest.raises(RuntimeError, match="simulated API failure"):
        clf.classify("codes stuff")
