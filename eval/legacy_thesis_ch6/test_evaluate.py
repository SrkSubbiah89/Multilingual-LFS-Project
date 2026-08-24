"""
eval/legacy_thesis_ch6/test_evaluate.py

Moved 2026-08-24 from backend/tests/test_evaluation.py (a documentation-
completeness audit found backend/evaluation/ had no real production
coupling -- only this test file imported it -- so it was relocated
under eval/ alongside every other evaluation-only module, matching the
eval/legacy824/-style convention of tests living next to the code they
cover). Included in `pytest backend/tests eval/ -q`, same as before;
only its location changed, not the test command.

16 unit tests for eval/legacy_thesis_ch6/evaluate.py.
Fully offline — no Qdrant, no Ollama, no model loading required.

Stub strategy
-------------
evaluate.py only imports `rank_bm25` and `sklearn` at module level.
All backend service imports (vector_store, isco_classifier) are lazy —
inside __init__ methods.  We therefore install lightweight stubs via an
autouse module-scoped fixture so they are active only during this test
module's execution and are cleaned up afterwards.  This prevents any
interference with test_vector_store.py and test_isco_classifier.py which
need the real modules.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# evaluate.py can be imported safely at collection time — its module-level
# code only touches rank_bm25 and sklearn, not any backend services.
# ---------------------------------------------------------------------------
from eval.legacy_thesis_ch6.evaluate import (
    TEST_CASES,
    HITL_THRESHOLD,
    BM25Baseline,
    FlatVectorBaseline,
    HierarchicalRAG,
    PredictionResult,
    SystemMetrics,
    evaluate_system,
    print_comparison_table,
    save_results_csv,
)

# ---------------------------------------------------------------------------
# Keys that the stubs occupy in sys.modules
# ---------------------------------------------------------------------------
_STUB_KEYS = [
    "backend.rag.vector_store",
    "backend.agents.isco_classifier",
]


def _build_stubs() -> dict:
    """Return a mapping of module-name → stub ModuleType."""
    stubs: dict[str, types.ModuleType] = {}

    # ── backend.rag.vector_store ─────────────────────────────────────────────
    vs_mod = types.ModuleType("backend.rag.vector_store")
    fake_data = [
        {"code": "2512", "level": 4, "title_en": "Software Developer",
         "title_ar": "مطور برمجيات", "description": "write backend APIs"},
        {"code": "2211", "level": 4, "title_en": "Medical Doctor",
         "title_ar": "طبيب", "description": "diagnose and treat patients"},
        {"code": "5120", "level": 4, "title_en": "Cook",
         "title_ar": "طاهٍ", "description": "prepare meals in a kitchen"},
        {"code": "7411", "level": 4, "title_en": "Electrician",
         "title_ar": "كهربائي", "description": "install electrical wiring"},
        {"code": "9112", "level": 4, "title_en": "Cleaner",
         "title_ar": "عامل نظافة", "description": "sweep and mop floors"},
    ]
    vs_mod._ISCO_DATA = fake_data

    flat_match = MagicMock()
    flat_match.code = "2512"
    flat_match.confidence = 0.88
    vs_instance = MagicMock()
    vs_instance.search.return_value = [flat_match]
    vs_mod.get_vector_store = MagicMock(return_value=vs_instance)
    stubs["backend.rag.vector_store"] = vs_mod

    # ── backend.agents.isco_classifier ──────────────────────────────────────
    clf_mod = types.ModuleType("backend.agents.isco_classifier")

    primary = MagicMock()
    primary.code = "2512"
    primary.confidence = 0.91

    clf_result = MagicMock()
    clf_result.primary = primary
    clf_result.alternatives = []

    clf_instance = MagicMock()
    clf_instance.classify.return_value = clf_result

    clf_mod.ISCOClassifier = MagicMock(return_value=clf_instance)
    stubs["backend.agents.isco_classifier"] = clf_mod

    return stubs


@pytest.fixture(autouse=True, scope="module")
def service_stubs():
    """Install lightweight stubs for the duration of this test module only."""
    # Save originals (may or may not exist)
    saved = {k: sys.modules.pop(k, None) for k in _STUB_KEYS}

    # Install stubs
    stubs = _build_stubs()
    sys.modules.update(stubs)

    yield

    # Restore originals; remove stubs
    for k in _STUB_KEYS:
        sys.modules.pop(k, None)
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v


# ===========================================================================
# 1. Test corpus integrity
# ===========================================================================

def test_corpus_has_100_cases():
    assert len(TEST_CASES) == 100


def test_each_case_is_2_tuple():
    for case in TEST_CASES:
        assert len(case) == 2, f"Expected 2-tuple, got: {case}"


def test_isco_codes_are_4_digits():
    bad = [code for _, code in TEST_CASES
           if not (len(code) == 4 and code.isdigit())]
    assert bad == [], f"Non 4-digit codes: {bad[:5]}"


def test_all_10_major_groups_covered():
    majors = {code[0] for _, code in TEST_CASES}
    assert majors == {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}


# ===========================================================================
# 2. BM25Baseline
# ===========================================================================

def test_bm25_predict_returns_prediction_result():
    bm25 = BM25Baseline()
    res  = bm25.predict("software developer")
    assert isinstance(res, PredictionResult)


def test_bm25_confidence_in_range():
    bm25 = BM25Baseline()
    res  = bm25.predict("cook preparing meals")
    assert 0.0 <= res.confidence <= 1.0


def test_bm25_top3_length_respects_top_k():
    bm25 = BM25Baseline()
    res  = bm25.predict("electrician", top_k=3)
    assert len(res.top3_codes) <= 3


def test_bm25_predicted_in_top3():
    bm25 = BM25Baseline()
    res  = bm25.predict("cleaner office floors")
    assert res.predicted_code in res.top3_codes


# ===========================================================================
# 3. FlatVectorBaseline
# ===========================================================================

def test_flat_predict_returns_prediction_result():
    flat = FlatVectorBaseline()
    res  = flat.predict("nurse")
    assert isinstance(res, PredictionResult)


def test_flat_predicted_in_top3():
    flat = FlatVectorBaseline()
    res  = flat.predict("developer", top_k=3)
    assert res.predicted_code in res.top3_codes


# ===========================================================================
# 4. HierarchicalRAG
# ===========================================================================

def test_hierarchical_predict_returns_prediction_result():
    hier = HierarchicalRAG()
    res  = hier.predict("software engineer")
    assert isinstance(res, PredictionResult)


def test_hierarchical_confidence_in_range():
    hier = HierarchicalRAG()
    res  = hier.predict("doctor at a clinic")
    assert 0.0 <= res.confidence <= 1.0


# ===========================================================================
# 5. evaluate_system
# ===========================================================================

def test_evaluate_system_returns_system_metrics():
    def perfect(text, top_k):
        code = TEST_CASES[0][1]
        return PredictionResult(code, [code], confidence=0.95, latency_ms=5.0)

    m = evaluate_system("test", perfect, [TEST_CASES[0]])
    assert isinstance(m, SystemMetrics)


def test_hitl_rate_is_one_when_all_low_confidence():
    def low_conf(text, top_k):
        return PredictionResult("2512", ["2512"], confidence=0.50, latency_ms=1.0)

    m = evaluate_system("low", low_conf, TEST_CASES[:10])
    assert m.hitl_rate == 1.0


def test_hitl_rate_is_zero_when_all_high_confidence():
    def high_conf(text, top_k):
        return PredictionResult("2512", ["2512"], confidence=0.99, latency_ms=1.0)

    m = evaluate_system("high", high_conf, TEST_CASES[:10])
    assert m.hitl_rate == 0.0


# ===========================================================================
# 6. print_comparison_table smoke test
# ===========================================================================

def test_comparison_table_printed(capsys):
    m = SystemMetrics(
        system_name="bm25",
        top1_accuracy=0.55,
        top3_accuracy=0.75,
        cohen_kappa=0.48,
        hitl_rate=0.30,
        avg_latency_ms=8.5,
        p95_latency_ms=12.0,
        p99_latency_ms=18.0,
        n_evaluated=100,
    )
    print_comparison_table({"bm25": m})
    out = capsys.readouterr().out
    assert "bm25" in out
    assert "Top-1 Accuracy" in out
    assert "Cohen" in out
