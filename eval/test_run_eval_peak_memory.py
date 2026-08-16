"""
Tests for real per-run peak-memory instrumentation in eval/run_eval.py
(Conference I Reviewer #2 response, Section D / reviewer comment 4:
computational analysis is missing). Closes the gap eval/manifest.py's
docstring used to describe as a permanent, pre-existing hole:
CaseResult.peak_memory_mb was a declared field that no code path ever
assigned. main()'s loop now samples process RSS (via _peak_rss_mb(), the
same optional-psutil pattern eval/dev_sweep.py already uses) right after
each run_one_case() call.

Fully hermetic: ISCOClassifier is always monkeypatched to a fake -- same
pattern as eval/test_flat_query_telemetry_serialization.py. No Qdrant,
embedding model, LLM, or WISCO artifact is used anywhere in this file.
"""

from __future__ import annotations

import csv as csv_module
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_eval  # noqa: E402


def _write_test_set(tmp_path: Path) -> Path:
    test_set = tmp_path / "test.csv"
    test_set.write_text(
        "case_id,input_text,input_language,gold_isco_4digit\n"
        "c1,software developer,en,2512\n"
        "c2,cleaner,en,9112\n",
        encoding="utf-8",
    )
    return test_set


def _fake_isco_instance():
    m = MagicMock()
    m.reranker_model_resolved = "none (reranking disabled)"

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        return SimpleNamespace(
            primary=SimpleNamespace(code="2512", title_en="Software Developers", title_ar="", confidence=0.9),
            method="flat_semantic", hitl_required=False, reasoning="test",
        )

    m.classify.side_effect = _classify
    return m


def _patch_fake_classifiers(monkeypatch):
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: _fake_isco_instance()))
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError("must not construct ISIC")))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError("must not construct ISCED")))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError("must not construct SRE")))


# ---------------------------------------------------------------------------
# _peak_rss_mb() itself
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not run_eval._HAVE_PSUTIL,
    reason="psutil is optional at runtime (pinned in requirements-dev.txt "
           "only, not requirements.txt -- a clean production install "
           "genuinely does not have it; found via Prompt 8's fresh-venv check)",
)
def test_peak_rss_mb_returns_a_real_positive_float_when_psutil_available():
    sample = run_eval._peak_rss_mb()
    assert isinstance(sample, float)
    assert sample > 0.0


def test_peak_rss_mb_returns_none_when_psutil_unavailable(monkeypatch):
    monkeypatch.setattr(run_eval, "_HAVE_PSUTIL", False)
    assert run_eval._peak_rss_mb() is None


@pytest.mark.skipif(
    not run_eval._HAVE_PSUTIL,
    reason="requires monkeypatching the real `psutil` module attribute, which "
           "run_eval.py never sets at all when the optional import fails",
)
def test_peak_rss_mb_returns_none_on_probe_failure(monkeypatch):
    monkeypatch.setattr(run_eval, "_HAVE_PSUTIL", True)
    broken_psutil = MagicMock()
    broken_psutil.Process.side_effect = RuntimeError("boom")
    monkeypatch.setattr(run_eval, "psutil", broken_psutil)
    assert run_eval._peak_rss_mb() is None


# ---------------------------------------------------------------------------
# CaseResult default is unaffected -- run_one_case() itself never samples;
# only main()'s loop does, after the call returns.
# ---------------------------------------------------------------------------

def test_case_result_peak_memory_mb_still_defaults_to_none():
    r = run_eval.CaseResult(case_id="c1", input_text="x", input_language="en")
    assert r.peak_memory_mb is None


# ---------------------------------------------------------------------------
# End-to-end via main(): a real (non-dry-run) run populates peak_memory_mb
# with a real number in the output CSV for every case row.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not run_eval._HAVE_PSUTIL,
    reason="psutil is optional at runtime (requirements-dev.txt only) -- "
           "not installed in this environment, so peak_memory_mb is "
           "expected to stay blank; see test_main_leaves_peak_memory_mb_blank_without_psutil",
)
def test_main_populates_peak_memory_mb_on_a_real_run(tmp_path, monkeypatch):
    test_set = _write_test_set(tmp_path)
    out_dir = tmp_path / "out"
    _patch_fake_classifiers(monkeypatch)

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "flat",
        "--use-llm-reranker", "off", "--output-dir", str(out_dir),
    ])
    run_eval.main()

    out_csv = next(out_dir.glob("*.csv"))
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv_module.DictReader(f))

    assert len(rows) == 2
    for row in rows:
        value = row["peak_memory_mb"]
        assert value not in ("", None), "peak_memory_mb must be populated on a real run with psutil available"
        assert float(value) > 0.0


def test_main_leaves_peak_memory_mb_blank_without_psutil(tmp_path, monkeypatch):
    """Mirrors test_main_populates_peak_memory_mb_on_a_real_run for the
    genuinely-no-psutil case (a clean `pip install -r requirements.txt`,
    with no requirements-dev.txt) -- runs regardless of this environment's
    real psutil availability by forcing _HAVE_PSUTIL False, so this test
    isn't itself skippable and always exercises the degrade path."""
    monkeypatch.setattr(run_eval, "_HAVE_PSUTIL", False)
    test_set = _write_test_set(tmp_path)
    out_dir = tmp_path / "out"
    _patch_fake_classifiers(monkeypatch)

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "flat",
        "--use-llm-reranker", "off", "--output-dir", str(out_dir),
    ])
    run_eval.main()

    out_csv = next(out_dir.glob("*.csv"))
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv_module.DictReader(f))

    assert len(rows) == 2
    for row in rows:
        assert row["peak_memory_mb"] in ("", None)


def test_main_dry_run_leaves_peak_memory_mb_blank(tmp_path, monkeypatch):
    """A dry run never invokes a classifier, so it must never sample or
    fabricate a memory figure either -- build_dry_run_case_result() is a
    fully separate code path from main()'s real-run loop and is untouched
    by this change."""
    test_set = _write_test_set(tmp_path)
    out_dir = tmp_path / "out"
    _patch_fake_classifiers(monkeypatch)

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "flat",
        "--use-llm-reranker", "off", "--output-dir", str(out_dir), "--dry-run",
    ])
    run_eval.main()

    out_csv = next(out_dir.glob("*.csv"))
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv_module.DictReader(f))

    assert len(rows) == 2
    for row in rows:
        assert row["peak_memory_mb"] in ("", None)
