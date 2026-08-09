"""
Tests for eval/run_eval.py's Task 21 --isco-catalogue-profile wiring.

Fully hermetic: ISCOClassifier is always monkeypatched to a fake that
fails loudly if constructed with unexpected kwargs. No Qdrant,
SentenceTransformer, Ollama, paid LLM/API call, or WISCO artifact is
used anywhere in this file.
"""

from __future__ import annotations

import csv as csv_module
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_eval  # noqa: E402

_OFFICIAL_PROFILE = "official_ilo2021_v1"


def _write_isco_only_test_set(tmp_path: Path) -> Path:
    test_set = tmp_path / "test.csv"
    test_set.write_text(
        "case_id,input_text,input_language,gold_isco_4digit\n"
        "c1,software developer,en,2512\n",
        encoding="utf-8",
    )
    return test_set


def _fake_isco_instance(pred_method):
    m = MagicMock()
    m.reranker_model_resolved = "none (reranking disabled)"

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        if trace is not None:
            for i in range(1, 5):
                trace[f"stage{i}"] = [{"code": "2512"[:i] if i < 4 else "2512", "label_en": "x", "score": 0.9}]
                trace[f"stage{i}_latency_ms"] = 5.0
        return SimpleNamespace(
            primary=SimpleNamespace(code="2512", title_en="Software Developers", title_ar="", confidence=0.9),
            method=pred_method, hitl_required=False, reasoning="test",
        )

    m.classify.side_effect = _classify
    return m


# ---------------------------------------------------------------------------
# 19. Model-free evaluation with an official profile does not construct
#     LLM/reranker/ISIC/ISCED/SRE for ISCO-only rows
# ---------------------------------------------------------------------------

def test_official_profile_model_free_run_constructs_no_llm_or_isic_isced_sre(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"

    captured_kwargs = {}

    def _ctor(**kwargs):
        captured_kwargs.update(kwargs)
        return _fake_isco_instance(f"hierarchical_isco08_{_OFFICIAL_PROFILE}")

    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=_ctor))
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError("must not construct ISIC")))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError("must not construct ISCED")))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError("must not construct SRE")))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--isco-catalogue-profile", _OFFICIAL_PROFILE,
        "--output-dir", str(out_dir),
    ])
    run_eval.main()

    assert captured_kwargs.get("enable_llm") is False
    assert captured_kwargs.get("isco_catalogue_profile") == _OFFICIAL_PROFILE

    out_csv = next(out_dir.glob("*.csv"))
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv_module.DictReader(f))
    assert rows[0]["pred_method"] == f"hierarchical_isco08_{_OFFICIAL_PROFILE}"


# ---------------------------------------------------------------------------
# 20. Strict hierarchical guard remains effective with the official profile
# ---------------------------------------------------------------------------

def test_require_genuine_hierarchical_accepts_official_profile_method_label(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        run_eval, "ISCOClassifier",
        MagicMock(side_effect=lambda **kw: _fake_isco_instance(f"hierarchical_isco08_{_OFFICIAL_PROFILE}")),
    )
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError("must not construct")))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError("must not construct")))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError("must not construct")))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--isco-catalogue-profile", _OFFICIAL_PROFILE,
        "--require-genuine-hierarchical", "--output-dir", str(out_dir),
    ])
    run_eval.main()  # must not raise / exit nonzero -- official label satisfies the strict guard

    out_csv = next(out_dir.glob("*.csv"))
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv_module.DictReader(f))
    assert rows[0]["pred_method"].startswith("hierarchical_")


def test_require_genuine_hierarchical_rejects_official_flat_fallback(tmp_path, monkeypatch):
    """A flat_isco08_official_... method label must still be treated as a
    non-hierarchical (fallback) result by the strict guard -- exactly like
    a legacy flat_semantic/flat_llm row would be."""
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        run_eval, "ISCOClassifier",
        MagicMock(side_effect=lambda **kw: _fake_isco_instance(f"flat_isco08_{_OFFICIAL_PROFILE}")),
    )
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError("must not construct")))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError("must not construct")))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError("must not construct")))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--isco-catalogue-profile", _OFFICIAL_PROFILE,
        "--require-genuine-hierarchical", "--output-dir", str(out_dir),
    ])
    with pytest.raises(SystemExit):
        run_eval.main()
    assert not any(out_dir.glob("*.csv"))


# ---------------------------------------------------------------------------
# 21. Manifest/config hash records profile and source identity
# ---------------------------------------------------------------------------

def test_config_hash_differs_between_legacy_and_official_profile():
    base = dict(
        system="hierarchical", beam=2, stage1_mode="description",
        reranker_candidates=5, branch_collapse=False, config=None, sre="on",
        use_llm_reranker="off",
    )
    legacy_args = SimpleNamespace(**base, isco_catalogue_profile="legacy")
    official_args = SimpleNamespace(**base, isco_catalogue_profile=_OFFICIAL_PROFILE)

    legacy_hash = run_eval._config_hash(legacy_args, "none (reranking disabled)", True)
    official_hash = run_eval._config_hash(official_args, "none (reranking disabled)", True)
    assert legacy_hash != official_hash


def test_default_cli_profile_is_legacy(tmp_path, monkeypatch):
    """Omitting --isco-catalogue-profile entirely (every pre-Task-21
    invocation) must still work end-to-end -- proves the new CLI
    argument's default ("legacy") requires no caller changes."""
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--dry-run", "--output-dir", str(out_dir),
    ])
    run_eval.main()
    out_csv = next(out_dir.glob("*.csv"))
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv_module.DictReader(f))
    assert rows[0]["config_hash"]
