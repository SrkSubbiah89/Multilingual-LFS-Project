"""
Tests for eval/run_eval.py's Task 09 correction: making a retrieval-only
ISCO evaluation (--use-llm-reranker off) genuinely model-free.

Covers: ISCOClassifier(enable_llm=...) wiring through build_system(), CLI
validation for --use-llm-reranker off (no --reranker-model required),
config-hash honesty (a retrieval-only run's hash must not look like a
reranked run), and conditional ISIC/ISCED/SRE construction based on
whether the loaded (post --limit) test-set rows actually carry paired
industry_text/education_text.

Fully hermetic: ISCOClassifier/ISICClassifier/ISCEDClassifier/
SemanticRelationEngine are always monkeypatched to fakes/mocks. No Qdrant,
SentenceTransformer download, Ollama, paid LLM/API call, benchmark run, or
real dataset access occurs anywhere in this file. See
backend/tests/test_isco_classifier.py's TestEnableLlmFalse/
TestEnableLlmTrueUnchanged for the classifier-level (non-eval-harness)
enable_llm tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_eval  # noqa: E402


def _fake_isco_instance(reranker_model_resolved="none (reranking disabled)"):
    m = MagicMock()
    m.reranker_model_resolved = reranker_model_resolved
    m.classify.return_value = SimpleNamespace(
        primary=SimpleNamespace(code="2512", title_en="Software Developers", title_ar="", confidence=0.9),
        method="flat_semantic", hitl_required=False, reasoning="test",
    )
    return m


# ---------------------------------------------------------------------------
# 3. build_system(): use_llm_reranker threading
# ---------------------------------------------------------------------------

def test_build_system_hierarchical_reranker_on_passes_model_no_enable_llm_override(monkeypatch):
    captured = {}
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: captured.update(kw) or MagicMock()))
    run_eval.build_system("hierarchical", reranker_model="ollama/llama3.2:1b", use_llm_reranker=True)
    assert captured["reranker_model"] == "ollama/llama3.2:1b"
    assert captured.get("enable_llm", True) is True  # reranking-on retains the model pin, default enable_llm


def test_build_system_hierarchical_reranker_off_passes_enable_llm_false_and_no_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: captured.update(kw) or MagicMock()))
    # Even if a stray reranker_model value is passed in, off must ignore it.
    run_eval.build_system("hierarchical", reranker_model="ollama/llama3.2:1b", use_llm_reranker=False)
    assert captured["enable_llm"] is False
    assert captured["reranker_model"] is None


def test_build_system_flat_reranker_on_passes_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: captured.update(kw) or MagicMock()))
    run_eval.build_system("flat", reranker_model="ollama/llama3.2:1b", use_llm_reranker=True)
    assert captured["force_flat"] is True
    assert captured["reranker_model"] == "ollama/llama3.2:1b"
    assert captured.get("enable_llm", True) is True


def test_build_system_flat_reranker_off_passes_enable_llm_false_and_no_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: captured.update(kw) or MagicMock()))
    run_eval.build_system("flat", reranker_model=None, use_llm_reranker=False)
    assert captured["force_flat"] is True
    assert captured["enable_llm"] is False
    assert captured["reranker_model"] is None


def test_build_system_use_llm_reranker_default_true(monkeypatch):
    """Omitting use_llm_reranker entirely (existing callers predating this
    parameter) must behave exactly as use_llm_reranker=True."""
    captured = {}
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: captured.update(kw) or MagicMock()))
    run_eval.build_system("hierarchical", reranker_model="ollama/llama3.2:1b")
    assert captured["reranker_model"] == "ollama/llama3.2:1b"
    assert captured.get("enable_llm", True) is True


# ---------------------------------------------------------------------------
# 4. CLI validation
# ---------------------------------------------------------------------------

def _write_isco_only_test_set(tmp_path: Path) -> Path:
    test_set = tmp_path / "test.csv"
    test_set.write_text(
        "case_id,input_text,input_language,gold_isco_4digit\n"
        "c1,software developer,en,2512\n",
        encoding="utf-8",
    )
    return test_set


def test_cli_rejects_reranking_on_hierarchical_without_pin(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--test-set", str(test_set), "--system", "hierarchical"])
    with pytest.raises(SystemExit):
        run_eval.main()


def test_cli_rejects_reranking_on_flat_without_pin(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--test-set", str(test_set), "--system", "flat"])
    with pytest.raises(SystemExit):
        run_eval.main()


def test_cli_accepts_use_llm_reranker_off_without_reranker_model_hierarchical(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"

    captured = {}
    monkeypatch.setattr(
        run_eval, "ISCOClassifier",
        MagicMock(side_effect=lambda **kw: captured.update(kw) or _fake_isco_instance()),
    )
    # 5. Mocks that must fail if constructed -- no row in this CSV has
    # industry_text/education_text at all, so none of these may fire.
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError("ISICClassifier must not be constructed")))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError("ISCEDClassifier must not be constructed")))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError("SemanticRelationEngine must not be constructed")))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--output-dir", str(out_dir),
    ])
    run_eval.main()  # must not raise / must not call parser.error()

    assert captured["enable_llm"] is False
    assert captured["reranker_model"] is None


def test_cli_accepts_use_llm_reranker_off_without_reranker_model_flat(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"

    captured = {}
    monkeypatch.setattr(
        run_eval, "ISCOClassifier",
        MagicMock(side_effect=lambda **kw: captured.update(kw) or _fake_isco_instance()),
    )
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError("must not construct")))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError("must not construct")))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError("must not construct")))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "flat",
        "--use-llm-reranker", "off", "--output-dir", str(out_dir),
    ])
    run_eval.main()

    assert captured["force_flat"] is True
    assert captured["enable_llm"] is False
    assert captured["reranker_model"] is None


def test_cli_dry_run_semantics_unchanged_still_needs_no_reranker_model(tmp_path, monkeypatch):
    """--dry-run's existing exemption (no --reranker-model needed) must
    remain true regardless of --use-llm-reranker -- item 7 of Scope B
    forbids modifying --dry-run's documented semantics."""
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--dry-run", "--output-dir", str(out_dir),
    ])
    run_eval.main()  # must not raise -- dry-run never required a reranker model


# ---------------------------------------------------------------------------
# 5 & 6. Conditional ISIC/ISCED/SRE construction
# ---------------------------------------------------------------------------

def _isic_result():
    return SimpleNamespace(
        section="J", section_title="", division_code="62", division_title="",
        group_code="620", group_title="", class_code="6201", class_title="",
        confidence=0.9, method="keyword", alternatives=[], raw_text="",
    )


def _isced_result():
    return SimpleNamespace(
        level=6, level_title="", broad_code="06", broad_title="",
        narrow_code="061", narrow_title="", detailed_code="0613", detailed_title="",
        confidence=0.8, method="keyword", raw_text="",
    )


def test_main_does_not_construct_isic_isced_sre_when_no_row_has_paired_text(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: _fake_isco_instance()))
    isic_ctor = MagicMock(side_effect=AssertionError("ISICClassifier must not be constructed"))
    isced_ctor = MagicMock(side_effect=AssertionError("ISCEDClassifier must not be constructed"))
    sre_ctor = MagicMock(side_effect=AssertionError("SemanticRelationEngine must not be constructed"))
    monkeypatch.setattr(run_eval, "ISICClassifier", isic_ctor)
    monkeypatch.setattr(run_eval, "ISCEDClassifier", isced_ctor)
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", sre_ctor)

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--output-dir", str(out_dir),
    ])
    run_eval.main()

    isic_ctor.assert_not_called()
    isced_ctor.assert_not_called()
    sre_ctor.assert_not_called()


def test_main_does_not_construct_isic_isced_sre_when_text_is_one_sided(tmp_path, monkeypatch):
    """A row with only industry_text (no education_text) does not count as
    'paired' -- both must be present, matching run_one_case()'s own
    per-row guard exactly."""
    test_set = tmp_path / "test.csv"
    test_set.write_text(
        "case_id,input_text,input_language,gold_isco_4digit,industry_text,education_text\n"
        "c1,software developer,en,2512,software company,\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: _fake_isco_instance()))
    isic_ctor = MagicMock(side_effect=AssertionError("must not construct"))
    monkeypatch.setattr(run_eval, "ISICClassifier", isic_ctor)
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError("must not construct")))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError("must not construct")))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--output-dir", str(out_dir),
    ])
    run_eval.main()
    isic_ctor.assert_not_called()


def test_main_constructs_isic_isced_sre_when_a_row_has_paired_text(tmp_path, monkeypatch):
    test_set = tmp_path / "test.csv"
    test_set.write_text(
        "case_id,input_text,input_language,gold_isco_4digit,industry_text,education_text\n"
        "c1,software developer,en,2512,software company,bachelor of science\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: _fake_isco_instance()))

    isic_instance = MagicMock()
    isic_instance.classify.return_value = _isic_result()
    isced_instance = MagicMock()
    isced_instance.classify.return_value = _isced_result()
    sre_instance = MagicMock()
    sre_instance.analyse.return_value = SimpleNamespace(score=0.9, violations=[])

    isic_ctor = MagicMock(return_value=isic_instance)
    isced_ctor = MagicMock(return_value=isced_instance)
    sre_ctor = MagicMock(return_value=sre_instance)
    monkeypatch.setattr(run_eval, "ISICClassifier", isic_ctor)
    monkeypatch.setattr(run_eval, "ISCEDClassifier", isced_ctor)
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", sre_ctor)

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--output-dir", str(out_dir),
    ])
    run_eval.main()

    isic_ctor.assert_called_once()
    isced_ctor.assert_called_once()
    sre_ctor.assert_called_once()
    isic_instance.classify.assert_called_once_with("software company")
    isced_instance.classify.assert_called_once_with("bachelor of science")


def test_main_sre_off_still_runs_isic_isced_only_sre_skipped(tmp_path, monkeypatch):
    """Task 05.1/Step 5.1 guarantee, re-verified through main(): --sre off
    disables only coherence analysis, never ISIC/ISCED classification."""
    test_set = tmp_path / "test.csv"
    test_set.write_text(
        "case_id,input_text,input_language,gold_isco_4digit,industry_text,education_text\n"
        "c1,software developer,en,2512,software company,bachelor of science\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: _fake_isco_instance()))

    isic_instance = MagicMock()
    isic_instance.classify.return_value = _isic_result()
    isced_instance = MagicMock()
    isced_instance.classify.return_value = _isced_result()
    sre_instance = MagicMock()
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(return_value=isic_instance))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(return_value=isced_instance))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(return_value=sre_instance))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--sre", "off", "--output-dir", str(out_dir),
    ])
    run_eval.main()

    isic_instance.classify.assert_called_once()
    isced_instance.classify.assert_called_once()
    sre_instance.analyse.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Config-hash / retrieval-only metadata honesty
# ---------------------------------------------------------------------------

def _hash_args(**overrides):
    base = dict(
        system="hierarchical", beam=2, stage1_mode="description",
        reranker_candidates=5, branch_collapse=False, config=None, sre="on",
        use_llm_reranker="on",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_config_hash_reflects_use_llm_reranker_off_not_hardcoded_true():
    hash_on = run_eval._config_hash(
        _hash_args(use_llm_reranker="on"), "anthropic/claude-3-5-sonnet-20241022", True,
    )
    hash_off = run_eval._config_hash(
        _hash_args(use_llm_reranker="off"), "none (reranking disabled)", True,
    )
    assert hash_on != hash_off


def test_config_hash_same_inputs_are_deterministic():
    a = run_eval._config_hash(_hash_args(use_llm_reranker="off"), "none (reranking disabled)", True)
    b = run_eval._config_hash(_hash_args(use_llm_reranker="off"), "none (reranking disabled)", True)
    assert a == b
