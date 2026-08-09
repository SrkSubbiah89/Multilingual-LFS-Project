"""
Tests for eval/analyze_official_tier1.py (Task 37).

Fully hermetic: every fixture is a small, hand-constructed temporary
CSV/catalogue/git-repo. No 18,747-row local WISCO data, network, live
Qdrant, model, LLM, or reranker resource is read anywhere in this file.
Check 10 (Task 36 report byte-identity) is exercised against a real,
purely-local, ephemeral git repository created per-test -- this is
still fully hermetic (no network, no remote) but avoids faking git
plumbing with a bypass flag.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_official_tier1 as aot  # noqa: E402
from analyze import wilson_score_interval, mcnemar_test  # noqa: E402

REFERENCE_FIELDS = ["case_id", "input_text", "input_language", "gold_isco_4digit", "gold_isic", "gold_isced"]

RESULT_FIELDS = [
    "case_id", "gold_isco_4digit", "pred_isco_4digit", "pred_method", "error",
    "reranker_fired", "reranker_model", "estimated_cost_usd", "prompt_tokens", "completion_tokens",
    "sre_status", "pred_isic_section", "pred_isced_level",
    "stage1_candidates", "stage2_candidates", "stage3_candidates", "stage4_candidates",
    "stage1_latency_ms", "stage2_latency_ms", "stage3_latency_ms", "stage4_latency_ms",
    "hier_stage_query_telemetry", "keyword_anchor_retry_used", "stage1_source",
    "flat_query_duration_ms", "flat_query_outcome", "flat_query_attempts",
]

_STAGE_JSON = '[{"code": "1", "label_en": "x", "score": 0.9}]'

_CLEAN_STAGE_TELEMETRY = json.dumps({
    f"stage{n}": {
        "queries": 1, "any_retry": False, "any_exception": False, "max_attempts_used": 1,
        "exception_types": [], "stage_budget_exhausted": False,
        "configured_query_timeout_seconds": 8.0, "initial_stage_budget_ms": 30000.0,
        "queries_detail": [{"outcome": "success", "attempts": 1, "attempt_durations_ms": [5.0],
                             "remaining_stage_budget_ms_at_entry": 30000.0}],
    }
    for n in (1, 2, 3, 4)
})

# a=both correct, b=flat-correct/hier-wrong, c=flat-wrong/hier-correct, d=both wrong
_CASES = [
    ("c1", "en", "1234", "1234", "1234"),
    ("c2", "ar", "1234", "1234", "5678"),
    ("c3", "hi", "5678", "1234", "5678"),
    ("c4", "tl", "5678", "1234", "1234"),
]

_CATALOGUE_COLUMNS = ["level", "code", "parent_code", "label"]
_CATALOGUE_ROWS = [
    {"level": "major", "code": "1", "parent_code": "", "label": "Major One"},
    {"level": "major", "code": "5", "parent_code": "", "label": "Major Five"},
    {"level": "submajor", "code": "12", "parent_code": "1", "label": "Submajor One-Two"},
    {"level": "submajor", "code": "56", "parent_code": "5", "label": "Submajor Five-Six"},
    {"level": "minor", "code": "123", "parent_code": "12", "label": "Minor One-Two-Three"},
    {"level": "minor", "code": "567", "parent_code": "56", "label": "Minor Five-Six-Seven"},
    {"level": "unit", "code": "1234", "parent_code": "123", "label": "Unit One-Two-Three-Four"},
    {"level": "unit", "code": "5678", "parent_code": "567", "label": "Unit Five-Six-Seven-Eight"},
]
_CATALOGUE_EXPECTED_COUNTS = {"major": 2, "submajor": 2, "minor": 2, "unit": 2}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _reference_rows(cases=_CASES):
    return [
        {"case_id": cid, "input_text": f"job {cid}", "input_language": lang,
         "gold_isco_4digit": gold, "gold_isic": "", "gold_isced": ""}
        for cid, lang, gold, _f, _h in cases
    ]


def _base_result_row(case_id, gold4, pred4, method) -> dict:
    return {
        "case_id": case_id, "gold_isco_4digit": gold4, "pred_isco_4digit": pred4,
        "pred_method": method, "error": "", "reranker_fired": "False",
        "reranker_model": "", "estimated_cost_usd": "0.0", "prompt_tokens": "0",
        "completion_tokens": "0", "sre_status": "not_applicable",
        "pred_isic_section": "", "pred_isced_level": "",
        "stage1_candidates": "null", "stage2_candidates": "null",
        "stage3_candidates": "null", "stage4_candidates": "null",
        "stage1_latency_ms": "", "stage2_latency_ms": "", "stage3_latency_ms": "", "stage4_latency_ms": "",
        "hier_stage_query_telemetry": "", "keyword_anchor_retry_used": "False", "stage1_source": "",
        "flat_query_duration_ms": "", "flat_query_outcome": "", "flat_query_attempts": "",
    }


def _flat_rows(cases=_CASES, method=aot.EXPECTED_FLAT_METHOD):
    rows = []
    for cid, _lang, gold, fpred, _hpred in cases:
        r = _base_result_row(cid, gold, fpred, method)
        r["flat_query_duration_ms"] = "8.5"
        r["flat_query_outcome"] = "success"
        r["flat_query_attempts"] = "1"
        rows.append(r)
    return rows


def _hier_rows(cases=_CASES, method=aot.EXPECTED_HIERARCHICAL_METHOD):
    rows = []
    for cid, _lang, gold, _fpred, hpred in cases:
        r = _base_result_row(cid, gold, hpred, method)
        for c in ("stage1_candidates", "stage2_candidates", "stage3_candidates", "stage4_candidates"):
            r[c] = _STAGE_JSON
        for c in ("stage1_latency_ms", "stage2_latency_ms", "stage3_latency_ms", "stage4_latency_ms"):
            r[c] = "10.0"
        r["hier_stage_query_telemetry"] = _CLEAN_STAGE_TELEMETRY
        r["stage1_source"] = "semantic_retrieval"
        rows.append(r)
    return rows


def _write_catalogue(tmp_path: Path) -> Path:
    p = tmp_path / "catalogue.csv"
    _write_csv(p, _CATALOGUE_COLUMNS, _CATALOGUE_ROWS)
    return p


def _write_catalogue_metadata(tmp_path: Path, catalogue_path: Path) -> Path:
    """A small fixture-equivalent of eval/verified_catalogue_counts.yaml,
    matching the fixture catalogue's own hash/counts -- never the real
    production metadata file (which records the real 619-row catalogue's
    hash and would never match a small test fixture)."""
    p = tmp_path / "catalogue_metadata.yaml"
    data = {
        "isco08": {
            "normalized_catalogue_sha256": aot.sha256_file(catalogue_path),
            "verified_counts": dict(_CATALOGUE_EXPECTED_COUNTS),
        }
    }
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _write_full_fixture(tmp_path: Path, cases=_CASES):
    heldout = tmp_path / "heldout.csv"
    flat = tmp_path / "flat.csv"
    hier = tmp_path / "hierarchical.csv"
    _write_csv(heldout, REFERENCE_FIELDS, _reference_rows(cases))
    _write_csv(flat, RESULT_FIELDS, _flat_rows(cases))
    _write_csv(hier, RESULT_FIELDS, _hier_rows(cases))
    catalogue = _write_catalogue(tmp_path)
    return heldout, flat, hier, catalogue


def _make_git_fixture(tmp_path: Path):
    """A real, purely-local, ephemeral git repo (no network/remote)
    containing a committed report file, used to hermetically exercise
    check 10 (byte-identity against a base commit)."""
    repo_root = tmp_path / "repo"
    report_rel = Path("Documentation/AI_HANDOFF/CLAUDE_TASK_36_FINAL_REPORT.md")
    report_path = repo_root / report_rel
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("OFFICIAL_TIER1_PRECISE_DEADLINE_FULL_RERUN_COMPLETED: yes\n", encoding="utf-8")
    run = lambda *args: subprocess.run(args, cwd=repo_root, check=True, capture_output=True, text=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.invalid")
    run("git", "config", "user.name", "test")
    run("git", "add", ".")
    run("git", "commit", "-q", "-m", "init")
    sha = run("git", "rev-parse", "HEAD").stdout.strip()
    return repo_root, report_path, sha


def _gate_kwargs(heldout, flat, hier, catalogue, repo_root, report_path, base_sha, tmp_path=None, **overrides):
    metadata_path = _write_catalogue_metadata(tmp_path or catalogue.parent, catalogue)
    kwargs = dict(
        flat_csv=flat, hierarchical_csv=hier, heldout_csv=heldout, catalogue_csv=catalogue,
        task36_report=report_path, task36_base_sha=base_sha,
        expected_flat_sha256=aot.sha256_file(flat),
        expected_hierarchical_sha256=aot.sha256_file(hier),
        expected_heldout_sha256=aot.sha256_file(heldout),
        expected_n=len(_CASES), max_stage_latency_ms=30000.0, repo_root=repo_root,
        catalogue_expected_counts=_CATALOGUE_EXPECTED_COUNTS,
        catalogue_metadata_path=metadata_path,
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# 1. Official profile method-label acceptance (and full happy-path gate)
# ---------------------------------------------------------------------------

def test_official_profile_method_labels_pass_gate(tmp_path):
    heldout, flat, hier, catalogue = _write_full_fixture(tmp_path)
    repo_root, report_path, base_sha = _make_git_fixture(tmp_path)

    flat_rows, hier_rows, heldout_rows, results = aot.run_eligibility_gate(
        **_gate_kwargs(heldout, flat, hier, catalogue, repo_root, report_path, base_sha)
    )
    by_name = {g["check"]: g for g in results}
    assert all(g["passed"] for g in results), results
    assert by_name["3_flat_method_label_exact"]["passed"] is True
    assert by_name["4_hierarchical_method_label_exact"]["passed"] is True
    assert len(flat_rows) == len(hier_rows) == len(heldout_rows) == len(_CASES)


# ---------------------------------------------------------------------------
# 2. SHA/integrity mismatch fail-closed behavior
# ---------------------------------------------------------------------------

def test_sha_mismatch_fails_closed(tmp_path):
    heldout, flat, hier, catalogue = _write_full_fixture(tmp_path)
    repo_root, report_path, base_sha = _make_git_fixture(tmp_path)

    kwargs = _gate_kwargs(heldout, flat, hier, catalogue, repo_root, report_path, base_sha)
    kwargs["expected_flat_sha256"] = "0" * 64  # deliberately wrong
    _, _, _, results = aot.run_eligibility_gate(**kwargs)
    by_name = {g["check"]: g for g in results}
    assert by_name["1_input_sha256_match"]["passed"] is False
    assert not all(g["passed"] for g in results)


# ---------------------------------------------------------------------------
# 3. Invalid/coarse code rejection
# ---------------------------------------------------------------------------

def test_coarse_or_invalid_code_rejected(tmp_path):
    cases = list(_CASES)
    heldout, flat, hier, catalogue = _write_full_fixture(tmp_path, cases)
    # Corrupt one flat prediction to a coarse 2-digit code (not a valid
    # official 4-digit unit-group code, and not in the fixture catalogue).
    flat_rows = list(csv.DictReader(flat.open(encoding="utf-8")))
    flat_rows[0]["pred_isco_4digit"] = "12"
    _write_csv(flat, RESULT_FIELDS, flat_rows)
    repo_root, report_path, base_sha = _make_git_fixture(tmp_path)

    kwargs = _gate_kwargs(heldout, flat, hier, catalogue, repo_root, report_path, base_sha)
    kwargs["expected_flat_sha256"] = aot.sha256_file(flat)  # re-hash after corruption
    _, _, _, results = aot.run_eligibility_gate(**kwargs)
    by_name = {g["check"]: g for g in results}
    assert by_name["5_predicted_codes_valid_and_in_catalogue"]["passed"] is False
    assert by_name["5_predicted_codes_valid_and_in_catalogue"]["detail"]["n_bad_flat_predictions"] == 1


# ---------------------------------------------------------------------------
# 4. Row-ID mismatch rejection
# ---------------------------------------------------------------------------

def test_row_id_mismatch_rejected(tmp_path):
    heldout, flat, hier, catalogue = _write_full_fixture(tmp_path)
    hier_rows = list(csv.DictReader(hier.open(encoding="utf-8")))
    hier_rows[0]["case_id"] = "not-a-real-heldout-id"
    _write_csv(hier, RESULT_FIELDS, hier_rows)
    repo_root, report_path, base_sha = _make_git_fixture(tmp_path)

    kwargs = _gate_kwargs(heldout, flat, hier, catalogue, repo_root, report_path, base_sha)
    kwargs["expected_hierarchical_sha256"] = aot.sha256_file(hier)
    _, _, _, results = aot.run_eligibility_gate(**kwargs)
    by_name = {g["check"]: g for g in results}
    assert by_name["2_row_counts_and_case_ids"]["passed"] is False
    assert by_name["2_row_counts_and_case_ids"]["detail"]["hierarchical_ids_match_heldout"] is False


# ---------------------------------------------------------------------------
# 5. Known Wilson interval result (independent reference formula)
# ---------------------------------------------------------------------------

def _reference_wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Independent, textbook-formula implementation used only to
    cross-check eval.analyze.wilson_score_interval -- not imported from
    the module under test."""
    p_hat = successes / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def test_known_wilson_interval_result():
    for successes, n in [(3, 4), (50, 100), (0, 10), (10, 10)]:
        expected_lo, expected_hi = _reference_wilson(successes, n)
        lo, hi = wilson_score_interval(successes, n)
        assert lo == pytest.approx(expected_lo, abs=1e-9)
        assert hi == pytest.approx(expected_hi, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. Known exact two-sided McNemar result (textbook example)
# ---------------------------------------------------------------------------

def test_known_mcnemar_exact_result():
    # Classic textbook example: b=1, c=9 discordant pairs (n=10).
    # Two-sided exact binomial p = 2 * P(X<=1 | X~Binom(10, 0.5)) = 22/1024.
    stat, p_value = mcnemar_test(1, 9)
    assert stat == 1.0
    assert p_value == pytest.approx(22 / 1024, abs=1e-12)

    # Symmetric discordant counts must give p=1.0 exactly.
    stat_sym, p_sym = mcnemar_test(3, 3)
    assert stat_sym == 3.0
    assert p_sym == pytest.approx(1.0, abs=1e-12)

    # Zero discordant pairs is defined as p=1.0 (no evidence either way).
    assert mcnemar_test(0, 0) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# 7. Paired-contingency accounting
# ---------------------------------------------------------------------------

def test_paired_contingency_accounting():
    flat_rows = _flat_rows()
    hier_rows = _hier_rows()
    result = aot.paired_contingency(flat_rows, hier_rows)
    assert result["n_pairs"] == 4
    assert result["both_correct"] == 1
    assert result["flat_only_correct"] == 1
    assert result["hierarchical_only_correct"] == 1
    assert result["both_incorrect"] == 1
    assert result["flat_correct_total"] == 2
    assert result["hierarchical_correct_total"] == 2
    assert result["accuracy_diff_hierarchical_minus_flat_pp"] == pytest.approx(0.0)
    # b == c == 1 -> symmetric -> exact two-sided p must be 1.0
    assert result["mcnemar_statistic_min_b_c"] == 1.0
    assert result["mcnemar_p_value_two_sided_exact"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. No analysis output before eligibility passes (and output IS written
#    once it does)
# ---------------------------------------------------------------------------

def test_no_output_before_eligibility_passes(tmp_path, capsys):
    heldout, flat, hier, catalogue = _write_full_fixture(tmp_path)
    repo_root, report_path, base_sha = _make_git_fixture(tmp_path)
    out_dir = tmp_path / "out"

    argv = [
        "analyze_official_tier1.py",
        "--flat-csv", str(flat), "--hierarchical-csv", str(hier),
        "--heldout-csv", str(heldout), "--catalogue-csv", str(catalogue),
        "--task36-report", str(report_path), "--task36-base-sha", base_sha,
        "--expected-flat-sha256", "0" * 64,  # wrong -- forces gate failure
        "--expected-hierarchical-sha256", aot.sha256_file(hier),
        "--expected-heldout-sha256", aot.sha256_file(heldout),
        "--expected-n", str(len(_CASES)), "--out", str(out_dir),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit) as exc_info:
            aot.main()
        assert exc_info.value.code != 0
    finally:
        sys.argv = old_argv

    assert not (out_dir / "official_tier1_analysis.json").exists()
    assert not (out_dir / "official_tier1_analysis.md").exists()
    assert not (out_dir / "analysis_manifest.json").exists()
    assert (out_dir / "analysis_failure_manifest.json").exists()
    failure = json.loads((out_dir / "analysis_failure_manifest.json").read_text(encoding="utf-8"))
    assert failure["status"] == "GATE_FAILURE"
    assert any(not g["passed"] for g in failure["eligibility_gate_results"])


def test_output_written_once_eligibility_passes(tmp_path):
    heldout, flat, hier, catalogue = _write_full_fixture(tmp_path)
    repo_root, report_path, base_sha = _make_git_fixture(tmp_path)
    out_dir = tmp_path / "out"

    argv = [
        "analyze_official_tier1.py",
        "--flat-csv", str(flat), "--hierarchical-csv", str(hier),
        "--heldout-csv", str(heldout), "--catalogue-csv", str(catalogue),
        "--task36-report", str(report_path), "--task36-base-sha", base_sha,
        "--expected-flat-sha256", aot.sha256_file(flat),
        "--expected-hierarchical-sha256", aot.sha256_file(hier),
        "--expected-heldout-sha256", aot.sha256_file(heldout),
        "--expected-n", str(len(_CASES)), "--out", str(out_dir),
    ]
    # main() reads args via argparse from sys.argv and does not accept a
    # catalogue_expected_counts override, so this test exercises
    # run_eligibility_gate() directly instead (main()'s CLI path always
    # uses the real 619-row official counts, correctly untestable here
    # without a real catalogue -- covered by the live Task 37 run itself).
    flat_rows, hier_rows, heldout_rows, results = aot.run_eligibility_gate(
        **_gate_kwargs(heldout, flat, hier, catalogue, repo_root, report_path, base_sha)
    )
    assert all(g["passed"] for g in results)

    lang_map = {r["case_id"]: r["input_language"] for r in heldout_rows}
    headline_flat = aot.compute_headline(flat_rows, "flat", aot.EXPECTED_FLAT_METHOD, aot.sha256_file(flat))
    headline_hier = aot.compute_headline(hier_rows, "hierarchical", aot.EXPECTED_HIERARCHICAL_METHOD, aot.sha256_file(hier))
    assert headline_flat.correct == 2 and headline_flat.n == 4
    assert headline_hier.correct == 2 and headline_hier.n == 4
    by_language = aot._subgroup_metrics(flat_rows, hier_rows, lambda r: lang_map[r["case_id"]])
    assert set(by_language.keys()) == {"en", "ar", "hi", "tl"}
    by_major = aot._subgroup_metrics(flat_rows, hier_rows, lambda r: r["gold_isco_4digit"].strip()[0])
    assert set(by_major.keys()) == {"1", "5"}
    operational = aot.compute_operational_summary(flat_rows, hier_rows)
    assert operational["n_hierarchical_rows_with_any_retry"] == 0
    assert operational["n_hierarchical_rows_with_any_exception"] == 0
    assert operational["n_hierarchical_rows_with_any_budget_exhaustion"] == 0
    assert operational["total_hierarchical_stage_query_count"] == 4 * 4  # 4 cases x 4 stages x 1 query each
