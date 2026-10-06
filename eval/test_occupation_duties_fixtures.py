"""Synthetic-only duties fixtures and read-only paired checker contracts.

No server, Qdrant, encoder, Ollama, cloud provider, or respondent data is used.
The only backend boundary is the lightweight input adapter with a fake
classifier; semantic expectations are deliberately allowed to miss.
"""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unicodedata
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.agents.occupation_inputs import (  # noqa: E402
    classify_occupation_input, MAX_DUTIES_CHARS, PARENT_METHOD, DUTIES_PARENT_METHOD,
)
from scripts import check_occupation_duties_runtime as checker  # noqa: E402

FIXTURE, FIXTURE_SHA = checker.load_fixture(checker.DEFAULT_FIXTURE)


def fake_classifier(calls):
    def classify(query, **kwargs):
        calls.append((query, kwargs))
        return SimpleNamespace(query=query, method=PARENT_METHOD, hitl_required=True,
                               primary=SimpleNamespace(code="2512", confidence=0.4))
    return SimpleNamespace(classify=classify)


def test_fixture_is_disclosed_synthetic_grouped_native_context_matrix():
    assert len(FIXTURE["cases"]) == 25
    assert FIXTURE["respondent_data"] is False
    assert FIXTURE["expert_validated"] is False
    assert FIXTURE["native_language_reviewed"] is False
    assert FIXTURE["accuracy_claim_allowed"] is False
    assert FIXTURE["split_role"] == "engineering_regression"
    assert len({case["scenario_family"] for case in FIXTURE["cases"]}) == 5
    assert len({case["case_id"] for case in FIXTURE["cases"]}) == 25
    for language in checker.LANGUAGES:
        cases = [case for case in FIXTURE["cases"] if case["language"] == language]
        assert len(cases) == 5
        assert len({case["title"] for case in cases}) == 1
        assert len({case["duties"] for case in cases}) == 5
        assert len({case["synthetic_expected_code"] for case in cases}) == 5
        assert all(case["title_alone_insufficient"] for case in cases)


def test_fixture_code_references_match_verified_official_source_when_available():
    source = FIXTURE["source_references"][0]
    assert source["issuer"] == "International Labour Organization"
    assert source["source_url"].startswith("https://webapps.ilo.org/")
    assert set(source["official_units"]) == checker.OFFICIAL_CODES
    path = ROOT / source["local_normalized_catalogue"]
    if not path.exists():
        return  # Official derived catalogues are intentionally ignored.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source["source_catalogue_sha256"]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        units = {row["code"]: row for row in csv.DictReader(stream) if row["level"] == "unit"}
    for case in FIXTURE["cases"]:
        code = case["synthetic_expected_code"]
        assert case["synthetic_expected_label"].casefold() == units[code]["label"].casefold()
        assert case["expected_code_reference"] == "ILO-ISCO08-2021#" + code
        assert units[code]["definition"]


@pytest.mark.parametrize("language", sorted(checker.LANGUAGES))
def test_native_fixture_text_survives_url_and_adapter_without_gold_or_metadata(language):
    for original in (row for row in FIXTURE["cases"] if row["language"] == language):
        url = checker.request_url("http://127.0.0.1:8000", original, True)
        parsed = urlsplit(url)
        params = parse_qs(parsed.query)
        assert unquote(parsed.path[len("/debug/isco/"):]) == original["title"]
        assert params == {"language": [language], "include_trace": ["true"], "duties": [original["duties"]]}
        mutated = deepcopy(original)
        mutated["synthetic_expected_code"] = "9999"
        mutated["metadata"]["industry_not_classifier_input"] = "DO_NOT_SEND_METADATA"
        mutated["expectation_rationale"] = "DO_NOT_SEND_LABEL_RATIONALE"
        assert checker.request_url("http://127.0.0.1:8000", mutated, True) == url
        calls, trace = [], {}
        result = classify_occupation_input(fake_classifier(calls), original["title"],
            duties=original["duties"], language=language, trace=trace, use_llm=False)
        assert calls[0][0] == original["title"] + " " + original["duties"]
        assert original["synthetic_expected_code"] not in calls[0][0]
        assert result.query == original["title"]
        assert result.method == DUTIES_PARENT_METHOD and result.hitl_required
        assert trace["duties_used"] is True
        assert trace["retrieval_query"] == calls[0][0]
        assert trace["duties_accuracy_evaluated"] is False
        script = "ARABIC" if language in {"ar", "ur"} else "DEVANAGARI" if language == "hi" else "LATIN"
        assert any(script in unicodedata.name(char, "") for char in original["duties"])


@pytest.mark.parametrize("duties", [None, "", "  ", "N/A", "unknown", "never_worked", "refused", "prefer_not_to_say"])
def test_missing_or_refused_duties_preserve_title_only_object_and_method(duties):
    calls, trace = [], {}
    result = classify_occupation_input(fake_classifier(calls), "Original title", duties=duties, trace=trace)
    assert calls[0][0] == "Original title"
    assert result.method == PARENT_METHOD and result.query == "Original title"
    assert trace["duties_used"] is False and trace["query_mode"] == "title_only"
    assert trace["retrieval_query"] == "Original title"
    assert trace["title_only_input"] is True


def test_duplicate_duties_preserve_raw_title_and_context_industry_cannot_enter_query():
    calls, trace = [], {}
    title = "  Analyst  "
    result = classify_occupation_input(fake_classifier(calls), title, duties="analyst",
                                       context="INDUSTRY_NOT_PRIMARY_INPUT", trace=trace)
    assert calls[0][0] == title
    assert calls[0][1]["context"] == "INDUSTRY_NOT_PRIMARY_INPUT"
    assert result.method == PARENT_METHOD
    assert trace["duties_used"] is False


def test_long_code_switched_duties_preserve_unicode_and_disclose_truncation():
    calls, trace = [], {}
    duties = ("मैं code लिखता हूँ اور test کرتا ہوں; nagte-test ako ng software. " * 80).strip()
    result = classify_occupation_input(fake_classifier(calls), "Staff member", duties=duties, trace=trace)
    expected = "Staff member " + duties[:MAX_DUTIES_CHARS].rstrip()
    assert calls[0][0] == expected
    assert trace["retrieval_query"] == expected and trace["duties_truncated"] is True
    assert "मैं" in expected and "اور" in expected and "nagte-test" in expected
    assert result.query == "Staff member" and result.hitl_required is True
    assert trace["duties_accuracy_evaluated"] is False


@pytest.mark.parametrize("duties", [1, ["code"], {"duties": "code"}])
def test_nontext_duties_never_reach_classifier(duties):
    calls = []
    with pytest.raises(ValueError):
        classify_occupation_input(fake_classifier(calls), "Staff", duties=duties)
    assert not calls


def fake_fetch(url, timeout):
    parsed = urlsplit(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    title = unquote(parsed.path[len("/debug/isco/"):])
    duties = params.get("duties", [""])[0]
    used = bool(duties)
    return {"code": "4110", "conf": 0.4, "ms": 1, "method": checker.DUTIES_METHOD if used else checker.TITLE_METHOD,
            "hitl_required": True, "duties_used": used,
            "trace": {"duties_used": used, "duties_truncated": False,
                      "query_mode": "title_and_duties" if used else "title_only",
                      "retrieval_query": title + " " + duties if used else title,
                      "title_only_input": not used, "duties_accuracy_evaluated": False,
                      "query_encoder": "intfloat/multilingual-e5-small", "selection_sha256": "frozen-selection",
                      "source_catalogue_sha256": "official-source", "candidates_scored": 436,
                      "reranker_fired": False, "score_calibrated": False,
                      "evidence": [{"code": "4110"}, {"code": "2512"}]},
            "unexpected_sensitive_field": "DO_NOT_RETAIN"}


def test_paired_checker_reports_all_semantic_misses_without_failing_valid_mechanism():
    requests = []
    def fetch(url, timeout):
        requests.append(url)
        return fake_fetch(url, timeout)
    report = checker.run_diagnostics(FIXTURE, FIXTURE_SHA, "http://127.0.0.1:8000", fetch=fetch)
    assert report["n_cases"] == 25 and report["n_readonly_requests"] == 35
    assert len(requests) == 35 and report["n_unique_title_queries"] == 5
    assert report["mandatory_checks_passed"] is True
    assert report["semantic_expectations_met"] == 0 and len(report["semantic_misses"]) == 25
    assert report["expert_validated"] is False and report["accuracy_claim_allowed"] is False
    assert all(len(row["title_and_duties"]["top_codes"]) == 2 for row in report["results"])
    assert "DO_NOT_RETAIN" not in json.dumps(report)
    assert all(urlsplit(url).path.startswith("/debug/isco/") for url in requests)
    assert all(set(parse_qs(urlsplit(url).query)) <= {"language", "include_trace", "duties"} for url in requests)


def test_checker_keeps_transport_errors_and_continues_all_cases():
    def fetch(url, timeout):
        params = parse_qs(urlsplit(url).query)
        if params["language"] == ["ar"] and params.get("duties") == [FIXTURE["cases"][6]["duties"]]:
            raise TimeoutError("Synthetic request timeout")
        return fake_fetch(url, timeout)
    report = checker.run_diagnostics(FIXTURE, FIXTURE_SHA, "http://localhost:8000", fetch=fetch)
    assert report["n_cases"] == 25 and report["n_readonly_requests"] == 35
    assert report["mandatory_checks_passed"] is False
    failed = [row for row in report["results"] if "error" in row["title_and_duties"]]
    assert len(failed) == 1
    assert failed[0]["title_and_duties"]["error"]["kind"] == "TimeoutError"
    assert any(item["case_id"] == failed[0]["case_id"] and item["check"] == "duties_request_succeeded"
               for item in report["mandatory_failures"])


def test_empty_duties_replay_detects_changed_title_only_prediction():
    def fetch(url, timeout):
        body = fake_fetch(url, timeout)
        if parse_qs(urlsplit(url).query, keep_blank_values=True).get("duties") == [""]:
            body["code"] = "5120"
        return body
    report = checker.run_diagnostics(FIXTURE, FIXTURE_SHA, "http://127.0.0.1:8000", fetch=fetch)
    assert not report["mandatory_checks_passed"]
    assert sum(row["check"] == "unchanged_with_explicit_empty_duties" for row in report["mandatory_failures"]) == 5


def test_checker_requires_review_and_does_not_silently_accept_untraced_context():
    case = FIXTURE["cases"][0]
    body = fake_fetch(checker.request_url("http://127.0.0.1:8000", case, True), 30)
    body["hitl_required"] = False
    body["trace"]["retrieval_query"] = case["title"]
    checks = checker.mechanism_checks(checker.sanitize_response(body), case, True)
    assert checks["human_review_required"] is False
    assert checks["original_query_preserved"] is False


@pytest.mark.parametrize("invalid_code", [4110, "٤١١٠", "not-a-code"])
def test_invalid_code_types_or_nonstandard_digits_fail_mechanism(invalid_code):
    case = FIXTURE["cases"][0]
    body = fake_fetch(checker.request_url("http://127.0.0.1:8000", case, True), 30)
    body["code"] = invalid_code
    checks = checker.mechanism_checks(checker.sanitize_response(body), case, True)
    assert checks["valid_primary_code"] is False


@pytest.mark.parametrize("base_url", ["https://example.com", "http://example.com", "http://127.0.0.1/api",
                                     "http://user:password@localhost:8000", "http://localhost:8000?token=secret"])
def test_checker_rejects_nonlocal_or_credentialed_targets(base_url):
    with pytest.raises(ValueError):
        checker.validate_base_url(base_url)


def test_synthetic_fixture_loader_rejects_relabelled_or_gold_injected_input(tmp_path):
    for mutate in (
        lambda data: data.update(respondent_data=True),
        lambda data: data.update(expert_validated=True),
        lambda data: data["cases"][0].update(duties="Use gold code 2512"),
        lambda data: data["cases"].append(deepcopy(data["cases"][0])),
    ):
        data = deepcopy(FIXTURE)
        mutate(data)
        path = tmp_path / "invalid_fixture.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError):
            checker.load_fixture(path)
