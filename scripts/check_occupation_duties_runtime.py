"""Read-only paired diagnostics for developer-authored synthetic occupation duties.

Uses only GET /debug/isco/{title}, with reranking disabled by that endpoint.
No survey writes, authentication, OTP, cloud calls, or model imports. Original
fixture text enters requests unchanged; gold expectations and metadata never do.
Semantic misses are reported and do not determine the process exit status.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "eval/fixtures/occupation_duties_synthetic_v1.json"
TITLE_METHOD = "isco_parent_document_rag"
DUTIES_METHOD = "isco_parent_document_duties_rag"
LANGUAGES = {"en", "ar", "ur", "hi", "tl"}
OFFICIAL_CODES = {"2512", "5120", "2411", "2341", "2221"}
MAX_DUTIES_CHARS = 1600


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "Redirect refused for local diagnostic", headers, fp)


def validate_base_url(base_url):
    parsed = urlsplit(base_url)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid local API port") from None
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
            or (port is not None and not 1 <= port <= 65535)):
        raise ValueError("Base URL must be a loopback HTTP origin without credentials or a path")
    return base_url.rstrip("/")


def load_fixture(path):
    payload = Path(path).read_bytes()
    fixture = json.loads(payload)
    if (fixture.get("schema_version") != 1
            or fixture.get("dataset_label") != "synthetic_or_operationally_realistic"
            or fixture.get("respondent_data") is not False
            or fixture.get("expert_validated") is not False
            or fixture.get("accuracy_claim_allowed") is not False
            or "SYNTHETIC" not in fixture.get("notice", "")):
        raise ValueError("Only explicitly synthetic engineering fixtures are accepted")
    cases = fixture.get("cases", [])
    if not 1 <= len(cases) <= 30:
        raise ValueError("Fixture must contain between one and 30 cases")
    identifiers = set()
    for case in cases:
        if (not isinstance(case.get("case_id"), str)
                or not case["case_id"].startswith("SYN-DUTIES-")
                or case["case_id"] in identifiers):
            raise ValueError("Fixture requires unique synthetic case identifiers")
        identifiers.add(case["case_id"])
        if case.get("language") not in LANGUAGES:
            raise ValueError("Unsupported fixture language")
        if not all(isinstance(case.get(key), str) and case[key].strip() for key in ("title", "duties")):
            raise ValueError("Fixture title and duties must be nonblank text")
        code = case.get("synthetic_expected_code")
        if code not in OFFICIAL_CODES or code in case["title"] or code in case["duties"]:
            raise ValueError("Expected code must be an official fixture unit and absent from query text")
    return fixture, hashlib.sha256(payload).hexdigest()


def request_url(base_url, case, with_duties=False, explicit_empty=False):
    # Deliberate whitelist: expectations, labels, rationale and industry metadata
    # cannot enter the endpoint even if the fixture grows additional fields.
    params = {"language": case["language"], "include_trace": "true"}
    if with_duties or explicit_empty:
        params["duties"] = case["duties"] if with_duties else ""
    return (validate_base_url(base_url) + "/debug/isco/" + quote(case["title"], safe="")
            + "?" + urlencode(params, encoding="utf-8"))


def fetch_json(url, timeout):
    # Ignore proxy environment settings and refuse redirects to other origins.
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sanitize_response(body):
    if not isinstance(body, dict):
        return {"error": {"kind": "invalid_response", "message": "Expected a JSON object"}}
    if "error" in body:
        # Never retain the backend traceback or arbitrary unrequested fields.
        return {"error": {"kind": "backend_error", "message": str(body["error"])[:500]}}
    trace = body.get("trace")
    trace = trace if isinstance(trace, dict) else {}
    fields = ("duties_used", "duties_truncated", "query_mode", "retrieval_query",
              "title_only_input", "duties_accuracy_evaluated", "query_encoder",
              "selection_sha256", "source_catalogue_sha256", "candidates_scored",
              "reranker_fired", "score_calibrated")
    safe_trace = {key: trace[key] for key in fields if key in trace}
    codes = []
    evidence = trace.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    for code in [body.get("code")] + [item.get("code") for item in evidence if isinstance(item, dict)]:
        if isinstance(code, str) and re.fullmatch(r"[0-9]{4}", code) and code not in codes:
            codes.append(code)
    return {"code": body.get("code"), "top_codes": codes, "method": body.get("method"),
            "hitl_required": body.get("hitl_required"), "duties_used": body.get("duties_used"),
            "confidence": body.get("conf") if _finite_number(body.get("conf")) else None,
            "milliseconds": body.get("ms") if _finite_number(body.get("ms")) else None,
            "trace": safe_trace}


def query(base_url, case, *, with_duties=False, explicit_empty=False, timeout=30, fetch=fetch_json):
    try:
        return sanitize_response(fetch(request_url(base_url, case, with_duties, explicit_empty), timeout))
    except Exception as exc:
        return {"error": {"kind": type(exc).__name__, "message": str(exc)[:500]}}


def mechanism_checks(result, case, with_duties):
    if "error" in result:
        return {"request_succeeded": False}
    trace = result["trace"]
    expected_query = (case["title"].strip() + " " + case["duties"].strip()[:MAX_DUTIES_CHARS].rstrip()
                      if with_duties else case["title"])
    checks = {
        "request_succeeded": True,
        "expected_method": result["method"] == (DUTIES_METHOD if with_duties else TITLE_METHOD),
        "valid_primary_code": isinstance(result["code"], str) and bool(re.fullmatch(r"[0-9]{4}", result["code"])),
        "human_review_required": result["hitl_required"] is True,
        "explicit_duties_flag": result["duties_used"] is with_duties and trace.get("duties_used") is with_duties,
        "query_mode": trace.get("query_mode") == ("title_and_duties" if with_duties else "title_only"),
        "original_query_preserved": trace.get("retrieval_query") == expected_query,
        "truncation_recorded": trace.get("duties_truncated") is (with_duties and len(case["duties"].strip()) > MAX_DUTIES_CHARS),
        "complete_official_candidate_pool": trace.get("candidates_scored") == 436,
        "reranking_disabled": trace.get("reranker_fired") is False,
        "score_not_calibrated": trace.get("score_calibrated") is False,
    }
    if "title_only_input" in trace:
        checks["input_scope_recorded"] = trace["title_only_input"] is not with_duties
    if with_duties:
        checks["duties_accuracy_not_claimed"] = trace.get("duties_accuracy_evaluated") is False
    return checks


def _same_title_result(first, replay):
    if "error" in first or "error" in replay:
        return False
    return (first["top_codes"] == replay["top_codes"] and first["method"] == replay["method"]
            and first["hitl_required"] == replay["hitl_required"]
            and first["confidence"] is not None and replay["confidence"] is not None
            and math.isclose(first["confidence"], replay["confidence"], rel_tol=0, abs_tol=1e-8))


def run_diagnostics(fixture, fixture_sha256, base_url, *, timeout=30, fetch=fetch_json):
    base_url = validate_base_url(base_url)
    baselines = {}
    results = []
    failures = []
    request_count = 0
    for case in fixture["cases"]:
        key = (case["language"], case["title"])
        if key not in baselines:
            baselines[key] = query(base_url, case, timeout=timeout, fetch=fetch)
            request_count += 1
        baseline = baselines[key]
        contextual = query(base_url, case, with_duties=True, timeout=timeout, fetch=fetch)
        request_count += 1
        checks = {**{"title_only_" + key: value for key, value in mechanism_checks(baseline, case, False).items()},
                  **{"duties_" + key: value for key, value in mechanism_checks(contextual, case, True).items()}}
        for identity in ("query_encoder", "selection_sha256", "source_catalogue_sha256"):
            checks["same_" + identity] = (baseline.get("trace", {}).get(identity) is not None
                                         and baseline.get("trace", {}).get(identity) == contextual.get("trace", {}).get(identity))
        misses = [check for check, passed in checks.items() if not passed]
        failures.extend({"case_id": case["case_id"], "check": check} for check in misses)
        results.append({
            "case_id": case["case_id"], "scenario_family": case["scenario_family"], "language": case["language"],
            "title": case["title"], "duties": case["duties"],
            "synthetic_expected_code": case["synthetic_expected_code"],
            "title_only": baseline, "title_and_duties": contextual, "mandatory_checks": checks,
            "semantic_expectation_met": contextual.get("code") == case["synthetic_expected_code"],
            "primary_code_changed": (baseline.get("code") is not None and contextual.get("code") is not None
                                     and baseline["code"] != contextual["code"]),
        })
    parity = []
    for (language, title), baseline in baselines.items():
        case = {"language": language, "title": title}
        replay = query(base_url, case, explicit_empty=True, timeout=timeout, fetch=fetch)
        request_count += 1
        checks = mechanism_checks(replay, case, False)
        checks["unchanged_with_explicit_empty_duties"] = _same_title_result(baseline, replay)
        failures.extend({"case_id": f"PARITY-{language}", "check": key} for key, value in checks.items() if not value)
        parity.append({"language": language, "title": title, "result": replay, "mandatory_checks": checks})
    return {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "Read-only local debug GETs; explicit synthetic title/duties diagnostics, no survey/auth/OTP/LLM calls",
        "dataset_label": "synthetic_or_operationally_realistic", "fixture_id": fixture["fixture_id"],
        "fixture_sha256": fixture_sha256, "base_url": base_url,
        "expert_validated": False, "labour_force_survey_field_validation": False, "accuracy_claim_allowed": False,
        "interpretation": "Semantic expectations are developer-authored diagnostics, not independent gold accuracy. Misses do not fail mechanism verification.",
        "n_cases": len(results), "n_scenario_families": len({row["scenario_family"] for row in results}),
        "n_unique_title_queries": len(baselines), "n_readonly_requests": request_count,
        "semantic_expectations_met": sum(row["semantic_expectation_met"] for row in results),
        "semantic_misses": [row["case_id"] for row in results if not row["semantic_expectation_met"]],
        "mandatory_checks_passed": not failures, "mandatory_failures": failures,
        "per_language": {language: {"n_cases": sum(row["language"] == language for row in results),
                                   "semantic_expectations_met": sum(row["language"] == language and row["semantic_expectation_met"] for row in results)}
                         for language in sorted({row["language"] for row in results})},
        "results": results, "title_only_parity": parity,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.output.exists() or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("A new output path and positive finite timeout are required")
    fixture, digest = load_fixture(args.fixture)
    report = run_diagnostics(fixture, digest, args.base_url, timeout=args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("n_cases", "n_readonly_requests", "semantic_expectations_met",
                                                  "mandatory_checks_passed", "mandatory_failures")}))
    return 0 if report["mandatory_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
