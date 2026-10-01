"""
eval/run_synthetic_pilot_n30_live.py

Synthetic, single-arm (AI only), n=30 pilot-scale operational validation --
built after an explicit, disclosed scope decision (2026-10-01) to finish
the project with synthetic data rather than the ethics-approved real
human-subjects pilot Module E's protocol describes
(Documentation/Phase_2/Week_1/module_e_pilot_protocol_draft.md). CLAUDE.md
explicitly says "Do not drop the pilot study (Module E) or substitute
synthetic data for it" -- that standing instruction was not silently
overridden; the user was told directly, before this script was written,
that this caps what the project can honestly claim (no CSAT, no AI-vs-
traditional-interviewer comparison -- neither has a synthetic substitute).
This script produces what IS honestly measurable from synthetic data:
completion time, conversational friction (clarification turns), and
cross-standard internal-contradiction rate, for the AI arm only.

Driven against the REAL, LIVE running backend (http://localhost:8000),
not mocked -- same discipline as every other live-verification in this
project's history. Deliberately does NOT touch the classifier
configuration: the live server stays on its current, stable default
config (see CLAUDE.md's 2026-10-01 entry on why the better-tested
e5-large config was attempted and reverted -- a real memory-exhaustion
crash risk on this machine). Classification ACCURACY against the
best-tested config is measured separately, offline, by
run_synthetic_pilot_n30_accuracy.py -- deliberately never in the same
process as this script, so a crash in one can never take down the other,
and never through the live HTTP path at all.

Synthetic case source: eval/results/synthetic_coordination_benchmark/
benchmark.csv -- 60 already-generated, already quality-checked
(job_title, industry_text, education_text) triples, grounded in real
ISCO-08 official definitions, built and disclosed 2026-09-12 for a
different evaluation (Item 1 cross-standard coordination). Reused here
rather than generating new text, for two reasons: (1) this text is
already validated clean (no refusal patterns, no script mixing -- the
same quality bar this project's synthetic-generation work has held
throughout its history), (2) avoiding a fresh local-LLM generation pass
sidesteps this session's own newly-reconfirmed low-memory risk on this
machine. English-only, same disclosed scope limitation as that benchmark
itself and the project's other synthetic-generation work (local models
have shown real, disqualifying multilingual quality problems -- see
CLAUDE.md's ISIC/ISCED-F benchmark entry).

All other required survey fields (the ~35 non-free-text fields on the
employed path) are answered with schema-valid canonical values via
ConversationManager.correction_schema_for()'s own _CORRECTION_FIELD_SCHEMA
-- the exact same source of truth the live system's own free-text
correction feature uses, so these are not arbitrary made-up strings.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agents.conversation_manager import _CORRECTION_FIELD_SCHEMA  # noqa: E402

BASE = "http://localhost:8000"
BENCHMARK_CSV = Path(__file__).resolve().parent / "results" / "synthetic_coordination_benchmark" / "benchmark.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "synthetic_pilot_n30"


def call(method: str, path: str, body: dict | None = None, token: str | None = None, timeout: int = 60):
    url = BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {}
    except Exception as exc:
        return 0, {"_transport_error": str(exc)}


def canonical_answer(field: str) -> str:
    """Same logic as this session's earlier e2e_continue.py driver -- the
    first accepted enum value, a round numeric, or a short disclosed
    placeholder for genuinely free-text fields not already sourced from
    the benchmark CSV (job_title/industry/education_level handled
    separately by the caller)."""
    schema = _CORRECTION_FIELD_SCHEMA.get(field, {})
    values = schema.get("values", "")
    if "free text" in values:
        examples = {
            "nationality": "Indian", "field_of_study": "Engineering",
            "job_duties": "general duties as described",
            "main_skills": "professional skills related to the role",
            "job_search_methods": "online portals",
            "last_job_title": "Accountant",
            "labour_market_barriers": "none", "platform_names": "Upwork",
            "survey_comments": "no comments",
        }
        return examples.get(field, "none")
    if "numeric" in values:
        return "40"
    return values.split("|")[0].strip()


def load_benchmark_cases(n: int) -> list[dict]:
    with BENCHMARK_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < n:
        raise RuntimeError(f"Benchmark only has {len(rows)} rows, need {n}")
    return rows[:n]


def run_one_case(case_idx: int, case: dict) -> dict:
    email = f"synthetic_pilot_case_{case_idx:03d}@example.com"
    out = {
        "case_idx": case_idx,
        "gold_isco_4digit": case["gold_isco_4digit"],
        "job_title_text": case["job_title"],
        "industry_text": case["industry_text"],
        "education_text": case["education_text"],
        "turns": 0,
        "clarifying_turns": 0,
        "transport_errors": 0,
        "completed": False,
        "final_state": None,
        "sre_high_severity_seen": False,
        "sre_is_coherent": None,
        "report_quality_status": None,
        "report_flagged_count": None,
        "isco_code_assigned": None,
        "isco_method": None,
        "error": None,
    }
    t_start = time.perf_counter()

    status, res = call("POST", "/auth/request-otp", {"email": email})
    status, res = call("POST", "/auth/request-otp", {"email": email})
    otp = res.get("dev_otp")
    if not otp:
        out["error"] = f"no dev_otp in request-otp response: {res}"
        return out
    status, res = call("POST", "/auth/verify-otp", {"email": email, "code": otp})
    token = res.get("access_token") or res.get("token")
    if not token:
        out["error"] = f"no token from verify-otp: {res}"
        return out

    status, res = call("POST", "/survey/sessions", {"language": "en"}, token=token)
    if status != 201:
        out["error"] = f"session create failed: {status} {res}"
        return out
    sid = res["id"]

    status, res = call("POST", f"/survey/sessions/{sid}/message", {"message": "hello"}, token=token)
    out["turns"] += 1

    max_turns = 60
    while out["turns"] < max_turns:
        state = res.get("state")
        out["final_state"] = state
        if res.get("_transport_error"):
            out["transport_errors"] += 1
        if state == "clarifying":
            out["clarifying_turns"] += 1
        if res.get("session_completed") or state == "completing":
            out["completed"] = True
            break
        if state == "validating":
            status, res = call("POST", f"/survey/sessions/{sid}/message",
                                {"message": "Yes, that's correct"}, token=token)
            out["turns"] += 1
            continue

        next_field = res.get("next_field")
        if next_field == "job_title":
            ans = case["job_title"]
        elif next_field == "industry":
            ans = case["industry_text"]
        elif next_field == "education_level":
            ans = case["education_text"]
        elif next_field:
            ans = canonical_answer(next_field)
        else:
            ans = "yes"
        status, res = call("POST", f"/survey/sessions/{sid}/message", {"message": ans}, token=token)
        out["turns"] += 1

        for c in (res.get("isco_classifications") or []):
            out["isco_code_assigned"] = c.get("primary_code")
            out["isco_method"] = c.get("method")
        sc = res.get("semantic_coherence")
        if sc:
            out["sre_is_coherent"] = sc.get("is_coherent")
            if any(v.get("severity") == "HIGH" for v in (sc.get("violations") or [])):
                out["sre_high_severity_seen"] = True

    out["wall_clock_seconds"] = round(time.perf_counter() - t_start, 2)

    if out["completed"]:
        status, res = call("GET", f"/survey/sessions/{sid}/report?regenerate=true", token=token, timeout=90)
        if status == 200:
            out["report_quality_status"] = res.get("quality_status")
            out["report_flagged_count"] = res.get("flagged_count")
            sc = res.get("semantic_coherence")
            if sc:
                out["sre_is_coherent"] = sc.get("is_coherent")
                if any(v.get("severity") == "HIGH" for v in (sc.get("violations") or [])):
                    out["sre_high_severity_seen"] = True

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "live_results.csv")
    args = ap.parse_args()

    status, health = call("GET", "/health")
    if status != 200:
        print(f"ERROR: backend not healthy at {BASE} (status={status}): {health}", file=sys.stderr)
        sys.exit(1)

    cases = load_benchmark_cases(args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] running case (gold_isco={case['gold_isco_4digit']})...", flush=True)
        r = run_one_case(i, case)
        results.append(r)
        status_str = "OK" if r["completed"] else f"INCOMPLETE ({r.get('error') or r.get('final_state')})"
        print(f"  -> {status_str}, turns={r['turns']}, time={r.get('wall_clock_seconds', '?')}s, "
              f"isco={r.get('isco_code_assigned')}/{r.get('isco_method')}, "
              f"coherent={r.get('sre_is_coherent')}", flush=True)

    fieldnames = list(results[0].keys()) if results else []
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    n_completed = sum(1 for r in results if r["completed"])
    n_high_sev = sum(1 for r in results if r["sre_high_severity_seen"])
    n_incoherent = sum(1 for r in results if r["sre_is_coherent"] is False)
    avg_time = (sum(r.get("wall_clock_seconds", 0) for r in results if r["completed"]) / max(n_completed, 1))
    avg_turns = sum(r["turns"] for r in results) / max(len(results), 1)
    n_clarifying = sum(1 for r in results if r["clarifying_turns"] > 0)

    print("\n=== SUMMARY ===")
    print(f"n = {len(results)}")
    print(f"Completed: {n_completed}/{len(results)} ({100*n_completed/len(results):.1f}%)")
    print(f"Mean completion time (system-side): {avg_time:.1f}s")
    print(f"Mean turns per session: {avg_turns:.1f}")
    print(f"Sessions with >=1 clarification turn: {n_clarifying}/{len(results)}")
    print(f"HIGH-severity SRE contradiction detected: {n_high_sev}/{len(results)}")
    print(f"Sessions flagged incoherent by SRE: {n_incoherent}/{len(results)}")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
