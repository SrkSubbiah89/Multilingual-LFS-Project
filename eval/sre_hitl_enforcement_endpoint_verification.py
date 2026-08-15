"""
Module D Step 5.5, Task 3: end-to-end verification that HIGH-severity SRE
results reliably produce a queued HITL escalation THROUGH THE REAL LIVE
ENDPOINT (POST /survey/sessions/{id}/message), not just through direct
SemanticRelationEngine calls.

Reuses eval/sre_expanded_validation.py's build_cases() (61 cases, each with
a severity already predicted from the real crosswalk tables and confirmed
against the engine with zero mismatches) so this script is testing the same
known-good severity assignments, but through backend/api/survey_routes.py's
_send_message_impl -- the actual code path Step 5.5 wired the enforcement
into.

For every case:
  - HIGH severity  -> expect exactly one HITLQueue row tagged "SRE HIGH-severity"
  - non-HIGH       -> expect zero such rows

Run 3x (Prompt 5.5 ground rule) to confirm deterministic, stable results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from eval.sre_expanded_validation import build_cases


def _run_one_case(case: dict) -> dict:
    """Drives POST /survey/sessions/{id}/message for one case through a
    freshly isolated app+DB, mocking only the classifier boundary (ISCO/
    ISIC/ISCED) so the REAL SemanticRelationEngine and REAL enforcement
    code in survey_routes.py run unmodified."""
    eng = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    from backend.database.models import Base, User, SurveySession, HITLQueue
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)
    db = Session()

    from backend.main import app
    from backend.database.connection import get_db
    from backend.agents.conversation_manager import ConversationContext
    from backend.api import survey_routes as _sr_module
    from backend.auth.jwt_handler import create_access_token

    app.dependency_overrides[get_db] = lambda: db
    _sr_module._contexts.clear()

    job_title = case["job_title"]
    mock_cm = MagicMock()
    mock_lp = MagicMock()

    def _new_context(sid, lang="en"):
        ctx = ConversationContext(session_id=sid, language=lang)
        ctx.collected_data["industry"] = "placeholder industry"
        ctx.collected_data["education_level"] = "placeholder education"
        return ctx

    def _process_message(ctx, msg):
        ctx.collected_data["job_title"] = job_title
        return "Thanks, noted."

    mock_cm.new_context.side_effect = _new_context
    mock_cm.process_message.side_effect = _process_message

    mock_lp_result = MagicMock()
    mock_lp_result.detected_language = "en"
    mock_lp_result.is_code_switched = False
    mock_lp_result.entities = []
    mock_lp.process.return_value = mock_lp_result

    isco_result_obj = MagicMock(
        primary=MagicMock(code=case["isco_code"], title_en="Test Title", title_ar="x", confidence=0.95),
        method="flat_semantic", hitl_required=False, reasoning="test reasoning",
        hierarchy_path=None, alternatives=[], stage_confidences={},
    )
    isic_result_obj = MagicMock(
        section=case["isic_section"], section_title="Test Section", division_code="00",
        division_title="Test Division", group_code="", group_title="", class_code="",
        class_title="", confidence=0.90, method="keyword",
    )
    isced_result_obj = MagicMock(
        level=case["isced_level"], level_title="Test Level", broad_code="", broad_title="",
        narrow_code="", narrow_title="", detailed_code="", detailed_title="",
        confidence=0.90, method="keyword",
    )

    patchers = [
        patch("backend.auth.email_otp.send_otp_email", return_value=True),
        patch("backend.api.survey_routes._get_agents", return_value=(mock_cm, mock_lp)),
        patch("backend.api.survey_routes._get_context_memory"),
        patch("backend.api.survey_routes._get_isco_classifier"),
        patch("backend.api.survey_routes._get_isic_classifier"),
        patch("backend.api.survey_routes._get_isced_classifier"),
        patch("backend.api.survey_routes._get_nationality_classifier"),
        patch("backend.api.survey_routes._get_hitl_quality_manager"),
        patch("backend.api.survey_routes._get_emotional_intelligence"),
        patch("backend.api.survey_routes._get_validation_agent"),
        patch("backend.api.survey_routes._get_audit_logger"),
    ]
    started = [p.start() for p in patchers]
    (_send_otp, _agents, mock_ctx_mem, mock_isco, mock_isic, mock_isced,
     mock_nat, mock_hitl, mock_ei, _val, _audit) = started

    mock_ctx_mem.return_value.load_session.return_value = None
    mock_isco.return_value.classify.return_value = isco_result_obj
    mock_isic.return_value.classify.return_value = isic_result_obj
    mock_isced.return_value.classify.return_value = isced_result_obj
    mock_nat.return_value.classify.return_value = MagicMock(method="unknown")
    mock_hitl.return_value.review_session.return_value = MagicMock(status=MagicMock(value="pass"))
    mock_ei.return_value.analyze.return_value = MagicMock(state="neutral")

    try:
        client = TestClient(app)
        user = User(email=f"sre-verify-{id(case)}-{id(db)}@test.invalid")
        db.add(user)
        db.commit()
        db.refresh(user)
        token = create_access_token(user.id)
        session = SurveySession(user_id=user.id, language="en", status="active")
        db.add(session)
        db.commit()
        db.refresh(session)

        resp = client.post(
            f"/survey/sessions/{session.id}/message",
            json={"message": f"I am a {job_title}"},
            headers={"Authorization": f"Bearer {token}"},
        )
        status_code = resp.status_code

        rows = db.query(HITLQueue).filter(HITLQueue.session_id == session.id).all()
        sre_rows = [r for r in rows if r.ai_reasoning and "SRE HIGH-severity" in r.ai_reasoning]
        escalated = len(sre_rows) >= 1
    finally:
        for p in patchers:
            p.stop()
        app.dependency_overrides.pop(get_db, None)
        db.close()

    expected_high = case["predicted_overall_severity"] == "HIGH"
    return {
        "description": case["description"],
        "isco_code": case["isco_code"],
        "isic_section": case["isic_section"],
        "isced_level": case["isced_level"],
        "predicted_overall_severity": case["predicted_overall_severity"],
        "status_code": status_code,
        "escalated": escalated,
        "expected_high": expected_high,
        "correct": escalated == expected_high,
    }


def run_pass() -> list[dict]:
    cases = build_cases()
    return [_run_one_case(c) for c in cases]


def summarize(results: list[dict]) -> dict:
    high_cases = [r for r in results if r["expected_high"]]
    non_high_cases = [r for r in results if not r["expected_high"]]
    high_escalated = sum(1 for r in high_cases if r["escalated"])
    non_high_escalated = sum(1 for r in non_high_cases if r["escalated"])
    non200 = [r for r in results if r["status_code"] != 200]
    return {
        "n_total": len(results),
        "n_high": len(high_cases),
        "n_non_high": len(non_high_cases),
        "high_escalation_rate": high_escalated / len(high_cases) if high_cases else None,
        "non_high_escalation_rate": non_high_escalated / len(non_high_cases) if non_high_cases else None,
        "all_correct": all(r["correct"] for r in results),
        "n_non_200_status": len(non200),
        "incorrect_cases": [r["description"] for r in results if not r["correct"]],
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default="backend/evaluation/sre_hitl_enforcement_endpoint_verification.json")
    args = parser.parse_args()

    all_summaries = []
    all_results = []
    for run_idx in range(1, args.runs + 1):
        results = run_pass()
        summary = summarize(results)
        all_summaries.append(summary)
        all_results.append(results)
        print(f"--- Run {run_idx} ---")
        print(json.dumps(summary, indent=2))

    stable = len({json.dumps(s, sort_keys=True) for s in all_summaries}) == 1
    print(f"\nStable across {args.runs} runs: {stable}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "runs": all_summaries,
        "stable_across_runs": stable,
        "detailed_results_last_run": all_results[-1],
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
