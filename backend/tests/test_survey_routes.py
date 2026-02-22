"""
Tests for /survey/* endpoints.

Fixtures from conftest.py:
  db            - in-memory SQLite session
  user          - User ORM object
  other_user    - second User ORM object
  auth_client   - TestClient authenticated as `user`
  survey_session - freshly created in-progress session (dict)
  survey_response - response submitted to survey_session (dict)
"""

import pytest
from backend.api.survey_routes import get_current_user
from backend.database.models import SurveyResponse, SurveySession


# ---------------------------------------------------------------------------
# POST /survey/sessions
# ---------------------------------------------------------------------------

class TestCreateSession:
    def test_returns_201_with_correct_fields(self, auth_client):
        resp = auth_client.post("/survey/sessions", json={"language": "en"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["language"] == "en"
        assert data["status"] == "in_progress"
        assert data["completed_at"] is None
        assert "id" in data
        assert "started_at" in data

    def test_sets_user_id_to_current_user(self, auth_client, user):
        resp = auth_client.post("/survey/sessions", json={"language": "fr"})
        assert resp.json()["user_id"] == user.id

    def test_missing_language_returns_422(self, auth_client):
        resp = auth_client.post("/survey/sessions", json={})
        assert resp.status_code == 422

    def test_no_auth_returns_403(self, db):
        from backend.database.connection import get_db
        from backend.main import app
        from fastapi.testclient import TestClient

        saved = dict(app.dependency_overrides)
        try:
            app.dependency_overrides[get_db] = lambda: (yield db)
            # deliberately omit get_current_user override
            if get_current_user in app.dependency_overrides:
                del app.dependency_overrides[get_current_user]
            with TestClient(app) as c:
                resp = c.post("/survey/sessions", json={"language": "en"})
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(saved)
        assert resp.status_code in (401, 403)  # HTTPBearer raises 401/403 when no token provided


# ---------------------------------------------------------------------------
# GET /survey/sessions
# ---------------------------------------------------------------------------

class TestListSessions:
    def test_returns_empty_list_when_no_sessions(self, auth_client):
        resp = auth_client.get("/survey/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_all_sessions_for_user(self, auth_client):
        auth_client.post("/survey/sessions", json={"language": "en"})
        auth_client.post("/survey/sessions", json={"language": "fr"})
        resp = auth_client.get("/survey/sessions")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_does_not_return_other_users_sessions(self, auth_client, db, other_user):
        # Create a session directly in DB for other_user
        other_session = SurveySession(user_id=other_user.id, status="in_progress", language="de")
        db.add(other_session)
        db.commit()

        resp = auth_client.get("/survey/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_sessions_newest_first(self, auth_client):
        auth_client.post("/survey/sessions", json={"language": "en"})
        auth_client.post("/survey/sessions", json={"language": "fr"})
        sessions = auth_client.get("/survey/sessions").json()
        assert sessions[0]["language"] == "fr"
        assert sessions[1]["language"] == "en"


# ---------------------------------------------------------------------------
# GET /survey/sessions/{session_id}
# ---------------------------------------------------------------------------

class TestGetSession:
    def test_returns_session(self, auth_client, survey_session):
        resp = auth_client.get(f"/survey/sessions/{survey_session['id']}")
        assert resp.status_code == 200
        assert resp.json()["id"] == survey_session["id"]

    def test_not_found_returns_404(self, auth_client):
        resp = auth_client.get("/survey/sessions/9999")
        assert resp.status_code == 404

    def test_other_users_session_returns_404(self, auth_client, db, other_user):
        other_session = SurveySession(user_id=other_user.id, status="in_progress", language="en")
        db.add(other_session)
        db.commit()
        resp = auth_client.get(f"/survey/sessions/{other_session.id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /survey/sessions/{session_id}/complete
# ---------------------------------------------------------------------------

class TestCompleteSession:
    def test_marks_session_completed(self, auth_client, survey_session):
        resp = auth_client.patch(f"/survey/sessions/{survey_session['id']}/complete")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["completed_at"] is not None

    def test_already_completed_returns_409(self, auth_client, survey_session):
        sid = survey_session["id"]
        auth_client.patch(f"/survey/sessions/{sid}/complete")
        resp = auth_client.patch(f"/survey/sessions/{sid}/complete")
        assert resp.status_code == 409
        assert "already completed" in resp.json()["detail"]

    def test_not_found_returns_404(self, auth_client):
        resp = auth_client.patch("/survey/sessions/9999/complete")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /survey/sessions/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteSession:
    def test_deletes_session(self, auth_client, survey_session):
        sid = survey_session["id"]
        resp = auth_client.delete(f"/survey/sessions/{sid}")
        assert resp.status_code == 200
        assert str(sid) in resp.json()["message"]

    def test_deleted_session_returns_404(self, auth_client, survey_session):
        sid = survey_session["id"]
        auth_client.delete(f"/survey/sessions/{sid}")
        resp = auth_client.get(f"/survey/sessions/{sid}")
        assert resp.status_code == 404

    def test_delete_cascades_responses(self, auth_client, survey_session, survey_response, db):
        sid = survey_session["id"]
        rid = survey_response["id"]
        auth_client.delete(f"/survey/sessions/{sid}")
        assert db.get(SurveyResponse, rid) is None

    def test_not_found_returns_404(self, auth_client):
        resp = auth_client.delete("/survey/sessions/9999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /survey/sessions/{session_id}/responses
# ---------------------------------------------------------------------------

class TestSubmitResponse:
    def test_creates_response_with_all_fields(self, auth_client, survey_session):
        sid = survey_session["id"]
        resp = auth_client.post(
            f"/survey/sessions/{sid}/responses",
            json={"question_id": "q1", "answer": "Nurse", "isco_code": "2221", "confidence_score": 0.91},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["question_id"] == "q1"
        assert data["answer"] == "Nurse"
        assert data["isco_code"] == "2221"
        assert data["confidence_score"] == 0.91
        assert data["session_id"] == sid

    def test_creates_response_without_optional_fields(self, auth_client, survey_session):
        sid = survey_session["id"]
        resp = auth_client.post(
            f"/survey/sessions/{sid}/responses",
            json={"question_id": "q2", "answer": "Teacher"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["isco_code"] is None
        assert data["confidence_score"] is None

    def test_completed_session_returns_409(self, auth_client, survey_session):
        sid = survey_session["id"]
        auth_client.patch(f"/survey/sessions/{sid}/complete")
        resp = auth_client.post(
            f"/survey/sessions/{sid}/responses",
            json={"question_id": "q1", "answer": "Should fail"},
        )
        assert resp.status_code == 409
        assert "completed" in resp.json()["detail"]

    def test_nonexistent_session_returns_404(self, auth_client):
        resp = auth_client.post(
            "/survey/sessions/9999/responses",
            json={"question_id": "q1", "answer": "fail"},
        )
        assert resp.status_code == 404

    def test_missing_required_fields_returns_422(self, auth_client, survey_session):
        sid = survey_session["id"]
        resp = auth_client.post(f"/survey/sessions/{sid}/responses", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /survey/sessions/{session_id}/responses
# ---------------------------------------------------------------------------

class TestListResponses:
    def test_returns_empty_list(self, auth_client, survey_session):
        resp = auth_client.get(f"/survey/sessions/{survey_session['id']}/responses")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_all_responses(self, auth_client, survey_session):
        sid = survey_session["id"]
        auth_client.post(f"/survey/sessions/{sid}/responses", json={"question_id": "q1", "answer": "A"})
        auth_client.post(f"/survey/sessions/{sid}/responses", json={"question_id": "q2", "answer": "B"})
        resp = auth_client.get(f"/survey/sessions/{sid}/responses")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_nonexistent_session_returns_404(self, auth_client):
        resp = auth_client.get("/survey/sessions/9999/responses")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /survey/sessions/{session_id}/responses/{response_id}
# ---------------------------------------------------------------------------

class TestUpdateResponse:
    def test_updates_all_fields(self, auth_client, survey_session, survey_response):
        sid = survey_session["id"]
        rid = survey_response["id"]
        resp = auth_client.patch(
            f"/survey/sessions/{sid}/responses/{rid}",
            json={"question_id": "q1", "answer": "Senior Engineer", "isco_code": "2513", "confidence_score": 0.99},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Senior Engineer"
        assert data["isco_code"] == "2513"
        assert data["confidence_score"] == 0.99

    def test_nonexistent_response_returns_404(self, auth_client, survey_session):
        sid = survey_session["id"]
        resp = auth_client.patch(
            f"/survey/sessions/{sid}/responses/9999",
            json={"question_id": "q1", "answer": "fail", "isco_code": None, "confidence_score": None},
        )
        assert resp.status_code == 404

    def test_nonexistent_session_returns_404(self, auth_client):
        resp = auth_client.patch(
            "/survey/sessions/9999/responses/1",
            json={"question_id": "q1", "answer": "fail", "isco_code": None, "confidence_score": None},
        )
        assert resp.status_code == 404
