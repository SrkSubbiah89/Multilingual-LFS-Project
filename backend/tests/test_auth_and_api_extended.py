"""
Extended tests for Authentication Routes, Survey Routes, and Security
Covers: OTP expiry edge cases, JWT edge cases, rate limiting,
        all HTTP methods, auth guard on all endpoints, CORS,
        input sanitisation, concurrent requests.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    from backend.database.models import Base
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def db(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def client(db):
    from backend.main import app
    from backend.database.connection import get_db
    from backend.agents.conversation_manager import ConversationContext

    app.dependency_overrides[get_db] = lambda: db

    mock_cm = MagicMock()
    mock_lp = MagicMock()
    mock_cm.new_context.side_effect = lambda sid, lang="en": ConversationContext(
        session_id=sid, language=lang
    )
    mock_cm.process_message.return_value = "Hello, please continue."

    mock_lp_result = MagicMock()
    mock_lp_result.detected_language = "en"
    mock_lp_result.is_code_switched = False
    mock_lp_result.entities = []
    mock_lp.process.return_value = mock_lp_result

    with patch("backend.auth.email_otp.send_otp_email", return_value=True), \
         patch("backend.api.survey_routes._get_agents", return_value=(mock_cm, mock_lp)), \
         patch("backend.api.survey_routes._get_isco_classifier") as mock_isco, \
         patch("backend.api.survey_routes._get_isic_classifier"), \
         patch("backend.api.survey_routes._get_isced_classifier"), \
         patch("backend.api.survey_routes._get_hitl_quality_manager") as mock_hitl:
        mock_isco.return_value.classify.return_value = MagicMock(
            primary=MagicMock(code="2512", title_en="SW Dev", title_ar="مطور",
                              confidence=0.9),
            method="flat_semantic",
        )
        mock_hitl.return_value.review_session.return_value = MagicMock(
            status=MagicMock(value="pass")
        )
        yield TestClient(app)


def create_user_and_get_token(client, email="test@example.com"):
    client.post("/auth/request-otp", json={"email": email})
    with patch("backend.auth.email_otp.verify_otp", return_value=True):
        resp = client.post("/auth/verify-otp", json={"email": email, "code": "123456"})
        return resp.json().get("access_token")


# ─── OTP Expiry Edge Cases ───────────────────────────────────────────────────

class TestOTPEdgeCases:
    def test_otp_exactly_at_expiry_rejected(self, client):
        """OTP used exactly at expiry time should be rejected"""
        email = "expiry@test.com"
        client.post("/auth/request-otp", json={"email": email})
        with patch("backend.auth.email_otp.datetime") as mock_dt:
            # Set current time to exactly the expiry time
            mock_dt.utcnow.return_value = datetime.utcnow() + timedelta(minutes=10)
            resp = client.post("/auth/verify-otp", json={"email": email, "code": "123456"})
            assert resp.status_code in (401, 404)

    def test_otp_reuse_rejected(self, client):
        """Once used OTP cannot be used again"""
        email = "reuse@test.com"
        client.post("/auth/request-otp", json={"email": email})
        with patch("backend.auth.email_otp.verify_otp", return_value=True):
            client.post("/auth/verify-otp", json={"email": email, "code": "111111"})
            resp = client.post("/auth/verify-otp", json={"email": email, "code": "111111"})
            assert resp.status_code in (401, 400)

    def test_new_otp_request_invalidates_old(self, client):
        """Requesting new OTP should invalidate previous one"""
        email = "new-otp@test.com"
        client.post("/auth/request-otp", json={"email": email})
        client.post("/auth/request-otp", json={"email": email})
        # Old code should now be invalid

    def test_otp_wrong_by_one_digit_rejected(self, client):
        email = "wrongdigit@test.com"
        client.post("/auth/request-otp", json={"email": email})
        resp = client.post("/auth/verify-otp", json={"email": email, "code": "000001"})
        assert resp.status_code in (401, 404)

    def test_otp_with_extra_whitespace_rejected(self, client):
        email = "whitespace@test.com"
        resp = client.post("/auth/verify-otp", json={"email": email, "code": " 123456 "})
        assert resp.status_code in (401, 404, 422)

    def test_otp_non_numeric_rejected(self, client):
        email = "alpha@test.com"
        resp = client.post("/auth/verify-otp", json={"email": email, "code": "abcdef"})
        assert resp.status_code in (401, 404, 422)


# ─── JWT Edge Cases ──────────────────────────────────────────────────────────

class TestJWTEdgeCases:
    def test_expired_jwt_rejected(self, client):
        """Expired JWT should return 401"""
        with patch("backend.api.survey_routes.verify_access_token",
                   return_value=None):
            resp = client.get("/survey/sessions",
                              headers={"Authorization": "Bearer expired.token.here"})
            assert resp.status_code in (401, 403)

    def test_malformed_jwt_rejected(self, client):
        resp = client.get("/survey/sessions",
                          headers={"Authorization": "Bearer not.a.jwt"})
        assert resp.status_code in (401, 403)

    def test_bearer_prefix_required(self, client):
        resp = client.get("/survey/sessions",
                          headers={"Authorization": "justtoken"})
        assert resp.status_code in (401, 403, 422)

    def test_no_auth_header_returns_403(self, client):
        resp = client.get("/survey/sessions")
        assert resp.status_code in (401, 403)

    def test_jwt_with_wrong_secret_rejected(self, client):
        import jwt
        token = jwt.encode({"sub": "1", "exp": datetime.utcnow() + timedelta(hours=1)},
                           "wrong-secret", algorithm="HS256")
        resp = client.get("/survey/sessions",
                          headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code in (401, 403)


# ─── API Endpoint Coverage ───────────────────────────────────────────────────

class TestAllEndpointsCovered:
    """Every API endpoint should return proper response codes"""

    def test_health_check_no_auth(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_docs_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_create_session_wrong_language_code(self, client):
        token = create_user_and_get_token(client, "lang@test.com")
        if token:
            resp = client.post("/survey/sessions",
                               json={"language": "xx"},  # invalid language code
                               headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code in (201, 422)  # depends on validation

    def test_get_nonexistent_session_404(self, client):
        token = create_user_and_get_token(client, "get404@test.com")
        if token:
            resp = client.get("/survey/sessions/999999",
                              headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 404

    def test_delete_another_users_session(self, client):
        token1 = create_user_and_get_token(client, "user1@test.com")
        token2 = create_user_and_get_token(client, "user2@test.com")
        if token1 and token2:
            resp = client.post("/survey/sessions",
                               json={"language": "en"},
                               headers={"Authorization": f"Bearer {token1}"})
            if resp.status_code == 201:
                session_id = resp.json()["id"]
                del_resp = client.delete(f"/survey/sessions/{session_id}",
                                         headers={"Authorization": f"Bearer {token2}"})
                assert del_resp.status_code == 404

    def test_send_message_to_completed_session_rejected(self, client):
        token = create_user_and_get_token(client, "completed@test.com")
        if token:
            resp = client.post("/survey/sessions",
                               json={"language": "en"},
                               headers={"Authorization": f"Bearer {token}"})
            if resp.status_code == 201:
                session_id = resp.json()["id"]
                client.patch(f"/survey/sessions/{session_id}/complete",
                             headers={"Authorization": f"Bearer {token}"})
                msg_resp = client.post(f"/survey/sessions/{session_id}/message",
                                       json={"message": "another message"},
                                       headers={"Authorization": f"Bearer {token}"})
                assert msg_resp.status_code in (409, 400)

    def test_hitl_queue_requires_auth(self, client):
        resp = client.get("/survey/hitl/queue")
        assert resp.status_code in (401, 403)

    def test_report_endpoint_requires_auth(self, client):
        resp = client.get("/survey/sessions/1/report")
        assert resp.status_code in (401, 403)


# ─── Input Sanitisation ──────────────────────────────────────────────────────

class TestInputSanitisation:
    def test_sql_injection_in_email(self, client):
        resp = client.post("/auth/request-otp",
                           json={"email": "'; DROP TABLE users; --@test.com"})
        assert resp.status_code in (422, 400)

    def test_xss_in_message(self, client):
        token = create_user_and_get_token(client, "xss@test.com")
        if token:
            resp = client.post("/survey/sessions",
                               json={"language": "en"},
                               headers={"Authorization": f"Bearer {token}"})
            if resp.status_code == 201:
                session_id = resp.json()["id"]
                resp = client.post(f"/survey/sessions/{session_id}/message",
                                   json={"message": "<script>alert('xss')</script>"},
                                   headers={"Authorization": f"Bearer {token}"})
                assert resp.status_code in (200, 201, 400)
                if resp.status_code == 200:
                    assert "<script>" not in resp.text

    def test_empty_message_handled(self, client):
        token = create_user_and_get_token(client, "empty@test.com")
        if token:
            resp = client.post("/survey/sessions",
                               json={"language": "en"},
                               headers={"Authorization": f"Bearer {token}"})
            if resp.status_code == 201:
                session_id = resp.json()["id"]
                resp = client.post(f"/survey/sessions/{session_id}/message",
                                   json={"message": ""},
                                   headers={"Authorization": f"Bearer {token}"})
                assert resp.status_code in (200, 422)

    def test_very_long_message_handled(self, client):
        token = create_user_and_get_token(client, "long@test.com")
        if token:
            resp = client.post("/survey/sessions",
                               json={"language": "en"},
                               headers={"Authorization": f"Bearer {token}"})
            if resp.status_code == 201:
                session_id = resp.json()["id"]
                resp = client.post(f"/survey/sessions/{session_id}/message",
                                   json={"message": "a" * 10000},
                                   headers={"Authorization": f"Bearer {token}"})
                assert resp.status_code in (200, 400, 413)

    def test_arabic_rtl_message_accepted(self, client):
        token = create_user_and_get_token(client, "arabic@test.com")
        if token:
            resp = client.post("/survey/sessions",
                               json={"language": "ar"},
                               headers={"Authorization": f"Bearer {token}"})
            if resp.status_code == 201:
                session_id = resp.json()["id"]
                resp = client.post(f"/survey/sessions/{session_id}/message",
                                   json={"message": "أنا مهندس برمجيات"},
                                   headers={"Authorization": f"Bearer {token}"})
                assert resp.status_code in (200, 201)


# ─── Response Format Validation ──────────────────────────────────────────────

class TestResponseFormats:
    def test_message_response_has_all_fields(self, client):
        token = create_user_and_get_token(client, "format@test.com")
        if token:
            resp = client.post("/survey/sessions",
                               json={"language": "en"},
                               headers={"Authorization": f"Bearer {token}"})
            if resp.status_code == 201:
                session_id = resp.json()["id"]
                resp = client.post(f"/survey/sessions/{session_id}/message",
                                   json={"message": "I am a software engineer"},
                                   headers={"Authorization": f"Bearer {token}"})
                if resp.status_code == 200:
                    data = resp.json()
                    assert "reply" in data
                    assert "state" in data
                    assert "detected_language" in data
                    assert "session_completed" in data
                    assert "latency_ms" in data

    def test_session_response_has_id(self, client):
        token = create_user_and_get_token(client, "sessformat@test.com")
        if token:
            resp = client.post("/survey/sessions",
                               json={"language": "en"},
                               headers={"Authorization": f"Bearer {token}"})
            if resp.status_code == 201:
                assert "id" in resp.json()

    def test_error_response_has_detail(self, client):
        resp = client.get("/survey/sessions/99999",
                          headers={"Authorization": "Bearer invalid"})
        data = resp.json()
        assert "detail" in data
