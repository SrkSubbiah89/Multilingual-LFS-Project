import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.connection import Base, get_db
from backend.database.models import User
from backend.main import app
from backend.api.survey_routes import get_current_user


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Clear rate-limiter state (in-process + Redis) before/after every test.

    Without this, tests that call /auth/request-otp share the same 'testclient'
    IP bucket and hit the 5-requests/10-min ceiling after a few tests.
    The Redis flush handles the distributed rate limiter when Docker Redis is up.
    """
    from backend.auth.email_otp import _rl_windows, _get_rl_redis

    def _clear_all():
        _rl_windows.clear()
        try:
            r = _get_rl_redis()
            if r:
                keys = list(r.scan_iter("lfs:rl:*"))
                if keys:
                    r.delete(*keys)
        except Exception:
            pass

    _clear_all()
    yield
    _clear_all()


@pytest.fixture()
def db():
    """Isolated in-memory SQLite database per test.

    StaticPool ensures every SQLAlchemy checkout reuses the same underlying
    connection, which is required for in-memory SQLite (each real connection
    would otherwise get its own empty database).
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def user(db):
    u = User(email="user@test.com", is_active=True)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture()
def other_user(db):
    u = User(email="other@test.com", is_active=True)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture()
def client(db):
    """Unauthenticated TestClient with in-memory DB — used for auth endpoint tests."""
    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture()
def auth_client(db, user):
    """TestClient authenticated as `user` with in-memory DB."""
    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: user

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture()
def survey_session(auth_client):
    """A freshly created in-progress survey session."""
    resp = auth_client.post("/survey/sessions", json={"language": "en"})
    assert resp.status_code == 201
    return resp.json()


@pytest.fixture()
def survey_response(auth_client, survey_session):
    """A response submitted to `survey_session`."""
    resp = auth_client.post(
        f"/survey/sessions/{survey_session['id']}/responses",
        json={
            "question_id": "q1",
            "answer": "Software Engineer",
            "isco_code": "2512",
            "confidence_score": 0.95,
        },
    )
    assert resp.status_code == 201
    return resp.json()
