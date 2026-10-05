"""Required-service readiness and diagnostic probes, without live connections."""

from unittest.mock import MagicMock

import pytest
import redis

from backend import main


def managed_mock(**attributes):
    resource = MagicMock(**attributes)
    resource.__enter__.return_value = resource
    resource.close.return_value = None

    def close_on_exit(*args):
        resource.close()
        return False

    resource.__exit__.side_effect = close_on_exit
    return resource


@pytest.fixture
def required_services(monkeypatch):
    database = managed_mock()
    storage = managed_mock()
    storage.ping.return_value = True
    session_factory = MagicMock(return_value=database)
    redis_factory = MagicMock(return_value=storage)
    monkeypatch.setattr(main, "SessionLocal", session_factory)
    monkeypatch.setattr(redis.Redis, "from_url", redis_factory)
    return database, storage, session_factory, redis_factory


def test_ready_requires_both_services_and_closes_resources(client, required_services):
    database, storage, _, _ = required_services
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready", "database": "connected", "redis": "connected",
    }
    assert str(database.execute.call_args.args[0]) == "SELECT 1"
    database.close.assert_called_once()
    storage.close.assert_called_once()


def test_database_query_failure_returns_503_and_closes_session(client, required_services):
    database, _, _, redis_factory = required_services
    database.execute.side_effect = RuntimeError("test connection failure")

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == "Database is not ready"
    database.close.assert_called_once()
    redis_factory.assert_not_called()


def test_database_session_creation_failure_returns_503(client, required_services):
    _, _, session_factory, redis_factory = required_services
    session_factory.side_effect = RuntimeError("test unavailable database")

    response = client.get("/ready")

    assert response.status_code == 503
    redis_factory.assert_not_called()


def test_redis_creation_failure_returns_503_after_closing_database(client, required_services):
    database, _, _, redis_factory = required_services
    redis_factory.side_effect = redis.ConnectionError("test unavailable Redis")

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == "Redis interview storage is not ready"
    database.close.assert_called_once()


@pytest.mark.parametrize("failure", [redis.ConnectionError, redis.TimeoutError])
def test_redis_probe_failure_returns_503_and_closes_client(client, required_services, failure):
    database, storage, _, _ = required_services
    storage.ping.side_effect = failure("test readiness failure")

    response = client.get("/ready")

    assert response.status_code == 503
    database.close.assert_called_once()
    storage.close.assert_called_once()


def test_unacknowledged_redis_ping_is_not_ready(client, required_services):
    _, storage, _, _ = required_services
    storage.ping.return_value = False

    assert client.get("/ready").status_code == 503
    storage.close.assert_called_once()


def test_redis_probe_has_finite_connect_and_read_timeouts(client, required_services):
    _, _, _, redis_factory = required_services

    assert client.get("/ready").status_code == 200

    options = redis_factory.call_args.kwargs
    assert 0 < options["socket_connect_timeout"] <= 2
    assert 0 < options["socket_timeout"] <= 2
    assert options["retry"].get_retries() == 0


@pytest.mark.parametrize("override,expected_url", [
    (None, "http://qdrant:7333/healthz"),
    ("https://vector.example.invalid/custom/", "https://vector.example.invalid/custom/healthz"),
])
def test_health_uses_configured_qdrant_address_and_closes_probes(
    client, required_services, monkeypatch, override, expected_url,
):
    import urllib.request

    _, storage, _, _ = required_services
    monkeypatch.setenv("QDRANT_HOST", "qdrant")
    monkeypatch.setenv("QDRANT_PORT", "7333")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434/")
    if override is None:
        monkeypatch.delenv("QDRANT_URL", raising=False)
    else:
        monkeypatch.setenv("QDRANT_URL", override)
    qdrant_response = managed_mock(status=200)
    ollama_response = managed_mock(status=200)
    open_url = MagicMock(side_effect=[qdrant_response, ollama_response])
    monkeypatch.setattr(urllib.request, "urlopen", open_url)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["services"] == {"redis": "ok", "qdrant": "ok", "ollama": "ok"}
    assert open_url.call_args_list[0].args[0] == expected_url
    assert open_url.call_args_list[1].args[0] == "http://ollama:11434/api/tags"
    assert all(call.kwargs["timeout"] == 2 for call in open_url.call_args_list)
    storage.close.assert_called_once()
    qdrant_response.close.assert_called_once()
    ollama_response.close.assert_called_once()


def test_health_stays_diagnostic_when_services_fail(client, required_services, monkeypatch):
    import urllib.request

    _, storage, _, _ = required_services
    storage.ping.side_effect = redis.ConnectionError("test storage unavailable")
    monkeypatch.setattr(urllib.request, "urlopen", MagicMock(side_effect=TimeoutError("test timeout")))

    response = client.get("/health")

    assert response.status_code == 200
    assert all(value.startswith("error:") for value in response.json()["services"].values())
    storage.close.assert_called_once()
