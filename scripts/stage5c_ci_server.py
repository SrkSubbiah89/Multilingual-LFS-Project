"""Disposable API for the GitHub-hosted Stage 5C job.

Credentials come from the job environment, which generates them at runtime.
This process refuses the project .env, refuses protected database names, and
does not print passwords. It does not read the laptop or its WSL services.
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ENV = (ROOT / ".env").resolve()
EXPECTED_REVISION = "a6c8f42e7d93"
EXPECTED_DATABASE = "lfs_stage5c_ci"
EXPECTED_USER = "lfs_stage5c_ci"
PROTECTED = {"lfs_native_test", "lfs_db"}
EXPECTED_TABLES = {
    "users", "otp_codes", "survey_sessions", "survey_responses", "audit_logs",
    "data_access_logs", "agent_decision_logs", "quality_reviews", "hitl_queue",
    "person_register", "survey_report_records", "alembic_version",
}


def fail(message):
    print("BLOCKED: " + message, file=sys.stderr)
    raise SystemExit(2)


def assert_ci_context():
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("STAGE5C_CI") != "1":
        fail("this server runs only inside the GitHub-hosted Stage 5C job")
    if PROJECT_ENV.exists() or (ROOT / "frontend" / ".env.local").exists():
        fail("checkout contains a local environment file")


def assert_targets():
    database = urlparse(os.environ.get("DATABASE_URL", "").replace("postgresql+psycopg2", "postgresql", 1))
    redis = urlparse(os.environ.get("REDIS_URL", ""))
    name = (database.path or "").lstrip("/")
    if database.hostname != "127.0.0.1" or database.port != 5432:
        fail("database listener is not 127.0.0.1:5432")
    if database.username != EXPECTED_USER or name != EXPECTED_DATABASE:
        fail("database identity is not the CI disposable database")
    if name in PROTECTED or "15432" in os.environ.get("DATABASE_URL", "") or "25432" in os.environ.get("DATABASE_URL", ""):
        fail("refusing a protected or laptop database")
    if redis.hostname != "127.0.0.1" or redis.port != 6379:
        fail("Redis listener is not 127.0.0.1:6379")
    if not database.password or not redis.password or not os.environ.get("JWT_SECRET"):
        fail("runtime credentials are incomplete")


def install_environment():
    os.environ.update({
        "APP_ENV": "development",
        "APP_HOST": "127.0.0.1",
        "LFS_ISOLATED_RUNTIME": "true",
        "LFS_ENABLE_ISCO_DEBUG": "false",
        "LFS_FAST_MODE": "false",
        "LFS_LOCAL_ONLY": "true",
        "ISCO_RETRIEVAL_STRATEGY": "parent_document",
        "OLLAMA_MODEL": "qwen2.5:3b",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "QDRANT_HOST": "127.0.0.1",
        "QDRANT_PORT": "6333",
        "ENABLE_SURVEY_CLASSIFICATION_CREW": "false",
        "CORS_ORIGINS": "http://127.0.0.1:13000",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "USE_TF": "0",
        "CREWAI_DISABLE_TELEMETRY": "true",
        "OTEL_SDK_DISABLED": "true",
        "ANTHROPIC_API_KEY": "",
        "OPENAI_API_KEY": "",
        "GROQ_API_KEY": "",
        "GEMINI_API_KEY": "",
        "OPENROUTER_API_KEY": "",
        "SENDGRID_API_KEY": "",
        "GMAIL_APP_PASSWORD": "",
        "TWILIO_ACCOUNT_SID": "",
        "TWILIO_AUTH_TOKEN": "",
    })
    import dotenv
    original = dotenv.main.load_dotenv

    def isolated_load_dotenv(*args, **kwargs):
        explicit = kwargs.get("dotenv_path", args[0] if args else None)
        if not explicit:
            explicit = dotenv.main.find_dotenv()
        if explicit and Path(str(explicit)).resolve() == PROJECT_ENV:
            return False
        return original(*args, **kwargs)

    dotenv.load_dotenv = isolated_load_dotenv
    dotenv.main.load_dotenv = isolated_load_dotenv


def connect():
    import psycopg2
    database = urlparse(os.environ["DATABASE_URL"].replace("postgresql+psycopg2", "postgresql", 1))
    try:
        return psycopg2.connect(
            host=database.hostname,
            port=database.port,
            dbname=database.path.lstrip("/"),
            user=database.username,
            password=database.password,
            connect_timeout=5,
        )
    except Exception as exc:
        fail("database connection failed: " + type(exc).__name__)


def assert_live_identity():
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_user")
            current = cursor.fetchone()
    if tuple(current) != (EXPECTED_DATABASE, EXPECTED_USER):
        fail("live database identity does not match the CI disposable database")
    print("DATABASE_IDENTITY_OK")


def upgrade():
    assert_live_identity()
    from alembic.config import Config
    from alembic import command
    os.chdir(ROOT)
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT version_num FROM alembic_version")
            revision = cursor.fetchone()[0]
            cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            tables = {row[0] for row in cursor.fetchall()}
            cursor.execute("SELECT count(*) FROM users")
            users = cursor.fetchone()[0]
    if revision != EXPECTED_REVISION:
        fail("Alembic revision is " + revision)
    missing = EXPECTED_TABLES - tables
    if missing:
        fail("schema is missing " + ", ".join(sorted(missing)))
    if users != 0:
        fail("CI database was not empty after migration")
    print("SCHEMA_OK revision=" + revision + " tables=" + str(len(tables)))


def serve():
    assert_live_identity()
    from backend.main import app

    @asynccontextmanager
    async def no_model_warmup(_app):
        yield

    app.router.lifespan_context = no_model_warmup
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=18000, log_level="info")


PUBLISHED_PORTS = (("5432", "1538"), ("6379", "18EB"))
LOOPBACK_V4_PREFIX = "0100007F:"
LOOPBACK_V6_PREFIX = "00000000000000000000000001000000:"


def listening_rows(text):
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) > 3 and parts[3] == "0A":
            rows.append(parts[1].upper())
    return rows


def nonloopback_ports(ipv4_rows, ipv6_rows):
    """Return published ports that have a non-loopback listener. An absent row is not a pass."""
    found = []
    for label, port in PUBLISHED_PORTS:
        for item in ipv4_rows:
            if item.endswith(":" + port) and not item.startswith(LOOPBACK_V4_PREFIX):
                found.append(label)
        for item in ipv6_rows:
            if item.endswith(":" + port) and not item.startswith(LOOPBACK_V6_PREFIX):
                found.append(label)
    return found


def bindings_are_loopback(bindings):
    """Accept only 127.0.0.1 published to the same host port. Empty or 0.0.0.0 host IPs fail."""
    needed = {"5432": False, "6379": False}
    for container_port, host_ip, host_port in bindings:
        if container_port not in needed:
            continue
        if host_ip != "127.0.0.1" or host_port != container_port:
            return False
        needed[container_port] = True
    return all(needed.values())


def loopback_accepts(port):
    import socket
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=3):
            return True
    except OSError:
        return False


def docker_bindings(container):
    import json
    import subprocess
    result = subprocess.run(
        ["docker", "inspect", container, "--format", "{{json .NetworkSettings.Ports}}"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        fail("docker port inspection failed")
    payload = json.loads(result.stdout)
    bindings = []
    for key, values in (payload or {}).items():
        container_port = key.split("/")[0]
        for value in values or []:
            bindings.append((container_port, value.get("HostIp", ""), value.get("HostPort", "")))
    return bindings


def assert_loopback_listeners():
    ipv4_path = Path("/proc/net/tcp")
    ipv6_path = Path("/proc/net/tcp6")
    ipv4 = listening_rows(ipv4_path.read_text(encoding="utf-8") if ipv4_path.exists() else "")
    ipv6 = listening_rows(ipv6_path.read_text(encoding="utf-8") if ipv6_path.exists() else "")
    outside = nonloopback_ports(ipv4, ipv6)
    if outside:
        fail("port " + outside[0] + " is listening outside loopback")
    bindings = docker_bindings("stage5c-postgres") + docker_bindings("stage5c-redis")
    if not bindings_are_loopback(bindings):
        fail("docker published a port outside 127.0.0.1")
    for label, _port in PUBLISHED_PORTS:
        if not loopback_accepts(label):
            fail("port " + label + " did not accept a loopback connection")
    print("LOOPBACK_LISTENERS_OK")


def probe_http(url, predicate):
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            body = response.read()
            status = response.status
    except Exception:
        return False
    return predicate(status, body, json)


def health():
    def ready(status, body, json):
        payload = json.loads(body.decode("utf-8"))
        required = payload.get("required", {})
        return status == 200 and required.get("database") == "ok" and required.get("redis") == "ok"
    if not probe_http("http://127.0.0.1:18000/health", ready):
        raise SystemExit(1)
    print("API_HEALTH_OK")


def ready_web():
    if not probe_http("http://127.0.0.1:13000/", lambda status, _body, _json: status == 200):
        raise SystemExit(1)
    print("FRONTEND_READY")


def write_restricted_text(target, text):
    """Create a secret file at mode 0600. The mode is applied by open, not afterward."""
    path = Path(target)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    previous = os.umask(0o077)
    descriptor = None
    try:
        descriptor = os.open(str(path), flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            descriptor = None
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.umask(previous)
        if descriptor is not None:
            os.close(descriptor)
    if os.name == "posix":
        mode = os.stat(path).st_mode & 0o777
        if mode != 0o600:
            path.unlink(missing_ok=True)
            fail("credential file mode was not 0600 at creation")


def write_credentials():
    import secrets
    database_password = secrets.token_urlsafe(24)
    redis_password = secrets.token_urlsafe(24)
    jwt_secret = secrets.token_urlsafe(32)
    target = Path(os.environ.get("STAGE5C_CREDENTIAL_FILE", "/tmp/stage5c-ci.env"))
    lines = [
        "PGPASSWORD='" + database_password + "'",
        "REDIS_PASSWORD='" + redis_password + "'",
        "JWT_SECRET='" + jwt_secret + "'",
        "DATABASE_URL='postgresql+psycopg2://" + EXPECTED_USER + ":" + database_password + "@127.0.0.1:5432/" + EXPECTED_DATABASE + "'",
        "REDIS_URL='redis://:" + redis_password + "@127.0.0.1:6379/0'",
    ]
    try:
        write_restricted_text(target, "\n".join(lines) + "\n")
    except FileExistsError:
        fail("credential file already exists")
    print("CREDENTIALS_WRITTEN")


def main():
    assert_ci_context()
    commands = {"credentials", "listeners", "health", "ready-web", "upgrade", "serve"}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        fail("usage: stage5c_ci_server.py credentials|listeners|health|ready-web|upgrade|serve")
    if sys.argv[1] == "credentials":
        write_credentials()
        return
    if sys.argv[1] == "listeners":
        assert_loopback_listeners()
        return
    if sys.argv[1] == "health":
        health()
        return
    if sys.argv[1] == "ready-web":
        ready_web()
        return
    sys.path.insert(0, str(ROOT))
    assert_targets()
    install_environment()
    if sys.argv[1] == "upgrade":
        upgrade()
    else:
        serve()


if __name__ == "__main__":
    main()
