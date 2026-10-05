"""Verify ContextMemory against a task-owned, disposable real Redis container.

Run: python scripts/verify_fresh_redis.py
Requires the already installed redis:7 Docker image; never pulls an image,
reads .env, or connects to an existing Redis service.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread, local
import time
from datetime import datetime, timedelta, timezone
import uuid


ROOT = Path(__file__).resolve().parents[1]
OWNER_LABEL = "org.openai.codex.disposable-redis-verification"


def docker(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *arguments], check=check, capture_output=True,
        text=True, encoding="utf-8", timeout=45,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def inspect_owned(container: str, owner: str, expected_id: str | None = None) -> dict | None:
    result = docker("inspect", container, check=False)
    if result.returncode:
        if "no such object" in result.stderr.lower() or "no such container" in result.stderr.lower():
            return None
        raise RuntimeError(f"Container identity inspection failed: {result.stderr.strip()}")
    records = json.loads(result.stdout)
    require(len(records) == 1, "Container inspection returned an unexpected count.")
    record = records[0]
    require(record["Config"].get("Labels", {}).get(OWNER_LABEL) == owner,
            "Container ownership label does not match this verifier.")
    if expected_id is not None:
        require(record["Id"] == expected_id, "Container ID changed during verification.")
    return record


class FastLockClient:
    """Use actual Redis locks with a short wait for the contention check."""

    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def lock(self, name, **options):
        options.update(blocking_timeout=0.25, sleep=0.02)
        return self.client.lock(name, **options)


def verify(output: Path) -> dict:
    owner = uuid.uuid4().hex
    name = f"codex-lfs-redis-{owner[:12]}"
    evidence = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "client_timezone": "Asia/Dubai",
        "started_at_client_time": datetime.now(timezone(timedelta(hours=4))).isoformat(),
        "isolation": {
            "image": "redis:7", "image_pull": "never", "existing_redis_used": False,
            "dotenv_loaded": False, "llm_constructed": False,
            "container_name": name, "ownership_label": OWNER_LABEL, "ownership_value": owner,
        },
        "checks": {},
        "cleanup": {"removed": False},
    }
    container_id = None
    clients = []
    try:
        image = docker("image", "inspect", "redis:7")
        evidence["isolation"]["image_id"] = json.loads(image.stdout)[0]["Id"]
        launched = docker(
            "run", "--detach", "--pull=never", "--rm", "--name", name,
            "--label", f"{OWNER_LABEL}={owner}",
            "--publish", "127.0.0.1::6379", "redis:7",
            "redis-server", "--save", "", "--appendonly", "no",
        )
        container_id = launched.stdout.strip()
        record = inspect_owned(container_id, owner, expected_id=container_id)
        require(record is not None, "The disposable Redis container did not start.")
        bindings = record["NetworkSettings"]["Ports"]["6379/tcp"]
        require(len(bindings) == 1 and bindings[0]["HostIp"] == "127.0.0.1",
                "Redis must be exposed only on the loopback interface.")
        port = int(bindings[0]["HostPort"])
        evidence["isolation"].update(container_id=container_id, host="127.0.0.1", port=port)

        # Disable .env loading before importing any application modules.
        import dotenv
        dotenv.load_dotenv = lambda *args, **kwargs: False
        os.environ.update({
            "CREWAI_TRACING_ENABLED": "false", "OTEL_SDK_DISABLED": "true",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "DATABASE_URL": "sqlite:///:memory:", "APP_ENV": "development",
            "REDIS_URL": f"redis://127.0.0.1:{port}",
            "REDIS_HOST": "127.0.0.1", "REDIS_PORT": str(port),
        })
        # Some application imports declare a database engine. Keep it in
        # memory, with the queue-pool options expected by connection.py.
        import sqlalchemy
        from sqlalchemy.pool import QueuePool
        create_engine = sqlalchemy.create_engine

        def isolated_engine(url, **options):
            if str(url) == "sqlite:///:memory:" and "max_overflow" in options:
                options.setdefault("poolclass", QueuePool)
            return create_engine(url, **options)

        sqlalchemy.create_engine = isolated_engine
        import socket
        connect = socket.socket.connect
        socketpair = socket.socketpair
        internal_socketpair = local()

        def isolated_socketpair(*args, **options):
            # Windows implements its private asyncio wake-up pipe by opening
            # a socket pair on loopback. Permit only this owned internal call.
            internal_socketpair.active = True
            try:
                return socketpair(*args, **options)
            finally:
                internal_socketpair.active = False

        def isolated_connect(sock, address):
            if getattr(internal_socketpair, "active", False):
                return connect(sock, address)
            if not (isinstance(address, tuple) and address[0] in ("127.0.0.1", "localhost")
                    and address[1] == port):
                raise ConnectionRefusedError("Only this verifier's disposable Redis may be accessed.")
            return connect(sock, address)

        socket.socket.connect = isolated_connect
        socket.socketpair = isolated_socketpair
        evidence["isolation"].update(database="sqlite:///:memory:", network_restricted_to_disposable_redis=True)
        sys.path.insert(0, str(ROOT))
        import redis
        from backend.agents.context_memory import ContextMemory

        clients = [redis.Redis(host="127.0.0.1", port=port, decode_responses=True,
                               socket_connect_timeout=1, socket_timeout=1) for _ in range(2)]
        deadline = time.monotonic() + 20
        while True:
            try:
                if clients[0].ping():
                    break
            except redis.exceptions.ConnectionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        memories = []
        for client in clients:
            memory = object.__new__(ContextMemory)
            memory._redis = FastLockClient(client)
            memory._ttl = 120
            memories.append(memory)
        first, second = memories
        sid = int(owner[:12], 16)

        with first.session_lock(sid) as checkpoint:
            checkpoint()
            saved = first.save_session(
                sid, "clarifying", "hi",
                {"employment_status": "employed", "job_title": "Engineer"},
                [{"role": "assistant", "content": "आपके मुख्य कार्य क्या हैं?"}],
                clarification_target="job_duties", clarification_count=2,
                is_returning=True, correction_applied=True,
                corrected_fields={"job_title", "education_level"},
                correction_rejected_field="industry", correction_no_target=True,
                prefilled_fields=["employment_status", "job_title"],
            )
            restored = second.load_session(sid)
            require(restored == saved, "The second worker did not restore the complete serialized state.")
            require(0 < clients[1].ttl(first._key(sid)) <= 120, "Session TTL was not applied.")
        require(not clients[1].exists(f"lfs:session-lock:{sid}"), "Session lock was not released.")
        evidence["checks"]["shared_state_restoration"] = {
            "passed": True, "state": restored.state, "language": restored.language,
            "clarification_target": restored.clarification_target,
            "clarification_count": restored.clarification_count,
            "correction_fields": sorted(restored.corrected_fields),
            "history_restored": True, "returning_and_correction_flags_restored": True,
            "ttl_applied": True,
        }

        with second.session_lock(sid) as checkpoint:
            checkpoint()
            updated = restored.model_dump()
            updated["collected_fields"]["job_duties"] = "I design mechanical systems."
            updated["history"].append({"role": "user", "content": "I design mechanical systems."})
            second.save_session(
                sid, "collecting_info", updated["language"],
                updated["collected_fields"], updated["history"],
                clarification_target=None, clarification_count=0,
                is_returning=updated["is_returning"], correction_applied=False,
                corrected_fields={"job_duties"}, prefilled_fields=updated["prefilled_fields"],
            )
        latest = first.load_session(sid)
        require(latest.collected_fields["job_duties"] == "I design mechanical systems."
                and latest.clarification_target is None and latest.clarification_count == 0
                and latest.turn_count == 1 and latest.created_at == saved.created_at,
                "The first worker did not see the second worker's latest state.")
        evidence["checks"]["second_worker_update_visible"] = {"passed": True}

        contention = []
        with first.session_lock(sid) as checkpoint:
            checkpoint()
            require(clients[1].pttl(f"lfs:session-lock:{sid}") > 0, "The owned lock has no lease.")

            def contender():
                started = time.monotonic()
                try:
                    with second.session_lock(sid):
                        contention.append({"blocked": False})
                except redis.exceptions.LockError:
                    contention.append({"blocked": True, "wait_seconds": round(time.monotonic() - started, 3)})
                except Exception as exc:
                    contention.append({"error": str(exc)})

            thread = Thread(target=contender, daemon=True)
            thread.start()
            thread.join(timeout=3)
            require(not thread.is_alive() and contention and contention[0].get("blocked"),
                    "The second worker was not excluded while the first owned the lock.")
            checkpoint()
        with second.session_lock(sid) as checkpoint:
            checkpoint()
        require(not clients[0].exists(f"lfs:session-lock:{sid}"), "The second worker did not release its lock.")
        evidence["checks"]["mutual_exclusion_and_release"] = {"passed": True, **contention[0]}

        lost_sid = sid + 1
        lost_key = f"lfs:session-lock:{lost_sid}"
        rejected = False
        release_rejected = False
        try:
            with first.session_lock(lost_sid) as checkpoint:
                checkpoint()
                # Delete only the unique lock owned by this run's disposable Redis.
                require(inspect_owned(container_id, owner, container_id) is not None,
                        "Container identity could not be verified before the controlled lock deletion.")
                require(clients[1].delete(lost_key) == 1, "The controlled lock was not present.")
                try:
                    checkpoint()
                except redis.exceptions.LockError:
                    rejected = True
        except redis.exceptions.LockNotOwnedError:
            release_rejected = True
        require(rejected and release_rejected, "Lost ownership was not rejected at checkpoint and release.")
        with second.session_lock(lost_sid) as checkpoint:
            checkpoint()
        evidence["checks"]["lost_owner_checkpoint"] = {
            "passed": True, "checkpoint_rejected": rejected,
            "release_rejected": release_rejected, "second_worker_can_reacquire": True,
            "controlled_deletion_key": lost_key,
        }
        evidence["status"] = "passed"
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        for client in clients:
            client.close()
        try:
            record = inspect_owned(container_id or name, owner, expected_id=container_id)
            if record is not None:
                # Use the verified immutable ID, never a broad container selector.
                docker("rm", "--force", record["Id"])
                require(inspect_owned(record["Id"], owner, record["Id"]) is None,
                        "The disposable container remained after removal.")
                evidence["cleanup"] = {"removed": True, "identity_verified": True,
                                       "container_id": record["Id"]}
            else:
                evidence["cleanup"] = {"removed": True, "already_absent": True}
        except Exception as exc:
            evidence["cleanup"] = {"removed": False, "error": str(exc)}
            evidence["status"] = "failed"
        evidence["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "Documentation/CODE_FIXES_2026-10-05_REDIS_RESULTS.json")
    args = parser.parse_args()
    evidence = verify(args.output.resolve())
    print(json.dumps({"status": evidence["status"], "checks": len(evidence["checks"]),
                      "container_removed": evidence["cleanup"]["removed"],
                      "evidence": str(args.output.resolve())}))
    if evidence["status"] != "passed":
        print(json.dumps(evidence.get("error", evidence["cleanup"])))
    return 0 if evidence["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
