"""Start or stop isolated database services for a local browser smoke check.

Run ``py -3.11 scripts/manage_browser_smoke_services.py start`` and then
``... stop``. Requires locally installed postgres:15 and redis:7 Docker images.
This script does not read .env, pull images, mount user volumes, start the app,
or access existing databases. Connection strings stay in ignored private state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import quote
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
STATE_DIRECTORY = ROOT / "Software" / "browser_smoke"
STATE_PATH = STATE_DIRECTORY / "services.json"
OWNER_LABEL = "org.openai.codex.browser-smoke"
ROLE_LABEL = "org.openai.codex.browser-smoke-service"
IMAGES = {"postgres": "postgres:15", "redis": "redis:7"}
CONTAINER_PORTS = {"postgres": "5432/tcp", "redis": "6379/tcp"}


def docker(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["docker", *arguments], capture_output=True, text=True,
        encoding="utf-8", timeout=45,
    )
    if check and result.returncode:
        # Do not include command arguments: launch arguments contain dummy
        # database credentials which should still stay out of console logs.
        raise RuntimeError(f"Docker {arguments[0]} failed: {result.stderr.strip()}")
    return result


def write_state(state: dict) -> None:
    STATE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def inspect_owned(service: dict, owner: str, role: str) -> dict | None:
    identifier = service.get("id", "")
    if len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
        raise RuntimeError("Private service state lacks an immutable container ID.")
    result = docker("inspect", identifier, check=False)
    if result.returncode:
        if "no such object" in result.stderr.lower() or "no such container" in result.stderr.lower():
            return None
        raise RuntimeError("Could not inspect the task-owned container.")
    records = json.loads(result.stdout)
    if len(records) != 1:
        raise RuntimeError("Unexpected container inspection count.")
    record = records[0]
    labels = record["Config"].get("Labels", {}) or {}
    if not (
        record["Id"] == identifier
        and record["Name"].lstrip("/") == service["name"]
        and labels.get(OWNER_LABEL) == owner
        and labels.get(ROLE_LABEL) == role
        and record["Config"]["Image"] == IMAGES[role]
    ):
        raise RuntimeError("Container identity or ownership does not match private state; refusing cleanup.")
    bindings = record["NetworkSettings"].get("Ports", {}).get(CONTAINER_PORTS[role])
    if record["State"]["Running"]:
        if not bindings or len(bindings) != 1 or bindings[0]["HostIp"] != "127.0.0.1":
            raise RuntimeError("Task services must have one loopback-only published port.")
        if service.get("port") is not None and int(bindings[0]["HostPort"]) != service["port"]:
            raise RuntimeError("Task-owned service port differs from private state.")
    return record


def wait_ready(service: dict, role: str, timeout: float = 25) -> None:
    command = (
        ("pg_isready", "-h", "127.0.0.1", "-U", "browser_smoke_user", "-d", "browser_smoke_db")
        if role == "postgres" else ("redis-cli", "ping")
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = docker("exec", service["id"], *command, check=False)
        if result.returncode == 0 and (role != "redis" or result.stdout.strip() == "PONG"):
            return
        time.sleep(0.2)
    raise RuntimeError(f"The disposable {role} service did not become ready.")


def stop_services(state: dict) -> None:
    owner = state["owner"]
    # Verify every recorded identity before stopping any container. A corrupted
    # state file must never cause partial cleanup of unverified containers.
    owned = []
    for role, service in state.get("services", {}).items():
        if role not in IMAGES:
            raise RuntimeError("Private state contains an unknown service role.")
        record = inspect_owned(service, owner, role)
        if record is not None:
            owned.append(service)
    for service in reversed(owned):
        docker("stop", "--timeout", "3", service["id"])
        # --rm must have removed the container, including anonymous volumes.
        result = docker("inspect", service["id"], check=False)
        if result.returncode == 0:
            raise RuntimeError("A task-owned container remained after stopping.")
    state.update(status="stopped", stopped_at_utc=datetime.now(timezone.utc).isoformat())
    state.pop("DATABASE_URL", None)
    state.pop("REDIS_URL", None)
    write_state(state)


def start_services() -> dict:
    if STATE_PATH.exists():
        existing = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if existing.get("status") != "stopped":
            if existing.get("status") == "ready" and set(existing.get("services", {})) == set(IMAGES):
                for role, service in existing["services"].items():
                    record = inspect_owned(service, existing["owner"], role)
                    if record is None or not record["State"]["Running"]:
                        raise RuntimeError("Recorded smoke services are unavailable; run stop before start.")
                    wait_ready(service, role)
                return existing
            raise RuntimeError("Unfinished smoke-service state exists; run stop before start.")
    for image in IMAGES.values():
        docker("image", "inspect", image)
    owner = uuid4().hex
    password = f"smoke_only_{uuid4().hex}"
    state = {
        "owner": owner, "ownership_label": OWNER_LABEL, "status": "starting",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "services": {}, "existing_services_used": False, "images_pulled": False,
    }
    write_state(state)
    try:
        for role, image in IMAGES.items():
            name = f"lfs-browser-smoke-{role}-{owner[:12]}"
            arguments = [
                "run", "--detach", "--rm", "--pull=never", "--name", name,
                "--label", f"{OWNER_LABEL}={owner}", "--label", f"{ROLE_LABEL}={role}",
                "--publish", f"127.0.0.1::{CONTAINER_PORTS[role].split('/')[0]}",
            ]
            if role == "postgres":
                arguments.extend([
                    "--env", "POSTGRES_USER=browser_smoke_user",
                    "--env", f"POSTGRES_PASSWORD={password}",
                    "--env", "POSTGRES_DB=browser_smoke_db", image,
                ])
            else:
                arguments.extend([image, "redis-server", "--save", "", "--appendonly", "no"])
            identifier = docker(*arguments).stdout.strip()
            service = {"id": identifier, "name": name, "image": image}
            state["services"][role] = service
            write_state(state)
            record = inspect_owned(service, owner, role)
            if record is None or not record["State"]["Running"]:
                raise RuntimeError("Disposable service exited during launch.")
            service["port"] = int(record["NetworkSettings"]["Ports"][CONTAINER_PORTS[role]][0]["HostPort"])
            service["host"] = "127.0.0.1"
            write_state(state)
            wait_ready(service, role)
        state.update(
            status="ready",
            DATABASE_URL=(f"postgresql://browser_smoke_user:{quote(password, safe='')}"
                          f"@127.0.0.1:{state['services']['postgres']['port']}/browser_smoke_db"),
            REDIS_URL=f"redis://127.0.0.1:{state['services']['redis']['port']}/0",
        )
        write_state(state)
        return state
    except BaseException:
        stop_services(state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop"))
    args = parser.parse_args()
    if args.action == "start":
        state = start_services()
        print(json.dumps({
            "status": state["status"], "state_file": str(STATE_PATH),
            "postgres_port": state["services"]["postgres"]["port"],
            "redis_port": state["services"]["redis"]["port"],
        }, indent=2))
    else:
        if not STATE_PATH.exists():
            print("No browser-smoke service state exists; no containers changed.")
            return
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        stop_services(state)
        print("Task-owned browser-smoke containers are stopped and removed.")


if __name__ == "__main__":
    main()
