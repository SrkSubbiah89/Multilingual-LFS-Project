"""Verify Alembic against a new disposable PostgreSQL container.

Requires Docker and an already downloaded postgres:15 image. Never loads .env,
uses no user volumes, and removes only the container created by this invocation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def verify():
    name = f"lfs-review-db-{uuid4().hex[:12]}"
    label = f"lfs.review={name}"
    password = "review_only%password"
    container_id = docker(
        "run", "--detach", "--rm", "--pull=never", "--name", name, "--label", label,
        "--publish", "127.0.0.1::5432", "--env", "POSTGRES_USER=review_user",
        "--env", f"POSTGRES_PASSWORD={password}", "--env", "POSTGRES_DB=review_db",
        "postgres:15",
    )
    try:
        container = json.loads(docker("inspect", container_id))[0]
        binding = container["NetworkSettings"]["Ports"]["5432/tcp"][0]
        assert binding["HostIp"] == "127.0.0.1"
        for _ in range(60):
            ready = subprocess.run(
                ["docker", "exec", container_id, "pg_isready", "-U", "review_user", "-d", "review_db"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Disposable PostgreSQL did not become ready.")

        sys.path.insert(0, str(ROOT))
        os.environ["DATABASE_URL"] = (
            f"postgresql://review_user:{quote(password, safe='')}@127.0.0.1:{binding['HostPort']}/review_db"
        )
        import dotenv
        dotenv.load_dotenv = lambda *args, **kwargs: False
        from alembic import command
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        from sqlalchemy import inspect, text
        from backend.database.connection import engine, Base
        import backend.database.models  # Register every model before comparison.

        try:
            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "backend/database/migrations"))
            command.upgrade(config, "head")
            command.upgrade(config, "head")
            inspector = inspect(engine)
            tables = set(inspector.get_table_names()) - {"alembic_version"}
            assert tables == set(Base.metadata.tables), (tables, set(Base.metadata.tables))
            for table_name, table in Base.metadata.tables.items():
                columns = {column["name"] for column in inspector.get_columns(table_name)}
                assert columns == set(table.columns.keys()), table_name
            with engine.connect() as connection:
                revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            assert revision == ScriptDirectory.from_config(config).get_current_head()
            result = {
                "database": "disposable PostgreSQL 15", "tables": len(tables), "revision": revision,
                "all_model_columns_match": True, "second_upgrade_idempotent": True,
                "percent_encoded_password_verified": True, "user_database_accessed": False,
            }
        finally:
            engine.dispose()
    finally:
        # Validate identity before cleanup; never select containers by a broad pattern.
        owned = json.loads(docker("inspect", container_id))[0]
        assert owned["Name"].lstrip("/") == name
        assert owned["Config"]["Labels"]["lfs.review"] == name
        docker("stop", "--timeout", "3", container_id)
    result["container_removed"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON evidence file.")
    args = parser.parse_args()
    result = verify()
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
