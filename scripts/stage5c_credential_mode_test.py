"""Prove Stage 5C credential files are created at mode 0600."""
import importlib.util
import os
import stat
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stage5c_ci_server", ROOT / "stage5c_ci_server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def main():
    if os.name != "posix":
        print("CREDENTIAL_MODE_NOT_POSIX")
        raise SystemExit(2)
    directory = Path(tempfile.mkdtemp(prefix="stage5c-cred-"))
    target = directory / "ci.env"
    previous = os.umask(0)
    try:
        server.write_restricted_text(target, "EXAMPLE=value\n")
    finally:
        os.umask(previous)
    mode = stat.S_IMODE(target.stat().st_mode)
    if mode != 0o600:
        print("CREDENTIAL_MODE_FAIL")
        raise SystemExit(1)
    try:
        server.write_restricted_text(target, "AGAIN=value\n")
    except FileExistsError:
        print("CREDENTIAL_MODE_PASS")
        return
    print("CREDENTIAL_MODE_FAIL exclusive create was not enforced")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
