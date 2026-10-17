"""Decisions for the Stage 5C loopback publication check. No sockets are opened."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stage5c_ci_server", ROOT / "stage5c_ci_server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def main():
    proc = "\n".join([
        "sl local_address rem_address st",
        "0: 0100007F:1538 00000000:0000 0A",
        "1: 0100007F:18EB 00000000:0000 0A",
        "2: 00000000:0050 00000000:0000 0A",
    ])
    rows = server.listening_rows(proc)
    assert server.nonloopback_ports(rows, []) == []
    exposed = server.listening_rows("sl local_address rem_address st\n0: 00000000:1538 00000000:0000 0A\n")
    assert server.nonloopback_ports(exposed, []) == ["5432"]
    ipv6 = server.listening_rows(
        "sl local_address rem_address st\n0: 00000000000000000000000000000000:18EB 00000000:0000 0A\n"
    )
    assert "6379" in server.nonloopback_ports([], ipv6)
    assert server.bindings_are_loopback([
        ("5432", "127.0.0.1", "5432"),
        ("6379", "127.0.0.1", "6379"),
    ])
    assert not server.bindings_are_loopback([
        ("5432", "0.0.0.0", "5432"),
        ("6379", "127.0.0.1", "6379"),
    ])
    assert not server.bindings_are_loopback([
        ("5432", "", "5432"),
        ("6379", "127.0.0.1", "6379"),
    ])
    assert not server.bindings_are_loopback([("5432", "127.0.0.1", "5432")])
    print("LOOPBACK_DECISION_PASS")


if __name__ == "__main__":
    main()
