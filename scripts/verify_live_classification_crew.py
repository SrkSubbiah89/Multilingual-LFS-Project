"""Probe real local CrewAI collaboration using synthetic, read-only inputs.

The occupation classifier bridges to the running backend's semantic-only
debug endpoint so this probe does not load a second embedding model. Industry
and education use the actual classifiers in this process. This is execution
evidence, not a classification accuracy benchmark or a stored survey test.
"""

from __future__ import annotations

import argparse
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "ollama/qwen2.5:3b"
ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}
ALLOWED_PORTS = {8000, 11434}


def isolate_process(model=DEFAULT_MODEL, request_timeout=30):
    """Never inherit real database/provider credentials or load the local .env."""
    sys.path.insert(0, str(ROOT))
    os.environ.update({
        "DATABASE_URL": "sqlite:///:memory:",
        "JWT_SECRET": "live-crew-probe-only-with-at-least-thirty-two-characters",
        "APP_ENV": "development", "LFS_FAST_MODE": "false",
        "REDIS_URL": "redis://127.0.0.1:1", "REDIS_HOST": "127.0.0.1", "REDIS_PORT": "1",
        "QDRANT_HOST": "127.0.0.1", "QDRANT_PORT": "1", "QDRANT_URL": "", "QDRANT_API_KEY": "",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434", "OLLAMA_MODEL": model.split("/", 1)[1],
        "LLM_FALLBACK_EXCLUDE": "anthropic,gemini,groq,openrouter",
        "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "", "GROQ_API_KEY": "",
        "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "OPENROUTER_API_KEY": "",
        "SENDGRID_API_KEY": "", "TWILIO_ACCOUNT_SID": "", "TWILIO_AUTH_TOKEN": "",
        "HITL_REVIEWER_USER_IDS": "", "CREWAI_TRACING_ENABLED": "false",
        "OTEL_SDK_DISABLED": "true", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    })
    # A proxy inherited from a shell would defeat the loopback-only target.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    import dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False

    # Connection construction imports the application's model definitions but
    # no engine is used. Its PostgreSQL pool options need adapting for SQLite.
    import sqlalchemy
    from sqlalchemy.pool import QueuePool
    original_engine = sqlalchemy.create_engine

    def isolated_engine(url, **kwargs):
        if str(url) == "sqlite:///:memory:" and "max_overflow" in kwargs:
            kwargs.setdefault("poolclass", QueuePool)
        return original_engine(url, **kwargs)

    sqlalchemy.create_engine = isolated_engine
    blocked = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_socketpair = socket.socketpair
    internal_socketpair = ContextVar("live_probe_internal_socketpair", default=False)

    def local_socketpair(*args, **kwargs):
        # Windows implements asyncio's internal socketpair with an ephemeral
        # loopback connection. This is interpreter IPC, not a service call.
        token = internal_socketpair.set(True)
        try:
            return original_socketpair(*args, **kwargs)
        finally:
            internal_socketpair.reset(token)

    def check_target(address):
        if internal_socketpair.get() and isinstance(address, tuple) and address[0] in ALLOWED_HOSTS:
            return
        if not isinstance(address, tuple) or address[0] not in ALLOWED_HOSTS or address[1] not in ALLOWED_PORTS:
            blocked.append({"host": str(address[0]) if isinstance(address, tuple) else "non-IP socket",
                            "port": address[1] if isinstance(address, tuple) else None})
            raise OSError("Live crew probe permits only loopback backend/Ollama connections")

    def isolated_connect(sock, address):
        check_target(address)
        return original_connect(sock, address)

    def isolated_connect_ex(sock, address):
        check_target(address)
        return original_connect_ex(sock, address)

    socket.socket.connect = isolated_connect
    socket.socket.connect_ex = isolated_connect_ex
    socket.socketpair = local_socketpair
    original_urlopen = urllib.request.urlopen

    def bounded_urlopen(url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, **kwargs):
        bounded_timeout = request_timeout if timeout is None or timeout is socket._GLOBAL_DEFAULT_TIMEOUT else min(timeout, request_timeout)
        return original_urlopen(url, data=data, timeout=bounded_timeout, **kwargs)

    # The native CrewAI adapter sets its own timeout. Enforce a smaller CLI
    # limit at its actual HTTP boundary too, rather than only on the LLM object.
    urllib.request.urlopen = bounded_urlopen
    return blocked


class ReadOnlyOccupationBridge:
    """Use only fields actually returned by the semantic debug endpoint."""

    def __init__(self, timeout=15, on_progress=None):
        self.calls = []
        self.timeout = timeout
        self.on_progress = on_progress

    def classify(self, text, *, context="", language="en", use_llm=True):
        print("Occupation tool: requesting existing backend semantic retrieval", flush=True)
        started = time.perf_counter()
        url = "http://127.0.0.1:8000/debug/isco/" + urllib.parse.quote(text, safe="")
        with urllib.request.urlopen(url, timeout=self.timeout) as response:
            data = json.load(response)
        if "error" in data or not re.fullmatch(r"\d{4}", str(data.get("code", ""))):
            raise RuntimeError("Existing backend returned no usable occupation code")
        self.calls.append({"input": text, "endpoint": "/debug/isco/{job_title}",
                           "response": data, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                           "context_forwarded": False, "language_forwarded": False,
                           "llm_reranking": False})
        if self.on_progress:
            self.on_progress()
        return SimpleNamespace(
            primary=SimpleNamespace(code=data["code"], title_en=data["title"], title_ar="",
                                    confidence=data["conf"]),
            method="read_only_debug_isco_semantic_bridge", hitl_required=None,
            hierarchy_path=None,
        )


class ObservedClassifier:
    def __init__(self, classifier, dimension, on_progress=None):
        self.classifier = classifier
        self.dimension = dimension
        self.calls = []
        self.on_progress = on_progress

    def classify(self, text, **kwargs):
        print(f"{self.dimension.upper()} tool: invoking actual classifier", flush=True)
        started = time.perf_counter()
        result = self.classifier.classify(text, **kwargs)
        self.calls.append({"input": text, "method": result.method,
                           "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})
        if self.on_progress:
            self.on_progress()
        return result


def dump_value(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, SimpleNamespace):
        return {key: dump_value(item) for key, item in vars(value).items()}
    if is_dataclass(value):
        return asdict(value)
    return value


def write_evidence(path, evidence):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-date", default="2026-10-06")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="An already-pulled ollama/<model> only")
    parser.add_argument("--inference-timeout", type=float, default=30, help="Local LLM request timeout, 2--60 seconds")
    parser.add_argument("--total-timeout", type=float, default=180, help="Hard subprocess deadline, 15--240 seconds")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path,
                        help="Evidence output; existing runs are preserved automatically")
    args = parser.parse_args()
    if not args.model.startswith("ollama/") or not args.model.split("/", 1)[1].strip():
        parser.error("Only an already-pulled local Ollama model can be used")
    if not 2 <= args.inference_timeout <= 60 or not 15 <= args.total_timeout <= 240:
        parser.error("Inference timeout must be 2--60 seconds and total deadline 15--240 seconds")
    try:
        datetime.strptime(args.verification_date, "%Y-%m-%d")
    except ValueError:
        parser.error("Verification date must use YYYY-MM-DD")
    args.output = args.output or ROOT / f"Documentation/TITLE_ALIGNMENT_{args.verification_date}_LIVE_CREW_RESULTS.json"
    if not args.worker and args.output.exists():
        suffix = datetime.now(timezone.utc).strftime("%H%M%S%f")
        args.output = args.output.with_name(f"{args.output.stem}_{suffix}{args.output.suffix}")
    return args


def run_probe(args):
    blocked = isolate_process(args.model, args.inference_timeout)
    runtime_observations = []

    class ProbeDiagnostics(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            observation = {"logger": record.name, "level": record.levelname, "message": message[:500]}
            if record.levelno >= logging.WARNING and observation not in runtime_observations:
                runtime_observations.append(observation)

    logging.getLogger("LiteLLM").addHandler(ProbeDiagnostics())
    evidence = {
        "verification_date": args.verification_date, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_pin": args.model, "success": False, "status": "running",
        "limits": {"inference_timeout_seconds": args.inference_timeout,
                   "hard_process_deadline_seconds": args.total_timeout},
        "scope": {
            "inputs": "one synthetic English employment profile",
            "crewai": "real sequential specialist agents and evidence auditor; no mocked CrewAI runtime",
            "occupation": "real running backend semantic retrieval, bridged through read-only /debug/isco",
            "industry_and_education": "actual project keyword/rule classifiers, with local-only LLM configuration",
            "occupation_bridge_limits": ["no context or language forwarded", "no occupation LLM reranking",
                                          "debug endpoint does not expose full classifier trace, hierarchy or HITL decision"],
            "respondent_database_writes": False, "otp_or_external_messages": False,
            "accuracy_evaluation": False, "survey_route_persistence_test": False,
        },
        "synthetic_profile": {"job_title": "software developer",
                              "industry_text": "computer programming and software development company",
                              "education_text": "bachelor degree in computer science", "language": "en"},
    }
    started = time.perf_counter()
    occupation = industry = education = None

    def persist_progress():
        evidence["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        evidence["observed_classifier_calls"] = {
            "isco": occupation.calls if occupation else [],
            "isic": industry.calls if industry else [],
            "isced": education.calls if education else [],
        }
        evidence["blocked_network_targets"] = list(blocked)
        evidence["runtime_logging_observations"] = list(runtime_observations)
        write_evidence(args.output, evidence)

    persist_progress()
    try:
        from backend.llm import get_llm_strict
        from importlib.metadata import version
        from backend.agents.isic_classifier import ISICClassifier
        from backend.agents.isced_classifier import ISCEDClassifier
        from backend.agents.survey_classification_crew import SurveyClassificationCrew
        evidence["package_versions"] = {name: version(name) for name in ("crewai", "litellm")}
        llm = get_llm_strict(args.model, temperature=0.0)
        llm.timeout = args.inference_timeout
        occupation = ReadOnlyOccupationBridge(timeout=min(args.inference_timeout, 15), on_progress=persist_progress)
        industry_classifier = ISICClassifier(reranker_model=args.model)
        industry_classifier._llm.timeout = args.inference_timeout
        industry = ObservedClassifier(industry_classifier, "isic", on_progress=persist_progress)
        education = ObservedClassifier(ISCEDClassifier(), "isced", on_progress=persist_progress)
        crew = SurveyClassificationCrew(occupation, industry, education, llm=llm)
        print("Starting real local CrewAI specialist and auditor probe", flush=True)
        result = crew.classify(**evidence["synthetic_profile"], use_llm=False)
        evidence["execution"] = result.execution
        evidence["classification_results"] = {
            "isco": dump_value(result.isco), "isic": dump_value(result.isic), "isced": dump_value(result.isced),
        }
        evidence["observed_classifier_calls"] = {
            "isco": occupation.calls, "isic": industry.calls, "isced": education.calls,
        }
        evidence["success"] = bool(result.execution["cooperation_verified"])
    except Exception as exc:
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
    evidence["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    evidence["blocked_network_targets"] = blocked
    evidence["runtime_logging_observations"] = runtime_observations
    evidence["status"] = "completed"
    write_evidence(args.output, evidence)
    print(json.dumps({"success": evidence["success"], "elapsed_seconds": evidence["elapsed_seconds"],
                      "output": str(args.output)}, indent=2), flush=True)
    return 0 if evidence["success"] else 1


def main():
    args = parse_args()
    if args.worker:
        return run_probe(args)
    # Terminate only this probe's owned Python child when the entire operation
    # exceeds its deadline. Model/provider retries cannot extend this bound.
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--model", args.model, "--inference-timeout", str(args.inference_timeout),
               "--total-timeout", str(args.total_timeout), "--verification-date", args.verification_date,
               "--output", str(args.output)]
    started = time.perf_counter()
    try:
        child = subprocess.run(command, stdin=subprocess.DEVNULL, timeout=args.total_timeout,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return child.returncode
    except subprocess.TimeoutExpired:
        evidence = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
        evidence.update({"verification_date": args.verification_date, "model_pin": args.model,
                         "success": False, "status": "hard_deadline_exceeded",
                         "owned_probe_process_terminated": True,
                         "elapsed_seconds": round(time.perf_counter() - started, 3)})
        write_evidence(args.output, evidence)
        print(json.dumps({"success": False, "status": evidence["status"], "output": str(args.output)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
