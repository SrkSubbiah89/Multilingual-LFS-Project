import os

# Real, confirmed finding (2026-08-28): this project never uses TensorFlow
# directly -- it isn't even in requirements.txt/requirements-dev.txt -- but
# sentence-transformers' underlying `transformers` dependency auto-imports it
# anyway when both backends are installed, and that import alone costs a real
# chunk of memory (verified directly: `sentence_transformers` imports cleanly
# in ~6s with `tensorflow` never appearing in `sys.modules` once USE_TF=0 is
# set). This mattered in practice: two consecutive live eval runs on this
# session's 8GB dev machine segfaulted during embedding-model load with only
# ~1.6GB free RAM, crashing immediately after TensorFlow's own import
# warnings and before any of this project's own code ran. `setdefault` (not
# a blind assignment) so an operator/CI environment that deliberately wants
# TensorFlow available is never silently overridden. This must be set here,
# in this package's own __init__, because `backend.rag.vector_store` (below)
# is this package's first sentence-transformers import, and Python always
# executes a package's __init__.py before any of its submodules -- setting
# it in any individual submodule would be too late for every OTHER submodule
# that happens to import first.
os.environ.setdefault("USE_TF", "0")

from .vector_store import OccupationMatch, VectorStore, get_vector_store

__all__ = ["OccupationMatch", "VectorStore", "get_vector_store", "make_qdrant_client"]


def make_qdrant_client(host: str | None = None, port: int | None = None, client_cls=None, **kwargs):
    """Single shared factory for every QdrantClient in this project.

    Added 2026-09-04 for the cloud-hosting migration: every one of the 9
    call sites that previously built `QdrantClient(host=..., port=...)`
    directly (backend/rag/vector_store.py, hierarchical_store.py,
    standard_hierarchical_store.py, load_full_isco.py, and 5
    build_official_isco08_collections*.py / build_standard_hierarchical_
    collections.py scripts) had no way to authenticate against a hosted
    Qdrant instance at all -- bare host/port implies plain, unauthenticated
    HTTP, and several of those scripts' own docstrings explicitly said so
    ("Local-only Qdrant target... no remote URL/token option exists in this
    script"). Rather than patch each of the 9 call sites with its own
    ad-hoc branching, they now all call this one shared function.

    If QDRANT_URL is set, connects to it with QDRANT_API_KEY (the Qdrant
    Cloud path) -- QdrantClient's own `url=` parameter auto-enables TLS
    when the URL's scheme is https://, so no separate https=True is needed.
    Otherwise, preserves every caller's exact prior behavior: host/port
    (explicit args, or QDRANT_HOST/QDRANT_PORT, or "localhost"/6333) with no
    auth, unchanged. **kwargs passes through additional constructor args
    individual callers already relied on (timeout=, check_compatibility=,
    etc.) unchanged in both branches.

    `client_cls` (added the same day, fixing a real regression this
    function's first version caused): several existing tests
    (test_vector_store.py, test_hierarchical_store.py,
    test_official_isco08_profiles.py, test_qdrant_retry_resilience.py,
    test_stage_budget_enforcement.py, test_official_profile_label_en_fix.py,
    test_flat_query_telemetry.py) monkeypatch the *caller module's own*
    `QdrantClient` name (e.g. `backend.rag.vector_store.QdrantClient`) and
    expect the store's constructor to use that patched class -- they never
    patch this function directly. When this function did its own internal
    `from qdrant_client import QdrantClient`, those patches became no-ops
    and every such test tried to hit a real, non-existent local Qdrant.
    Callers that need to stay patchable must pass their own module-level
    `QdrantClient` name (not a hardcoded import) as `client_cls`; callers
    with no such test dependency (the build scripts, which patch entire
    factory functions instead) can omit it and get the real class via a
    lazy import here, same as before.
    """
    if client_cls is None:
        from qdrant_client import QdrantClient  # lazy: matches this package's own lazy-import convention

        client_cls = QdrantClient

    url = os.getenv("QDRANT_URL")
    if url:
        return client_cls(url=url, api_key=os.getenv("QDRANT_API_KEY"), **kwargs)

    _host = host or os.getenv("QDRANT_HOST", "localhost")
    _port = int(port or os.getenv("QDRANT_PORT", 6333))
    return client_cls(host=_host, port=_port, **kwargs)
