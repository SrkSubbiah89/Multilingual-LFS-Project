"""
eval/legacy824/historical_loader.py

Task 39: loads backend.agents.isco_classifier (and its two direct
dependencies, backend.rag.vector_store and backend.llm.llm_client)
directly from a detached git worktree checked out at LEGACY_SHA
(824fcf235ae2f8787706cf479a07620519c914de), WITHOUT ever adding the
worktree to sys.path permanently and WITHOUT modifying any file in it.

Why this exists
----------------
The historical source files' own `from backend.llm import ...` /
`from backend.rag import ...` statements are literal and (per Task 39's
explicit instruction) unmodifiable. To make those statements resolve to
the WORKTREE's copies -- not this repository's own current backend.*
packages, which may already be imported elsewhere in the same process
under the same top-level name -- this loader temporarily binds
`backend`, `backend.rag`, `backend.llm` in sys.modules to freshly-loaded
worktree modules just long enough to execute isco_classifier.py's own
import statements, then restores whatever was bound before (usually
nothing, since this repository's tests/tools never import bare
`backend` as a standalone top-level package -- they use
`backend.agents...`/`backend.rag...` etc., which is a different
sys.modules key). The loaded classifier module is registered only under
a private name (`_legacy824_isco_classifier`), never under
`backend.agents.isco_classifier`, so it can never be mistaken for or
interfere with this repository's own current classifier.

This is environment/loading isolation only. It does not alter a single
byte of the historical source.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import NamedTuple


class LoadedHistoricalModules(NamedTuple):
    isco_classifier: types.ModuleType
    vector_store: types.ModuleType
    llm_client: types.ModuleType


def _load_module_from_path(name: str, path: Path) -> types.ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"historical source file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {name} at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_historical_isco_classifier(worktree_path: Path) -> LoadedHistoricalModules:
    """
    Loads the historical ISCOClassifier (and its two direct
    dependencies) verbatim from *worktree_path*, which must be a
    detached checkout of LEGACY_SHA. Returns the three loaded modules
    so a caller (a hermetic test, or the real development-run driver)
    can monkeypatch `vector_store.get_vector_store` / `llm_client.
    get_llm` / the classifier module's own `Crew` name before ever
    instantiating `isco_classifier.ISCOClassifier` -- no network,
    Qdrant, or model call happens merely by loading these modules.
    """
    worktree_path = Path(worktree_path)

    vector_store_mod = _load_module_from_path(
        "_legacy824_vector_store", worktree_path / "backend" / "rag" / "vector_store.py",
    )
    llm_client_mod = _load_module_from_path(
        "_legacy824_llm_client", worktree_path / "backend" / "llm" / "llm_client.py",
    )

    # Minimal stand-ins for backend/rag/__init__.py and backend/llm/__init__.py
    # at LEGACY_SHA, exposing exactly the names those files exposed
    # (confirmed by direct source read -- see the Task 39 final report).
    rag_pkg = types.ModuleType("backend.rag")
    rag_pkg.OccupationMatch = vector_store_mod.OccupationMatch
    rag_pkg.VectorStore = vector_store_mod.VectorStore
    rag_pkg.get_vector_store = vector_store_mod.get_vector_store

    llm_pkg = types.ModuleType("backend.llm")
    llm_pkg.TaskType = llm_client_mod.TaskType
    llm_pkg.get_llm = llm_client_mod.get_llm

    backend_pkg = types.ModuleType("backend")

    saved = {k: sys.modules.get(k) for k in ("backend", "backend.rag", "backend.llm")}
    sys.modules["backend"] = backend_pkg
    sys.modules["backend.rag"] = rag_pkg
    sys.modules["backend.llm"] = llm_pkg
    try:
        isco_mod = _load_module_from_path(
            "_legacy824_isco_classifier",
            worktree_path / "backend" / "agents" / "isco_classifier.py",
        )
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return LoadedHistoricalModules(isco_mod, vector_store_mod, llm_client_mod)
