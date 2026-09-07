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

__all__ = ["OccupationMatch", "VectorStore", "get_vector_store"]
