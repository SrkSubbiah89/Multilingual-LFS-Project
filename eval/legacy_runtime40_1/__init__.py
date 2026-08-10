"""
eval/legacy_runtime40_1/

Task 40.1: bounded, read-only archival provenance search for the historical
`qdrant-client` version needed to run LEGACY_SHA's byte-identical
`VectorStore.search()` (LEGACY_SHA = 824fcf235ae2f8787706cf479a07620519c914de,
reused unmodified from Task 39/Task 40). This package never starts, queries,
or connects to Qdrant, never creates a Python environment, never installs a
package, and never calls a classifier/LLM/Anthropic. It only classifies
archival evidence (local archives, Python environments, Docker image/
container metadata, GitHub) a caller supplies, following the same
never-guess, never-infer-from-non-contemporaneous-evidence discipline as
Task 40's `eval/legacy_runtime40/provenance.py`, extended with a Grade A/B/C/D
hierarchy so a preserved Docker image (not just a requirements-file pin) can
also establish a version.
"""

from eval.legacy_runtime40 import LEGACY_SHA  # noqa: F401 -- re-exported, single source of truth
