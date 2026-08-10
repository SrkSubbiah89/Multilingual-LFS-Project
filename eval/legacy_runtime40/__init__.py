"""
eval/legacy_runtime40/

Task 40: Historical Qdrant Runtime Reconstruction for Literal Legacy
ISCO Reproduction. This package never modifies, shims, wraps, or
monkeypatches the historical VectorStore.search() method or any legacy
classifier logic (LEGACY_SHA = 824fcf235ae2f8787706cf479a07620519c914de,
reused unmodified from Task 39's detached worktree). It only determines
-- from real, dated repository-history evidence -- whether a defensible
historical qdrant-client version can be established at all.
"""

LEGACY_SHA = "824fcf235ae2f8787706cf479a07620519c914de"
