"""
backend/rag/standard_hierarchical_store.py

Conference I Reviewer #2 response, Task 05. Standard-specific hierarchical
retrieval stores for ISIC Rev.4 and ISCED-F 2013, built on the SAME generic
``backend.rag.hierarchy_engine.HierarchyBeamSearchEngine`` ISCO already
uses -- the beam-search algorithm itself is reused verbatim, never copied
or forked (see hierarchy_engine.py's own module docstring). This module
adds no changes to ``backend/rag/hierarchical_store.py`` or ISCO's
production behaviour whatsoever.

Embedding convention (matches ``backend/rag/hierarchical_store.py``
exactly): ``intfloat/multilingual-e5-small`` (384-dim), ``"query: "``
prefix at search time (this module), ``"passage: "`` prefix at index time
(``backend/rag/build_standard_hierarchical_collections.py``).

Collection names (stable, standard-specific, never overlapping ISCO's own):

    isic_rev4_sections / isic_rev4_divisions / isic_rev4_groups / isic_rev4_classes
    iscedf2013_broad_fields / iscedf2013_narrow_fields / iscedf2013_detailed_fields

Readiness contract: a ``StandardHierarchicalStore`` checks, ONCE at
construction, whether every one of its required collections exists in
Qdrant. ``search()`` NEVER silently reports a completed hierarchical result
when that check failed -- it returns an explicit ``ready=False`` result
with a non-empty ``unavailable_reason`` instead of attempting (and
therefore never actually running) the engine search. A second, distinct
"ready but this query returned nothing" state is also explicit (``ready=
True`` with a non-empty ``unavailable_reason``) -- callers (ISICClassifier/
ISCEDClassifier) must branch on ``unavailable_reason`` being non-empty to
decide whether to fall back, never on ``code`` being non-empty alone.

Operational resilience (Task 05.1): the readiness check itself is an
external call (``QdrantClient.get_collections()``) and can fail if Qdrant
is unreachable or erroring, not just report collections as missing. That
failure is caught at construction time and folded into the SAME
``ready=False`` / non-empty ``unavailable_reason`` contract above -- a
caller cannot tell "collections missing" apart from "Qdrant unreachable"
without reading the reason text, and does not need to; both mean "do not
attempt a hierarchical query, use the explicit fallback path instead".
When readiness fails this way, the embedding model is never loaded. Two
more operational boundaries -- embedding the query text, and running the
engine search itself -- are also caught around this same external-call
boundary in ``search()``: on failure, an explicit no-code
``StandardHierarchyResult`` with a non-empty ``unavailable_reason`` is
returned rather than letting the exception propagate into the classifier.
No score, candidate, or hierarchy path is ever fabricated on any of these
paths. Exception handling here is deliberately narrow -- it wraps ONLY
these three external-call boundaries (readiness check, embedding,
engine search), never any other classifier logic.

Model initialization resilience (Task 05.2): ``SentenceTransformer(MODEL_
NAME)`` construction is LAZY -- it happens inside ``_embed_query()``, on
first use, not in ``__init__``. This means it is automatically covered by
the same protected try/except ``search()`` already wraps around
``_embed_query()`` (added in Task 05.1) -- a model-construction failure
is indistinguishable, from the caller's perspective, from a query-encoding
failure: both surface as a ready-store, no-code result with a non-empty,
embedding-related ``unavailable_reason``, never a raised exception. It
also means a store whose readiness check failed or whose collections are
missing never attempts model construction at all, since ``search()``
returns before ``_embed_query()`` is ever called. A dependency-injected
``embedder=`` is stored as-is and never replaced by a real model.

Dependency injection for hermetic tests: both ``client`` and ``embedder``
are constructor parameters. Passing either causes
``get_isic_hierarchical_store()``/``get_iscedf_hierarchical_store()`` to
construct and return a FRESH, non-cached instance (never the module-level
singleton) -- tests never share state with, or accidentally warm, the
production singleton. Omitting both (the normal production call pattern)
uses the same lazily-constructed, cached-singleton pattern
``backend.rag.hierarchical_store.get_hierarchical_store()`` already
established.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from backend.rag.hierarchy_engine import EngineCandidate, HierarchyBeamSearchEngine, StageConfig

_logger = logging.getLogger(__name__)

MODEL_NAME = "intfloat/multilingual-e5-small"
VECTOR_DIM = 384
HITL_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Collection names -- stable, unambiguous, standard-specific
# ---------------------------------------------------------------------------

ISIC_COLLECTIONS = {
    "sections": "isic_rev4_sections",
    "divisions": "isic_rev4_divisions",
    "groups": "isic_rev4_groups",
    "classes": "isic_rev4_classes",
}

ISCEDF_COLLECTIONS = {
    "broad_fields": "iscedf2013_broad_fields",
    "narrow_fields": "iscedf2013_narrow_fields",
    "detailed_fields": "iscedf2013_detailed_fields",
}

# ---------------------------------------------------------------------------
# Stage weights -- explicit, documented, sum to 1.0. These are engineering
# defaults (roughly following ISCO's own "later stages weighted higher"
# shape), NOT tuned by any measured evaluation -- no ISIC/ISCED-F accuracy
# evidence exists yet (see Documentation/Conference_I_Reviewer_2/
# ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md). Do not describe
# these as tuned/optimal in any manuscript-facing text.
# ---------------------------------------------------------------------------

ISIC_STAGE_WEIGHTS = (0.10, 0.20, 0.25, 0.45)
ISCEDF_STAGE_WEIGHTS = (0.20, 0.30, 0.50)

assert abs(sum(ISIC_STAGE_WEIGHTS) - 1.0) < 1e-9, "ISIC_STAGE_WEIGHTS must sum to 1.0"
assert abs(sum(ISCEDF_STAGE_WEIGHTS) - 1.0) < 1e-9, "ISCEDF_STAGE_WEIGHTS must sum to 1.0"


def isic_stages() -> list[StageConfig]:
    names = ("sections", "divisions", "groups", "classes")
    return [
        StageConfig(name=n, collection=ISIC_COLLECTIONS[n], weight=w)
        for n, w in zip(names, ISIC_STAGE_WEIGHTS)
    ]


def iscedf_stages() -> list[StageConfig]:
    names = ("broad_fields", "narrow_fields", "detailed_fields")
    return [
        StageConfig(name=n, collection=ISCEDF_COLLECTIONS[n], weight=w)
        for n, w in zip(names, ISCEDF_STAGE_WEIGHTS)
    ]


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class StandardHierarchyResult:
    code: str
    label_en: str
    label_ar: str
    confidence: float
    stage_confidences: dict
    hierarchy_path: list = field(default_factory=list)
    top_candidates: list = field(default_factory=list)  # list[EngineCandidate]
    hitl_required: bool = True
    ready: bool = False
    unavailable_reason: str = ""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class StandardHierarchicalStore:
    """Hierarchical retrieval store for ONE classification standard (ISIC
    Rev.4 or ISCED-F 2013), parameterized by its stage configuration. Both
    factory functions below construct one of these each -- this class is
    not a base class either ISCO's or any future standard's store must
    inherit from, just a reusable parameterization to avoid duplicating
    the readiness-check/embed/translate logic twice."""

    def __init__(
        self,
        standard: str,
        stages: list[StageConfig],
        hitl_threshold: float = HITL_THRESHOLD,
        client: Optional[QdrantClient] = None,
        embedder=None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        self.standard = standard
        self.stages = stages
        self._required_collections = [s.collection for s in stages]

        if client is not None:
            self._client = client
        else:
            _host = host or os.getenv("QDRANT_HOST", "localhost")
            _port = int(port or os.getenv("QDRANT_PORT", 6333))
            self._client = QdrantClient(host=_host, port=_port)

        # Upfront, one-time collection-readiness check -- see module
        # docstring's "Readiness contract" and "Operational resilience".
        # This is an external call and can fail outright (Qdrant
        # unreachable/erroring), not just report collections as missing --
        # both outcomes fold into the same ready=False / unavailable_reason
        # contract, since neither state permits a hierarchical query.
        self._missing_collections: list[str] = []
        self._unavailable_reason: str = ""
        try:
            existing = {c.name for c in self._client.get_collections().collections}
            self._missing_collections = [c for c in self._required_collections if c not in existing]
            self.ready = not self._missing_collections
            if not self.ready:
                self._unavailable_reason = (
                    f"Required Qdrant collection(s) missing for {self.standard}: "
                    f"{', '.join(self._missing_collections)}. Build them with "
                    f"'python -m backend.rag.build_standard_hierarchical_collections' "
                    f"before this store can run a real search."
                )
                _logger.warning(
                    "StandardHierarchicalStore(%s): required Qdrant collection(s) missing: %s. "
                    "search() will report ready=False and never attempt a hierarchical query.",
                    self.standard, ", ".join(self._missing_collections),
                )
        except Exception as exc:
            self.ready = False
            self._unavailable_reason = (
                f"Qdrant readiness check failed for {self.standard}: {exc}. "
                f"search() will report this as an operational unavailability and never "
                f"attempt a hierarchical query."
            )
            _logger.warning(
                "StandardHierarchicalStore(%s): Qdrant readiness check failed: %s. "
                "search() will report ready=False and never attempt a hierarchical query.",
                self.standard, exc,
            )

        # Embedding model construction is LAZY (Task 05.2) -- never done in
        # __init__. An injected fake embedder is stored directly and used
        # as-is; otherwise self._embedder stays None until _embed_query()
        # constructs the real SentenceTransformer on first use, inside the
        # same protected try/except boundary search() already wraps
        # _embed_query() in. This means a store that is not `ready` never
        # attempts model construction at all (search() returns before ever
        # calling _embed_query()), and a construction failure on a ready
        # store is caught exactly like any other embedding failure -- never
        # raised to the classifier.
        self._embedder = embedder

        self._engine = HierarchyBeamSearchEngine(
            client=self._client, stages=stages, hitl_threshold=hitl_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        text: str,
        top_k: int = 5,
        beam: int = 2,
        reranker_candidates: int = 5,
    ) -> StandardHierarchyResult:
        """Run parent-filtered hierarchical retrieval for *text*. Returns
        an EXPLICIT not-ready/no-result state (never a fabricated or
        borrowed result) whenever the collections are absent or the engine
        found nothing -- see this module's "Readiness contract"."""
        if not self.ready:
            return StandardHierarchyResult(
                code="", label_en="", label_ar="", confidence=0.0,
                stage_confidences={}, hierarchy_path=[], top_candidates=[],
                hitl_required=True, ready=False,
                unavailable_reason=self._unavailable_reason,
            )

        text = (text or "").strip()
        if not text:
            return StandardHierarchyResult(
                code="", label_en="", label_ar="", confidence=0.0,
                stage_confidences={}, hierarchy_path=[], top_candidates=[],
                hitl_required=True, ready=True,
                unavailable_reason="empty input text",
            )

        try:
            query_vec = self._embed_query(text)
        except Exception as exc:
            _logger.warning(
                "StandardHierarchicalStore(%s): embedding failed: %s. "
                "search() reports this as an operational unavailability.",
                self.standard, exc,
            )
            return StandardHierarchyResult(
                code="", label_en="", label_ar="", confidence=0.0,
                stage_confidences={}, hierarchy_path=[], top_candidates=[],
                hitl_required=True, ready=True,
                unavailable_reason=f"{self.standard} query embedding failed: {exc}",
            )

        try:
            engine_result = self._engine.search(
                query_vec, top_k=top_k, beam=beam, reranker_candidates=reranker_candidates,
            )
        except Exception as exc:
            _logger.warning(
                "StandardHierarchicalStore(%s): engine search failed: %s. "
                "search() reports this as an operational unavailability.",
                self.standard, exc,
            )
            return StandardHierarchyResult(
                code="", label_en="", label_ar="", confidence=0.0,
                stage_confidences={}, hierarchy_path=[], top_candidates=[],
                hitl_required=True, ready=True,
                unavailable_reason=f"{self.standard} hierarchical engine search failed: {exc}",
            )

        if engine_result is None:
            return StandardHierarchyResult(
                code="", label_en="", label_ar="", confidence=0.0,
                stage_confidences={}, hierarchy_path=[], top_candidates=[],
                hitl_required=True, ready=True,
                unavailable_reason=(
                    f"{self.standard} hierarchical search returned no candidates for this query "
                    f"(every explored branch had zero hits at some stage)"
                ),
            )

        return StandardHierarchyResult(
            code=engine_result.code,
            label_en=engine_result.label_en,
            label_ar=engine_result.label_ar,
            confidence=engine_result.confidence,
            stage_confidences=engine_result.stage_confidences,
            hierarchy_path=engine_result.hierarchy_path,
            top_candidates=engine_result.top_candidates,
            hitl_required=engine_result.hitl_required,
            ready=True,
            unavailable_reason="",
        )

    # ------------------------------------------------------------------
    # Embedding (E5 convention -- matches hierarchical_store.py exactly)
    # ------------------------------------------------------------------

    def _embed_query(self, text: str) -> list[float]:
        # Lazy construction (Task 05.2) -- see __init__'s comment. Runs
        # inside search()'s try/except around _embed_query(), so a
        # SentenceTransformer(MODEL_NAME) construction failure is caught
        # exactly like a query-encoding failure, never raised to the
        # classifier. An injected embedder is never overwritten.
        if self._embedder is None:
            self._embedder = SentenceTransformer(MODEL_NAME)
        prefixed = f"query: {text.strip()}"
        vec = self._embedder.encode(
            [prefixed], normalize_embeddings=True, show_progress_bar=False, batch_size=1,
        )
        return vec[0].tolist()


# ---------------------------------------------------------------------------
# Module-level singletons + DI-aware factories
# ---------------------------------------------------------------------------

_isic_store: Optional[StandardHierarchicalStore] = None
_iscedf_store: Optional[StandardHierarchicalStore] = None


def get_isic_hierarchical_store(client=None, embedder=None) -> StandardHierarchicalStore:
    """Production callers (no args): lazily-constructed, cached singleton,
    same pattern as backend.rag.hierarchical_store.get_hierarchical_store().
    Tests (either arg passed): always a fresh, non-cached instance built
    from the supplied fake(s) -- never touches or warms the production
    singleton."""
    global _isic_store
    if client is not None or embedder is not None:
        return StandardHierarchicalStore(
            standard="ISIC Rev.4", stages=isic_stages(), hitl_threshold=HITL_THRESHOLD,
            client=client, embedder=embedder,
        )
    if _isic_store is None:
        _isic_store = StandardHierarchicalStore(
            standard="ISIC Rev.4", stages=isic_stages(), hitl_threshold=HITL_THRESHOLD,
        )
    return _isic_store


def get_iscedf_hierarchical_store(client=None, embedder=None) -> StandardHierarchicalStore:
    """See get_isic_hierarchical_store()'s docstring -- identical contract."""
    global _iscedf_store
    if client is not None or embedder is not None:
        return StandardHierarchicalStore(
            standard="ISCED-F 2013", stages=iscedf_stages(), hitl_threshold=HITL_THRESHOLD,
            client=client, embedder=embedder,
        )
    if _iscedf_store is None:
        _iscedf_store = StandardHierarchicalStore(
            standard="ISCED-F 2013", stages=iscedf_stages(), hitl_threshold=HITL_THRESHOLD,
        )
    return _iscedf_store
