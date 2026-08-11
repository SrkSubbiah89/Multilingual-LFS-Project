"""
backend/rag/hierarchical_store.py

4-stage hierarchical ISCO-08 retrieval using Qdrant.

Architecture
------------
Uses four Qdrant collections, one per ISCO-08 hierarchy level:

  isco08_major_groups    – 10 major groups   (1-digit code)
  isco08_submajor_groups – ~40 sub-major groups (2-digit code)
  isco08_minor_groups    – ~130 minor groups  (3-digit code)
  isco08_unit_groups     – ~430 unit groups   (4-digit code)

Each entry carries a ``parent_code`` payload field so Qdrant
field-condition filters can prune the search at every stage.

Pipeline
--------
1. Search ``isco08_major_groups`` (10 entries, no filter) → top-1 major code.
2. Search ``isco08_submajor_groups`` filtered by ``parent_code == major_code`` → top-1.
3. Search ``isco08_minor_groups`` filtered by ``parent_code == submajor_code`` → top-1.
4. Search ``isco08_unit_groups`` filtered by ``parent_code == minor_code`` → top-5.

Returns ``HierarchicalResult`` with the best unit-group code, confidence,
hierarchy path, top candidates, and a HITL flag.

Retrieval outcomes (Task 13 revision)
--------------------------------------
Every call to ``search()`` ends in exactly one of four distinct,
explicitly labelled states -- see ``HierarchicalResult.fallback_used`` and
each ``ISCOClassification.method``'s ``"hierarchical_"``/``"flat_"``
prefix (backend/agents/isco_classifier.py):

1. **Successful keyword-anchor route.** ``ISCOClassifier`` supplies a
   keyword-derived ``major_hint`` (``_keyword_major_hint()``); the anchor
   reaches a complete stage 1-4 path on the first attempt. ``fallback_used
   =False``. ``trace["stage1_source"] == "keyword_map"``,
   ``trace["keyword_anchor_retry"] == False``.
2. **Recovered semantic-stage route after a failed keyword anchor.** The
   keyword anchor reached no stage-4 result (e.g. the hint's submajor/
   minor/unit branch had zero children in the collection), so
   ``_hierarchical_search()`` retries the SAME generic engine exactly
   once with no seed (normal semantic stage-1 retrieval) before
   considering flat fallback. If that retry succeeds, ``fallback_used=
   False`` and the case is still genuine hierarchical retrieval --
   ``trace["stage1_source"] == "semantic_retrieval"`` (the true final
   path) and ``trace["keyword_anchor_retry"] == True`` +
   ``trace["keyword_anchor_original_hint"]`` (the failed hint) record
   what was tried and abandoned, without ever conflating the failed
   attempt's stage evidence with the winning retry's. This state did not
   exist before Task 13 -- previously, a failed keyword anchor fell
   straight through to flat fallback (see state 4), which the Task 12
   full WISCO run showed happening for 21/18,747 cases.
3. **Explicit flat fallback.** Neither the keyword-anchor attempt (if
   any) nor the unseeded retry (if attempted) produced a complete 4-stage
   path, OR a hierarchical collection is absent, OR a Qdrant query
   exception occurred (see "Bounded Qdrant requests" below). The flat
   ``isco_occupations`` collection is searched instead.
   ``fallback_used=True``, ``ISCOClassification.method`` gets a
   ``"flat_"`` prefix, and the trace records the SAME candidate list
   under all four ``stageN`` keys (documented, intentional -- there are
   no real stages 1-3 for a flat search) so a fallback result's trace is
   never mistaken for four genuinely distinct hierarchical stages.
4. **Total unavailability.** Neither hierarchical nor flat collections are
   usable (or both are empty for this query) -- ``_empty_result()``,
   ``code=""``, ``fallback_used=True``.

``eval/run_eval.py --require-genuine-hierarchical`` (Task 13, evaluation-
only) refuses to accept a run containing ANY case in states 3 or 4, or
missing stage evidence, or (with ``--max-stage-latency-ms``) an excessive
per-stage latency -- it aborts immediately rather than writing a result
CSV that could be mistaken for complete, valid hierarchical benchmark
evidence. Production behaviour (states 1-4 above) is completely
unaffected by that flag; it only changes whether ``eval/run_eval.py``
itself accepts the run's output.

Bounded Qdrant requests (Task 13)
------------------------------------
``HierarchicalISCOStore`` passes an explicit, finite request timeout to
its ``QdrantClient`` (default 30s, override via ``QDRANT_TIMEOUT_SECONDS``
-- see ``_resolve_qdrant_timeout_seconds()``). This does NOT prove or fix
the unconfirmed root cause of the multi-minute-to-multi-hour individual
stage stalls the Task 12 full WISCO run observed (see
Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md)
-- it only bounds a *class* of indefinitely-blocked Qdrant requests so a
future stall raises a catchable exception (routed through the existing
query-exception → zero-hits → fallback/retry path, state 3/4 above)
instead of hanging forever, making the failure diagnosable rather than
silent.

Embedding
---------
Same model as vector_store.py: ``intfloat/multilingual-e5-small`` (384-dim).
E5 prefix convention:  "query: " for queries, "passage: " for indexing.

Usage
-----
from backend.rag.hierarchical_store import get_hierarchical_store

hs     = get_hierarchical_store()
result = hs.search("software engineer")
result = hs.search("مهندس برمجيات")
print(result.code, result.label_en, result.confidence)
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

from backend.rag import hierarchy_engine
from backend.rag.hierarchy_engine import HierarchyBeamSearchEngine, SeedSpec, StageConfig, StageOverride
from backend.rag.official_isco08_catalogue import PROFILE_COLLECTION_NAMES

load_dotenv()

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME  = "intfloat/multilingual-e5-small"
VECTOR_DIM  = 384
_BATCH_SIZE = 64

HITL_THRESHOLD = 0.70  # below this → human-in-the-loop review required

# Confidence weights for the four stages
_W1, _W2, _W3, _W4 = 0.10, 0.20, 0.20, 0.50

# Collection names (legacy profile -- unchanged since before Task 21)
_COL_MAJOR    = "isco08_major_groups"
_COL_SUBMAJOR = "isco08_submajor_groups"
_COL_MINOR    = "isco08_minor_groups"
_COL_UNIT     = "isco08_unit_groups"
_COL_FLAT     = "isco_occupations"  # fallback

# Task 21: profile support. LEGACY_PROFILE (the default) preserves the
# exact module-level constants above -- every existing caller that omits
# `profile=` sees byte-identical collection names and behaviour to
# before this task. OFFICIAL_PROFILE_ILO2021_V1 points at the versioned,
# primary-ILO-sourced collections (Task 20/21) instead, including its
# OWN flat collection (isco08_unit_groups_flat_ilo2021_v1) -- an official
# profile never falls back to the legacy, mixed-granularity
# `isco_occupations` collection, so a caller can never receive a
# legacy-sourced result while believing they selected the official
# profile.
LEGACY_PROFILE = "legacy"
OFFICIAL_PROFILE_ILO2021_V1 = "official_ilo2021_v1"

_PROFILE_COLLECTIONS: dict[str, dict[str, str]] = {
    LEGACY_PROFILE: {
        "major": _COL_MAJOR, "submajor": _COL_SUBMAJOR,
        "minor": _COL_MINOR, "unit": _COL_UNIT, "flat": _COL_FLAT,
    },
    **{profile: names for profile, names in PROFILE_COLLECTION_NAMES.items()},
}


class UnknownISCOCatalogueProfileError(Exception):
    """Raised by HierarchicalISCOStore.__init__ when an unrecognized
    `profile` is requested -- fail closed rather than silently falling
    back to the legacy collection names."""


_ISCO4_RE = re.compile(r"^[0-9]{4}$")

# Task 13: bounded Qdrant request timeout. Finite and conservative by
# default -- the Task 12 full WISCO run observed individual stage queries
# blocking for minutes to hours with no client-side bound at all. This
# timeout does NOT prove or fix that stall's root cause (unconfirmed --
# see Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md);
# it only bounds a *class* of indefinitely-blocked Qdrant requests so a
# future stall raises a catchable exception (routed through the existing,
# unchanged query-exception -> zero-hits -> fallback/retry path -- see
# HierarchyBeamSearchEngine._query()) instead of hanging forever, making
# the failure diagnosable rather than silent.
QDRANT_DEFAULT_TIMEOUT_SECONDS = 30


def _resolve_qdrant_timeout_seconds() -> int:
    """Read QDRANT_TIMEOUT_SECONDS from the environment; fail safe to
    QDRANT_DEFAULT_TIMEOUT_SECONDS (never raise, never crash startup) on
    a missing, malformed, or non-positive value."""
    raw = os.getenv("QDRANT_TIMEOUT_SECONDS")
    if raw is None or not raw.strip():
        return QDRANT_DEFAULT_TIMEOUT_SECONDS
    try:
        value = int(raw.strip())
    except ValueError:
        _logger.warning(
            "HierarchicalISCOStore: QDRANT_TIMEOUT_SECONDS=%r is not a valid "
            "integer; using the default %ss.",
            raw, QDRANT_DEFAULT_TIMEOUT_SECONDS,
        )
        return QDRANT_DEFAULT_TIMEOUT_SECONDS
    if value <= 0:
        _logger.warning(
            "HierarchicalISCOStore: QDRANT_TIMEOUT_SECONDS=%r must be positive; "
            "using the default %ss.",
            raw, QDRANT_DEFAULT_TIMEOUT_SECONDS,
        )
        return QDRANT_DEFAULT_TIMEOUT_SECONDS
    return value


def _resolve_max_query_attempts() -> int:
    """Task 27: read QDRANT_QUERY_MAX_ATTEMPTS from the environment; fail
    SAFE to hierarchy_engine.DEFAULT_MAX_QUERY_ATTEMPTS (1 -- no retry,
    never raise, never crash startup) on a missing, malformed, or
    out-of-[1, MAX_QUERY_ATTEMPTS_HARD_CAP]-range value. This mirrors
    _resolve_qdrant_timeout_seconds()'s exact precedent: an ambient
    environment variable degrades gracefully. A value passed directly as
    a HierarchicalISCOStore/HierarchyBeamSearchEngine constructor
    argument is validated differently -- see
    HierarchyBeamSearchEngine.__init__, which raises ValueError
    immediately on an invalid explicit value instead."""
    raw = os.getenv("QDRANT_QUERY_MAX_ATTEMPTS")
    if raw is None or not raw.strip():
        return hierarchy_engine.DEFAULT_MAX_QUERY_ATTEMPTS
    try:
        value = int(raw.strip())
    except ValueError:
        _logger.warning(
            "HierarchicalISCOStore: QDRANT_QUERY_MAX_ATTEMPTS=%r is not a valid "
            "integer; using the default %s (no retry).",
            raw, hierarchy_engine.DEFAULT_MAX_QUERY_ATTEMPTS,
        )
        return hierarchy_engine.DEFAULT_MAX_QUERY_ATTEMPTS
    if not (1 <= value <= hierarchy_engine.MAX_QUERY_ATTEMPTS_HARD_CAP):
        _logger.warning(
            "HierarchicalISCOStore: QDRANT_QUERY_MAX_ATTEMPTS=%r must be in "
            "[1, %s]; using the default %s (no retry).",
            raw, hierarchy_engine.MAX_QUERY_ATTEMPTS_HARD_CAP,
            hierarchy_engine.DEFAULT_MAX_QUERY_ATTEMPTS,
        )
        return hierarchy_engine.DEFAULT_MAX_QUERY_ATTEMPTS
    return value


def _resolve_retry_backoff_seconds() -> float:
    """Task 27: read QDRANT_QUERY_RETRY_BACKOFF_SECONDS from the
    environment; fail SAFE to hierarchy_engine.DEFAULT_RETRY_BACKOFF_SECONDS
    (0.0) on a missing, malformed, or out-of-range value. Same precedent
    as _resolve_max_query_attempts()/_resolve_qdrant_timeout_seconds()."""
    raw = os.getenv("QDRANT_QUERY_RETRY_BACKOFF_SECONDS")
    if raw is None or not raw.strip():
        return hierarchy_engine.DEFAULT_RETRY_BACKOFF_SECONDS
    try:
        value = float(raw.strip())
    except ValueError:
        _logger.warning(
            "HierarchicalISCOStore: QDRANT_QUERY_RETRY_BACKOFF_SECONDS=%r is not "
            "a valid number; using the default %ss.",
            raw, hierarchy_engine.DEFAULT_RETRY_BACKOFF_SECONDS,
        )
        return hierarchy_engine.DEFAULT_RETRY_BACKOFF_SECONDS
    if not (0 <= value <= hierarchy_engine.MAX_RETRY_BACKOFF_SECONDS_HARD_CAP):
        _logger.warning(
            "HierarchicalISCOStore: QDRANT_QUERY_RETRY_BACKOFF_SECONDS=%r must be "
            "in [0, %s]; using the default %ss.",
            raw, hierarchy_engine.MAX_RETRY_BACKOFF_SECONDS_HARD_CAP,
            hierarchy_engine.DEFAULT_RETRY_BACKOFF_SECONDS,
        )
        return hierarchy_engine.DEFAULT_RETRY_BACKOFF_SECONDS
    return value


# ---------------------------------------------------------------------------
# Task 34 / Task 34.1: bounded client-side deadline pool
# ---------------------------------------------------------------------------
#
# Audited finding (qdrant-client 1.17.0, REST transport): the public
# ``query_points(timeout=...)`` parameter is a Qdrant SERVER-side
# operation-timeout hint only -- confirmed by direct source reading of
# ``qdrant_client/http/api/search_api.py``'s generated
# ``_build_for_query_points()``, which places it in the REST request's
# query string (``?timeout=N``), never as an httpx per-call timeout
# override. The actual client-side connect/read/write/pool timeout is
# governed exclusively by the ``httpx.Client`` a ``QdrantClient`` was
# CONSTRUCTED with (``QdrantClient(timeout=N)``). qdrant-client's own
# public call chain (``ApiClient.send_inner()`` -> ``self._client.send(
# request)``) never forwards any per-call timeout override into httpx,
# so there is no public, per-call way to shrink an ALREADY-CONSTRUCTED
# client's timeout for one specific request.
#
# Task 34.1 correction: Task 34's report stated that "a bare int/float
# applies uniformly" to the constructed client's timeout. That is true
# only up to a rounding step Task 34 did not audit deeply enough: direct
# source reading of ``qdrant_client.qdrant_remote.QdrantRemote.__init__``
# shows the constructor's ``timeout`` value is unconditionally passed
# through ``math.ceil()`` before it ever reaches httpx --
#
#     _timeout = math.ceil(timeout) if timeout is not None else None
#     # it has been changed from float to int.
#     # convert it to the closest greater or equal int value (e.g. 0.5 -> 1)
#
# -- so a fractional remaining stage budget (e.g. 7.01s) previously
# produced ``ceil(7.01) == 8``, an 8-second client deadline that EXCEEDS
# the 7.01s actually remaining. This pool now selects
# ``floor(deadline_seconds)`` (never ``ceil``) as the integer second
# count requested from the constructor: because it is already an
# integer, qdrant-client's own internal ``math.ceil()`` is a no-op on
# it, so the resulting client timeout is always
# ``floor(deadline_seconds) <= deadline_seconds`` -- never rounded
# upward past the caller's requested deadline. Below
# ``hierarchy_engine.MIN_PRACTICAL_CLIENT_DEADLINE_SECONDS`` (1 whole
# second -- the smallest non-zero value ``floor()`` can produce),
# ``get()`` refuses outright (see its docstring) rather than construct
# a zero/invalid timeout or round up; callers (``_query()``) check the
# remaining budget against that same constant BEFORE calling ``get()``
# and report ``stage_budget_exhausted`` themselves, so ``get()``'s
# refusal is a defensive backstop, not the primary enforcement point.
#
# The only public, documented way to get a genuinely different
# client-side timeout is therefore to construct a client with that
# timeout. This pool does exactly that, bounded and cleanly closed:
# requesting a deadline that is EXACTLY EQUAL to the store's own
# already-resolved default deadline returns the store's existing
# primary client (zero behaviour change, zero new construction, for
# the overwhelming common case) -- this equality check now compares
# the raw requested deadline directly, not a rounded cache key, so a
# deadline merely rounding to the same integer as the primary (e.g.
# 7.99s when the primary is 8s) is never mistaken for the true default
# and correctly gets its own, smaller, dedicated client instead. Any
# other (necessarily smaller, stage-budget-capped) deadline gets its
# own client, constructed with the public ``check_compatibility=False``
# constructor parameter (skips qdrant-client's own default
# construction-time server-version HTTP probe -- confirmed by direct
# source reading of ``QdrantRemote.__init__`` -- so acquiring a new
# pooled client never itself performs an extra, unbounded network call
# before our own deadline is even in effect), cached by its floored
# integer-second value, bounded to at most ``max_size`` distinct
# additional clients with least-recently-used eviction. Every evicted
# or explicitly-closed client's connection is closed via its own public
# ``QdrantClient.close()``.
class _QdrantClientDeadlinePool:
    """Task 34/34.1: see the module-level comment above for the full
    audited rationale. Not used by any caller that omits
    `client_for_deadline` when constructing `HierarchyBeamSearchEngine`
    -- purely additive."""

    def __init__(
        self,
        host: str,
        port: int,
        primary_client: QdrantClient,
        primary_deadline_seconds: int,
        max_size: int = 8,
    ) -> None:
        self._host = host
        self._port = port
        self._primary_client = primary_client
        self._primary_deadline_seconds = primary_deadline_seconds
        self._max_size = max_size
        self._pool: dict[int, QdrantClient] = {}
        self._order: list[int] = []  # least-recently-used first

    def get(self, deadline_seconds: float) -> QdrantClient:
        """Return a QdrantClient whose own constructor-level timeout is
        `floor(deadline_seconds)` (Task 34.1: never `ceil` -- see the
        module-level comment above for why `floor` is required to keep
        the resulting client timeout from ever exceeding the caller's
        requested deadline). Never constructs a new client for a
        deadline that is exactly equal to the store's own
        primary/default deadline -- returns the existing primary client
        directly in that case.

        Raises ``ValueError`` if `deadline_seconds` is below
        `hierarchy_engine.MIN_PRACTICAL_CLIENT_DEADLINE_SECONDS`: no
        positive integer-second client timeout can be constructed below
        that threshold without either rounding up past the requested
        deadline or being zero/invalid. Callers are expected to check
        the remaining budget against that same constant themselves
        BEFORE calling `get()` (see `hierarchy_engine._query()`) and
        report `stage_budget_exhausted` directly -- this exception is a
        defensive backstop, not the primary enforcement point."""
        if deadline_seconds < hierarchy_engine.MIN_PRACTICAL_CLIENT_DEADLINE_SECONDS:
            raise ValueError(
                f"_QdrantClientDeadlinePool.get(): deadline_seconds={deadline_seconds!r} "
                "is below hierarchy_engine.MIN_PRACTICAL_CLIENT_DEADLINE_SECONDS="
                f"{hierarchy_engine.MIN_PRACTICAL_CLIENT_DEADLINE_SECONDS!r}; callers must "
                "check this themselves and treat it as stage-budget-exhausted rather "
                "than call get()."
            )
        if deadline_seconds == self._primary_deadline_seconds:
            return self._primary_client
        key = math.floor(deadline_seconds)
        if key in self._pool:
            self._order.remove(key)
            self._order.append(key)
            return self._pool[key]
        client = QdrantClient(
            host=self._host, port=self._port, timeout=key, check_compatibility=False,
        )
        self._pool[key] = client
        self._order.append(key)
        if len(self._pool) > self._max_size:
            oldest_key = self._order.pop(0)
            evicted = self._pool.pop(oldest_key)
            try:
                evicted.close()
            except Exception as exc:  # noqa: BLE001 - cleanup best-effort, never fatal
                _logger.debug("_QdrantClientDeadlinePool: failed to close evicted client: %s", exc)
        return client

    def close_all(self) -> None:
        """Close every pooled (non-primary) client. Safe to call more
        than once; safe even if some clients were already closed."""
        for client in self._pool.values():
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001 - cleanup best-effort, never fatal
                _logger.debug("_QdrantClientDeadlinePool: failed to close pooled client: %s", exc)
        self._pool.clear()
        self._order.clear()


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class UnitCandidate:
    """One of the top-N unit-group candidates returned from stage 4."""
    code: str
    label_en: str
    label_ar: str
    score: float


@dataclass
class HierarchicalResult:
    """
    Structured output from ``HierarchicalISCOStore.search()``.

    Attributes
    ----------
    code : str
        Best-matching 4-digit ISCO-08 unit-group code (or major/minor code
        when the search terminates early due to collection absence).
    label_en : str
        English label of the selected occupation.
    label_ar : str
        Arabic label of the selected occupation.
    confidence : float
        Weighted mean of stage confidences:
        stage4×0.5 + stage3×0.2 + stage2×0.2 + stage1×0.1
    stage_confidences : dict
        Raw cosine-similarity scores at each stage
        {stage1, stage2, stage3, stage4}.
    hierarchy_path : list[str]
        Ordered list of ISCO codes traversed, e.g. ["2", "25", "251", "2512"].
    top_candidates : list[UnitCandidate]
        Top-3 unit-group candidates from stage 4.
    hitl_required : bool
        True when ``confidence < HITL_THRESHOLD`` (0.70).
    fallback_used : bool
        True when the flat ``isco_occupations`` collection was used instead
        of the hierarchical pipeline.
    """
    code: str
    label_en: str
    label_ar: str
    confidence: float
    stage_confidences: dict
    hierarchy_path: list = field(default_factory=list)
    top_candidates: list = field(default_factory=list)
    hitl_required: bool = False
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# HierarchicalISCOStore
# ---------------------------------------------------------------------------

class HierarchicalISCOStore:
    """
    4-stage hierarchical ISCO-08 retrieval backed by Qdrant.

    Instantiation probes Qdrant and loads the SentenceTransformer model.
    All four hierarchical collections must already exist and be populated;
    this class does *not* create or populate them (use the companion
    ``build_hierarchical_collections.py`` script for that).

    When collections are absent the instance degrades gracefully to
    flat search on ``isco_occupations``.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        profile: str = LEGACY_PROFILE,
        max_query_attempts: Optional[int] = None,
        retry_backoff_seconds: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        timeout_seconds : int, optional
            Task 13: explicit Qdrant request timeout (seconds), passed
            straight through to ``QdrantClient(timeout=...)``. Existing
            callers that omit this (default None) get the value resolved
            from the ``QDRANT_TIMEOUT_SECONDS`` environment variable, or
            ``QDRANT_DEFAULT_TIMEOUT_SECONDS`` (30s) if that variable is
            absent, empty, malformed, or non-positive -- see
            ``_resolve_qdrant_timeout_seconds()``. This bounds how long a
            single Qdrant request can block; it does not change what a
            successful or failed query means to the rest of this class.
        profile : str, default "legacy"
            Task 21: which set of Qdrant collection names to use.
            ``"legacy"`` (default) is byte-identical to every prior
            version of this class -- existing callers that omit this
            parameter see zero behavioural change. ``"official_ilo2021_v1"``
            (see ``OFFICIAL_PROFILE_ILO2021_V1``) points at the versioned,
            primary-ILO-sourced collections instead, including a
            dedicated official flat collection
            (``isco08_unit_groups_flat_ilo2021_v1``) used for THIS
            profile's own fallback -- an official profile never falls
            back to the legacy ``isco_occupations`` collection. An
            unrecognized profile raises ``UnknownISCOCatalogueProfileError``
            immediately (fail closed; never silently falls back to
            legacy collection names).
        max_query_attempts : int, optional (Task 27)
            Total attempts per Qdrant query call before it is treated as
            failed. Existing callers that omit this (default None) get
            the value resolved from the ``QDRANT_QUERY_MAX_ATTEMPTS``
            environment variable, or 1 (no retry -- byte-for-byte prior
            behaviour) if that variable is absent, empty, malformed, or
            outside ``[1, hierarchy_engine.MAX_QUERY_ATTEMPTS_HARD_CAP]``
            -- see ``_resolve_max_query_attempts()``. An explicit value
            passed here instead is validated (and raises ``ValueError``
            on an invalid one) by ``HierarchyBeamSearchEngine.__init__``.
        retry_backoff_seconds : float, optional (Task 27)
            Fixed delay between attempts, only consulted when
            ``max_query_attempts`` > 1. Existing callers that omit this
            get the value resolved from
            ``QDRANT_QUERY_RETRY_BACKOFF_SECONDS``, or 0.0 if absent/
            invalid -- see ``_resolve_retry_backoff_seconds()``.
        """
        if profile not in _PROFILE_COLLECTIONS:
            raise UnknownISCOCatalogueProfileError(
                f"profile {profile!r} is not a known ISCO-08 catalogue profile; "
                f"known profiles: {sorted(_PROFILE_COLLECTIONS)}"
            )
        self.profile = profile
        _cols = _PROFILE_COLLECTIONS[profile]
        col_major, col_submajor, col_minor, col_unit, col_flat = (
            _cols["major"], _cols["submajor"], _cols["minor"], _cols["unit"], _cols["flat"],
        )

        _host = host or os.getenv("QDRANT_HOST", "localhost")
        _port = int(port or os.getenv("QDRANT_PORT", 6333))
        _timeout = timeout_seconds if timeout_seconds is not None else _resolve_qdrant_timeout_seconds()

        self._client = QdrantClient(host=_host, port=_port, timeout=_timeout)
        self._model  = SentenceTransformer(MODEL_NAME)
        self._col_flat = col_flat

        # Pre-check which collections exist so we know upfront whether to
        # run the hierarchical pipeline or fall back immediately.
        existing = {c.name for c in self._client.get_collections().collections}
        self._hierarchical_ready = all(
            col in existing
            for col in (col_major, col_submajor, col_minor, col_unit)
        )
        self._flat_ready = col_flat in existing

        if not self._hierarchical_ready:
            _logger.warning(
                "HierarchicalISCOStore(profile=%r): one or more hierarchical Qdrant "
                "collections are missing (%s). Will fall back to flat search on '%s'.",
                profile,
                ", ".join(
                    c for c in (col_major, col_submajor, col_minor, col_unit)
                    if c not in existing
                ),
                col_flat,
            )

        # Generic beam-search engine, configured with ISCO's 4 stages and
        # weights (see backend/rag/hierarchy_engine.py). _hierarchical_search
        # below is a thin translator between this store's public
        # HierarchicalResult and the engine's stage-count-agnostic
        # EngineResult -- the actual beam-search logic lives in the engine.
        # Same engine class for every profile (Task 21 adapts collection
        # names only; it does not implement a new search algorithm).
        _resolved_max_query_attempts = (
            max_query_attempts if max_query_attempts is not None else _resolve_max_query_attempts()
        )
        _resolved_retry_backoff_seconds = (
            retry_backoff_seconds if retry_backoff_seconds is not None else _resolve_retry_backoff_seconds()
        )
        # Task 34: bounded pool providing a genuinely deadline-specific
        # client (public QdrantClient(timeout=...) constructor, never a
        # private-attribute override) whenever a query's effective
        # per-attempt timeout is smaller than this store's own resolved
        # default (i.e. a shared stage budget has genuinely capped it) --
        # see _QdrantClientDeadlinePool's module-level docstring above
        # for the full audited rationale. Requesting the default deadline
        # itself always returns self._client unchanged.
        self._client_deadline_pool = _QdrantClientDeadlinePool(
            host=_host, port=_port,
            primary_client=self._client, primary_deadline_seconds=_timeout,
        )
        self._engine = HierarchyBeamSearchEngine(
            client=self._client,
            stages=[
                StageConfig(name="major", collection=col_major, weight=_W1),
                StageConfig(name="submajor", collection=col_submajor, weight=_W2),
                StageConfig(name="minor", collection=col_minor, weight=_W3),
                StageConfig(name="unit", collection=col_unit, weight=_W4),
            ],
            hitl_threshold=HITL_THRESHOLD,
            max_query_attempts=_resolved_max_query_attempts,
            retry_backoff_seconds=_resolved_retry_backoff_seconds,
            # Task 31: reliable per-request timeout propagation -- reuses
            # the SAME resolved value already used to construct this
            # store's QdrantClient(timeout=_timeout) above, so every
            # query_points() call now also carries an explicit,
            # request-level timeout matching the client's own configured
            # timeout (see HierarchyBeamSearchEngine._query()'s docstring
            # for exactly what this Qdrant REST parameter controls).
            query_timeout_seconds=_timeout,
            # Task 34: genuine client-side connect/read/write/pool
            # deadline enforcement -- see _QdrantClientDeadlinePool above.
            client_for_deadline=self._client_deadline_pool.get,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Task 34: close every pooled deadline-specific Qdrant client
        (never the primary client, which callers may still be using).
        Optional -- this store has no other lifecycle management and
        nothing in this codebase currently calls this automatically
        (existing singleton/long-lived usage is unaffected); provided so
        a caller that wants to release pooled connections explicitly
        can. Safe to call more than once."""
        self._client_deadline_pool.close_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        major_hint: str = "",
        beam: int = 2,
        stage1_mode: str = "description",
        reranker_candidates: int = 5,
        branch_collapse: bool = False,
        capture_pool_metadata: bool = False,
        trace: Optional[dict] = None,
        max_stage_latency_ms: Optional[float] = None,
    ) -> HierarchicalResult:
        """
        Run 4-stage hierarchical ISCO-08 retrieval for *query*.

        Falls back to the flat ``isco_occupations`` collection if the
        hierarchical collections are unavailable or any stage yields
        zero results.

        Parameters
        ----------
        query : str
            Free-text job title, description, or occupation-related text
            (English, Arabic, or code-switched).
        top_k : int
            Number of unit-group candidates to retrieve per branch in
            stage 4 (i.e. the Qdrant query limit for each explored
            minor-group parent). See reranker_candidates for how many of
            the union across branches end up in
            ``HierarchicalResult.top_candidates``.
        major_hint : str
            Optional 1-digit ISCO major group code (e.g. "5" for Service
            workers).  When provided, stage 1 is skipped and this code is
            used directly, preventing semantic drift at the top of the
            hierarchy (e.g. "chef" embedding to "Professionals" instead of
            "Service and Sales Workers"). Takes priority over stage1_mode
            unconditionally -- unaffected by this parameter's value.
        beam : int
            Beam width at stages 1–3 (default 2).  Set to 1 for strict
            greedy top-1 selection; used by the ablation study to measure
            the accuracy gain attributable to beam search.
        stage1_mode : str, default "description"
            "description" (default): unchanged prior behaviour -- stage 1
            searches isco08_major_groups directly (each major group
            embedded from its short official label only, e.g.
            "Professionals").
            "leaf_vote": bottom-up alternative -- retrieves the top-20
            isco08_unit_groups leaves directly (no major-group anchoring),
            aggregates by major group via summed cosine of that group's
            leaf hits, and takes the top-`beam` majors forward. Added
            after the Step 1 audit found major-group description text too
            abstract to embed near specific job titles (e.g. "primary
            school teacher" never placed major group 2 in its stage-1
            candidates under "description" mode). Only consulted when
            major_hint is empty -- the keyword map, when it fires, still
            skips stage 1 entirely regardless of this parameter.
        reranker_candidates : int, default 5
            How many unit-group candidates end up in
            ``HierarchicalResult.top_candidates`` (which is exactly what
            gets shown to the re-ranking LLM -- see isco_classifier.py's
            _classify_hierarchical()). Default behaviour as of the
            candidate-pooling fix: candidates are pooled across every
            explored beam branch (not just the winning one), deduplicated
            by code keeping the highest score seen, sorted globally, and
            the top `reranker_candidates` are kept. See branch_collapse
            for why this changed.
        branch_collapse : bool, default False
            When True, reproduces the PRE-FIX behaviour: only the winning
            branch's own top-N children (N = reranker_candidates) are
            shown to the reranker, discarding every candidate found in
            every other explored branch. This was the only behaviour that
            existed before the fix, and is a real, quantified defect --
            on the 130-case full set, 49/82 errors were cases where the
            correct code was found in a losing branch (competitive score,
            e.g. 0.792 vs the winning branch's 0.800) and silently
            discarded before the reranker ever ran, regardless of which
            model executed it. Kept as an explicit opt-in (not the
            default) purely so the before/after comparison can still be
            run and reported -- not because the old behaviour is
            considered acceptable for production use.
        capture_pool_metadata : bool, default False
            B2 instrumentation, OBSERVATIONAL ONLY. When True, populates
            trace["stage4_pool_enriched"] with the full pooled candidate
            list (all codes, not truncated to reranker_candidates) each
            carrying branch_id, source_rank (its rank within its own
            branch), path (major/submajor/minor/unit codes for that
            branch), raw_score, and normalized_score. Computed via
            backend.rag.candidate_pool.pool_and_rank_candidates() using
            sort_mode="raw_score" -- i.e. a richer LOGGING view of exactly
            the same B1 pool/order, never a different selection. Does not
            alter pool_hits/top_candidates/the reranker's actual input in
            any way, and defaults to False so B0/B1 callers (which never
            pass this) see zero behavioural change and pay zero extra
            cost. See candidate_pool.py's module docstring for why this
            logic lives in a separate, independently unit-tested module
            rather than being added inline here.
        trace : dict, optional
            Instrumentation-only. When a dict is passed, it is populated
            in-place with the full candidate list considered at each stage
            (not just the winning beam path) under keys "stage1".."stage4",
            each a list of {"code", "label_en", "score"}. Does not affect
            the return value or any selection/branching decision — this is
            purely additive logging for the evaluation harness (see
            eval/run_eval.py). Existing callers that omit this parameter
            see zero behavioural change.
        max_stage_latency_ms : float, optional (Task 31)
            Opt-in strict stage deadline budget in milliseconds, passed
            straight through to
            ``HierarchyBeamSearchEngine.search(max_stage_latency_ms=...)``
            -- see that method's docstring for the full semantics.
            Default ``None`` (every existing caller that omits this)
            reproduces prior behaviour exactly -- no deadline is ever
            established. Intended to be threaded only from
            ``eval/run_eval.py``'s strict ``--max-stage-latency-ms``
            configuration (via ``ISCOClassifier.classify()``), never
            silently imposed on production/default classification calls.

        Returns
        -------
        HierarchicalResult
        """
        query = query.strip()
        if not query:
            return self._empty_result()

        query_vec = self._embed_query(query)

        if self._hierarchical_ready:
            result = self._hierarchical_search(
                query_vec, top_k=max(top_k, 5), major_hint=major_hint.strip(),
                beam=max(1, int(beam)), stage1_mode=stage1_mode,
                reranker_candidates=max(1, int(reranker_candidates)),
                branch_collapse=branch_collapse,
                capture_pool_metadata=capture_pool_metadata, trace=trace,
                max_stage_latency_ms=max_stage_latency_ms,
            )
            if result is not None:
                return result
            # At least one stage returned 0 results; fall through to flat.
            _logger.info(
                "HierarchicalISCOStore: hierarchical pipeline returned no results "
                "for query=%r; falling back to flat search.",
                query,
            )

        return self._flat_search(query_vec, top_k=top_k, trace=trace)

    # ------------------------------------------------------------------
    # Hierarchical pipeline
    # ------------------------------------------------------------------

    def _hierarchical_search(
        self,
        query_vec: list[float],
        top_k: int,
        major_hint: str = "",
        beam: int = 2,
        stage1_mode: str = "description",
        reranker_candidates: int = 5,
        branch_collapse: bool = False,
        capture_pool_metadata: bool = False,
        trace: Optional[dict] = None,
        max_stage_latency_ms: Optional[float] = None,
    ) -> Optional[HierarchicalResult]:
        """
        Execute a beam-search hierarchical pipeline across 4 stages.

        Instead of strict greedy top-1 at every stage (which cascades a
        single wrong choice into a completely wrong result), we keep up to
        ``beam`` candidates at stages 1–3 and select the path whose
        stage-4 unit-group score is highest.

        Returns ``None`` only if every beam path returns 0 unit-group hits
        (caller falls back to flat search in that case).

        This is a thin translator over the generic
        ``HierarchyBeamSearchEngine`` (backend/rag/hierarchy_engine.py):
        ``major_hint`` becomes a ``SeedSpec`` (stage-0 bypass),
        ``stage1_mode="leaf_vote"`` becomes a ``StageOverride`` wrapping
        ``self._leaf_vote_stage1``, and the engine's stage-count-agnostic
        ``EngineResult`` is converted back to this store's public
        ``HierarchicalResult``. The actual beam-search/pooling/trace logic
        lives entirely in the engine now.

        Task 13 keyword-anchor recovery: a keyword ``major_hint`` anchors
        the engine to a single stage-0 branch (see ``SeedSpec`` below). If
        that one anchored branch reaches no stage-4 hit anywhere (engine
        returns ``None``), the anchor itself -- not genuine hierarchical
        unavailability -- was the cause, so this method retries the SAME
        engine exactly once with no seed (normal semantic stage-1
        retrieval) before this store falls through to flat search. Each
        attempt is traced into its own throwaway dict; only the winning
        attempt's stage evidence is ever copied into the caller's
        ``trace``, so a failed seeded attempt can never be mistaken for
        the final retrieval path (see the docstring on ``trace`` above).
        """
        seed = None
        stage_overrides = None
        used_keyword_anchor = bool(major_hint)

        # When a keyword hint is supplied (e.g. "chef" → "5") we trust it
        # and skip semantic search entirely for stage 1.
        if major_hint:
            seed = SeedSpec(
                candidates=[(major_hint, 1.0)],
                source_label="keyword_map",
                candidate_label="(keyword hint, search skipped)",
            )
            _logger.debug(
                "HierarchicalISCOStore: keyword major_hint=%r, stage 1 skipped.",
                major_hint,
            )
        elif stage1_mode == "leaf_vote":
            stage_overrides = {
                0: StageOverride(
                    fn=lambda qv, b: self._leaf_vote_stage1(qv, beam=b),
                    source_label="leaf_vote",
                    candidate_label="(leaf-vote aggregate over top-20 unit groups)",
                )
            }

        want_trace = trace is not None
        attempt_trace: dict = {} if want_trace else None
        engine_result = self._engine.search(
            query_vec,
            top_k=top_k,
            beam=beam,
            seed=seed,
            stage_overrides=stage_overrides,
            reranker_candidates=reranker_candidates,
            branch_collapse=branch_collapse,
            capture_pool_metadata=capture_pool_metadata,
            trace=attempt_trace,
            max_stage_latency_ms=max_stage_latency_ms,
        )

        retried = False
        if engine_result is None and used_keyword_anchor:
            _logger.info(
                "HierarchicalISCOStore: keyword major_hint=%r reached no complete "
                "hierarchical path; retrying once with normal semantic stage-1 "
                "retrieval (no seed) before falling back to flat search.",
                major_hint,
            )
            retried = True
            attempt_trace = {} if want_trace else None
            engine_result = self._engine.search(
                query_vec,
                top_k=top_k,
                beam=beam,
                seed=None,
                stage_overrides=stage_overrides,
                reranker_candidates=reranker_candidates,
                branch_collapse=branch_collapse,
                capture_pool_metadata=capture_pool_metadata,
                trace=attempt_trace,
                max_stage_latency_ms=max_stage_latency_ms,
            )

        if trace is not None:
            # Only the winning attempt's stage evidence is copied in --
            # a failed seeded attempt's throwaway trace is discarded
            # entirely, never merged with the retry's genuine evidence.
            trace.update(attempt_trace)
            if used_keyword_anchor:
                trace["keyword_anchor_retry"] = retried
                trace["keyword_anchor_original_hint"] = major_hint

        if engine_result is None:
            return None

        top_candidates = [
            UnitCandidate(code=c.code, label_en=c.label_en, label_ar=c.label_ar, score=c.score)
            for c in engine_result.top_candidates
        ]

        return HierarchicalResult(
            code=engine_result.code,
            label_en=engine_result.label_en,
            label_ar=engine_result.label_ar,
            confidence=engine_result.confidence,
            stage_confidences=engine_result.stage_confidences,
            hierarchy_path=engine_result.hierarchy_path,
            top_candidates=top_candidates,
            hitl_required=engine_result.hitl_required,
            fallback_used=engine_result.fallback_used,
        )

    # ------------------------------------------------------------------
    # Flat fallback
    # ------------------------------------------------------------------

    def _flat_search(
        self,
        query_vec: list[float],
        top_k: int,
        trace: Optional[dict] = None,
    ) -> HierarchicalResult:
        """
        Search this store's profile-resolved flat collection (legacy:
        ``isco_occupations``; official: ``isco08_unit_groups_flat_ilo2021_v1``)
        and wrap the result as a ``HierarchicalResult`` so callers get a
        consistent return type.
        """
        if not self._flat_ready:
            _logger.error(
                "HierarchicalISCOStore(profile=%r): neither hierarchical "
                "collections nor flat '%s' collection is available.",
                self.profile, self._col_flat,
            )
            return self._empty_result()

        # Task 25: capture whether this query genuinely succeeded (with
        # zero or more hits) or raised an exception (e.g. a Qdrant
        # timeout) -- previously both were collapsed into an identical
        # empty `hits` list with no telemetry at all, making a real
        # zero-hit response and a swallowed exception indistinguishable
        # in the evaluation trace/CSV. Recorded before the `not hits`
        # early return below so telemetry survives on every outcome,
        # including the unavailable-result path.
        query_telemetry: dict = {}
        hits = self._query(
            collection=self._col_flat, query_vec=query_vec, limit=top_k,
            query_telemetry=query_telemetry,
        )
        if trace is not None:
            trace["flat_query_outcome"] = query_telemetry.get("outcome", "")
            trace["flat_query_duration_ms"] = query_telemetry.get("duration_ms")
            trace["flat_query_exception_type"] = query_telemetry.get("exception_type", "")
            trace["flat_query_exception_message"] = query_telemetry.get("exception_message", "")
            # Task 27: additive -- attempts is 1 and attempt_durations_ms
            # is a single-element list for every existing (no-retry,
            # default max_query_attempts=1) caller, so flat_query_duration_ms
            # above remains byte-identical to Task 25's original semantics
            # in that default case.
            trace["flat_query_attempts"] = query_telemetry.get("attempts")
            trace["flat_query_attempt_durations_ms"] = query_telemetry.get("attempt_durations_ms", [])
        if not hits:
            return self._empty_result()
        if trace is not None:
            # Flat fallback has no real stage 1-3; record the same candidate
            # set under all four keys so callers can see this was a flat, not
            # hierarchical, retrieval (HierarchicalResult.fallback_used=True
            # is the authoritative flag; this just keeps the trace non-empty).
            flat_bucket = [
                {"code": h.payload.get("code", ""), "label_en": hierarchy_engine.extract_label_en(h.payload),
                 "score": round(float(h.score), 4)}
                for h in hits
            ]
            for key in ("stage1", "stage2", "stage3", "stage4"):
                trace[key] = flat_bucket

        best    = hits[0]
        score   = round(float(best.score), 4)
        p       = best.payload

        top_candidates = [
            UnitCandidate(
                code=hit.payload.get("code", ""),
                label_en=hierarchy_engine.extract_label_en(hit.payload),
                label_ar=hit.payload.get("label_ar", ""),
                score=round(float(hit.score), 4),
            )
            for hit in hits[:3]
        ]

        code = p.get("code", "")

        # Task 21: an official profile's flat collection is built to
        # contain ONLY 4-digit unit-group codes (unlike the legacy,
        # intentionally mixed-granularity `isco_occupations` collection,
        # which may legitimately return a coarser code -- see
        # Documentation/Conference_I_Reviewer_2/FLAT_BASELINE_COVERAGE_AUDIT.md).
        # This is a defensive runtime check, not expected to ever fire in
        # practice given the builder's own guarantees, but the official
        # flat profile must never silently return a coarse code as if it
        # were a genuine four-digit prediction.
        if self.profile != LEGACY_PROFILE and not _ISCO4_RE.match(code):
            _logger.error(
                "HierarchicalISCOStore(profile=%r): flat collection '%s' returned "
                "a non-4-digit candidate code %r; refusing to report it as a "
                "genuine official flat prediction.",
                self.profile, self._col_flat, code,
            )
            return self._empty_result()

        hierarchy_path = _infer_path(code)

        return HierarchicalResult(
            code=code,
            label_en=hierarchy_engine.extract_label_en(p),
            label_ar=p.get("label_ar", ""),
            confidence=score,
            stage_confidences={
                "stage1": score,
                "stage2": score,
                "stage3": score,
                "stage4": score,
            },
            hierarchy_path=hierarchy_path,
            top_candidates=top_candidates,
            hitl_required=score < HITL_THRESHOLD,
            fallback_used=True,
        )

    # ------------------------------------------------------------------
    # Bottom-up stage 1 (leaf-anchored ancestor voting)
    # ------------------------------------------------------------------

    def _leaf_vote_stage1(
        self, query_vec: list[float], top_n: int = 20, beam: int = 2,
    ) -> list[tuple[str, float]]:
        """
        Retrieve the top-`top_n` isco08_unit_groups leaves directly
        (isco08_unit_groups is a dedicated 436-entry, 4-digit-only
        collection -- no mixed-granularity filtering needed, unlike the
        separate flat isco_occupations fallback collection), then rank
        major groups by the SUMMED cosine score of their leaf hits within
        that top-n set (more supporting leaves near the query outweighs a
        single strong hit -- a genuine vote, not a max-pool).

        Returns up to `beam` (major_code, score) pairs, sorted by that
        summed-cosine ranking. The returned score is each major group's
        MEAN (not summed) cosine among its hits in the top-n, so it stays
        in the same ~[0,1] range as a single cosine score and is safe to
        feed into the existing stage-confidence weighted-average formula
        (_W1 * s1_score + ...) without separate rescaling -- summing
        would blow past 1.0 for any major group with several hits.
        """
        hits = self._query(collection=self._engine.stages[-1].collection, query_vec=query_vec, limit=top_n)
        if not hits:
            return []

        score_sum: dict[str, float] = {}
        score_count: dict[str, int] = {}
        for hit in hits:
            code = hit.payload.get("code", "")
            if not code:
                continue
            major = code[0]
            score_sum[major] = score_sum.get(major, 0.0) + float(hit.score)
            score_count[major] = score_count.get(major, 0) + 1

        ranked = sorted(score_sum.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (major, score_sum[major] / score_count[major])
            for major, _ in ranked[:beam]
        ]

    # ------------------------------------------------------------------
    # Qdrant helpers
    # ------------------------------------------------------------------

    def _query(
        self,
        collection: str,
        query_vec: list[float],
        limit: int,
        parent_code: Optional[str] = None,
        query_telemetry: Optional[dict] = None,
    ):
        """
        Wrap ``client.query_points()`` with an optional parent_code filter.

        Returns a list of ``ScoredPoint`` objects (empty list on any error).
        Delegates to the generic engine's ``_query`` (same implementation,
        moved to backend/rag/hierarchy_engine.py) so ``_flat_search`` and
        ``_leaf_vote_stage1`` -- which query outside the engine's own
        stage-based beam loop -- keep working unchanged. ``query_telemetry``
        (Task 25) is passed straight through -- see
        ``HierarchyBeamSearchEngine._query()``'s docstring.
        """
        return self._engine._query(
            collection=collection, query_vec=query_vec, limit=limit,
            parent_code=parent_code, query_telemetry=query_telemetry,
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_query(self, text: str) -> list[float]:
        """Encode *text* with the E5 query prefix, returns a normalised vector."""
        prefixed = f"query: {text.strip()}"
        vec = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=1,
        )
        return vec[0].tolist()

    # ------------------------------------------------------------------
    # Sentinel
    # ------------------------------------------------------------------

    def search_flat_only(
        self,
        query: str,
        top_k: int = 5,
        trace: Optional[dict] = None,
    ) -> HierarchicalResult:
        """
        Task 21: bypass the hierarchical stages entirely and query only
        this store's profile-resolved flat collection directly -- reuses
        the existing ``_flat_search``/``_embed_query`` implementations
        unchanged (no new retrieval algorithm). For the official profile
        this queries ONLY ``isco08_unit_groups_flat_ilo2021_v1`` (never
        the legacy ``isco_occupations`` collection); for the legacy
        profile it reproduces the legacy flat-fallback collection's
        existing behaviour, standalone.
        """
        query = query.strip()
        if not query:
            return self._empty_result()
        query_vec = self._embed_query(query)
        return self._flat_search(query_vec, top_k=top_k, trace=trace)

    @staticmethod
    def _empty_result() -> HierarchicalResult:
        return HierarchicalResult(
            code="",
            label_en="Unknown",
            label_ar="غير معروف",
            confidence=0.0,
            stage_confidences={"stage1": 0.0, "stage2": 0.0,
                               "stage3": 0.0, "stage4": 0.0},
            hierarchy_path=[],
            top_candidates=[],
            hitl_required=True,
            fallback_used=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_path(code: str) -> list[str]:
    """
    Derive the hierarchy path from an ISCO code by string prefix.

    Examples
    --------
    "2512" → ["2", "25", "251", "2512"]
    "25"   → ["2", "25"]
    "2"    → ["2"]
    ""     → []
    """
    if not code:
        return []
    path: list[str] = []
    for length in range(1, len(code) + 1):
        path.append(code[:length])
    return path


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_hierarchical_store: Optional[HierarchicalISCOStore] = None


def get_hierarchical_store() -> HierarchicalISCOStore:
    """
    Return the module-level ``HierarchicalISCOStore`` singleton.

    The first call creates the instance (loads the SentenceTransformer model
    and connects to Qdrant).  Subsequent calls return the cached instance.

    Raises
    ------
    Exception
        Propagated from ``HierarchicalISCOStore.__init__()`` if Qdrant is
        unreachable or the model cannot be loaded.
    """
    global _hierarchical_store
    if _hierarchical_store is None:
        _hierarchical_store = HierarchicalISCOStore()
    return _hierarchical_store
