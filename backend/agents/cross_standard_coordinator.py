"""
backend/agents/cross_standard_coordinator.py

Deterministic (non-LLM) backward-direction cross-standard coordination
(Item 1b of the 2026-09-12 multi-agent RAG work): once ISIC/ISCED results
are known, reconsider an already-uncertain ISCO primary result using
evidence that wasn't available when ISCO committed.

This is real coordination -- it can actually change ``isco_result``'s
effective primary code -- as distinct from ``SemanticRelationEngine.
analyse()``, which only ever *scores* the already-fixed ISCO/ISIC/ISCED
combination and has never revised anything (confirmed directly: its
``inferred_isco`` field is LLM-suggested advisory text shown in the
explanation string, never read back to replace ``isco_result.primary_code``
anywhere in ``survey_routes.py``).

Reuses, rather than reimplements, SemanticRelationEngine's own validated
ISCO<->ISIC / ISCO<->ISCED crosswalk logic (the ``isco_isic_compatible``/
``isco_isced_compatible`` booleans already returned by ``analyse()``) as
the single source of truth for "compatible" -- no second hand-built
crosswalk table.

Opt-in: only ever called from ``survey_routes.py`` when
``ENABLE_COORDINATED_CLASSIFICATION`` is set. Every failure mode degrades
to "no revision" rather than raising or blocking classification.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger(__name__)


def maybe_revise_isco_with_cross_signal(
    isco_result: Any,
    isic_section: Optional[str],
    isced_level: Optional[int],
) -> tuple[Optional[Any], str]:
    """Return ``(alternative_to_promote, reason)``, or ``(None, "")``.

    Fires only when ALL of the following hold:

    1. ``isco_result.hitl_required`` is True -- ISCOClassifier's own
       confidence-based uncertainty gate (``HITL_THRESHOLD`` /
       ``_MIN_TRUSTED_CANDIDATE_GAP`` in ``isco_classifier.py``). This
       coordinator never reconsiders a primary ISCO already considers
       confident.
    2. At least one of ``isic_section``/``isced_level`` is known.
    3. The current primary is NOT compatible with the known ISIC section
       and/or ISCED level, per SemanticRelationEngine's own crosswalk.
    4. At least one of ``isco_result.alternatives`` (already computed by
       ISCOClassifier.classify() -- zero new retrieval) IS compatible.

    Never re-retrieves. Never fires on an already-coherent primary (guard
    1) or an already cross-standard-compatible one (guard 3) -- this is a
    genuine correction path, not an arbitrary reshuffle. Any exception
    (missing attributes, engine failure) is caught and degrades to
    ``(None, "")``, leaving the caller's ISCO result untouched -- same
    fallback contract as every other opt-in addition in this codebase.
    """
    try:
        if not getattr(isco_result, "hitl_required", False):
            return None, ""
        if isic_section is None and isced_level is None:
            return None, ""

        primary_code = getattr(isco_result, "primary_code", None)
        if not primary_code:
            return None, ""

        from backend.agents.semantic_relation import get_semantic_relation_engine

        engine = get_semantic_relation_engine(use_llm=False)

        primary_check = engine.analyse(
            isco_code=primary_code,
            isic_section=isic_section,
            isced_level=isced_level,
        )
        if primary_check.isco_isic_compatible and primary_check.isco_isced_compatible:
            return None, ""  # already coherent -- nothing to revise

        for alt in getattr(isco_result, "alternatives", None) or []:
            alt_code = getattr(alt, "code", None)
            if alt_code is None and isinstance(alt, dict):
                alt_code = alt.get("code")
            if not alt_code:
                continue

            alt_check = engine.analyse(
                isco_code=alt_code,
                isic_section=isic_section,
                isced_level=isced_level,
            )
            if alt_check.isco_isic_compatible and alt_check.isco_isced_compatible:
                reason = (
                    f"Primary ISCO {primary_code} was uncertain (hitl_required) and "
                    f"incompatible with ISIC={isic_section!r}/ISCED={isced_level!r}; "
                    f"alternative {alt_code} is compatible with both."
                )
                return alt, reason

        return None, ""
    except Exception as exc:
        log.warning(
            "cross_standard_coordinator: revision check failed (%s); leaving ISCO primary unchanged.",
            exc,
        )
        return None, ""
