"""
eval/legacy_decision_policy41/hierarchical_retrieval_adapter.py

Task 43 (extended): wires Task 41's decision-policy component to the
current maintained HIERARCHICAL (4-stage) retrieval, mirroring
`flat_retrieval_adapter.py`'s flat-retrieval version.

Unlike the flat adapter, this one does NOT bypass the store's own
retrieval method -- `HierarchicalISCOStore.search(query, reranker_
candidates=5)` already returns exactly 5 pooled, deduplicated unit-group
candidates via `HierarchicalResult.top_candidates` (the same beam-search/
candidate-pooling algorithm the real hierarchical evaluation path uses,
completely unmodified and unre-implemented here).

Known gap this module works around, not fixes (same root cause already
disclosed for the flat path in `flat_retrieval_adapter.py`): both
`hierarchical_store.py::_hierarchical_search()` and
`hierarchy_engine.py` build `UnitCandidate`/`EngineCandidate.label_en`/
`label_ar` from `hit.payload.get("label_en"/"label_ar", "")`, but the
official hierarchical collections' payloads
(`build_official_isco08_collections.py::_build_payload()`) only ever
write `title_en` -- so `top_candidates[i].label_en` is always blank for
the official profile here too. Rather than touch `hierarchical_store.py`
or `hierarchy_engine.py` (both exercised by the already-measured Task 36
evaluation path), this module looks up each candidate's real `title_en`
by `code` from the already-verified, already-loaded official catalogue
(`backend/rag/official_isco08_catalogue.py::load_official_catalogue()`)
-- a read-only, independent, already-hash-verified source of the exact
same titles, not a guess or a fabrication.
"""

from __future__ import annotations

from pathlib import Path

from eval.legacy_decision_policy41.flat_retrieval_adapter import InsufficientCandidatesError
from eval.legacy_decision_policy41.policy import PolicyCandidate

REQUIRED_CANDIDATE_COUNT = 5

DEFAULT_CATALOGUE_PATH = (
    Path(__file__).resolve().parents[2]
    / "eval" / "local_catalogues" / "ilo_isco08_2021" / "normalized" / "isco08_official_normalized.csv"
)


def build_unit_code_to_title_en(catalogue_records) -> dict[str, str]:
    """*catalogue_records* is the list returned by
    `official_isco08_catalogue.load_official_catalogue()`. Returns
    {unit_code: title_en} for the 436 unit-level records only."""
    return {r.code: r.title_en for r in catalogue_records if r.level == "unit"}


def fetch_five_official_hierarchical_candidates(
    store, query_text: str, code_to_title_en: dict[str, str],
) -> list[PolicyCandidate]:
    """
    Calls *store*.search(query_text, reranker_candidates=5) -- the real,
    unmodified, already-tested 4-stage hierarchical retrieval -- and
    converts its 5 pooled unit-group candidates to `PolicyCandidate`s.

    `title_en` is looked up by code from *code_to_title_en* (never from
    the candidate's own blank `label_en`). `title_ar`/`description` are
    honest empty strings (same official-catalogue limitation as the flat
    adapter). `level` is always 4 (every candidate here is a 4-digit
    unit-group code, by construction of `reranker_candidates`).
    """
    result = store.search(query_text, reranker_candidates=REQUIRED_CANDIDATE_COUNT)
    candidates = result.top_candidates
    if len(candidates) < REQUIRED_CANDIDATE_COUNT:
        raise InsufficientCandidatesError(
            f"expected {REQUIRED_CANDIDATE_COUNT} hierarchical candidates, "
            f"got {len(candidates)} for query {query_text!r}"
        )
    return [
        PolicyCandidate(
            code=c.code,
            title_en=code_to_title_en.get(c.code, ""),
            title_ar="",
            level=4,
            confidence=float(c.score),
            description="",
        )
        for c in candidates[:REQUIRED_CANDIDATE_COUNT]
    ]


def make_hierarchical_candidate_fetcher(code_to_title_en: dict[str, str]):
    """Returns a `candidate_fetcher(store, query_text) -> list[PolicyCandidate]`
    closure bound to *code_to_title_en*, matching the signature
    `live_reranker.run_dev_with_live_reranker`'s `candidate_fetcher`
    parameter expects (same shape as
    `flat_retrieval_adapter.fetch_five_official_flat_candidates`)."""

    def fetcher(store, query_text: str) -> list[PolicyCandidate]:
        return fetch_five_official_hierarchical_candidates(store, query_text, code_to_title_en)

    return fetcher
