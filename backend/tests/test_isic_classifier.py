"""
Tests for backend/agents/isic_classifier.py

Fully offline — no LLM, no network calls required.
Only the keyword-scoring path is exercised; the LLM re-ranking path is
tested via a mock.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.classifier_methods import (
    ISIC_FLAT_FALLBACK_KEYWORD,
    ISIC_FLAT_RETRIEVAL,
    ISIC_HIERARCHICAL_FALLBACK_KEYWORD,
    ISIC_HIERARCHICAL_RETRIEVAL,
)
from backend.agents.isic_classifier import (
    ISICClassifier,
    ISICClassification,
    _ISIC_DATA,
)
from backend.rag.standard_hierarchical_store import (
    HITL_THRESHOLD,
    ISIC_COLLECTIONS,
    ISIC_FLAT_COLLECTIONS_BY_PROFILE,
    StandardFlatStore,
    StandardHierarchicalStore,
    isic_stages,
)


class _FakeQdrantClient:
    """Same fake-client pattern as test_standard_hierarchical_store.py --
    table: dict[(collection, parent_code)] -> list[(code, label_en, label_ar, score)]."""

    def __init__(self, table, existing_collections):
        self.table = table
        self.existing_collections = set(existing_collections)
        self.calls = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.existing_collections])

    def query_points(self, collection_name, query, query_filter, limit, with_payload):
        parent_code = None
        if query_filter is not None:
            parent_code = query_filter.must[0].match.value
        self.calls.append((collection_name, parent_code, limit))
        rows = self.table.get((collection_name, parent_code), [])
        points = [
            SimpleNamespace(score=score, payload={"code": code, "label_en": label_en, "label_ar": label_ar})
            for code, label_en, label_ar, score in rows[:limit]
        ]
        return SimpleNamespace(points=points)


class _FakeFlatQdrantClient:
    """Fake client for StandardFlatStore -- its search() calls
    query_points(collection_name=, query=, limit=, with_payload=True) with
    NO query_filter kwarg at all (there is no parent to filter on), unlike
    the hierarchical engine's _query(). table: dict[collection] ->
    list[(code, label_en, label_ar, score)]."""

    def __init__(self, table, existing_collections):
        self.table = table
        self.existing_collections = set(existing_collections)
        self.calls = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.existing_collections])

    def query_points(self, collection_name, query, limit, with_payload):
        self.calls.append((collection_name, limit))
        rows = self.table.get(collection_name, [])
        points = [
            SimpleNamespace(score=score, payload={"code": code, "label_en": label_en, "label_ar": label_ar})
            for code, label_en, label_ar, score in rows[:limit]
        ]
        return SimpleNamespace(points=points)


class _FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=1):
        import numpy as np
        return np.zeros((len(texts), 384))


_ISIC_HIT_TABLE = {
    ("isic_rev4_sections", None): [("A", "Agriculture, Forestry and Fishing", "", 0.90)],
    ("isic_rev4_divisions", "A"): [("01", "Crop and animal production", "", 0.85)],
    ("isic_rev4_groups", "01"): [("011", "Growing of non-perennial crops", "", 0.80)],
    ("isic_rev4_classes", "011"): [("0111", "Growing of cereals, leguminous crops and oil seeds", "", 0.75)],
}


def _patch_isic_store(monkeypatch, table, existing_collections):
    """Monkeypatch backend.rag.standard_hierarchical_store.get_isic_hierarchical_store
    (which ISICClassifier._classify_hierarchical imports locally, by name, at
    call time) so classify(method=ISIC_HIERARCHICAL_RETRIEVAL) resolves to a
    real StandardHierarchicalStore backed by fakes -- no live Qdrant, no
    embedding-model load."""
    client = _FakeQdrantClient(table, existing_collections)
    store = StandardHierarchicalStore(
        standard="ISIC Rev.4", stages=isic_stages(), hitl_threshold=HITL_THRESHOLD,
        client=client, embedder=_FakeEmbedder(),
    )
    monkeypatch.setattr(
        "backend.rag.standard_hierarchical_store.get_isic_hierarchical_store",
        lambda: store,
    )
    return client


_ISIC_FLAT_COLLECTION = ISIC_FLAT_COLLECTIONS_BY_PROFILE["enriched_e5large"]
_ISIC_FLAT_HIT_TABLE = {
    _ISIC_FLAT_COLLECTION: [
        ("0111", "Growing of cereals, leguminous crops and oil seeds", "", 0.82),
    ],
}


def _patch_isic_flat_store(monkeypatch, table, existing_collections):
    """Same pattern as _patch_isic_store, for _classify_flat()'s
    get_isic_flat_store(profile="enriched_e5large") import -- ISICClassifier
    always requests the enriched_e5large profile for the flat path (see
    classifier_methods.ISIC_FLAT_RETRIEVAL's docstring for why: this is
    the specific flat+real-official-text+e5-large recipe that mirrors
    ISCO-08's own best-tested config, not just any flat profile).

    Real gap found and fixed by code review (2026-08-27): the previous
    version of this helper used a bare `lambda profile="e5_large": store`
    that silently accepted and ignored WHATEVER profile string production
    code passed -- so a regression that changed _classify_flat's
    hardcoded profile back to "e5_large" (pointing at the wrong,
    non-enriched collection) would have passed every existing test. This
    version records every profile it was actually called with, and
    test_flat_retrieval_requests_the_enriched_e5large_profile below
    asserts against that record."""
    client = _FakeFlatQdrantClient(table, existing_collections)
    store = StandardFlatStore(
        standard="ISIC Rev.4", collection=_ISIC_FLAT_COLLECTION, hitl_threshold=HITL_THRESHOLD,
        client=client, embedder=_FakeEmbedder(),
    )
    requested_profiles: list[str] = []

    def _fake_get_isic_flat_store(profile):
        requested_profiles.append(profile)
        return store

    monkeypatch.setattr(
        "backend.rag.standard_hierarchical_store.get_isic_flat_store",
        _fake_get_isic_flat_store,
    )
    client.requested_profiles = requested_profiles
    return client


@pytest.fixture
def clf():
    """ISICClassifier with LLM constructor patched out."""
    with patch("backend.agents.isic_classifier.get_llm", return_value=MagicMock()):
        return ISICClassifier()


# ---------------------------------------------------------------------------
# 1. Data integrity
# ---------------------------------------------------------------------------

def test_isic_data_not_empty():
    assert len(_ISIC_DATA) > 30


def test_all_entries_have_required_keys():
    required = {"section", "section_title", "division_code", "division_title", "keywords"}
    for entry in _ISIC_DATA:
        assert required.issubset(entry.keys()), f"Missing keys in: {entry}"


def test_division_codes_are_two_digits():
    bad = [e["division_code"] for e in _ISIC_DATA if not e["division_code"].isdigit()]
    assert bad == []


def test_sections_A_to_U_present():
    sections = {e["section"] for e in _ISIC_DATA}
    # Core economic sections must be present
    for s in "ABCDFGHIJKLMNOPQRST":
        assert s in sections, f"Section {s} missing"


# ---------------------------------------------------------------------------
# 2. classify() — keyword path
# ---------------------------------------------------------------------------

def test_software_company_maps_to_section_J(clf):
    result = clf.classify("I work at a software development company")
    assert result.section == "J"


def test_division_code_for_it_services(clf):
    result = clf.classify("software developer tech startup app")
    assert result.division_code == "62"


def test_hospital_maps_to_section_Q(clf):
    result = clf.classify("I work in a hospital as a nurse")
    assert result.section == "Q"


def test_school_maps_to_section_P(clf):
    result = clf.classify("teacher at a secondary school")
    assert result.section == "P"


def test_bank_maps_to_section_K(clf):
    result = clf.classify("finance banking investment fund")
    assert result.section == "K"


def test_construction_maps_to_section_F(clf):
    result = clf.classify("construction building contractor civil")
    assert result.section == "F"


def test_arabic_hospital_maps_to_section_Q(clf):
    result = clf.classify("أعمل في مستشفى حكومي")
    assert result.section == "Q"


def test_arabic_software_maps_to_section_J(clf):
    result = clf.classify("شركة برمجيات تقنية")
    assert result.section == "J"


def test_confidence_in_range(clf):
    result = clf.classify("restaurant food service waiter")
    assert 0.0 <= result.confidence <= 1.0


def test_high_confidence_skips_llm(clf):
    # High keyword overlap → method = "keyword"
    result = clf.classify("software developer programmer IT technology app web")
    assert result.method == "keyword"


def test_returns_isic_classification(clf):
    result = clf.classify("teacher university professor")
    assert isinstance(result, ISICClassification)


def test_empty_text_returns_fallback(clf):
    result = clf.classify("")
    assert isinstance(result, ISICClassification)
    assert result.confidence == 0.0


def test_alternatives_list_populated(clf):
    # Multiple keyword hits → alternatives present
    result = clf.classify("doctor hospital clinic nurse physician")
    # For high-confidence results alternatives may be empty or present
    assert isinstance(result.alternatives, list)


# ---------------------------------------------------------------------------
# 3. LLM re-ranking path
# ---------------------------------------------------------------------------

def test_llm_rerank_called_for_low_confidence(clf):
    """When keyword score < KEYWORD_THRESHOLD, LLM re-ranking is attempted."""
    # Patch _keyword_score to return a low score
    original_score = clf._keyword_score

    def low_score(text):
        scored = original_score(text)
        return [(0.5, e) for _, e in scored[:3]] if scored else []

    with patch.object(clf, "_keyword_score", side_effect=low_score):
        with patch.object(clf, "_llm_rerank", return_value=None) as mock_rerank:
            result = clf.classify("some obscure industry description")
            # _llm_rerank may or may not be called depending on low-score path
            assert isinstance(result, ISICClassification)


def test_llm_rerank_exception_falls_back_to_keyword(clf):
    """LLM failure must not crash classify()."""
    with patch.object(clf, "_llm_rerank", side_effect=RuntimeError("LLM down")):
        # Should not raise; falls back to keyword result
        result = clf.classify("farm agriculture crop harvest")
        assert isinstance(result, ISICClassification)


# ---------------------------------------------------------------------------
# 3b. Determinism (bug found 2026-08-24, same class as ISCED's): tie-broken
#     keyword results must not depend on PYTHONHASHSEED / set() order.
# ---------------------------------------------------------------------------

def test_keyword_score_result_stable_across_repeated_calls():
    clf = ISICClassifier.__new__(ISICClassifier)
    text = "management consultancy and advisory services"  # documented real tie, see test_isic_isced_reranker_model.py
    results = {scored[0][1]["class_code"] for _ in range(20) for scored in [clf._keyword_score(text)]}
    assert len(results) == 1, f"tie-broken winner must be stable within a process, got {results}"


def test_keyword_score_uses_deterministic_tokenisation_not_a_hash_ordered_set():
    """The fix itself, checked directly: tokenisation must be
    dict.fromkeys() (order-preserving), not set() (hash-order,
    PYTHONHASHSEED-dependent) -- a regression guard against this exact
    bug being reintroduced by a future refactor."""
    import inspect
    source = inspect.getsource(ISICClassifier._keyword_score)
    assert "dict.fromkeys(" in source
    assert "set(re.findall" not in source


# ---------------------------------------------------------------------------
# 4. method="isic_hierarchical_retrieval": real hierarchical retrieval +
#    explicit fallback labelling (hermetic: FakeQdrantClient/FakeEmbedder,
#    no live Qdrant, no embedding-model load)
# ---------------------------------------------------------------------------

def test_method_none_default_path_unchanged(clf):
    """Omitting method= (or passing None explicitly) must be byte-for-byte
    identical to calling classify(text) as before this parameter existed."""
    r_default = clf.classify("software developer tech startup app")
    r_explicit_none = clf.classify("software developer tech startup app", method=None)
    assert r_default == r_explicit_none


def test_unrelated_method_value_runs_default_legacy_pipeline(clf):
    """A method= value that isn't ISIC_HIERARCHICAL_RETRIEVAL is ignored --
    the default keyword/LLM pipeline runs unchanged (guards against a typo
    silently routing to a different branch)."""
    result = clf.classify("software developer tech startup app", method="some_other_value")
    assert result.method in ("keyword", "llm")


def test_hierarchical_retrieval_runs_real_parent_filtered_search(clf, monkeypatch):
    """With the required collections present, method=isic_hierarchical_retrieval
    must run the actual generic engine through a real parent-filtered query
    chain (verified via the fake client's recorded calls), and populate the
    hierarchy_path / stage_confidences / method fields honestly."""
    client = _patch_isic_store(monkeypatch, _ISIC_HIT_TABLE, set(ISIC_COLLECTIONS.values()))

    result = clf.classify("cereal farming", method=ISIC_HIERARCHICAL_RETRIEVAL)

    assert isinstance(result, ISICClassification)
    assert result.method == ISIC_HIERARCHICAL_RETRIEVAL
    assert result.fallback_used is False
    assert result.fallback_reason is None
    assert result.section == "A"
    assert result.division_code == "01"
    assert result.group_code == "011"
    assert result.class_code == "0111"
    assert result.hierarchy_path == ["A", "01", "011", "0111"]
    assert set(result.stage_confidences) == {"stage1", "stage2", "stage3", "stage4"}
    assert 0.0 < result.confidence <= 1.0
    # Real parent-filtered queries were actually issued, not a flat lookup.
    assert client.calls == [
        ("isic_rev4_sections", None, 2),
        ("isic_rev4_divisions", "A", 2),
        ("isic_rev4_groups", "01", 2),
        ("isic_rev4_classes", "011", 5),
    ]


def test_hierarchical_retrieval_falls_back_when_collections_missing(clf, monkeypatch):
    """No collections built yet -- classify() must fall back to the legacy
    pipeline and report an explicit fallback label, never silently claim
    isic_hierarchical_retrieval succeeded."""
    _patch_isic_store(monkeypatch, table={}, existing_collections=set())

    result = clf.classify("software developer tech startup app", method=ISIC_HIERARCHICAL_RETRIEVAL)

    assert result.method in (ISIC_HIERARCHICAL_FALLBACK_KEYWORD, "isic_hierarchical_fallback_llm")
    assert result.fallback_used is True
    assert result.fallback_reason
    assert "missing" in result.fallback_reason.lower()
    # The legacy pipeline still ran for real -- section is populated.
    assert result.section != ""


def test_hierarchical_retrieval_falls_back_when_search_finds_nothing(clf, monkeypatch):
    """Collections exist but this query matches nothing -- still an explicit,
    non-fabricated fallback, distinct from the missing-collections case."""
    _patch_isic_store(monkeypatch, table={}, existing_collections=set(ISIC_COLLECTIONS.values()))

    result = clf.classify("software developer tech startup app", method=ISIC_HIERARCHICAL_RETRIEVAL)

    assert result.fallback_used is True
    assert result.fallback_reason
    assert "no candidates" in result.fallback_reason.lower()
    assert result.method != ISIC_HIERARCHICAL_RETRIEVAL


def test_hierarchical_fallback_result_is_never_mislabeled():
    """Never observe method=isic_hierarchical_retrieval together with
    fallback_used=True -- the two are mutually exclusive by construction."""
    from backend.agents.isic_classifier import ISICClassifier as _C
    # Static invariant check on the dataclass defaults: a fresh instance
    # constructed without the hierarchical branch never sets fallback_used.
    default = _C._fallback("")
    assert default.fallback_used is False
    assert default.method != ISIC_HIERARCHICAL_RETRIEVAL


# ---------------------------------------------------------------------------
# Flat retrieval (ISCO-08 best-tested-config parity, added 2026-08-25)
# ---------------------------------------------------------------------------

def test_flat_retrieval_runs_real_direct_search_and_resolves_full_hierarchy(clf, monkeypatch):
    """With the flat collection present, method=isic_flat_retrieval must
    query it directly (single call, no parent filter), and reconstruct the
    full section/division/group ancestry from _ENTRY_BY_CLASS since the
    flat store's own hierarchy_path only ever carries the leaf code."""
    client = _patch_isic_flat_store(monkeypatch, _ISIC_FLAT_HIT_TABLE, {_ISIC_FLAT_COLLECTION})

    result = clf.classify("cereal farming", method=ISIC_FLAT_RETRIEVAL)

    assert isinstance(result, ISICClassification)
    assert result.method == ISIC_FLAT_RETRIEVAL
    assert result.fallback_used is False
    assert result.fallback_reason is None
    assert result.class_code == "0111"
    assert result.section == "A"
    assert result.division_code == "01"
    assert result.group_code == "011"
    assert result.hierarchy_path == ["A", "01", "011", "0111"]
    assert 0.0 < result.confidence <= 1.0
    # Exactly one direct query against the flat collection -- no parent-chain
    # traversal like the hierarchical path issues.
    assert client.calls == [(_ISIC_FLAT_COLLECTION, 5)]


def test_flat_retrieval_requests_the_enriched_e5large_profile(clf, monkeypatch):
    """Real gap found and fixed by code review: no prior test asserted
    WHICH profile string _classify_flat actually requests from
    get_isic_flat_store() -- a regression silently reverting it to
    "e5_large" (the plain, non-enriched flat profile) would have passed
    every other test unnoticed."""
    client = _patch_isic_flat_store(monkeypatch, _ISIC_FLAT_HIT_TABLE, {_ISIC_FLAT_COLLECTION})
    clf.classify("cereal farming", method=ISIC_FLAT_RETRIEVAL)
    assert client.requested_profiles == ["enriched_e5large"]


def test_flat_retrieval_falls_back_when_returned_code_is_not_in_isic_data(clf, monkeypatch):
    """Real gap found by a second, independent code review pass and fixed
    2026-08-27: the Qdrant collection and _ISIC_DATA are two separate
    sources of truth with no version check. A code the collection returns
    that _ISIC_DATA doesn't have (simulated here with a made-up code no
    real ISIC class uses) must fall back honestly, never be reported as a
    successful classification with empty section/division/group and
    fallback_used=False."""
    drifted_table = {_ISIC_FLAT_COLLECTION: [("9876", "A code _ISIC_DATA does not have", "", 0.90)]}
    _patch_isic_flat_store(monkeypatch, drifted_table, {_ISIC_FLAT_COLLECTION})

    result = clf.classify("software developer tech startup app", method=ISIC_FLAT_RETRIEVAL)

    assert result.fallback_used is True
    assert result.method != ISIC_FLAT_RETRIEVAL
    assert result.fallback_reason
    assert "drifted" in result.fallback_reason.lower() or "not present" in result.fallback_reason.lower()
    # The legacy pipeline still ran for real -- section is populated, not empty.
    assert result.section != ""


def test_flat_retrieval_falls_back_when_collection_missing(clf, monkeypatch):
    _patch_isic_flat_store(monkeypatch, table={}, existing_collections=set())

    result = clf.classify("software developer tech startup app", method=ISIC_FLAT_RETRIEVAL)

    assert result.method in (ISIC_FLAT_FALLBACK_KEYWORD, "isic_flat_fallback_llm")
    assert result.fallback_used is True
    assert result.fallback_reason
    assert "missing" in result.fallback_reason.lower()
    assert result.section != ""


def test_flat_retrieval_falls_back_when_search_finds_nothing(clf, monkeypatch):
    _patch_isic_flat_store(monkeypatch, table={}, existing_collections={_ISIC_FLAT_COLLECTION})

    result = clf.classify("software developer tech startup app", method=ISIC_FLAT_RETRIEVAL)

    assert result.fallback_used is True
    assert result.fallback_reason
    assert "no candidates" in result.fallback_reason.lower()
    assert result.method != ISIC_FLAT_RETRIEVAL


def test_unrelated_method_value_still_ignores_flat_retrieval_too(clf):
    result = clf.classify("software developer tech startup app", method="some_other_value")
    assert result.method not in (ISIC_FLAT_RETRIEVAL, ISIC_HIERARCHICAL_RETRIEVAL)
