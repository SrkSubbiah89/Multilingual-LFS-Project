"""
Tests for backend/rag/standard_hierarchical_store.py -- ISIC Rev.4 / ISCED-F
2013 hierarchical retrieval stores built on the shared, generic
HierarchyBeamSearchEngine. Hermetic: FakeQdrantClient/FakeEmbedder stand in
for qdrant_client.QdrantClient/SentenceTransformer -- no live Qdrant
connection, no embedding-model load, no network call anywhere in this file.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.rag.standard_hierarchical_store import (
    HITL_THRESHOLD,
    ISCEDF_COLLECTIONS,
    ISCEDF_STAGE_WEIGHTS,
    ISIC_COLLECTIONS,
    ISIC_STAGE_WEIGHTS,
    StandardHierarchicalStore,
    get_iscedf_hierarchical_store,
    get_isic_hierarchical_store,
    iscedf_stages,
    isic_stages,
)


class FakeQdrantClient:
    """table: dict[(collection, parent_code_or_None)] -> list[(code, label_en, label_ar, score)].
    existing_collections: set of collection names get_collections() should report as present."""

    def __init__(self, table, existing_collections):
        self.table = table
        self.existing_collections = set(existing_collections)
        self.calls = []  # (collection_name, parent_code, limit) per call, in order

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


class FakeEmbedder:
    """Stands in for SentenceTransformer -- returns a fixed-shape zero vector,
    never loads a real model."""

    def __init__(self, dim=384):
        self.dim = dim
        self.calls = []

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=1):
        self.calls.append(list(texts))
        import numpy as np
        return np.zeros((len(texts), self.dim))


# ---------------------------------------------------------------------------
# 4. Stage configuration: collection names + weights sum to 1.0
# ---------------------------------------------------------------------------

def test_isic_stage_weights_sum_to_one():
    assert abs(sum(ISIC_STAGE_WEIGHTS) - 1.0) < 1e-9


def test_iscedf_stage_weights_sum_to_one():
    assert abs(sum(ISCEDF_STAGE_WEIGHTS) - 1.0) < 1e-9


def test_isic_stages_use_required_collection_names():
    stages = isic_stages()
    names = [s.name for s in stages]
    collections = [s.collection for s in stages]
    assert names == ["sections", "divisions", "groups", "classes"]
    assert collections == [
        "isic_rev4_sections", "isic_rev4_divisions", "isic_rev4_groups", "isic_rev4_classes",
    ]
    assert [s.weight for s in stages] == list(ISIC_STAGE_WEIGHTS)


def test_iscedf_stages_use_required_collection_names():
    stages = iscedf_stages()
    names = [s.name for s in stages]
    collections = [s.collection for s in stages]
    assert names == ["broad_fields", "narrow_fields", "detailed_fields"]
    assert collections == [
        "iscedf2013_broad_fields", "iscedf2013_narrow_fields", "iscedf2013_detailed_fields",
    ]
    assert [s.weight for s in stages] == list(ISCEDF_STAGE_WEIGHTS)


def test_collection_name_maps_match_stage_configs():
    assert set(ISIC_COLLECTIONS.values()) == {s.collection for s in isic_stages()}
    assert set(ISCEDF_COLLECTIONS.values()) == {s.collection for s in iscedf_stages()}


# ---------------------------------------------------------------------------
# 5. ISIC parent-filtered path: A -> 01 -> 011 -> 0111
# ---------------------------------------------------------------------------

_ISIC_TABLE = {
    ("isic_rev4_sections", None): [("A", "Agriculture, Forestry and Fishing", "", 0.90)],
    ("isic_rev4_divisions", "A"): [("01", "Crop and animal production", "", 0.85)],
    ("isic_rev4_groups", "01"): [("011", "Growing of non-perennial crops", "", 0.80)],
    ("isic_rev4_classes", "011"): [("0111", "Growing of cereals, leguminous crops and oil seeds", "", 0.75)],
}


def _make_isic_store(table=None, existing=None):
    table = table if table is not None else _ISIC_TABLE
    existing = existing if existing is not None else set(ISIC_COLLECTIONS.values())
    client = FakeQdrantClient(table, existing)
    embedder = FakeEmbedder()
    store = StandardHierarchicalStore(
        standard="ISIC Rev.4", stages=isic_stages(), hitl_threshold=HITL_THRESHOLD,
        client=client, embedder=embedder,
    )
    return store, client, embedder


def test_isic_search_follows_real_parent_filtered_path():
    store, client, embedder = _make_isic_store()
    result = store.search("cereal farming")

    assert result.ready is True
    assert result.unavailable_reason == ""
    assert result.code == "0111"
    assert result.hierarchy_path == ["A", "01", "011", "0111"]

    # Every stage after the first must have queried with the previous stage's
    # chosen code as parent_code -- proof this is genuine parent-filtered
    # retrieval, not a flat lookup.
    assert client.calls == [
        ("isic_rev4_sections", None, 2),
        ("isic_rev4_divisions", "A", 2),
        ("isic_rev4_groups", "01", 2),
        ("isic_rev4_classes", "011", 5),
    ]
    assert embedder.calls == [["query: cereal farming"]]


def test_isic_search_confidence_is_weighted_by_stage():
    store, _, _ = _make_isic_store()
    result = store.search("cereal farming")
    expected = round(
        ISIC_STAGE_WEIGHTS[0] * 0.90 + ISIC_STAGE_WEIGHTS[1] * 0.85
        + ISIC_STAGE_WEIGHTS[2] * 0.80 + ISIC_STAGE_WEIGHTS[3] * 0.75,
        4,
    )
    assert result.confidence == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 6. ISCED-F parent-filtered path: 06 -> 061 -> 0613
# ---------------------------------------------------------------------------

_ISCEDF_TABLE = {
    ("iscedf2013_broad_fields", None): [("06", "Information and Communication Technologies", "", 0.92)],
    ("iscedf2013_narrow_fields", "06"): [("061", "Information and communication technologies", "", 0.88)],
    ("iscedf2013_detailed_fields", "061"): [("0613", "Software and applications development and analysis", "", 0.83)],
}


def _make_iscedf_store(table=None, existing=None):
    table = table if table is not None else _ISCEDF_TABLE
    existing = existing if existing is not None else set(ISCEDF_COLLECTIONS.values())
    client = FakeQdrantClient(table, existing)
    embedder = FakeEmbedder()
    store = StandardHierarchicalStore(
        standard="ISCED-F 2013", stages=iscedf_stages(), hitl_threshold=HITL_THRESHOLD,
        client=client, embedder=embedder,
    )
    return store, client, embedder


def test_iscedf_search_follows_real_parent_filtered_path():
    store, client, embedder = _make_iscedf_store()
    result = store.search("software developer degree")

    assert result.ready is True
    assert result.unavailable_reason == ""
    assert result.code == "0613"
    assert result.hierarchy_path == ["06", "061", "0613"]
    assert client.calls == [
        ("iscedf2013_broad_fields", None, 2),
        ("iscedf2013_narrow_fields", "06", 2),
        ("iscedf2013_detailed_fields", "061", 5),
    ]


def test_iscedf_search_confidence_is_weighted_by_stage():
    store, _, _ = _make_iscedf_store()
    result = store.search("software developer degree")
    expected = round(
        ISCEDF_STAGE_WEIGHTS[0] * 0.92 + ISCEDF_STAGE_WEIGHTS[1] * 0.88 + ISCEDF_STAGE_WEIGHTS[2] * 0.83,
        4,
    )
    assert result.confidence == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Readiness contract: missing collections / empty result / empty text
# ---------------------------------------------------------------------------

def test_missing_collections_never_attempts_search():
    client = FakeQdrantClient(_ISIC_TABLE, existing_collections=set())  # nothing built
    store = StandardHierarchicalStore(
        standard="ISIC Rev.4", stages=isic_stages(), client=client, embedder=FakeEmbedder(),
    )
    assert store.ready is False

    result = store.search("cereal farming")
    assert result.ready is False
    assert result.code == ""
    assert result.unavailable_reason != ""
    assert client.calls == []  # never even attempted a query


def test_ready_but_zero_hits_is_distinct_from_missing_collections():
    client = FakeQdrantClient({}, existing_collections=set(ISIC_COLLECTIONS.values()))
    store = StandardHierarchicalStore(
        standard="ISIC Rev.4", stages=isic_stages(), client=client, embedder=FakeEmbedder(),
    )
    assert store.ready is True

    result = store.search("cereal farming")
    assert result.ready is True          # collections existed
    assert result.code == ""             # but nothing was found
    assert result.unavailable_reason != ""


def test_empty_text_is_explicit_not_ready_search():
    store, _, embedder = _make_isic_store()
    result = store.search("   ")
    assert result.ready is True
    assert result.unavailable_reason == "empty input text"
    assert embedder.calls == []  # never embedded an empty query


# ---------------------------------------------------------------------------
# Factory DI contract: injecting client/embedder never touches the singleton
# ---------------------------------------------------------------------------

def test_factory_with_injected_args_returns_fresh_instance_each_time():
    client = FakeQdrantClient(_ISIC_TABLE, set(ISIC_COLLECTIONS.values()))
    embedder = FakeEmbedder()
    a = get_isic_hierarchical_store(client=client, embedder=embedder)
    b = get_isic_hierarchical_store(client=client, embedder=embedder)
    assert a is not b


def test_iscedf_factory_with_injected_args_returns_fresh_instance_each_time():
    client = FakeQdrantClient(_ISCEDF_TABLE, set(ISCEDF_COLLECTIONS.values()))
    embedder = FakeEmbedder()
    a = get_iscedf_hierarchical_store(client=client, embedder=embedder)
    b = get_iscedf_hierarchical_store(client=client, embedder=embedder)
    assert a is not b


# ---------------------------------------------------------------------------
# Operational resilience (Task 05.1): readiness-check / embedding /
# engine-search failures must never raise out of search() -- always an
# explicit, no-code StandardHierarchyResult instead.
# ---------------------------------------------------------------------------

class _RaisingGetCollectionsClient:
    """get_collections() raises -- simulates Qdrant being unreachable or
    erroring at readiness-check time. query_points() must never be called."""

    def __init__(self, exc=None):
        self._exc = exc or ConnectionError("Qdrant unreachable")
        self.query_points_called = False

    def get_collections(self):
        raise self._exc

    def query_points(self, *args, **kwargs):
        self.query_points_called = True
        raise AssertionError("query_points() must never be called when readiness check failed")


class _RaisingEmbedder:
    def __init__(self, exc=None):
        self._exc = exc or RuntimeError("embedding model failed to load")

    def encode(self, *args, **kwargs):
        raise self._exc


def test_readiness_check_failure_yields_explicit_unavailable_result_and_never_queries():
    """1. A fake client whose get_collections() raises yields an explicit
    unavailable result and never runs a hierarchy query."""
    client = _RaisingGetCollectionsClient()
    store = StandardHierarchicalStore(
        standard="ISIC Rev.4", stages=isic_stages(), client=client, embedder=FakeEmbedder(),
    )
    assert store.ready is False
    assert store._unavailable_reason != ""

    result = store.search("cereal farming")
    assert result.ready is False
    assert result.code == ""
    assert result.unavailable_reason != ""
    assert "qdrant" in result.unavailable_reason.lower() or "readiness" in result.unavailable_reason.lower()
    assert client.query_points_called is False


def test_readiness_check_failure_never_loads_embedding_model():
    """No SentenceTransformer construction attempted when readiness already
    failed and no embedder was injected -- verified via the internal
    _embedder attribute staying None (no real model, no crash)."""
    client = _RaisingGetCollectionsClient()
    store = StandardHierarchicalStore(standard="ISIC Rev.4", stages=isic_stages(), client=client)
    assert store.ready is False
    assert store._embedder is None


def test_readiness_check_failure_still_honours_injected_embedder():
    """Dependency injection keeps working even on the failure path -- an
    explicitly injected fake embedder is still stored (just never called,
    since search() short-circuits before reaching _embed_query)."""
    client = _RaisingGetCollectionsClient()
    embedder = FakeEmbedder()
    store = StandardHierarchicalStore(standard="ISIC Rev.4", stages=isic_stages(), client=client, embedder=embedder)
    assert store._embedder is embedder
    store.search("cereal farming")
    assert embedder.calls == []  # never reached -- readiness failure short-circuits first


def test_isic_classifier_hierarchical_mode_falls_back_on_readiness_failure(monkeypatch):
    """2. ISIC hierarchical mode falls back with the correct explicit label
    and metadata when the readiness check fails."""
    from backend.agents.classifier_methods import (
        ISIC_HIERARCHICAL_RETRIEVAL,
        ISIC_HIERARCHICAL_FALLBACK_KEYWORD,
        ISIC_HIERARCHICAL_FALLBACK_LLM,
    )
    from backend.agents.isic_classifier import ISICClassifier
    from unittest.mock import MagicMock, patch

    client = _RaisingGetCollectionsClient()
    store = StandardHierarchicalStore(
        standard="ISIC Rev.4", stages=isic_stages(), client=client, embedder=FakeEmbedder(),
    )
    monkeypatch.setattr(
        "backend.rag.standard_hierarchical_store.get_isic_hierarchical_store", lambda: store,
    )

    with patch("backend.agents.isic_classifier.get_llm", return_value=MagicMock()):
        clf = ISICClassifier()
    result = clf.classify("software developer tech startup app", method=ISIC_HIERARCHICAL_RETRIEVAL)

    assert result.method in (ISIC_HIERARCHICAL_FALLBACK_KEYWORD, ISIC_HIERARCHICAL_FALLBACK_LLM)
    assert result.fallback_used is True
    assert result.fallback_reason
    assert result.section != ""  # legacy pipeline genuinely ran


def test_isced_classifier_hierarchical_mode_falls_back_on_readiness_failure_and_keeps_level(monkeypatch):
    """3. ISCED-F hierarchical mode falls back with the correct explicit
    label and metadata when the readiness check fails, while retaining
    independently classified ISCED 2011 level output."""
    from backend.agents.classifier_methods import (
        ISCEDF_HIERARCHICAL_RETRIEVAL,
        ISCEDF_HIERARCHICAL_FALLBACK_KEYWORD,
    )
    from backend.agents.isced_classifier import ISCEDClassifier

    client = _RaisingGetCollectionsClient()
    store = StandardHierarchicalStore(
        standard="ISCED-F 2013", stages=iscedf_stages(), client=client, embedder=FakeEmbedder(),
    )
    monkeypatch.setattr(
        "backend.rag.standard_hierarchical_store.get_iscedf_hierarchical_store", lambda: store,
    )

    clf = ISCEDClassifier()
    result = clf.classify("Bachelor of Science BSc university", method=ISCEDF_HIERARCHICAL_RETRIEVAL)

    assert result.method == ISCEDF_HIERARCHICAL_FALLBACK_KEYWORD
    assert result.fallback_used is True
    assert result.fallback_reason
    assert result.level == 6  # independently classified, unaffected by the store failure


def test_embedder_failure_does_not_fabricate_result():
    """4a. An embedder failure does not fabricate a result and is explicitly
    surfaced to the fallback path."""
    client = FakeQdrantClient(_ISIC_TABLE, existing_collections=set(ISIC_COLLECTIONS.values()))
    store = StandardHierarchicalStore(
        standard="ISIC Rev.4", stages=isic_stages(), client=client, embedder=_RaisingEmbedder(),
    )
    assert store.ready is True  # collections were fine -- only embedding fails

    result = store.search("cereal farming")
    assert result.ready is True
    assert result.code == ""
    assert result.unavailable_reason != ""
    assert "embed" in result.unavailable_reason.lower()
    assert client.calls == []  # never reached a Qdrant query


def test_engine_search_failure_does_not_fabricate_result():
    """4b. An engine/search failure does not fabricate a result and is
    explicitly surfaced to the fallback path."""
    client = FakeQdrantClient(_ISIC_TABLE, existing_collections=set(ISIC_COLLECTIONS.values()))
    store = StandardHierarchicalStore(
        standard="ISIC Rev.4", stages=isic_stages(), client=client, embedder=FakeEmbedder(),
    )
    assert store.ready is True

    def _boom(*args, **kwargs):
        raise RuntimeError("unexpected engine failure")

    store._engine.search = _boom  # simulate a failure the engine itself doesn't already swallow

    result = store.search("cereal farming")
    assert result.ready is True
    assert result.code == ""
    assert result.unavailable_reason != ""
    assert "search" in result.unavailable_reason.lower()


def test_positive_path_parent_filtering_unaffected_by_resilience_changes():
    """5. Existing positive-path parent filtering still passes unchanged."""
    store, client, _ = _make_isic_store()
    result = store.search("cereal farming")
    assert result.ready is True
    assert result.unavailable_reason == ""
    assert result.code == "0111"
    assert result.hierarchy_path == ["A", "01", "011", "0111"]
    assert client.calls == [
        ("isic_rev4_sections", None, 2),
        ("isic_rev4_divisions", "A", 2),
        ("isic_rev4_groups", "01", 2),
        ("isic_rev4_classes", "011", 5),
    ]


# ---------------------------------------------------------------------------
# Model-initialization resilience (Task 05.2): SentenceTransformer(MODEL_NAME)
# construction is lazy and happens only inside the protected embedding
# boundary in search() -- a construction failure must never raise to the
# classifier, and must never be attempted at all when not `ready` or when a
# fake embedder was injected.
# ---------------------------------------------------------------------------

class _RaisingSentenceTransformer:
    """Stand-in for sentence_transformers.SentenceTransformer whose
    constructor raises -- simulates a model-initialization failure (e.g.
    missing/corrupt local cache, out-of-memory, etc.)."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("embedding model failed to initialize")


def test_model_construction_never_attempted_in_init_no_embedder_injected():
    """Constructing a ready store WITHOUT an injected embedder must not
    call SentenceTransformer at all -- proof construction is lazy."""
    import backend.rag.standard_hierarchical_store as store_mod

    calls = {"count": 0}

    class _CountingSentenceTransformer:
        def __init__(self, *args, **kwargs):
            calls["count"] += 1

        def encode(self, *args, **kwargs):
            import numpy as np
            return np.zeros((1, 384))

    original = store_mod.SentenceTransformer
    store_mod.SentenceTransformer = _CountingSentenceTransformer
    try:
        client = FakeQdrantClient(_ISIC_TABLE, set(ISIC_COLLECTIONS.values()))
        store = StandardHierarchicalStore(standard="ISIC Rev.4", stages=isic_stages(), client=client)
        assert calls["count"] == 0  # not constructed in __init__
        assert store._embedder is None

        store.search("cereal farming")
        assert calls["count"] == 1  # constructed lazily, on first real use
    finally:
        store_mod.SentenceTransformer = original


def test_model_construction_failure_yields_explicit_embedding_unavailable_result(monkeypatch):
    """1. When SentenceTransformer construction raises, a ready-store
    search() returns an explicit no-code embedding-unavailable result
    rather than raising."""
    import backend.rag.standard_hierarchical_store as store_mod
    monkeypatch.setattr(store_mod, "SentenceTransformer", _RaisingSentenceTransformer)

    client = FakeQdrantClient(_ISIC_TABLE, set(ISIC_COLLECTIONS.values()))
    store = StandardHierarchicalStore(standard="ISIC Rev.4", stages=isic_stages(), client=client)
    assert store.ready is True  # collections fine -- only model init fails

    result = store.search("cereal farming")
    assert result.ready is True
    assert result.code == ""
    assert result.unavailable_reason != ""
    assert "embed" in result.unavailable_reason.lower()
    assert client.calls == []  # never reached a Qdrant query


def test_isic_classifier_falls_back_on_model_construction_failure(monkeypatch):
    """2. ISICClassifier.classify(..., method=isic_hierarchical_retrieval)
    receives a model-construction failure as the correct explicitly
    labelled ISIC fallback result."""
    import backend.rag.standard_hierarchical_store as store_mod
    from backend.agents.classifier_methods import (
        ISIC_HIERARCHICAL_RETRIEVAL,
        ISIC_HIERARCHICAL_FALLBACK_KEYWORD,
        ISIC_HIERARCHICAL_FALLBACK_LLM,
    )
    from backend.agents.isic_classifier import ISICClassifier
    from unittest.mock import MagicMock, patch

    monkeypatch.setattr(store_mod, "SentenceTransformer", _RaisingSentenceTransformer)
    client = FakeQdrantClient(_ISIC_TABLE, set(ISIC_COLLECTIONS.values()))
    store = StandardHierarchicalStore(standard="ISIC Rev.4", stages=isic_stages(), client=client)
    monkeypatch.setattr(
        "backend.rag.standard_hierarchical_store.get_isic_hierarchical_store", lambda: store,
    )

    with patch("backend.agents.isic_classifier.get_llm", return_value=MagicMock()):
        clf = ISICClassifier()
    result = clf.classify("software developer tech startup app", method=ISIC_HIERARCHICAL_RETRIEVAL)

    assert result.method in (ISIC_HIERARCHICAL_FALLBACK_KEYWORD, ISIC_HIERARCHICAL_FALLBACK_LLM)
    assert result.fallback_used is True
    assert result.fallback_reason
    assert result.section != ""  # legacy pipeline genuinely ran


def test_isced_classifier_falls_back_on_model_construction_failure_and_keeps_level(monkeypatch):
    """3. ISCEDClassifier.classify(..., method=iscedf_hierarchical_retrieval)
    receives a model-construction failure as the correct explicitly
    labelled ISCED-F fallback result and retains independent ISCED 2011
    level classification."""
    import backend.rag.standard_hierarchical_store as store_mod
    from backend.agents.classifier_methods import (
        ISCEDF_HIERARCHICAL_RETRIEVAL,
        ISCEDF_HIERARCHICAL_FALLBACK_KEYWORD,
    )
    from backend.agents.isced_classifier import ISCEDClassifier

    monkeypatch.setattr(store_mod, "SentenceTransformer", _RaisingSentenceTransformer)
    client = FakeQdrantClient(_ISCEDF_TABLE, set(ISCEDF_COLLECTIONS.values()))
    store = StandardHierarchicalStore(standard="ISCED-F 2013", stages=iscedf_stages(), client=client)
    monkeypatch.setattr(
        "backend.rag.standard_hierarchical_store.get_iscedf_hierarchical_store", lambda: store,
    )

    clf = ISCEDClassifier()
    result = clf.classify("Bachelor of Science BSc university", method=ISCEDF_HIERARCHICAL_RETRIEVAL)

    assert result.method == ISCEDF_HIERARCHICAL_FALLBACK_KEYWORD
    assert result.fallback_used is True
    assert result.fallback_reason
    assert result.level == 6  # independently classified, unaffected by the model-init failure


def test_injected_embedder_bypasses_model_construction_even_if_it_would_fail(monkeypatch):
    """4. A fake injected embedder bypasses model construction entirely --
    proven by monkeypatching SentenceTransformer to always raise and
    confirming the injected-embedder positive path still succeeds."""
    import backend.rag.standard_hierarchical_store as store_mod
    monkeypatch.setattr(store_mod, "SentenceTransformer", _RaisingSentenceTransformer)

    store, client, embedder = _make_isic_store()  # embedder=FakeEmbedder() injected
    result = store.search("cereal farming")

    assert result.ready is True
    assert result.unavailable_reason == ""
    assert result.code == "0111"
    assert embedder.calls == [["query: cereal farming"]]
