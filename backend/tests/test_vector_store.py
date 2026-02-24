"""
Tests for backend/rag/vector_store.py

QdrantClient and SentenceTransformer are patched out so no running Qdrant
instance or model download is required.  All tests exercise the VectorStore
logic (collection management, embedding dispatch, search result mapping).
"""

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from backend.rag.vector_store import (
    COLLECTION_NAME,
    VECTOR_DIM,
    OccupationMatch,
    VectorStore,
    _stable_id,
    _ISCO_DATA,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_qdrant_client():
    """A QdrantClient mock that reports the collection already exists and is populated."""
    client = MagicMock()
    # Collection already exists
    existing = MagicMock()
    existing.name = COLLECTION_NAME
    client.get_collections.return_value = MagicMock(collections=[existing])
    # Collection is already populated → skip upsert
    client.count.return_value = MagicMock(count=len(_ISCO_DATA))
    return client


@pytest.fixture
def mock_model():
    """A SentenceTransformer mock whose encode() returns L2-normalised zero vectors."""
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.zeros((len(texts), VECTOR_DIM))
    return model


@pytest.fixture
def vs(monkeypatch, mock_qdrant_client, mock_model):
    """VectorStore with Qdrant and SentenceTransformer patched out."""
    monkeypatch.setattr(
        "backend.rag.vector_store.QdrantClient",
        lambda host, port: mock_qdrant_client,
    )
    monkeypatch.setattr(
        "backend.rag.vector_store.SentenceTransformer",
        lambda name: mock_model,
    )
    return VectorStore()


# ---------------------------------------------------------------------------
# _stable_id
# ---------------------------------------------------------------------------

class TestStableId:
    def test_same_code_gives_same_id(self):
        assert _stable_id("2512") == _stable_id("2512")

    def test_different_codes_give_different_ids(self):
        assert _stable_id("2512") != _stable_id("2511")

    def test_returns_non_negative_integer(self):
        assert _stable_id("1") >= 0

    def test_fits_in_uint64(self):
        # Qdrant uint64 max = 2^64 - 1
        assert _stable_id("9999") < 2**64

    def test_all_isco_codes_have_unique_ids(self):
        ids = [_stable_id(d["code"]) for d in _ISCO_DATA]
        assert len(ids) == len(set(ids)), "Collision in _stable_id across ISCO codes"


# ---------------------------------------------------------------------------
# _ensure_collection
# ---------------------------------------------------------------------------

class TestEnsureCollection:
    def test_skips_creation_when_collection_exists(self, vs, mock_qdrant_client):
        # Already exists — create_collection should NOT have been called
        mock_qdrant_client.create_collection.assert_not_called()

    def test_creates_collection_when_absent(self, monkeypatch, mock_model):
        client = MagicMock()
        client.get_collections.return_value = MagicMock(collections=[])
        client.count.return_value = MagicMock(count=len(_ISCO_DATA))  # pre-populated after create
        monkeypatch.setattr("backend.rag.vector_store.QdrantClient", lambda **kw: client)
        monkeypatch.setattr("backend.rag.vector_store.SentenceTransformer", lambda _: mock_model)
        VectorStore()
        client.create_collection.assert_called_once()

    def test_recreate_deletes_then_creates(self, monkeypatch, mock_model):
        client = MagicMock()
        existing = MagicMock()
        existing.name = COLLECTION_NAME
        client.get_collections.return_value = MagicMock(collections=[existing])
        client.count.return_value = MagicMock(count=len(_ISCO_DATA))
        monkeypatch.setattr("backend.rag.vector_store.QdrantClient", lambda **kw: client)
        monkeypatch.setattr("backend.rag.vector_store.SentenceTransformer", lambda _: mock_model)
        VectorStore(recreate=True)
        client.delete_collection.assert_called_once_with(COLLECTION_NAME)
        client.create_collection.assert_called_once()


# ---------------------------------------------------------------------------
# _ensure_populated
# ---------------------------------------------------------------------------

class TestEnsurePopulated:
    def test_skips_upsert_when_already_populated(self, vs, mock_qdrant_client):
        mock_qdrant_client.upsert.assert_not_called()

    def test_upserts_when_collection_is_empty(self, monkeypatch, mock_model):
        client = MagicMock()
        existing = MagicMock()
        existing.name = COLLECTION_NAME
        client.get_collections.return_value = MagicMock(collections=[existing])
        client.count.return_value = MagicMock(count=0)   # empty → trigger upsert
        monkeypatch.setattr("backend.rag.vector_store.QdrantClient", lambda **kw: client)
        monkeypatch.setattr("backend.rag.vector_store.SentenceTransformer", lambda _: mock_model)
        VectorStore()
        assert client.upsert.called

    def test_upserts_all_isco_entries(self, monkeypatch, mock_model):
        client = MagicMock()
        client.get_collections.return_value = MagicMock(collections=[])
        client.count.return_value = MagicMock(count=0)
        monkeypatch.setattr("backend.rag.vector_store.QdrantClient", lambda **kw: client)
        monkeypatch.setattr("backend.rag.vector_store.SentenceTransformer", lambda _: mock_model)
        VectorStore()
        total_upserted = sum(
            len(c.kwargs.get("points", c.args[1] if len(c.args) > 1 else []))
            for c in client.upsert.call_args_list
        )
        assert total_upserted == len(_ISCO_DATA)

    def test_embed_called_with_all_isco_texts(self, monkeypatch, mock_model):
        client = MagicMock()
        client.get_collections.return_value = MagicMock(collections=[])
        client.count.return_value = MagicMock(count=0)
        monkeypatch.setattr("backend.rag.vector_store.QdrantClient", lambda **kw: client)
        monkeypatch.setattr("backend.rag.vector_store.SentenceTransformer", lambda _: mock_model)
        VectorStore()
        total_encoded = sum(
            len(c.args[0]) for c in mock_model.encode.call_args_list
        )
        assert total_encoded == len(_ISCO_DATA)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def _make_hit(code="2512", title_en="Software Developers",
              title_ar="مطورو البرمجيات", score=0.85):
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "code":        code,
        "title_en":    title_en,
        "title_ar":    title_ar,
        "level":       4,
        "description": "Design, develop and maintain software applications",
    }
    return hit


class TestSearch:
    def test_empty_query_returns_empty_list(self, vs, mock_qdrant_client):
        results = vs.search("")
        assert results == []
        mock_qdrant_client.search.assert_not_called()

    def test_whitespace_query_returns_empty_list(self, vs, mock_qdrant_client):
        results = vs.search("   ")
        assert results == []

    def test_returns_occupation_match_objects(self, vs, mock_qdrant_client):
        mock_qdrant_client.search.return_value = [_make_hit()]
        results = vs.search("software developer")
        assert len(results) == 1
        assert isinstance(results[0], OccupationMatch)

    def test_result_fields_match_payload(self, vs, mock_qdrant_client):
        mock_qdrant_client.search.return_value = [
            _make_hit(code="2512", title_en="Software Developers", score=0.85)
        ]
        result = vs.search("developer")[0]
        assert result.code     == "2512"
        assert result.title_en == "Software Developers"
        assert result.confidence == pytest.approx(0.85, abs=1e-4)

    def test_confidence_clamped_below_zero(self, vs, mock_qdrant_client):
        mock_qdrant_client.search.return_value = [_make_hit(score=-0.1)]
        result = vs.search("query")[0]
        assert result.confidence == 0.0

    def test_confidence_clamped_above_one(self, vs, mock_qdrant_client):
        mock_qdrant_client.search.return_value = [_make_hit(score=1.1)]
        result = vs.search("query")[0]
        assert result.confidence == 1.0

    def test_confidence_rounded_to_4_decimal_places(self, vs, mock_qdrant_client):
        mock_qdrant_client.search.return_value = [_make_hit(score=0.123456789)]
        result = vs.search("query")[0]
        assert result.confidence == pytest.approx(0.1235, abs=1e-4)

    def test_top_k_passed_to_qdrant(self, vs, mock_qdrant_client):
        mock_qdrant_client.search.return_value = []
        vs.search("nurse", top_k=3)
        _, call_kwargs = mock_qdrant_client.search.call_args
        assert call_kwargs.get("limit") == 3

    def test_default_top_k_is_five(self, vs, mock_qdrant_client):
        mock_qdrant_client.search.return_value = []
        vs.search("doctor")
        _, call_kwargs = mock_qdrant_client.search.call_args
        assert call_kwargs.get("limit") == 5

    def test_multiple_results_returned(self, vs, mock_qdrant_client):
        mock_qdrant_client.search.return_value = [
            _make_hit("2512", score=0.90),
            _make_hit("2511", score=0.80),
            _make_hit("2513", score=0.70),
        ]
        results = vs.search("developer", top_k=3)
        assert len(results) == 3

    def test_query_prefixed_with_query_tag(self, vs, mock_qdrant_client, mock_model):
        mock_qdrant_client.search.return_value = []
        vs.search("nurse")
        encode_call_texts = mock_model.encode.call_args[0][0]
        assert encode_call_texts[0].startswith("query: ")


# ---------------------------------------------------------------------------
# ISCO dataset integrity
# ---------------------------------------------------------------------------

class TestIscoDataIntegrity:
    def test_all_entries_have_required_keys(self):
        required = {"code", "level", "title_en", "title_ar", "description"}
        for entry in _ISCO_DATA:
            assert required.issubset(entry.keys()), f"Missing keys in entry: {entry}"

    def test_levels_are_valid(self):
        for entry in _ISCO_DATA:
            assert entry["level"] in (1, 2, 3, 4), f"Invalid level: {entry}"

    def test_codes_are_non_empty_strings(self):
        for entry in _ISCO_DATA:
            assert isinstance(entry["code"], str) and entry["code"]

    def test_all_codes_unique(self):
        codes = [d["code"] for d in _ISCO_DATA]
        assert len(codes) == len(set(codes)), "Duplicate ISCO codes in dataset"

    def test_dataset_has_all_major_groups(self):
        major_codes = {d["code"] for d in _ISCO_DATA if d["level"] == 1}
        for code in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            assert code in major_codes, f"Major group {code} missing from dataset"

    def test_dataset_contains_unit_groups(self):
        unit_groups = [d for d in _ISCO_DATA if d["level"] == 4]
        assert len(unit_groups) >= 30, "Expected at least 30 unit-group entries"
