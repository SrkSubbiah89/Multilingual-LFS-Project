"""
Tests for backend/rag/build_official_isco08_collections.py (Task 21 dry
run + Task 22 guarded live execution).

Fully hermetic: small synthetic catalogue/metadata fixtures, a
hand-built FakeQdrantClient (no live Qdrant), and a FakeEmbedder (no
real SentenceTransformer). qdrant_client.models (Distance/PointStruct/
VectorParams) are plain data containers from the already-installed
qdrant_client package -- constructing them performs no I/O and no
network call, so using the real classes here (with a fake *client*)
keeps this file hermetic. No test in this file ever calls execute_build()
without injecting fakes for both qdrant_client_factory and
embedder_factory.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.rag.build_official_isco08_collections as builder  # noqa: E402
import backend.rag.official_isco08_catalogue as oic  # noqa: E402


_FIELDS = ["level", "code", "parent_code", "label"]
_SMALL_ROWS = [
    ("major", "1", "", "Managers"),
    ("submajor", "11", "1", "Chief Executives"),
    ("submajor", "12", "1", "Administrative Managers"),
    ("minor", "111", "11", "Legislators and Senior Officials"),
    ("minor", "121", "12", "Business Services Managers"),
    ("unit", "1111", "111", "Legislators"),
    ("unit", "1112", "111", "Senior Government Officials"),
    ("unit", "1211", "121", "Finance Managers"),
]
_SMALL_EXPECTED_COUNTS = {"major": 1, "submajor": 2, "minor": 2, "unit": 3}
_VALID_CONFIRM = oic.DEFAULT_PROFILE


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[tuple]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(_FIELDS)
        for r in rows:
            w.writerow(r)
    return path


def _write_metadata(path: Path, catalogue_sha256: str, counts: dict) -> Path:
    path.write_text(
        yaml.safe_dump({"isco08": {"normalized_catalogue_sha256": catalogue_sha256, "verified_counts": counts}}),
        encoding="utf-8",
    )
    return path


def _valid_fixture(tmp_path: Path):
    csv_path = _write_csv(tmp_path / "cat.csv", _SMALL_ROWS)
    meta_path = _write_metadata(tmp_path / "meta.yaml", _sha256(csv_path), _SMALL_EXPECTED_COUNTS)
    return csv_path, meta_path


_ALL_TARGET_NAMES = {
    "isco08_major_groups_ilo2021_v1", "isco08_submajor_groups_ilo2021_v1",
    "isco08_minor_groups_ilo2021_v1", "isco08_unit_groups_ilo2021_v1",
    "isco08_unit_groups_flat_ilo2021_v1",
}
_LEGACY_NAMES = {"isco08_major_groups", "isco08_submajor_groups", "isco08_minor_groups",
                  "isco08_unit_groups", "isco_occupations"}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeQdrantClient:
    def __init__(self, existing_collections=()):
        self.existing_collections = set(existing_collections)
        self.collections_data: dict[str, list] = {}
        self.created_order: list[str] = []

    def get_collections(self):
        names = self.existing_collections | set(self.collections_data)
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in names])

    def create_collection(self, collection_name, vectors_config):
        if collection_name in self.collections_data or collection_name in self.existing_collections:
            raise RuntimeError(f"collection {collection_name!r} already exists")
        self.collections_data[collection_name] = []
        self.created_order.append(collection_name)

    def upsert(self, collection_name, points):
        self.collections_data[collection_name].extend(points)

    def count(self, collection_name, exact=True):
        return SimpleNamespace(count=len(self.collections_data.get(collection_name, [])))

    def scroll(self, collection_name, limit, with_payload=True):
        pts = self.collections_data.get(collection_name, [])
        return list(pts[:limit]), None


class _RaisingOnUpsertClient(FakeQdrantClient):
    """Fails on the upsert into the 3rd target (index 2), simulating a
    mid-build failure after two collections were already created."""

    def upsert(self, collection_name, points):
        if collection_name == "isco08_minor_groups_ilo2021_v1":
            raise TimeoutError("simulated Qdrant write failure")
        super().upsert(collection_name, points)


class FakeEmbedder:
    def __init__(self):
        self.encoded_texts: list[list[str]] = []

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32):
        import numpy as np
        self.encoded_texts.append(list(texts))
        return np.zeros((len(texts), 384))


def _fakes(existing=()):
    client = FakeQdrantClient(existing)
    embedder = FakeEmbedder()
    return client, embedder


def _client_factory(client):
    return lambda host, port: client


def _embedder_factory(embedder):
    return lambda: embedder


# ---------------------------------------------------------------------------
# 1. Dry run validates source and has no Qdrant/embedder import or
#    construction
# ---------------------------------------------------------------------------

def test_dry_run_plan_counts_and_names(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    plan = builder.build_plan(csv_path, meta_path, profile=oic.DEFAULT_PROFILE, expected_counts=_SMALL_EXPECTED_COUNTS)

    by_level = {e.level: e for e in plan.entries if e.name != "isco08_unit_groups_flat_ilo2021_v1"}
    assert by_level["major"].record_count == 1
    assert by_level["submajor"].record_count == 2
    assert by_level["minor"].record_count == 2
    assert by_level["unit"].record_count == 3

    names = {e.name for e in plan.entries}
    assert names == _ALL_TARGET_NAMES
    flat_entry = next(e for e in plan.entries if e.name == "isco08_unit_groups_flat_ilo2021_v1")
    assert flat_entry.record_count == 3
    assert flat_entry.collection_role == "unit_flat"

    plan_dict = builder.plan_to_dict(plan)
    assert plan_dict["profile"] == oic.DEFAULT_PROFILE
    assert len(plan_dict["entries"]) == 5


def test_unknown_profile_rejected(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    with pytest.raises(builder.UnknownOfficialProfileError, match="no registered collection-name mapping"):
        builder.build_plan(csv_path, meta_path, profile="not_a_real_profile", expected_counts=_SMALL_EXPECTED_COUNTS)


def test_dry_run_never_constructs_qdrant_or_embedder_object(tmp_path, monkeypatch):
    csv_path, meta_path = _valid_fixture(tmp_path)

    def _boom(*args, **kwargs):
        raise AssertionError("Qdrant/embedder must never be constructed during a dry-run")

    monkeypatch.setattr(builder, "_default_qdrant_client_factory", _boom)
    monkeypatch.setattr(builder, "_default_embedder_factory", _boom)

    plan = builder.build_plan(csv_path, meta_path, profile=oic.DEFAULT_PROFILE, expected_counts=_SMALL_EXPECTED_COUNTS)
    assert len(plan.entries) == 5


def test_dry_run_module_top_level_has_no_qdrant_or_embedder_import():
    """Only TOP-LEVEL (module-body) import statements matter here -- the
    lazy imports inside _default_qdrant_client_factory()/
    _default_embedder_factory()/_create_and_populate_collection() are
    deliberately nested inside function bodies (Task 22's whole design),
    so they must be excluded from this check, not flagged by it."""
    source = Path(builder.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = ("qdrant", "sentence_transformers")
    for node in tree.body:  # top-level only, NOT ast.walk()
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(b in alias.name.lower() for b in banned), f"unexpected top-level import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").lower()
            assert not any(b in mod for b in banned), f"unexpected top-level import from: {node.module}"


def test_lazy_imports_are_confined_inside_function_bodies():
    """Positive control for the test above: confirms the qdrant/embedder
    imports DO exist somewhere in the file (so the top-level-only check
    isn't vacuously passing because the feature was deleted), but only
    ever inside a FunctionDef, never at module level."""
    source = Path(builder.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    def _contains_banned_import(node) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Import) and any("qdrant" in a.name.lower() or "sentence_transformers" in a.name.lower() for a in child.names):
                return True
            if isinstance(child, ast.ImportFrom) and child.module and ("qdrant" in child.module.lower() or "sentence_transformers" in child.module.lower()):
                return True
        return False

    functions_with_lazy_import = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and _contains_banned_import(node)
    ]
    assert set(functions_with_lazy_import) >= {"_default_qdrant_client_factory", "_default_embedder_factory", "_create_and_populate_collection"}


# ---------------------------------------------------------------------------
# 2. Execute missing either acknowledgement fails before dependency
#    construction
# ---------------------------------------------------------------------------

def test_execute_missing_confirm_profile_fails_before_dependency_construction(tmp_path, monkeypatch):
    csv_path, meta_path = _valid_fixture(tmp_path)
    boom = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not construct dependency"))
    with pytest.raises(builder.BuildAcknowledgementError, match="confirm-profile"):
        builder.execute_build(
            csv_path, meta_path, oic.DEFAULT_PROFILE,
            confirm_profile="", allow_local_qdrant_mutation=True,
            output_manifest_path=tmp_path / "manifest.json",
            expected_counts=_SMALL_EXPECTED_COUNTS,
            qdrant_client_factory=boom, embedder_factory=boom,
        )
    assert not (tmp_path / "manifest.json").exists()


def test_execute_missing_allow_mutation_fails_before_dependency_construction(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    boom = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not construct dependency"))
    with pytest.raises(builder.BuildAcknowledgementError, match="allow-local-qdrant-mutation"):
        builder.execute_build(
            csv_path, meta_path, oic.DEFAULT_PROFILE,
            confirm_profile=_VALID_CONFIRM, allow_local_qdrant_mutation=False,
            output_manifest_path=tmp_path / "manifest.json",
            expected_counts=_SMALL_EXPECTED_COUNTS,
            qdrant_client_factory=boom, embedder_factory=boom,
        )
    assert not (tmp_path / "manifest.json").exists()


# ---------------------------------------------------------------------------
# 3. Execute with wrong profile fails before dependency construction
# ---------------------------------------------------------------------------

def test_execute_wrong_confirm_profile_string_rejected(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    boom = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not construct dependency"))
    with pytest.raises(builder.BuildAcknowledgementError, match="official_ilo2021_v1"):
        builder.execute_build(
            csv_path, meta_path, oic.DEFAULT_PROFILE,
            confirm_profile="some_other_profile", allow_local_qdrant_mutation=True,
            output_manifest_path=tmp_path / "manifest.json",
            expected_counts=_SMALL_EXPECTED_COUNTS,
            qdrant_client_factory=boom, embedder_factory=boom,
        )


def test_execute_profile_confirm_profile_mismatch_rejected(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    boom = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not construct dependency"))
    with pytest.raises(builder.BuildAcknowledgementError, match="must exactly equal"):
        builder.execute_build(
            csv_path, meta_path, "some_other_profile",
            confirm_profile=_VALID_CONFIRM, allow_local_qdrant_mutation=True,
            output_manifest_path=tmp_path / "manifest.json",
            expected_counts=_SMALL_EXPECTED_COUNTS,
            qdrant_client_factory=boom, embedder_factory=boom,
        )


# ---------------------------------------------------------------------------
# 4. Invalid/missing/hash-mismatched catalogue fails before dependency
#    construction, even with valid acknowledgements
# ---------------------------------------------------------------------------

def test_execute_hash_mismatch_fails_before_dependency_construction(tmp_path):
    csv_path, _meta = _valid_fixture(tmp_path)
    bad_meta = _write_metadata(tmp_path / "bad_meta.yaml", "0" * 64, _SMALL_EXPECTED_COUNTS)
    boom = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not construct dependency"))
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="sha256 mismatch"):
        builder.execute_build(
            csv_path, bad_meta, oic.DEFAULT_PROFILE,
            confirm_profile=_VALID_CONFIRM, allow_local_qdrant_mutation=True,
            output_manifest_path=tmp_path / "manifest.json",
            expected_counts=_SMALL_EXPECTED_COUNTS,
            qdrant_client_factory=boom, embedder_factory=boom,
        )


def test_execute_missing_catalogue_fails_before_dependency_construction(tmp_path):
    _csv, meta_path = _valid_fixture(tmp_path)
    boom = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not construct dependency"))
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="not found"):
        builder.execute_build(
            tmp_path / "does_not_exist.csv", meta_path, oic.DEFAULT_PROFILE,
            confirm_profile=_VALID_CONFIRM, allow_local_qdrant_mutation=True,
            output_manifest_path=tmp_path / "manifest.json",
            expected_counts=_SMALL_EXPECTED_COUNTS,
            qdrant_client_factory=boom, embedder_factory=boom,
        )


# ---------------------------------------------------------------------------
# 5 & 16. Target names are exactly the five versioned official names;
#          legacy collection names are never targets
# ---------------------------------------------------------------------------

def test_target_builds_use_only_official_versioned_names(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    records = oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)
    by_level = oic.records_by_level(records)
    builds = builder.target_builds_for_profile(by_level, oic.DEFAULT_PROFILE)
    names = {b.name for b in builds}
    assert names == _ALL_TARGET_NAMES
    assert not (names & _LEGACY_NAMES)


# ---------------------------------------------------------------------------
# 6. Non-absent target collection causes a preflight failure before any
#    create/write
# ---------------------------------------------------------------------------

def test_existing_target_collection_blocks_before_any_mutation(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    client, embedder = _fakes(existing=("isco08_minor_groups_ilo2021_v1",))
    with pytest.raises(builder.BuildPreflightError, match="isco08_minor_groups_ilo2021_v1"):
        builder.execute_build(
            csv_path, meta_path, oic.DEFAULT_PROFILE,
            confirm_profile=_VALID_CONFIRM, allow_local_qdrant_mutation=True,
            output_manifest_path=tmp_path / "manifest.json",
            expected_counts=_SMALL_EXPECTED_COUNTS,
            qdrant_client_factory=_client_factory(client), embedder_factory=_embedder_factory(embedder),
        )
    assert client.created_order == []
    assert client.collections_data == {}
    manifest = __import__("json").loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "preflight_failed_existing_target"
    assert manifest["targets_created_or_partial"] == []


# ---------------------------------------------------------------------------
# 7 & 8. All five collections planned before the first mutation;
#         deterministic code/collection ordering
# ---------------------------------------------------------------------------

def test_all_five_targets_planned_before_first_mutation(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    records = oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)
    by_level = oic.records_by_level(records)
    builds = builder.target_builds_for_profile(by_level, oic.DEFAULT_PROFILE)
    assert len(builds) == 5  # entire plan derived before any client/embedder is touched


def test_collection_and_code_ordering_is_deterministic(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    records = oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)
    by_level = oic.records_by_level(records)
    b1 = builder.target_builds_for_profile(by_level, oic.DEFAULT_PROFILE)
    b2 = builder.target_builds_for_profile(by_level, oic.DEFAULT_PROFILE)
    assert [t.name for t in b1] == [t.name for t in b2]
    assert [t.name for t in b1] == [
        "isco08_major_groups_ilo2021_v1", "isco08_submajor_groups_ilo2021_v1",
        "isco08_minor_groups_ilo2021_v1", "isco08_unit_groups_ilo2021_v1",
        "isco08_unit_groups_flat_ilo2021_v1",
    ]
    for t1, t2 in zip(b1, b2):
        assert [r.code for r in t1.records] == [r.code for r in t2.records]


# ---------------------------------------------------------------------------
# 9 & 10 & 11. Level-specific counts, flat 4-digit-only rule, payload
#              schema, deterministic embedding text
# ---------------------------------------------------------------------------

def test_successful_execute_creates_verifies_and_writes_success_manifest(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    client, embedder = _fakes()
    manifest = builder.execute_build(
        csv_path, meta_path, oic.DEFAULT_PROFILE,
        confirm_profile=_VALID_CONFIRM, allow_local_qdrant_mutation=True,
        output_manifest_path=tmp_path / "manifest.json",
        expected_counts=_SMALL_EXPECTED_COUNTS,
        qdrant_client_factory=_client_factory(client), embedder_factory=_embedder_factory(embedder),
    )
    assert manifest["status"] == "success"
    assert client.created_order == [
        "isco08_major_groups_ilo2021_v1", "isco08_submajor_groups_ilo2021_v1",
        "isco08_minor_groups_ilo2021_v1", "isco08_unit_groups_ilo2021_v1",
        "isco08_unit_groups_flat_ilo2021_v1",
    ]
    assert len(client.collections_data["isco08_major_groups_ilo2021_v1"]) == 1
    assert len(client.collections_data["isco08_submajor_groups_ilo2021_v1"]) == 2
    assert len(client.collections_data["isco08_minor_groups_ilo2021_v1"]) == 2
    assert len(client.collections_data["isco08_unit_groups_ilo2021_v1"]) == 3
    assert len(client.collections_data["isco08_unit_groups_flat_ilo2021_v1"]) == 3

    flat_points = client.collections_data["isco08_unit_groups_flat_ilo2021_v1"]
    for p in flat_points:
        assert builder._ISCO4_RE.match(p.payload["code"])
        # Payload schema: all required identity/provenance fields present.
        for key in ("code", "level", "parent_code", "title_en", "profile",
                    "source_catalogue_sha256", "collection_role", "embedding_text"):
            assert key in p.payload
        assert p.payload["collection_role"] == "unit_flat"
        assert p.payload["level"] == "unit"
        assert p.payload["profile"] == oic.DEFAULT_PROFILE

    unit_hier_points = client.collections_data["isco08_unit_groups_ilo2021_v1"]
    assert {p.payload["code"] for p in unit_hier_points} == {p.payload["code"] for p in flat_points}
    assert {p.payload["collection_role"] for p in unit_hier_points} == {"unit_hierarchical"}

    assert manifest["embedding_model_identity"] == builder._EMBEDDING_MODEL_NAME
    assert manifest["payload_schema_version"] == builder._PAYLOAD_SCHEMA_VERSION
    assert len(manifest["verified_targets"]) == 5
    assert all(t["verified"] for t in manifest["verified_targets"])

    # Embedding text is deterministic and derived only from the record's
    # own embedding_text -- never a WISCO/benchmark label.
    all_texts = [t for batch in embedder.encoded_texts for t in batch]
    assert all(t.startswith("passage: ") for t in all_texts)
    assert any("Legislators" in t for t in all_texts)


def test_success_manifest_written_to_disk(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    client, embedder = _fakes()
    manifest_path = tmp_path / "out" / "manifest.json"
    builder.execute_build(
        csv_path, meta_path, oic.DEFAULT_PROFILE,
        confirm_profile=_VALID_CONFIRM, allow_local_qdrant_mutation=True,
        output_manifest_path=manifest_path,
        expected_counts=_SMALL_EXPECTED_COUNTS,
        qdrant_client_factory=_client_factory(client), embedder_factory=_embedder_factory(embedder),
    )
    assert manifest_path.exists()
    written = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    assert written["status"] == "success"
    assert "build_timestamp_utc" in written
    assert "builder_script_sha256" in written
    assert written["catalogue_sha256"] == _sha256(csv_path)


# ---------------------------------------------------------------------------
# 13 & 14. Partial failure: nonzero, failure manifest, no auto-deletion,
#           failed verification is not marked success
# ---------------------------------------------------------------------------

def test_partial_failure_writes_failure_manifest_and_raises(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    client = _RaisingOnUpsertClient()
    embedder = FakeEmbedder()
    manifest_path = tmp_path / "manifest.json"

    with pytest.raises(builder.BuildExecutionError, match="simulated Qdrant write failure"):
        builder.execute_build(
            csv_path, meta_path, oic.DEFAULT_PROFILE,
            confirm_profile=_VALID_CONFIRM, allow_local_qdrant_mutation=True,
            output_manifest_path=manifest_path,
            expected_counts=_SMALL_EXPECTED_COUNTS,
            qdrant_client_factory=_client_factory(client), embedder_factory=_embedder_factory(embedder),
        )

    # First two targets (major, submajor) were created; the third (minor)
    # failed on upsert -- nothing was deleted, nothing beyond that point
    # was attempted.
    assert client.created_order == ["isco08_major_groups_ilo2021_v1", "isco08_submajor_groups_ilo2021_v1", "isco08_minor_groups_ilo2021_v1"]
    assert "isco08_major_groups_ilo2021_v1" in client.collections_data
    assert "isco08_unit_groups_flat_ilo2021_v1" not in client.collections_data

    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed_partial_build"
    assert manifest["failure_reason"]
    assert "isco08_minor_groups_ilo2021_v1" in manifest["targets_created_or_partial"]
    assert "remediation_note" in manifest
    assert "separate" in manifest["remediation_note"].lower()


def test_failed_verification_is_not_marked_success(tmp_path):
    """A collection whose written payload doesn't match the plan (e.g. a
    hypothetical corrupted write) must raise and produce a failure
    manifest, never a success one."""
    csv_path, meta_path = _valid_fixture(tmp_path)

    class _CorruptingClient(FakeQdrantClient):
        def upsert(self, collection_name, points):
            if collection_name == "isco08_major_groups_ilo2021_v1" and points:
                points[0].payload["code"] = "9"  # corrupt the code post-hoc
            super().upsert(collection_name, points)

    client = _CorruptingClient()
    embedder = FakeEmbedder()
    manifest_path = tmp_path / "manifest.json"

    with pytest.raises(builder.BuildExecutionError, match="unexpected code"):
        builder.execute_build(
            csv_path, meta_path, oic.DEFAULT_PROFILE,
            confirm_profile=_VALID_CONFIRM, allow_local_qdrant_mutation=True,
            output_manifest_path=manifest_path,
            expected_counts=_SMALL_EXPECTED_COUNTS,
            qdrant_client_factory=_client_factory(client), embedder_factory=_embedder_factory(embedder),
        )
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed_partial_build"


# ---------------------------------------------------------------------------
# 15. Builder source remains free of WISCO/evaluation/classifier imports
# ---------------------------------------------------------------------------

def test_builder_module_has_no_wisco_evaluation_or_classifier_imports():
    source = Path(builder.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned_substrings = ("wisco", "isco_classifier", "isic_classifier", "isced_classifier",
                          "semantic_relation", "crewai")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(b in alias.name.lower() for b in banned_substrings), f"unexpected import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").lower()
            assert not any(b in mod for b in banned_substrings), f"unexpected import from: {node.module}"


# ---------------------------------------------------------------------------
# 17. No remote Qdrant URL/token configuration is accepted
# ---------------------------------------------------------------------------

def test_cli_has_no_remote_url_or_token_option():
    parser_source = Path(builder.__file__).read_text(encoding="utf-8")
    for banned in ("--qdrant-url", "--url", "--token", "--api-key", "--remote"):
        assert banned not in parser_source


def test_resolve_local_target_reads_only_host_and_port_env_vars(monkeypatch):
    monkeypatch.delenv("QDRANT_HOST", raising=False)
    monkeypatch.delenv("QDRANT_PORT", raising=False)
    host, port = builder._resolve_local_qdrant_target()
    assert host == "localhost"
    assert port == 6333

    monkeypatch.setenv("QDRANT_HOST", "127.0.0.1")
    monkeypatch.setenv("QDRANT_PORT", "7000")
    host, port = builder._resolve_local_qdrant_target()
    assert host == "127.0.0.1"
    assert port == 7000


# ---------------------------------------------------------------------------
# CLI-level acknowledgement/refusal tests
# ---------------------------------------------------------------------------

def test_cli_execute_without_output_manifest_errors(tmp_path, monkeypatch):
    csv_path, meta_path = _valid_fixture(tmp_path)
    argv = ["build_official_isco08_collections.py", "--catalogue", str(csv_path),
            "--metadata", str(meta_path), "--execute",
            "--confirm-profile", _VALID_CONFIRM, "--allow-local-qdrant-mutation"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        builder.main()


def test_cli_execute_missing_acknowledgement_errors_before_construction(tmp_path, monkeypatch, capsys):
    csv_path, meta_path = _valid_fixture(tmp_path)
    boom = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not construct"))
    monkeypatch.setattr(builder, "_default_qdrant_client_factory", boom)
    monkeypatch.setattr(builder, "_default_embedder_factory", boom)
    argv = ["build_official_isco08_collections.py", "--catalogue", str(csv_path),
            "--metadata", str(meta_path), "--execute", "--output-manifest", str(tmp_path / "m.json")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        builder.main()
    assert exc_info.value.code != 0
    out = capsys.readouterr().out
    assert "ACKNOWLEDGEMENT REQUIRED" in out


def test_cli_dry_run_flag_wins_even_with_execute(tmp_path, monkeypatch, capsys):
    """--dry-run alongside --execute must take the dry-run code path --
    proven by never touching the (boom-rigged) dependency factories, even
    though the CLI's dry-run path validates against the real production
    10/43/130/436 counts (not this test's small fixture) and therefore
    still exits nonzero for an unrelated reason (count mismatch)."""
    csv_path, meta_path = _valid_fixture(tmp_path)
    boom = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not construct -- --dry-run must win"))
    monkeypatch.setattr(builder, "_default_qdrant_client_factory", boom)
    monkeypatch.setattr(builder, "_default_embedder_factory", boom)
    argv = ["build_official_isco08_collections.py", "--catalogue", str(csv_path),
            "--metadata", str(meta_path), "--execute", "--dry-run",
            "--confirm-profile", _VALID_CONFIRM, "--allow-local-qdrant-mutation"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        builder.main()  # exits via the dry-run validation-failure path, never touching `boom`
    out = capsys.readouterr().out
    assert "CATALOGUE/PROFILE VALIDATION FAILURE" in out
    assert "ACKNOWLEDGEMENT REQUIRED" not in out
    assert "BUILD EXECUTION FAILURE" not in out


def test_cli_reports_validation_failure_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    rows = list(_SMALL_ROWS)
    rows[0] = ("major", "1", "", "")
    csv_path = _write_csv(tmp_path / "cat_bad2.csv", rows)
    meta_path = _write_metadata(tmp_path / "meta_bad2.yaml", _sha256(csv_path), _SMALL_EXPECTED_COUNTS)

    argv = ["build_official_isco08_collections.py", "--catalogue", str(csv_path), "--metadata", str(meta_path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        builder.main()
    assert exc_info.value.code != 0
    out = capsys.readouterr().out
    assert "CATALOGUE/PROFILE VALIDATION FAILURE" in out
