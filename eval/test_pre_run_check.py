"""
Tests for eval/pre_run_check.py -- the B2 pre-run gate (task E). Covers the
G-list requirements from the audit task: confirmed/inferred/wrong-override
beam evidence, model-identity and fingerprint mismatches, dirty-tree
gating, duplicate dev IDs, smoke20/full130 overlap, the full130
non-access guarantee, and the no-inference/no-retrieval guarantee.
"""

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pre_run_check as prc  # noqa: E402
import dev_sweep as ds  # noqa: E402
import validate_dev_set as vds  # noqa: E402


REAL_BASELINE = Path(__file__).resolve().parent / "configs" / "b1_frozen.json"
REAL_MANIFEST = Path(__file__).resolve().parent / "configs" / "full130_leakage_manifest.json"
REAL_SMOKE20 = Path(__file__).resolve().parent / "test_set_smoke20.csv"


def write_dev_set(path: Path, rows: list):
    fieldnames = ["case_id", "language", "respondent_text", "gold_isco_code", "gold_label_source",
                  "annotator_or_adjudication_reference", "dataset_split"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def dev_row(case_id="dev001", language="en", respondent_text="baker", gold_isco_code="7512"):
    return {
        "case_id": case_id, "language": language, "respondent_text": respondent_text,
        "gold_isco_code": gold_isco_code, "gold_label_source": "human_coder_single",
        "annotator_or_adjudication_reference": "AB", "dataset_split": "dev_v1",
    }


def make_valid_dev_rows(n_en=20, n_ar=15):
    rows = []
    for i in range(n_en):
        rows.append(dev_row(case_id=f"dev_en_{i}", language="en",
                             respondent_text=f"unique english occupation description {i}"))
    for i in range(n_ar):
        rows.append(dev_row(case_id=f"dev_ar_{i}", language="ar",
                             respondent_text=f"unique arabic occupation description {i}"))
    return rows


# ---------------------------------------------------------------------------
# check_full130_manifest_overlap -- pure function, no file access
# ---------------------------------------------------------------------------

def test_manifest_overlap_detects_case_id_collision():
    manifest = {"case_ids": ["ar1", "ar2"], "normalized_text_sha256": []}
    dev_rows = [{"case_id": "ar1", "respondent_text": "something else entirely"}]
    ok, msg = prc.check_full130_manifest_overlap(dev_rows, manifest)
    assert ok is False
    assert "ar1" in msg


def test_manifest_overlap_detects_text_hash_collision():
    import hashlib
    text = "software developer building mobile apps"
    h = hashlib.sha256(vds.normalize_text(text).encode()).hexdigest()
    manifest = {"case_ids": [], "normalized_text_sha256": [h]}
    dev_rows = [{"case_id": "dev099", "respondent_text": text}]
    ok, msg = prc.check_full130_manifest_overlap(dev_rows, manifest)
    assert ok is False
    assert "dev099" in msg


def test_manifest_overlap_passes_when_no_collision():
    manifest = {"case_ids": ["ar1"], "normalized_text_sha256": ["deadbeef"]}
    dev_rows = [{"case_id": "dev001", "respondent_text": "completely unrelated text"}]
    ok, msg = prc.check_full130_manifest_overlap(dev_rows, manifest)
    assert ok is True


def test_manifest_overlap_against_real_manifest_no_collision_for_synthetic_rows():
    manifest = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    dev_rows = make_valid_dev_rows()
    ok, msg = prc.check_full130_manifest_overlap(dev_rows, manifest)
    assert ok is True


def test_manifest_overlap_against_real_manifest_detects_real_full130_case_id():
    manifest = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    real_id = manifest["case_ids"][0]
    dev_rows = [dev_row(case_id=real_id, respondent_text="does not matter for this check")]
    ok, msg = prc.check_full130_manifest_overlap(dev_rows, manifest)
    assert ok is False
    assert real_id in msg


# ---------------------------------------------------------------------------
# guard_against_full130_access -- the runtime non-access guarantee
# ---------------------------------------------------------------------------

def test_guard_blocks_opening_a_full130_named_path(tmp_path):
    decoy = tmp_path / "test_set_full130.csv"
    decoy.write_text("case_id,input_text\n1,x\n", encoding="utf-8")
    with prc.guard_against_full130_access():
        with pytest.raises(prc.Full130AccessBlocked):
            open(decoy, encoding="utf-8")


def test_guard_allows_opening_unrelated_paths(tmp_path):
    ok_file = tmp_path / "dev_set_v1.csv"
    ok_file.write_text("hello", encoding="utf-8")
    with prc.guard_against_full130_access():
        with open(ok_file, encoding="utf-8") as f:
            assert f.read() == "hello"


def test_guard_restores_original_open_after_exit(tmp_path):
    import builtins
    original = builtins.open
    with prc.guard_against_full130_access():
        pass
    assert builtins.open is original


def test_guard_restores_original_open_even_on_exception(tmp_path):
    import builtins
    original = builtins.open
    try:
        with prc.guard_against_full130_access():
            raise ValueError("boom")
    except ValueError:
        pass
    assert builtins.open is original


def test_real_run_pre_run_checks_never_trips_the_full130_guard(tmp_path):
    """The actual, real checklist run (against real config/manifest files)
    must never attempt to open eval/test_set_full130.csv -- if it did, the
    guard would raise Full130AccessBlocked and this test would fail with
    that exception instead of returning normally."""
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    assert isinstance(overall_ok, bool)  # completed without Full130AccessBlocked propagating


def test_run_pre_run_checks_fails_if_manifest_missing(tmp_path):
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, tmp_path / "does_not_exist.json",
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = {name: (ok, msg) for name, ok, msg in checklist}
    assert names["dev_set_no_overlap_with_full130"][0] is False


# ---------------------------------------------------------------------------
# Beam-provenance gate, via the full checklist (task G)
# ---------------------------------------------------------------------------

def test_checklist_confirmed_beam_evidence_passes(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline = json.loads(REAL_BASELINE.read_text(encoding="utf-8"))
    baseline["beam_evidence"] = {"status": "confirmed", "source": "test", "detail": "test"}
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())

    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, baseline_path, REAL_SMOKE20, REAL_MANIFEST, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["beam_provenance_gate"] is True


def test_checklist_inferred_beam_evidence_fails_without_override(tmp_path):
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=None, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["beam_provenance_gate"] is False
    assert overall_ok is False


def test_checklist_wrong_beam_override_fails(tmp_path):
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=99, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["beam_provenance_gate"] is False
    assert overall_ok is False


def test_checklist_correct_beam_override_passes_that_gate(tmp_path):
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["beam_provenance_gate"] is True


# ---------------------------------------------------------------------------
# Model identity / composite fingerprint mismatches, via the full checklist
# ---------------------------------------------------------------------------

def test_checklist_model_identity_mismatch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ds, "resolve_ollama_model_identity",
        lambda tag: {"status": "confirmed", "tag": tag, "digest": "totally-different-digest"},
    )
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["codebase_check:ollama_model_identity"] is False
    assert overall_ok is False


def test_checklist_composite_fingerprint_mismatch_fails(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline = json.loads(REAL_BASELINE.read_text(encoding="utf-8"))
    baseline["implementation_fingerprint"] = {
        "components": {"x": "0000000000000000"}, "composite_sha256": "0" * 64,
    }
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())

    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, baseline_path, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["codebase_check:implementation_fingerprint"] is False
    assert overall_ok is False


# ---------------------------------------------------------------------------
# Dirty tree gating, via the full checklist
# ---------------------------------------------------------------------------

def test_checklist_dirty_tree_fails_without_override(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "check_git_tree_clean", lambda: (False, " M some/file.py"))
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=False,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["git_tree_clean"] is False
    assert overall_ok is False


def test_checklist_dirty_tree_passes_with_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "check_git_tree_clean", lambda: (False, " M some/file.py"))
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    messages = dict((n, msg) for n, _, msg in checklist)
    assert names["git_tree_clean"] is True
    assert "override" in messages["git_tree_clean"].lower()


# ---------------------------------------------------------------------------
# Duplicate dev IDs / smoke20 overlap, via the full checklist
# ---------------------------------------------------------------------------

def test_checklist_duplicate_dev_ids_fail(tmp_path):
    rows = make_valid_dev_rows()
    rows[1]["case_id"] = rows[0]["case_id"]  # force a duplicate
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, rows)
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["dev_set_validation_vs_smoke20"] is False
    assert overall_ok is False


def test_checklist_smoke20_overlap_fails(tmp_path):
    smoke20_rows = vds.load_csv_rows(REAL_SMOKE20)
    assert smoke20_rows, "expected eval/test_set_smoke20.csv to have rows"
    colliding_case_id = smoke20_rows[0]["case_id"]

    rows = make_valid_dev_rows()
    rows[0]["case_id"] = colliding_case_id
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, rows)

    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["dev_set_validation_vs_smoke20"] is False
    assert overall_ok is False


def test_checklist_full130_overlap_fails(tmp_path):
    manifest = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    colliding_case_id = manifest["case_ids"][0]

    rows = make_valid_dev_rows()
    rows[0]["case_id"] = colliding_case_id
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, rows)

    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["dev_set_no_overlap_with_full130"] is False
    assert overall_ok is False


# ---------------------------------------------------------------------------
# No Ollama inference / Qdrant retrieval invoked (task G)
# ---------------------------------------------------------------------------

def test_pre_run_check_never_constructs_isco_classifier(tmp_path, monkeypatch):
    """Patches only __init__ (construction), not the class itself -- the
    composite-fingerprint check legitimately introspects
    ISCOClassifier._llm_select_from_candidates' SOURCE via inspect.getsource,
    which must keep working; what must never happen is an actual instance
    being built (which would mean real classify() calls become possible)."""
    def _blow_up(self, *a, **k):
        raise AssertionError("ISCOClassifier must never be constructed by pre_run_check.py")
    monkeypatch.setattr(ds.run_eval.ISCOClassifier, "__init__", _blow_up)

    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    assert isinstance(overall_ok, bool)  # completed without __init__ ever being called


def test_pre_run_check_never_calls_hierarchical_search(tmp_path, monkeypatch):
    def _blow_up(*a, **k):
        raise AssertionError("HierarchicalISCOStore._hierarchical_search must never be CALLED "
                              "by pre_run_check.py (its SOURCE may be introspected for hashing)")
    monkeypatch.setattr(ds.HierarchicalISCOStore, "search", _blow_up)

    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    assert isinstance(overall_ok, bool)


def test_pre_run_check_module_does_not_call_build_system_or_run_one_case():
    """Structural guarantee: pre_run_check.py's CODE (not its module
    docstring, which mentions these names in prose for documentation)
    never calls run_eval.build_system() or run_eval.run_one_case() -- the
    functions that actually perform retrieval/inference."""
    src = Path(prc.__file__).read_text(encoding="utf-8")
    # Strip the leading module docstring (delimited by the first pair of
    # triple-quotes) so prose mentions of these names don't false-positive.
    parts = src.split('"""', 2)
    code_only = parts[2] if len(parts) == 3 else src
    assert "build_system(" not in code_only
    assert "run_one_case(" not in code_only


def test_urllib_calls_during_checklist_only_hit_metadata_endpoints(tmp_path, monkeypatch):
    """Every urllib.request.urlopen call made anywhere during the checklist
    (Ollama /api/tags, /api/version; Qdrant /) must target a metadata
    endpoint -- never /api/generate, /api/chat, or a Qdrant search path."""
    import urllib.request as _ur
    real_urlopen = _ur.urlopen
    seen_urls = []

    def spy_urlopen(req, *a, **k):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        seen_urls.append(url)
        return real_urlopen(req, *a, **k)

    monkeypatch.setattr(_ur, "urlopen", spy_urlopen)

    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )

    forbidden = ("/api/generate", "/api/chat", "/points/search", "/collections")
    for url in seen_urls:
        assert not any(f in url for f in forbidden), f"unexpected non-metadata call: {url}"


# ---------------------------------------------------------------------------
# New checklist items from the validator-strengthening pass: semantic ISCO
# catalogue, normalization integrity, CSV structure
# ---------------------------------------------------------------------------

def test_checklist_includes_isco_catalogue_loaded_item(tmp_path):
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert "isco_catalogue_loaded" in names
    assert names["isco_catalogue_loaded"] is True  # real catalogue, 436 codes


def test_checklist_isco_catalogue_missing_is_a_hard_failure(tmp_path, monkeypatch):
    """Fix 3's explicit requirement: isco_catalogue_loaded must remain a
    hard failure (blocks overall_ok), not an advisory-only item."""
    monkeypatch.setattr(vds, "load_isco_unit_group_catalogue", lambda *a, **k: set())
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["isco_catalogue_loaded"] is False
    assert overall_ok is False


def test_checklist_includes_dev_set_header_schema_item(tmp_path):
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["dev_set_header_schema"] is True


def test_checklist_malformed_header_blocks_overall(tmp_path):
    dev_set = tmp_path / "dev_set_v1.csv"
    reordered = ["language", "case_id", "respondent_text", "gold_isco_code",
                 "gold_label_source", "annotator_or_adjudication_reference", "dataset_split"]
    dev_set.write_text(",".join(reordered) + "\n", encoding="utf-8")
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["dev_set_header_schema"] is False
    assert overall_ok is False


def test_checklist_canonical_header_only_template_passes_header_item(tmp_path):
    dev_set = tmp_path / "dev_set_v1.csv"
    dev_set.write_text(",".join(vds.CANONICAL_HEADER) + "\n", encoding="utf-8")
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["dev_set_header_schema"] is True
    # overall still fails, but purely downstream (0 rows), not the header
    assert names["dev_set_validation_vs_smoke20"] is False


def test_checklist_semantic_isco_rejection_flows_through(tmp_path):
    """A gold_isco_code like 9999 (syntactically valid, doesn't exist) must
    fail dev_set_validation_vs_smoke20 once the real catalogue is wired in."""
    rows = make_valid_dev_rows()
    rows[0]["gold_isco_code"] = "9999"
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, rows)
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    messages = dict((n, msg) for n, _, msg in checklist)
    assert names["dev_set_validation_vs_smoke20"] is False
    assert "9999" in messages["dev_set_validation_vs_smoke20"]
    assert overall_ok is False


def test_checklist_includes_full130_manifest_normalization_integrity_item(tmp_path):
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["full130_manifest_normalization_integrity"] is True


def test_checklist_normalization_integrity_failure_blocks_overall(tmp_path):
    import json
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(json.dumps({
        "case_ids": ["a"], "normalized_text_sha256": ["b"],
        "normalization_version": "v1", "normalization_fingerprint": "0" * 64,
    }), encoding="utf-8")
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, bad_manifest,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["full130_manifest_normalization_integrity"] is False
    assert overall_ok is False


def test_checklist_includes_dev_set_csv_structure_item(tmp_path):
    dev_set = tmp_path / "dev_set_v1.csv"
    write_dev_set(dev_set, make_valid_dev_rows())
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["dev_set_csv_structure"] is True


def test_checklist_malformed_csv_structure_blocks_overall(tmp_path):
    dev_set = tmp_path / "dev_set_v1.csv"
    dev_set.write_text(
        "case_id,language,respondent_text,gold_isco_code,gold_label_source,"
        "annotator_or_adjudication_reference,dataset_split\n"
        "dev001,en\n",  # ragged row -- far too few fields
        encoding="utf-8",
    )
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, REAL_BASELINE, REAL_SMOKE20, REAL_MANIFEST,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    names = dict((n, ok) for n, ok, _ in checklist)
    assert names["dev_set_csv_structure"] is False
    assert overall_ok is False
