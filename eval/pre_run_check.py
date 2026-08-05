"""
eval/pre_run_check.py

B2 PRE-RUN GATE (task E). Standalone checklist script that must PASS before
eval/dev_sweep.py is ever invoked for real. Runs every check dev_sweep.py
itself enforces (baseline schema, beam-provenance gate, Ollama model
identity, composite implementation fingerprint, git provenance/clean tree)
PLUS dev-set leakage-safety validation, and prints a clear PASS/FAIL
checklist -- so a human can see exactly what's wrong in one place before
spending any wall-clock time on a live sweep.

This script performs NO classification, NO retrieval, and NO LLM calls.
It never imports/constructs ISCOClassifier or HierarchicalISCOStore, and
never calls run_eval.build_system()/run_one_case(). The only network calls
made anywhere in this file are metadata-only (Ollama GET /api/tags, GET
/api/version -- via eval.dev_sweep's existing functions; Qdrant GET / via
eval.dev_sweep.resolve_qdrant_version()) -- never /api/generate, /api/chat,
or a Qdrant /collections/*/points/search call.

full130 is never opened, read, or used here (task E). Dev-set overlap
against eval/test_set_full130.csv is checked via
eval/configs/full130_leakage_manifest.json -- a small, pre-built manifest
containing ONLY case_ids and sha256 hashes of normalized input_text (no
gold labels, no raw text) extracted from full130 once, offline (see that
file's own _description). This script additionally wraps its entire check
sequence in guard_against_full130_access(), a runtime monkeypatch of
builtins.open() that raises immediately if anything -- including a future
edit that reintroduces a direct full130 read -- ever tries to open a path
matching eval/test_set_full130.csv. This is a genuine, testable guarantee,
not just a code-review promise.

Usage
-----
    python eval/pre_run_check.py \\
        --baseline-config eval/configs/b1_frozen.json \\
        --dev-set eval/dev_set_v1.csv

    # If beam_evidence.status is "inferred" (see eval/configs/b1_frozen.json):
    python eval/pre_run_check.py \\
        --baseline-config eval/configs/b1_frozen.json \\
        --dev-set eval/dev_set_v1.csv \\
        --confirm-inferred-beam 3

Exit codes
----------
0   Every check passed. Safe to run eval/dev_sweep.py with the same
    --baseline-config/--dev-set/--confirm-inferred-beam/--allow-dirty-tree
    arguments.
1   At least one check failed. Do not run eval/dev_sweep.py until fixed.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dev_sweep as ds  # noqa: E402
import validate_dev_set as vds  # noqa: E402

_DEFAULT_SMOKE20 = Path(__file__).resolve().parent / "test_set_smoke20.csv"
_DEFAULT_FULL130_MANIFEST = Path(__file__).resolve().parent / "configs" / "full130_leakage_manifest.json"
_FULL130_FILENAME_PATTERN = "test_set_full130"


class Full130AccessBlocked(Exception):
    """Raised by guard_against_full130_access() if anything tries to open a
    path matching eval/test_set_full130.csv while the guard is active."""


@contextlib.contextmanager
def guard_against_full130_access():
    """Monkeypatches builtins.open() (which io.open()/Path.open()/
    Path.read_text()/Path.read_bytes() all funnel through in CPython) to
    raise Full130AccessBlocked immediately if the path contains
    "test_set_full130". Restores the original open() on exit, including on
    exception. This is what makes 'full130 path is never read by pre-run
    check' a checked runtime property instead of just a code-review claim."""
    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        path_str = str(file).replace("\\", "/")
        if _FULL130_FILENAME_PATTERN in path_str:
            raise Full130AccessBlocked(
                f"BLOCKED: attempted to open {path_str!r}, which matches the protected "
                f"full130 test-set filename pattern ({_FULL130_FILENAME_PATTERN!r}). "
                f"eval/pre_run_check.py must never open eval/test_set_full130.csv -- use "
                f"eval/configs/full130_leakage_manifest.json for overlap checks instead."
            )
        return original_open(file, *args, **kwargs)

    builtins.open = guarded_open
    try:
        yield
    finally:
        builtins.open = original_open


def check_full130_manifest_overlap(dev_rows: list, manifest: dict) -> tuple:
    """Verifies dev_rows (dev_set_v1.csv schema: case_id, respondent_text)
    don't collide with full130 -- via the pre-built manifest's case_ids and
    normalized-text sha256 hashes ONLY. Never touches
    eval/test_set_full130.csv itself. Returns (ok, message)."""
    manifest_ids = set(manifest.get("case_ids", []))
    manifest_hashes = set(manifest.get("normalized_text_sha256", []))

    id_collisions = []
    text_collisions = []
    for row in dev_rows:
        cid = row.get("case_id", "")
        if cid and cid in manifest_ids:
            id_collisions.append(cid)
        text = row.get("respondent_text", "")
        if text:
            h = hashlib.sha256(vds.normalize_text(text).encode()).hexdigest()
            if h in manifest_hashes:
                text_collisions.append(cid)

    if id_collisions or text_collisions:
        return False, (
            f"{len(id_collisions)} case_id collision(s) with full130 (via manifest): "
            f"{id_collisions}; {len(text_collisions)} respondent_text collision(s) with "
            f"full130 (via manifest, offending dev case_ids): {text_collisions}"
        )
    return True, (
        f"No case_id or normalized-text collisions with full130's {len(manifest_ids)} "
        f"case(s) (checked via manifest only -- full130 itself was never opened)"
    )


def run_pre_run_checks(
    dev_set_path: Path,
    baseline_config_path: Path,
    smoke20_path: Path = _DEFAULT_SMOKE20,
    full130_manifest_path: Path = _DEFAULT_FULL130_MANIFEST,
    confirm_inferred_beam: int = None,
    allow_dirty_tree: bool = False,
) -> tuple:
    """Runs the full B2 pre-run checklist, wrapped end-to-end in
    guard_against_full130_access(). Returns (overall_ok: bool, checklist:
    list[(name, ok, message)]) in the order checks ran. Short-circuits (does
    not run later checks) only when an earlier check makes later ones
    meaningless to attempt (missing files, malformed JSON, bad schema)."""
    checklist = []

    def record(name, ok, message):
        checklist.append((name, ok, message))
        return ok

    with guard_against_full130_access():
        if not dev_set_path.exists():
            record("dev_set_exists", False, f"Dev set not found: {dev_set_path}")
            return False, checklist
        record("dev_set_exists", True, str(dev_set_path))

        if not baseline_config_path.exists():
            record("baseline_config_exists", False, f"Baseline config not found: {baseline_config_path}")
            return False, checklist
        record("baseline_config_exists", True, str(baseline_config_path))

        try:
            baseline = ds.load_baseline_config(baseline_config_path)
        except (json.JSONDecodeError, OSError) as exc:
            record("baseline_config_parses", False, f"{type(exc).__name__}: {exc}")
            return False, checklist
        record("baseline_config_parses", True, "valid JSON")

        shape_errors = ds.validate_baseline_shape(baseline)
        shape_ok = record(
            "baseline_config_schema", not shape_errors,
            "; ".join(shape_errors) if shape_errors else "schema OK",
        )
        if not shape_ok:
            return False, checklist  # remaining checks assume a well-shaped baseline

        for name, check in ds.BASELINE_CODEBASE_CHECKS:
            ok, msg = check(baseline)
            record(f"codebase_check:{name}", ok, msg)

        beam_ok, beam_msg = ds.check_beam_evidence(baseline, confirm_inferred_beam)
        record("beam_provenance_gate", beam_ok, beam_msg)

        git_clean, git_dirty = ds.check_git_tree_clean()
        if git_clean:
            record("git_tree_clean", True, "clean")
        elif allow_dirty_tree:
            record("git_tree_clean", True, f"DIRTY but --allow-dirty-tree override in effect: {git_dirty}")
        else:
            record("git_tree_clean", False, f"dirty (pass --allow-dirty-tree to override): {git_dirty}")

        structure_errors = vds.validate_csv_structure(dev_set_path)
        record(
            "dev_set_csv_structure", not structure_errors,
            "; ".join(structure_errors) if structure_errors else "well-formed CSV",
        )
        if structure_errors:
            return False, checklist  # ragged/malformed rows make further parsing unreliable

        valid_isco_codes = vds.load_isco_unit_group_catalogue()
        record(
            "isco_catalogue_loaded", bool(valid_isco_codes),
            f"{len(valid_isco_codes)} unit-group code(s) loaded from "
            f"backend/rag/load_full_isco.py" if valid_isco_codes else
            "could not load the ISCO-08 catalogue -- semantic gold_isco_code validation "
            "will be skipped (format-only 4-digit validation still applies)",
        )

        manifest = {}
        if full130_manifest_path.exists():
            manifest = json.loads(full130_manifest_path.read_text(encoding="utf-8"))
            norm_ok, norm_msg = vds.check_manifest_normalization_integrity(manifest)
            record("full130_manifest_normalization_integrity", norm_ok, norm_msg)
        else:
            record(
                "full130_manifest_normalization_integrity", False,
                f"full130 leakage manifest not found: {full130_manifest_path} -- cannot verify "
                f"(this is NOT resolved by reading eval/test_set_full130.csv directly)",
            )

        dev_rows = vds.load_csv_rows(dev_set_path)
        smoke20_rows = vds.load_csv_rows(smoke20_path)
        smoke20_ids = vds.load_case_ids(smoke20_rows)
        smoke20_texts = vds.load_normalized_texts(smoke20_rows, "input_text")
        dev_report = vds.validate_dev_set(
            dev_rows, smoke20_ids, smoke20_texts, valid_isco_codes=valid_isco_codes,
        )
        dev_msg = f"{len(dev_report.errors)} error(s)"
        if dev_report.errors:
            dev_msg += ": " + "; ".join(dev_report.errors)
        if dev_report.warnings:
            dev_msg += f" ({len(dev_report.warnings)} warning(s): " + "; ".join(dev_report.warnings) + ")"
        record("dev_set_validation_vs_smoke20", dev_report.ok, dev_msg)

        if manifest:
            overlap_ok, overlap_msg = check_full130_manifest_overlap(dev_rows, manifest)
            record("dev_set_no_overlap_with_full130", overlap_ok, overlap_msg)
        else:
            record(
                "dev_set_no_overlap_with_full130", False,
                f"full130 leakage manifest not found or unusable: {full130_manifest_path} -- "
                f"cannot verify (this is NOT resolved by reading eval/test_set_full130.csv "
                f"directly)",
            )

    overall_ok = all(ok for _, ok, _ in checklist)
    return overall_ok, checklist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--dev-set", required=True, type=Path)
    parser.add_argument("--confirm-inferred-beam", type=int, default=None)
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--smoke20", type=Path, default=_DEFAULT_SMOKE20)
    parser.add_argument(
        "--full130-manifest", type=Path, default=_DEFAULT_FULL130_MANIFEST,
        help=(
            "Leakage-detection manifest (case_ids + normalized-text sha256 hashes only, no "
            "labels) -- NOT the full130 test set itself, which this script never opens."
        ),
    )
    args = parser.parse_args()

    overall_ok, checklist = run_pre_run_checks(
        args.dev_set, args.baseline_config, args.smoke20, args.full130_manifest,
        args.confirm_inferred_beam, args.allow_dirty_tree,
    )

    print("=" * 78)
    print("B2 PRE-RUN CHECKLIST")
    print("=" * 78)
    for name, ok, message in checklist:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {message}")
    print("=" * 78)
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    if overall_ok:
        print(
            "\nSafe to run eval/dev_sweep.py with the same --baseline-config/--dev-set"
            + (f"/--confirm-inferred-beam {args.confirm_inferred_beam}" if args.confirm_inferred_beam is not None else "")
            + (" /--allow-dirty-tree" if args.allow_dirty_tree else "")
            + "."
        )
    else:
        print("\nDo NOT run eval/dev_sweep.py until every check above passes.")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
