"""
Tests for eval/full130_access_guard.py -- the shared, robust runtime guard
against opening eval/test_set_full130.csv through any of Python's common
file-reading entry points. See that module's docstring for why patching
builtins.open() alone is not sufficient (io.open is a separate binding).
"""

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import full130_access_guard as guard  # noqa: E402


DECOY_PATH_STR = "eval/test_set_full130.csv"


# ---------------------------------------------------------------------------
# Each of the five entry points, blocked individually
# ---------------------------------------------------------------------------

def test_builtins_open_is_blocked(tmp_path):
    decoy = tmp_path / "test_set_full130.csv"
    decoy.write_text("case_id,input_text\n1,x\n", encoding="utf-8")
    with guard.guard_against_full130_access():
        with pytest.raises(guard.Full130AccessBlocked):
            open(decoy, encoding="utf-8")


def test_io_open_is_blocked(tmp_path):
    """The regression this module exists to fix: io.open is a SEPARATE
    name binding from builtins.open, so a builtins.open-only guard would
    not catch this."""
    decoy = tmp_path / "test_set_full130.csv"
    decoy.write_text("case_id,input_text\n1,x\n", encoding="utf-8")
    with guard.guard_against_full130_access():
        with pytest.raises(guard.Full130AccessBlocked):
            io.open(decoy, encoding="utf-8")


def test_path_open_is_blocked(tmp_path):
    decoy = tmp_path / "test_set_full130.csv"
    decoy.write_text("case_id,input_text\n1,x\n", encoding="utf-8")
    with guard.guard_against_full130_access():
        with pytest.raises(guard.Full130AccessBlocked):
            decoy.open(encoding="utf-8")


def test_path_read_text_is_blocked(tmp_path):
    """The specific vector this task asked to prove is blocked."""
    decoy = tmp_path / "test_set_full130.csv"
    decoy.write_text("case_id,input_text\n1,x\n", encoding="utf-8")
    with guard.guard_against_full130_access():
        with pytest.raises(guard.Full130AccessBlocked):
            decoy.read_text(encoding="utf-8")


def test_path_read_bytes_is_blocked(tmp_path):
    decoy = tmp_path / "test_set_full130.csv"
    decoy.write_text("case_id,input_text\n1,x\n", encoding="utf-8")
    with guard.guard_against_full130_access():
        with pytest.raises(guard.Full130AccessBlocked):
            decoy.read_bytes()


def test_blocks_regardless_of_path_type_str_or_path(tmp_path):
    """The forbidden-pattern check must fire whether the caller passes a
    plain string or a pathlib.Path object."""
    decoy_str = str(tmp_path / "test_set_full130.csv")
    Path(decoy_str).write_text("x", encoding="utf-8")
    with guard.guard_against_full130_access():
        with pytest.raises(guard.Full130AccessBlocked):
            open(decoy_str, encoding="utf-8")  # plain str, not Path


def test_blocks_windows_backslash_paths(tmp_path):
    """Path normalisation (backslash -> forward slash) must not create a
    bypass on Windows."""
    windows_style = str(tmp_path) + "\\test_set_full130.csv"
    with guard.guard_against_full130_access():
        with pytest.raises(guard.Full130AccessBlocked):
            open(windows_style, encoding="utf-8")


# ---------------------------------------------------------------------------
# Unrelated paths remain readable through every entry point
# ---------------------------------------------------------------------------

def test_unrelated_path_readable_via_builtins_open(tmp_path):
    ok_file = tmp_path / "dev_set_v1.csv"
    ok_file.write_text("hello", encoding="utf-8")
    with guard.guard_against_full130_access():
        with open(ok_file, encoding="utf-8") as f:
            assert f.read() == "hello"


def test_unrelated_path_readable_via_io_open(tmp_path):
    ok_file = tmp_path / "dev_set_v1.csv"
    ok_file.write_text("hello", encoding="utf-8")
    with guard.guard_against_full130_access():
        with io.open(ok_file, encoding="utf-8") as f:
            assert f.read() == "hello"


def test_unrelated_path_readable_via_path_read_text(tmp_path):
    ok_file = tmp_path / "dev_set_v1.csv"
    ok_file.write_text("hello", encoding="utf-8")
    with guard.guard_against_full130_access():
        assert ok_file.read_text(encoding="utf-8") == "hello"


def test_unrelated_path_readable_via_path_read_bytes(tmp_path):
    ok_file = tmp_path / "dev_set_v1.csv"
    ok_file.write_bytes(b"hello")
    with guard.guard_against_full130_access():
        assert ok_file.read_bytes() == b"hello"


def test_unrelated_path_readable_via_path_open(tmp_path):
    ok_file = tmp_path / "dev_set_v1.csv"
    ok_file.write_text("hello", encoding="utf-8")
    with guard.guard_against_full130_access():
        with ok_file.open(encoding="utf-8") as f:
            assert f.read() == "hello"


def test_real_leakage_manifest_is_readable_inside_the_guard():
    manifest_path = Path(__file__).resolve().parent / "configs" / "full130_leakage_manifest.json"
    with guard.guard_against_full130_access():
        text = manifest_path.read_text(encoding="utf-8")
        assert "case_ids" in text


def test_real_b1_frozen_config_is_readable_inside_the_guard():
    config_path = Path(__file__).resolve().parent / "configs" / "b1_frozen.json"
    with guard.guard_against_full130_access():
        with open(config_path, encoding="utf-8") as f:
            assert f.read()


# ---------------------------------------------------------------------------
# Restoration: all five patches are undone on exit, including on exception
# ---------------------------------------------------------------------------

def test_all_five_patches_restored_after_normal_exit():
    import builtins
    import pathlib as pl

    originals = (builtins.open, io.open, pl.Path.open, pl.Path.read_text, pl.Path.read_bytes)
    with guard.guard_against_full130_access():
        pass
    afters = (builtins.open, io.open, pl.Path.open, pl.Path.read_text, pl.Path.read_bytes)
    assert originals == afters


def test_all_five_patches_restored_after_exception():
    import builtins
    import pathlib as pl

    originals = (builtins.open, io.open, pl.Path.open, pl.Path.read_text, pl.Path.read_bytes)
    try:
        with guard.guard_against_full130_access():
            raise ValueError("boom")
    except ValueError:
        pass
    afters = (builtins.open, io.open, pl.Path.open, pl.Path.read_text, pl.Path.read_bytes)
    assert originals == afters


def test_nested_guard_entry_exit_does_not_corrupt_state(tmp_path):
    """Two sequential (non-overlapping) uses of the guard must each
    restore state correctly -- no leaked patch from one use into the next."""
    ok_file = tmp_path / "dev_set_v1.csv"
    ok_file.write_text("hello", encoding="utf-8")

    with guard.guard_against_full130_access():
        assert ok_file.read_text(encoding="utf-8") == "hello"
    with guard.guard_against_full130_access():
        assert ok_file.read_text(encoding="utf-8") == "hello"

    # Outside any guard, a real full130-named file (if it existed) would
    # simply behave like a normal file -- confirm the guard doesn't leak.
    decoy = tmp_path / "test_set_full130.csv"
    decoy.write_text("unblocked outside the guard", encoding="utf-8")
    assert decoy.read_text(encoding="utf-8") == "unblocked outside the guard"


# ---------------------------------------------------------------------------
# Real execution paths (pre_run_check / validate_dev_set) never trigger it
# ---------------------------------------------------------------------------

def test_real_pre_run_check_execution_does_not_trigger_the_guard(tmp_path):
    import pre_run_check as prc

    dev_set = tmp_path / "dev_set_v1.csv"
    dev_set.write_text(
        "case_id,language,respondent_text,gold_isco_code,gold_label_source,"
        "annotator_or_adjudication_reference,dataset_split\n",
        encoding="utf-8",
    )
    real_baseline = Path(__file__).resolve().parent / "configs" / "b1_frozen.json"
    real_manifest = Path(__file__).resolve().parent / "configs" / "full130_leakage_manifest.json"
    real_smoke20 = Path(__file__).resolve().parent / "test_set_smoke20.csv"

    # Completing without Full130AccessBlocked propagating IS the proof.
    overall_ok, checklist = prc.run_pre_run_checks(
        dev_set, real_baseline, real_smoke20, real_manifest,
        confirm_inferred_beam=3, allow_dirty_tree=True,
    )
    assert isinstance(overall_ok, bool)


def test_real_validate_dev_set_main_execution_does_not_trigger_the_guard(tmp_path, monkeypatch):
    import validate_dev_set as vds

    dev_set = tmp_path / "dev_set_v1.csv"
    dev_set.write_text(
        "case_id,language,respondent_text,gold_isco_code,gold_label_source,"
        "annotator_or_adjudication_reference,dataset_split\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["validate_dev_set.py", "--dev-set", str(dev_set)])
    with pytest.raises(SystemExit) as exc_info:
        vds.main()
    # Exits 1 because the set is empty (0 rows) -- NOT because the guard fired
    # (which would raise Full130AccessBlocked, a different exception entirely).
    assert exc_info.value.code == 1


def test_pre_run_check_module_reexports_the_shared_guard():
    """pre_run_check.py no longer defines its own guard -- confirms it
    re-exports the shared implementation rather than silently reintroducing
    a second, divergent copy."""
    import pre_run_check as prc
    assert prc.guard_against_full130_access is guard.guard_against_full130_access
    assert prc.Full130AccessBlocked is guard.Full130AccessBlocked


def test_validate_dev_set_module_uses_the_shared_guard():
    import validate_dev_set as vds
    assert vds.guard_against_full130_access is guard.guard_against_full130_access
