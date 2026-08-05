"""
eval/full130_access_guard.py

Shared, robust runtime guard against ever opening eval/test_set_full130.csv
through any of Python's common file-reading entry points. Used by
eval/pre_run_check.py and eval/validate_dev_set.py to make "full130 is
never read outside the manifest builder" a CHECKED runtime property, not
just a code-review promise or a single-entry-point patch that a different
read style could quietly bypass.

Why five entry points, not just builtins.open()
---------------------------------------------------------------------------
An earlier version of this guard patched only builtins.open(). That is NOT
sufficient: io.open is a SEPARATE name binding in the io module's own
namespace. CPython sets builtins.open = io.open once, at interpreter
startup -- but rebinding builtins.open afterwards (as any open()-patching
guard must do) does not touch io.open's own binding, so code that calls
io.open(...) directly (rather than the builtin open(...)) would sail
straight past a builtins.open-only guard. pathlib.Path.open()/.read_text()/
.read_bytes() are patched independently too, rather than trusting that
their internal implementation always delegates to io.open/builtins.open in
every supported Python version -- this guard does not want to depend on
that being true.

Usage
-----
    from eval.full130_access_guard import guard_against_full130_access

    with guard_against_full130_access():
        ...  # any code that must be proven to never open full130

The guard is OPT-IN (a context manager an execution path chooses to wrap
itself in), not a global patch applied on import. This is precisely why
eval/build_full130_leakage_manifest.py -- THE ONE script in this repo
authorised to read eval/test_set_full130.csv directly -- is exempt: it
simply never imports or activates this guard. See that script's own module
docstring for the authorisation rationale. Every other script that might
touch eval/test_set_full130.csv (eval/validate_dev_set.py,
eval/pre_run_check.py) must wrap its execution in this guard.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import pathlib

FORBIDDEN_PATTERN = "test_set_full130"


class Full130AccessBlocked(Exception):
    """Raised by guard_against_full130_access() if anything tries to open
    a path matching eval/test_set_full130.csv while the guard is active."""


def _path_str(candidate) -> str:
    return str(candidate).replace("\\", "/")


def _is_forbidden(candidate) -> bool:
    return FORBIDDEN_PATTERN in _path_str(candidate)


def _blocked(candidate) -> "Full130AccessBlocked":
    return Full130AccessBlocked(
        f"BLOCKED: attempted to access {_path_str(candidate)!r}, which matches the "
        f"protected full130 test-set filename pattern ({FORBIDDEN_PATTERN!r}). "
        f"eval/test_set_full130.csv must never be read outside "
        f"eval/build_full130_leakage_manifest.py (the sole authorised reader) -- use "
        f"eval/configs/full130_leakage_manifest.json for any overlap/leakage check instead."
    )


@contextlib.contextmanager
def guard_against_full130_access():
    """Patches (and unconditionally restores, including on exception) five
    independent file-reading entry points for the duration of the `with`
    block: builtins.open, io.open, pathlib.Path.open, pathlib.Path.
    read_text, pathlib.Path.read_bytes. Any call whose path argument
    contains "test_set_full130" raises Full130AccessBlocked immediately;
    every other path is passed through to the real implementation
    unchanged (e.g. eval/configs/full130_leakage_manifest.json reads
    normally inside this guard)."""
    original_builtins_open = builtins.open
    original_io_open = io.open
    original_path_open = pathlib.Path.open
    original_path_read_text = pathlib.Path.read_text
    original_path_read_bytes = pathlib.Path.read_bytes

    def guarded_builtins_open(file, *args, **kwargs):
        if _is_forbidden(file):
            raise _blocked(file)
        return original_builtins_open(file, *args, **kwargs)

    def guarded_io_open(file, *args, **kwargs):
        if _is_forbidden(file):
            raise _blocked(file)
        return original_io_open(file, *args, **kwargs)

    def guarded_path_open(self, *args, **kwargs):
        if _is_forbidden(self):
            raise _blocked(self)
        return original_path_open(self, *args, **kwargs)

    def guarded_path_read_text(self, *args, **kwargs):
        if _is_forbidden(self):
            raise _blocked(self)
        return original_path_read_text(self, *args, **kwargs)

    def guarded_path_read_bytes(self, *args, **kwargs):
        if _is_forbidden(self):
            raise _blocked(self)
        return original_path_read_bytes(self, *args, **kwargs)

    builtins.open = guarded_builtins_open
    io.open = guarded_io_open
    pathlib.Path.open = guarded_path_open
    pathlib.Path.read_text = guarded_path_read_text
    pathlib.Path.read_bytes = guarded_path_read_bytes
    try:
        yield
    finally:
        builtins.open = original_builtins_open
        io.open = original_io_open
        pathlib.Path.open = original_path_open
        pathlib.Path.read_text = original_path_read_text
        pathlib.Path.read_bytes = original_path_read_bytes
