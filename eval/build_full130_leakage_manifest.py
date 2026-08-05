"""
eval/build_full130_leakage_manifest.py

THE ONE authorised place in this repository that reads
eval/test_set_full130.csv directly. Every other script that needs to check
dev-set independence from full130 (eval/validate_dev_set.py,
eval/pre_run_check.py, eval/dev_sweep.py) reads ONLY the manifest this
script produces (eval/configs/full130_leakage_manifest.json) -- never the
raw file. See eval/dev_set_schema.md's "Independence requirements" section
and eval/PRE_RUN_B2_CHECKLIST.md's "full130 usage rule".

This is a manual, one-time (or rare-recompute) operation, NOT part of any
automated validation/CI path. Re-run it only when:
  - eval/test_set_full130.csv's content changes (should never happen --
    it's the frozen held-out confirmation set -- but if it ever legitimately
    does, e.g. a data-entry correction, the manifest must be rebuilt), or
  - eval/validate_dev_set.py's normalize_text() implementation changes
    (its source fingerprint is baked into the manifest -- see
    eval/validate_dev_set.py's check_manifest_normalization_integrity();
    every consumer fails closed if the live fingerprint no longer matches
    what's recorded here, rather than silently comparing hashes built by a
    normalization function that no longer exists).

The manifest contains ONLY:
  - case_ids (full130's case_id column values)
  - normalized_text_sha256 (sha256(normalize_text(input_text)) for each row)
  - normalization_version / normalization_fingerprint (see above)
  - counts and generation metadata

It NEVER contains gold_isco_4digit, raw input_text, or any other column --
this is a deliberate privacy/scope boundary, not an oversight: the manifest
must be safe to read by scripts that are themselves forbidden from ever
seeing full130's actual content.

Usage
-----
    python eval/build_full130_leakage_manifest.py

    # after review, if you intend to actually replace the checked-in file:
    python eval/build_full130_leakage_manifest.py --write

Without --write, this prints what WOULD be written (including a diff
against the existing manifest's case_id/hash counts) and exits without
touching eval/configs/full130_leakage_manifest.json -- so running this
script to double check the current manifest is still correct doesn't
accidentally overwrite it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse validate_dev_set.py's normalize_text()/NORMALIZATION_VERSION/
# compute_normalize_text_fingerprint() directly -- this script must hash
# full130's text with the EXACT SAME function every consumer will later
# compare against, not a separately-maintained copy that could drift.
import validate_dev_set as vds  # noqa: E402

_DEFAULT_DIR = Path(__file__).resolve().parent
_DEFAULT_FULL130 = _DEFAULT_DIR / "test_set_full130.csv"
_DEFAULT_MANIFEST_OUT = _DEFAULT_DIR / "configs" / "full130_leakage_manifest.json"


def build_manifest(full130_path: Path) -> dict:
    """Reads eval/test_set_full130.csv (the ONE permitted direct read of
    that file in this codebase) and returns the manifest dict. Does not
    write anything -- see main() for the --write gate."""
    with open(full130_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    case_ids = sorted({r["case_id"] for r in rows if r.get("case_id")})
    text_hashes = sorted({
        hashlib.sha256(vds.normalize_text(r["input_text"]).encode()).hexdigest()
        for r in rows if r.get("input_text")
    })

    return {
        "_description": (
            "Leakage-detection manifest for eval/test_set_full130.csv -- contains ONLY "
            "case_ids and sha256 hashes of normalized input_text, never gold labels or raw "
            "text. Built by eval/build_full130_leakage_manifest.py, the ONE script in this "
            "repo authorised to read eval/test_set_full130.csv directly, so every other "
            "consumer (eval/validate_dev_set.py, eval/pre_run_check.py, eval/dev_sweep.py) "
            "can verify dev-set independence WITHOUT ever opening that file."
        ),
        "_source_file": "eval/test_set_full130.csv",
        "_normalize_function": (
            "eval/validate_dev_set.py:normalize_text() -- "
            "re.sub(r'\\s+', ' ', s.strip().lower())"
        ),
        "_generated_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "normalization_version": vds.NORMALIZATION_VERSION,
        "normalization_fingerprint": vds.compute_normalize_text_fingerprint(),
        "n_rows": len(rows),
        "case_ids": case_ids,
        "normalized_text_sha256": text_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full130", type=Path, default=_DEFAULT_FULL130)
    parser.add_argument("--out", type=Path, default=_DEFAULT_MANIFEST_OUT)
    parser.add_argument("--write", action="store_true",
                         help="Actually overwrite --out. Without this flag, prints a summary "
                              "and exits without touching the checked-in manifest.")
    args = parser.parse_args()

    if not args.full130.exists():
        parser.error(f"full130 source not found: {args.full130}")

    manifest = build_manifest(args.full130)

    print(f"Read {manifest['n_rows']} row(s) from {args.full130}")
    print(f"  case_ids: {len(manifest['case_ids'])}")
    print(f"  normalized_text_sha256: {len(manifest['normalized_text_sha256'])}")
    print(f"  normalization_version: {manifest['normalization_version']}")
    print(f"  normalization_fingerprint: {manifest['normalization_fingerprint']}")

    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        if existing.get("case_ids") == manifest["case_ids"] and \
                existing.get("normalized_text_sha256") == manifest["normalized_text_sha256"] and \
                existing.get("normalization_fingerprint") == manifest["normalization_fingerprint"]:
            print(f"\n{args.out} is already up to date (case_ids, hashes, and "
                  f"normalization_fingerprint all match).")
        else:
            print(f"\n{args.out} DIFFERS from what would be generated now -- "
                  f"full130 content, this repo's normalize_text(), or both have changed "
                  f"since it was last built.")
    else:
        print(f"\n{args.out} does not exist yet.")

    if not args.write:
        print("\n(dry run -- pass --write to actually update the manifest file)")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
