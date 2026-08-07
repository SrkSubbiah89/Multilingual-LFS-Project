"""
eval/figure_exports/export_classifier_hierarchy.py

Exports ISCO-08's 4-stage hierarchy configuration (Conference I Reviewer #2
response, Section I: figure-support data exports) as clean CSV/JSON for a
classifier-hierarchy diagram -- NOT a screenshot, see
Documentation/Conference_I_Reviewer_2/FIGURE_DATA_EXPORT_GUIDE.md for how
to turn this into a vector PDF/SVG/TikZ figure.

Deliberately reconstructs the stage list from
backend.rag.hierarchical_store's module-level constants (_COL_MAJOR etc.,
_W1..._W4) rather than instantiating HierarchicalISCOStore itself --
instantiating that class connects to a live Qdrant instance
(QdrantClient(...).get_collections()), which this export script has no
need for and should not require just to describe the stage configuration.
If those constants ever change, this script's ISCO_STAGES list must be
updated to match (same single-source-of-truth risk as any other
constants-mirroring script; the alternative -- requiring live Qdrant for a
documentation export -- was judged worse).

Also exports the ISIC/ISCED-F hierarchical retrieval stage configuration
(Task 05: backend.rag.standard_hierarchical_store) as separate,
clearly-marked "implemented_unevaluated" entries -- real code, real
collection names/weights, but no accuracy measurement and no live Qdrant
collection built by this repository's own work -- so the exported diagram
data can show them as visually distinct from ISCO's implemented-and-
evaluated stages without either overstating or omitting them.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

from backend.agents.classifier_methods import (  # noqa: E402
    ISCEDF_HIERARCHICAL_RETRIEVAL,
    ISIC_HIERARCHICAL_RETRIEVAL,
)
from backend.rag.hierarchical_store import _COL_MAJOR, _COL_MINOR, _COL_SUBMAJOR, _COL_UNIT, _W1, _W2, _W3, _W4  # noqa: E402
from backend.rag.standard_hierarchical_store import (  # noqa: E402
    ISCEDF_COLLECTIONS,
    ISCEDF_STAGE_WEIGHTS,
    ISIC_COLLECTIONS,
    ISIC_STAGE_WEIGHTS,
)


@dataclass
class HierarchyStageRow:
    standard: str
    stage_index: int
    stage_name: str
    collection: str
    weight: float
    status: str  # "implemented" | "implemented_unevaluated"
    method_id: str = ""


ISCO_STAGES: list[HierarchyStageRow] = [
    HierarchyStageRow("ISCO-08", 1, "major", _COL_MAJOR, _W1, "implemented", "isco_hierarchical_rag"),
    HierarchyStageRow("ISCO-08", 2, "submajor", _COL_SUBMAJOR, _W2, "implemented", "isco_hierarchical_rag"),
    HierarchyStageRow("ISCO-08", 3, "minor", _COL_MINOR, _W3, "implemented", "isco_hierarchical_rag"),
    HierarchyStageRow("ISCO-08", 4, "unit", _COL_UNIT, _W4, "implemented", "isco_hierarchical_rag"),
]

# Real code, real collection names/weights (backend/rag/standard_hierarchical_store.py)
# -- but no accuracy measurement exists yet, and no live Qdrant collection
# has been built/populated by this repository's own work (a separate,
# explicit operator action; see build_standard_hierarchical_collections.py).
# method_id ties each row back to the authoritative constant in
# backend/agents/classifier_methods.py.
IMPLEMENTED_UNEVALUATED_STAGES: list[HierarchyStageRow] = [
    HierarchyStageRow("ISIC Rev.4", 1, "section", ISIC_COLLECTIONS["sections"], ISIC_STAGE_WEIGHTS[0], "implemented_unevaluated", ISIC_HIERARCHICAL_RETRIEVAL),
    HierarchyStageRow("ISIC Rev.4", 2, "division", ISIC_COLLECTIONS["divisions"], ISIC_STAGE_WEIGHTS[1], "implemented_unevaluated", ISIC_HIERARCHICAL_RETRIEVAL),
    HierarchyStageRow("ISIC Rev.4", 3, "group", ISIC_COLLECTIONS["groups"], ISIC_STAGE_WEIGHTS[2], "implemented_unevaluated", ISIC_HIERARCHICAL_RETRIEVAL),
    HierarchyStageRow("ISIC Rev.4", 4, "class", ISIC_COLLECTIONS["classes"], ISIC_STAGE_WEIGHTS[3], "implemented_unevaluated", ISIC_HIERARCHICAL_RETRIEVAL),
    HierarchyStageRow("ISCED-F 2013", 1, "broad", ISCEDF_COLLECTIONS["broad_fields"], ISCEDF_STAGE_WEIGHTS[0], "implemented_unevaluated", ISCEDF_HIERARCHICAL_RETRIEVAL),
    HierarchyStageRow("ISCED-F 2013", 2, "narrow", ISCEDF_COLLECTIONS["narrow_fields"], ISCEDF_STAGE_WEIGHTS[1], "implemented_unevaluated", ISCEDF_HIERARCHICAL_RETRIEVAL),
    HierarchyStageRow("ISCED-F 2013", 3, "detailed", ISCEDF_COLLECTIONS["detailed_fields"], ISCEDF_STAGE_WEIGHTS[2], "implemented_unevaluated", ISCEDF_HIERARCHICAL_RETRIEVAL),
]


def all_rows() -> list[HierarchyStageRow]:
    return ISCO_STAGES + IMPLEMENTED_UNEVALUATED_STAGES


def write_json(rows: list[HierarchyStageRow], path: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stages": [asdict(r) for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(rows: list[HierarchyStageRow], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["standard", "stage_index", "stage_name", "collection", "weight", "status", "method_id"])
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = all_rows()
    write_json(rows, args.out / "classifier_hierarchy.json")
    write_csv(rows, args.out / "classifier_hierarchy.csv")
    print(f"Wrote {len(rows)} stage row(s) to {args.out}")


if __name__ == "__main__":
    main()
