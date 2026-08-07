"""
eval/figure_exports/export_agent_role_diagram.py

Exports the classifier method registry (backend/agents/method_registry.py,
Conference I Reviewer #2 response, Section A) as a nodes/edges CSV/JSON
pair for an agent-role diagram (Section I, reviewer comment 6: multiple
LLM roles unclear -- this is real, computed data, not a hand-drawn guess
at what the system's components are).

nodes  : one row per component (LanguageProcessor, ISCOClassifier, ...)
edges  : one row per (component, method) -- carries category/model so the
         diagram can style edges by whether a method is deterministic/
         retrieval/llm/hybrid. Every REGISTRY row describes real, tested
         code as of Task 05 (the last "not yet implemented" stub -- ISIC/
         ISCED-F hierarchical retrieval -- was replaced with a genuine
         implementation); ``is_implemented`` is kept as an explicit,
         always-True field so the diagram schema doesn't silently change
         shape if a future stub is ever added again.

See Documentation/Conference_I_Reviewer_2/FIGURE_DATA_EXPORT_GUIDE.md for
how to turn this into a vector PDF/SVG/TikZ diagram -- no screenshots.
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

from backend.agents.method_registry import REGISTRY  # noqa: E402


@dataclass
class NodeRow:
    component: str
    method_count: int
    has_llm_method: bool
    affects_hitl_escalation: bool


@dataclass
class EdgeRow:
    component: str
    method_id: str
    category: str
    model_name: str
    is_implemented: bool


def build_nodes() -> list[NodeRow]:
    by_component: dict[str, list] = {}
    for entry in REGISTRY:
        by_component.setdefault(entry.component, []).append(entry)
    nodes = []
    for component, entries in by_component.items():
        nodes.append(NodeRow(
            component=component,
            method_count=len(entries),
            has_llm_method=any(e.category in ("llm", "hybrid") for e in entries),
            affects_hitl_escalation=any(e.affects_hitl_escalation for e in entries),
        ))
    return nodes


def build_edges() -> list[EdgeRow]:
    return [
        EdgeRow(
            component=e.component, method_id=e.method_id, category=e.category,
            model_name=e.model_name or "", is_implemented=True,
        )
        for e in REGISTRY
    ]


def write_json(nodes: list[NodeRow], edges: list[EdgeRow], path: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(rows: list, path: Path, fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    nodes = build_nodes()
    edges = build_edges()
    write_json(nodes, edges, args.out / "agent_role_diagram.json")
    write_csv(nodes, args.out / "agent_role_diagram_nodes.csv", ["component", "method_count", "has_llm_method", "affects_hitl_escalation"])
    write_csv(edges, args.out / "agent_role_diagram_edges.csv", ["component", "method_id", "category", "model_name", "is_implemented"])
    print(f"Wrote {len(nodes)} node(s) and {len(edges)} edge(s) to {args.out}")


if __name__ == "__main__":
    main()
