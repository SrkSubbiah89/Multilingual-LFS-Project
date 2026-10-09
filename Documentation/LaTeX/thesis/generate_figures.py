"""Verify the recorded WISCO runs and regenerate the thesis accuracy figure.

Run from any directory with Python 3.11 and matplotlib 3.10.1::

    py -3.11 Documentation/LaTeX/thesis/generate_figures.py

The default run checks each source CSV, recomputes exact four-digit accuracy,
and verifies identical case IDs and gold labels across the five per-case runs.
The served configuration publishes no per-case file and is verified for
internal consistency against its aggregate audit record instead.
Use --verify-only to check the data without plotting. A document-only checkout
can use --skip-source-check to plot the pinned, previously verified summary.
The source CSVs are read only; no model, database, or network service is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURE_DIR = Path(__file__).resolve().parent / "figures"
N_RECORDS = 18_747


@dataclass(frozen=True)
class Configuration:
    label: str
    encoder: str
    correct: int
    source: str
    # True when the count comes from a preserved aggregate audit record rather
    # than a per-case CSV. The served fragment-to-parent configuration does not
    # publish per-case predictions, so it cannot be checked for case-ID
    # identity against the others; it is verified for internal consistency
    # instead and drawn hatched so the two evidence classes stay visibly
    # distinct in the figure.
    summary: bool = False


CONFIGURATIONS = (
    Configuration(
        "Title-only / hierarchical",
        "E5-small",
        1_941,
        "eval/local_runs/wisco_official_tier1_precise_deadline_full_20260809T190821Z/"
        "hierarchical/20260809T192256Z_wisco_official_tier1_precise_deadline_full_hierarchical.csv",
    ),
    Configuration(
        "Title-only / flat",
        "E5-small",
        3_973,
        "eval/local_runs/wisco_official_tier1_precise_deadline_full_20260809T190821Z/"
        "flat/20260809T191133Z_wisco_official_tier1_precise_deadline_full_flat.csv",
    ),
    Configuration(
        "Title-only / flat",
        "E5-large",
        5_567,
        "eval/local_runs/e5large_full_heldout_20260824/results.csv",
    ),
    Configuration(
        "Enriched / flat",
        "E5-small",
        6_102,
        "eval/results/raw_runs/enriched_catalogue_heldout_20260824/20260824T113739Z_flat.csv",
    ),
    Configuration(
        "Fragment-to-parent / served",
        "E5-small",
        7_279,
        "Documentation/RAG_ACCURACY_AUDIT_2026-10-06_HISTORY_FINAL_RESULTS.json",
        summary=True,
    ),
    Configuration(
        "Enriched / flat",
        "E5-large",
        7_676,
        "eval/results/raw_runs/enriched_e5large_heldout_20260824/20260824T123941Z_flat.csv",
    ),
)


def _verify_summary_source(configuration: "Configuration", path: Path) -> None:
    """Check an aggregate audit record for internal consistency.

    Case-ID identity cannot be checked for a configuration that publishes no
    per-case predictions, so this asserts what the record itself can prove:
    the paired quadrants must reconstruct both arms' totals and the full
    population, the per-language counts must sum to the reported total, and
    the plotted count must equal the recorded one.
    """
    if not path.is_file():
        raise ValueError(
            f"Audit record is missing: {path}\n"
            "Restore the recorded evaluation artifacts, or explicitly use "
            "--skip-source-check for a document-only checkout."
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    served = record["methods"]["parent_document_rag"]
    dense = record["methods"]["historical_enriched_dense_small"]
    quadrants = record["paired_comparisons"]["current_dense_to_parent"]

    population = (
        quadrants["both_correct"] + quadrants["reference_only_correct"]
        + quadrants["candidate_only_correct"] + quadrants["both_wrong"]
    )
    if population != N_RECORDS or served["n"] != N_RECORDS:
        raise ValueError(f"{path}: population is not {N_RECORDS}")
    if quadrants["both_correct"] + quadrants["candidate_only_correct"] != served["top1_correct"]:
        raise ValueError(f"{path}: quadrants do not reconstruct the served total")
    if quadrants["both_correct"] + quadrants["reference_only_correct"] != dense["top1_correct"]:
        raise ValueError(f"{path}: quadrants do not reconstruct the baseline total")
    by_language = sum(
        block["parent_document_rag"]["top1_correct"] for block in record["per_language"].values()
    )
    if by_language != served["top1_correct"]:
        raise ValueError(f"{path}: per-language counts do not sum to the reported total")
    if served["top1_correct"] != configuration.correct:
        raise ValueError(
            f"{path}: expected {configuration.correct} exact matches, "
            f"found {served['top1_correct']}"
        )
    accuracy = 100 * served["top1_correct"] / N_RECORDS
    print(
        f"Verified {configuration.label}, {configuration.encoder}: "
        f"{served['top1_correct']:,}/{N_RECORDS:,} = {accuracy:.2f}% "
        "(aggregate audit record; internal consistency only)"
    )


def verify_sources() -> None:
    """Recompute counts and check that every run used the same labelled records."""
    reference_gold: dict[str, str] | None = None
    for configuration in CONFIGURATIONS:
        path = REPO_ROOT / configuration.source
        if configuration.summary:
            _verify_summary_source(configuration, path)
            continue
        if not path.is_file():
            raise ValueError(
                f"Source CSV is missing: {path}\n"
                "Restore the recorded evaluation artifacts, or explicitly use "
                "--skip-source-check for a document-only checkout."
            )
        gold_by_case: dict[str, str] = {}
        exact_correct = 0
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required_columns = {"case_id", "gold_isco_4digit", "pred_isco_4digit"}
            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing required columns in {path}")
            for line_number, row in enumerate(reader, start=2):
                case_id = row["case_id"].strip()
                gold = row["gold_isco_4digit"].strip()
                predicted = row["pred_isco_4digit"].strip()
                if not case_id or case_id in gold_by_case:
                    raise ValueError(f"Empty or duplicate case ID: {path}:{line_number}")
                if len(gold) != 4 or not gold.isdigit():
                    raise ValueError(f"Invalid gold ISCO code: {path}:{line_number}")
                if row.get("evaluation_status", "").strip().lower() == "dry_run":
                    raise ValueError(f"Dry-run record is not measured data: {path}:{line_number}")
                gold_by_case[case_id] = gold
                # Keep codes as strings so leading zeroes remain significant.
                exact_correct += bool(predicted) and predicted == gold
        if len(gold_by_case) != N_RECORDS:
            raise ValueError(f"{path}: expected {N_RECORDS} cases, found {len(gold_by_case)}")
        if exact_correct != configuration.correct:
            raise ValueError(
                f"{path}: expected {configuration.correct} exact matches, found {exact_correct}"
            )
        if reference_gold is None:
            reference_gold = gold_by_case
        elif gold_by_case != reference_gold:
            raise ValueError(f"Case IDs or gold labels differ from the first run: {path}")
        accuracy = 100 * exact_correct / N_RECORDS
        print(
            f"Verified {configuration.label}, {configuration.encoder}: "
            f"{exact_correct:,}/{N_RECORDS:,} = {accuracy:.2f}%"
        )


def draw_accuracy_figure() -> None:
    """Save vector PDF and 300 dpi PNG without unqualified uncertainty intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )
    palette = {"E5-small": "#8A9096", "E5-large": "#0072B2"}
    accuracies = [100 * item.correct / N_RECORDS for item in CONFIGURATIONS]
    positions = list(range(len(CONFIGURATIONS)))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.subplots_adjust(left=0.37, right=0.97, bottom=0.28, top=0.88)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E5E7E9", linewidth=0.65)
    ax.barh(
        positions,
        accuracies,
        height=0.58,
        color=[palette[item.encoder] for item in CONFIGURATIONS],
        # Hatch the bar whose count comes from an aggregate audit record
        # rather than released per-case predictions, so the figure does not
        # present two evidence classes as one.
        hatch=["//" if item.summary else "" for item in CONFIGURATIONS],
        edgecolor="white",
        linewidth=0,
    )
    ax.set_yticks(
        positions,
        [f"{item.label}\n{item.encoder}" for item in CONFIGURATIONS],
        fontsize=8.5,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 50)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("Exact four-digit ISCO accuracy (%)", fontsize=9, labelpad=7)
    ax.tick_params(axis="x", labelsize=8, length=3, color="#888888")
    ax.tick_params(axis="y", length=0, pad=8)
    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)
    ax.spines["bottom"].set_color("#A0A0A0")
    ax.spines["bottom"].set_linewidth(0.7)
    for position, accuracy in zip(positions, accuracies):
        ax.text(
            accuracy + 0.8,
            position,
            f"{accuracy:.2f}%",
            va="center",
            ha="left",
            fontsize=9,
            color="#222222",
        )
    fig.text(
        0.04,
        0.97,
        f"WISCO retrieval comparison (n = {N_RECORDS:,})",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="semibold",
        color="#222222",
    )
    fig.legend(
        handles=[Patch(facecolor=palette[name], label=name) for name in palette]
        + [
            Patch(
                facecolor="#FFFFFF",
                edgecolor="#555555",
                hatch="//",
                label="aggregate record",
            )
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.99),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        columnspacing=1,
    )
    fig.text(
        0.04,
        0.105,
        "Same partition reused across all six configurations; no LLM reranking.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.text(
        0.04,
        0.047,
        "Classification-title benchmark; these records are not real respondent interviews.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / "isco_retrieval_accuracy.pdf"
    png_path = FIGURE_DIR / "isco_retrieval_accuracy.png"
    fig.savefig(
        pdf_path,
        metadata={
            "Title": "WISCO exact four-digit ISCO retrieval accuracy",
            "Subject": "Five configurations evaluated on the same 18,747 classification-title records",
            "Creator": "generate_figures.py / matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-only", action="store_true", help="Check source CSVs without plotting")
    modes.add_argument(
        "--skip-source-check",
        action="store_true",
        help="Plot the pinned summary without reopening source CSVs",
    )
    args = parser.parse_args()
    if args.skip_source_check:
        print("Source checks skipped explicitly; plotting the pinned summary counts.")
    else:
        verify_sources()
    if not args.verify_only:
        draw_accuracy_figure()


if __name__ == "__main__":
    main()
