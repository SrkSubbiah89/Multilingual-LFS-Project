"""Recompute the thesis result counts from preserved local artifacts.

Run with Python 3.11+ from any directory. Uses only the standard library,
does not import the application, initialise models, or access the network.
The default output is JSON on stdout; --output writes a chosen report path.
Exact McNemar tails are summed as integers before Decimal division, avoiding
the floating-point underflow present in some historical analysis outputs.
This verifies record-level arithmetic, not independence or generalisation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = "eval/local_runs/wisco_official_tier1_precise_deadline_full_20260809T190821Z"
PATHS = {
    "flat_small": BASE_DIR + "/flat/20260809T191133Z_wisco_official_tier1_precise_deadline_full_flat.csv",
    "hierarchical_small": BASE_DIR + "/hierarchical/20260809T192256Z_wisco_official_tier1_precise_deadline_full_hierarchical.csv",
    "flat_large": "eval/local_runs/e5large_full_heldout_20260824/results.csv",
    "enriched_small": "eval/results/raw_runs/enriched_catalogue_heldout_20260824/20260824T113739Z_flat.csv",
    "enriched_large": "eval/results/raw_runs/enriched_e5large_heldout_20260824/20260824T123941Z_flat.csv",
}
ISCO_COLUMNS = {
    "case_id", "input_language", "gold_isco_4digit", "pred_isco_4digit",
    "end_to_end_latency_ms", "peak_memory_mb", "embedding_model_version",
    "error", "evaluation_status",
    "config",
}


def load_csv(relative: str, columns: set[str] | None = None) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) if columns is None else
                {key: value for key, value in row.items() if key in columns}
                for row in reader]


def file_identity(relative: str) -> dict:
    path = ROOT / relative
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def wilson(successes: int, n: int) -> list[float]:
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [100 * (centre - margin), 100 * (centre + margin)]


def exact_mcnemar(b: int, c: int) -> str:
    """Two-sided exact binomial tail, returned as an underflow-safe string."""
    n = b + c
    if not n:
        return "1"
    numerator = min(2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)), 1 << n)
    with localcontext() as context:
        context.prec = 40
        return format(Decimal(numerator) / Decimal(1 << n), ".12E")


def accuracy(rows: list[dict], gold: str, prediction: str) -> dict:
    successes = sum(row[gold] == row[prediction] for row in rows)
    return {"n": len(rows), "correct": successes,
            "accuracy_percent": 100 * successes / len(rows),
            "nominal_wilson_95_percent": wilson(successes, len(rows))}


def indexed(rows: list[dict]) -> dict[str, dict]:
    result = {row["case_id"]: row for row in rows}
    assert len(result) == len(rows), "Duplicate case identifiers"
    return result


def paired(first: list[dict], second: list[dict]) -> dict:
    a, b = indexed(first), indexed(second)
    assert a.keys() == b.keys(), "Paired case populations differ"
    outcomes = Counter()
    for case_id, left in a.items():
        right = b[case_id]
        assert left["gold_isco_4digit"] == right["gold_isco_4digit"], "Reference labels differ"
        x = left["pred_isco_4digit"] == left["gold_isco_4digit"]
        y = right["pred_isco_4digit"] == right["gold_isco_4digit"]
        outcomes[x, y] += 1
    b_count, c_count = outcomes[True, False], outcomes[False, True]
    return {"n": len(a), "both_correct": outcomes[True, True],
            "first_only_correct_b": b_count, "second_only_correct_c": c_count,
            "both_incorrect": outcomes[False, False],
            "exact_mcnemar_p": exact_mcnemar(b_count, c_count)}


SERVED_AUDIT = "Documentation/RAG_ACCURACY_AUDIT_2026-10-06_HISTORY_FINAL_RESULTS.json"


def served_configuration(sources: set[str]) -> dict:
    """Re-derive the served fragment-to-parent figures from its preserved audit.

    Unlike runs A, C and D this configuration publishes no per-case CSV, so the
    checks here are internal-consistency checks of the recorded summary: paired
    quadrants must reconstruct both totals and the population, percentages must
    follow from counts, and per-language counts must sum to the reported total.
    Passing does not re-execute retrieval or verify the encoder's runtime
    weights; it establishes that the reported numbers are mutually consistent
    and bound to a recorded serving identity.
    """
    sources.add(SERVED_AUDIT)
    record = json.loads((ROOT / SERVED_AUDIT).read_text(encoding="utf-8"))
    methods, pairs = record["methods"], record["paired_comparisons"]
    dense, parent = methods["historical_enriched_dense_small"], methods["parent_document_rag"]
    quad = pairs["current_dense_to_parent"]

    population = quad["both_correct"] + quad["reference_only_correct"] \
        + quad["candidate_only_correct"] + quad["both_wrong"]
    assert population == parent["n"] == dense["n"] == 18747, "served: population mismatch"
    assert quad["both_correct"] + quad["reference_only_correct"] == dense["top1_correct"], \
        "served: dense total not reconstructed by paired quadrants"
    assert quad["both_correct"] + quad["candidate_only_correct"] == parent["top1_correct"], \
        "served: parent total not reconstructed by paired quadrants"

    languages = {code: block["parent_document_rag"]["top1_correct"]
                 for code, block in record["per_language"].items()}
    assert sum(languages.values()) == parent["top1_correct"], \
        "served: per-language counts do not sum to the reported total"

    return {
        "n": parent["n"],
        "correct": parent["top1_correct"],
        "accuracy_percent": 100 * parent["top1_correct"] / parent["n"],
        "nominal_wilson_95_percent": wilson(parent["top1_correct"], parent["n"]),
        "same_encoder_dense_baseline_correct": dense["top1_correct"],
        "paired_vs_dense_baseline": {
            "b_baseline_only": quad["reference_only_correct"],
            "c_served_only": quad["candidate_only_correct"],
            "difference_correct": quad["difference_correct_candidate_minus_reference"],
            "difference_percentage_points": quad["difference_percentage_points_candidate_minus_reference"],
            "nominal_exact_mcnemar": exact_mcnemar(
                quad["reference_only_correct"], quad["candidate_only_correct"]),
        },
        "by_language_percent": {
            code: 100 * block["parent_document_rag"]["top1_correct"] / block["parent_document_rag"]["n"]
            for code, block in sorted(record["per_language"].items())},
        "serving_identity": record["current_frozen_reference_identity"],
        "limitation": (
            "Per-case predictions are not published; figures are re-derived from the preserved "
            "audit summary. Internal consistency only, not re-execution or encoder verification."),
    }


def audit() -> dict:
    sources: set[str] = set(PATHS.values())
    runs = {name: load_csv(path, ISCO_COLUMNS) for name, path in PATHS.items()}
    expected = {"flat_small": 3973, "hierarchical_small": 1941, "flat_large": 5567,
                "enriched_small": 6102, "enriched_large": 7676}
    report: dict = {"scope": "Offline record-level arithmetic audit; nominal uncertainty only",
                    "wisco": {}, "paired_against_flat_small": {}}
    for name, rows in runs.items():
        assert len(rows) == 18747, name + ": unexpected population"
        assert not any(row.get("error", "") for row in rows), name + ": classifier errors"
        assert all(row.get("evaluation_status", "measured") == "measured" for row in rows)
        metric = accuracy(rows, "gold_isco_4digit", "pred_isco_4digit")
        assert metric["correct"] == expected[name], name + ": unexpected correct count"
        report["wisco"][name] = metric
        if name != "flat_small":
            report["paired_against_flat_small"][name] = paired(runs["flat_small"], rows)
        times = sorted(float(row["end_to_end_latency_ms"]) for row in rows
                       if row.get("end_to_end_latency_ms"))
        if times:
            metric["latency_ms"] = {"mean": statistics.mean(times),
                "median": statistics.median(times), "p95_nearest_rank": times[math.ceil(.95 * len(times)) - 1]}
    report["paired_enriched_small_vs_large"] = paired(runs["enriched_small"], runs["enriched_large"])
    report["wisco_by_language"] = {
        language: {name: accuracy([row for row in rows if row["input_language"] == language],
                                "gold_isco_4digit", "pred_isco_4digit")
                   for name, rows in runs.items() if name in ("flat_small", "enriched_large")}
        for language in ("ar", "en", "hi", "tl", "ur")}
    memory = [float(row["peak_memory_mb"]) for row in runs["enriched_large"]]
    report["enriched_large_sampled_rss_mib"] = {"maximum": max(memory), "mean": statistics.mean(memory)}
    report["historical_encoder_metadata"] = sorted({row["embedding_model_version"]
                                                   for row in runs["enriched_large"]})
    report["encoder_metadata_limitation"] = (
        "Historical enriched-large CSV labels e5-small; catalogue profile mapping selects e5-large. "
        "This audit preserves historical files and does not verify the encoder's runtime weights.")
    report["served_fragment_to_parent"] = served_configuration(sources)

    zero_path = "eval/results/wisco_groq_zeroshot/results.jsonl"
    sources.add(zero_path)
    zero_original = [json.loads(line) for line in (ROOT / zero_path).read_text(encoding="utf-8").splitlines() if line]
    zero = [{"case_id": row["case_id"], "gold_isco_4digit": row["gold"],
             "pred_isco_4digit": row["pred"]} for row in zero_original]
    baseline = indexed(runs["flat_small"])
    matched = [baseline[row["case_id"]] for row in zero]
    report["zero_shot_matched"] = {"zero_shot": accuracy(zero, "gold_isco_4digit", "pred_isco_4digit"),
        "retrieval_same_cases": accuracy(matched, "gold_isco_4digit", "pred_isco_4digit"),
        "paired_zero_shot_first": paired(zero, matched)}
    assert (len(zero), report["zero_shot_matched"]["zero_shot"]["correct"],
            report["zero_shot_matched"]["retrieval_same_cases"]["correct"]) == (324, 61, 86)

    translation_paths = ["eval/results/wisco_groq_zeroshot/translation_baseline/20260909T202439Z_flat.csv",
                         "eval/results/wisco_groq_zeroshot/translation_translated/20260909T202802Z_flat.csv"]
    sources.update(translation_paths)
    original, translated = [load_csv(path, ISCO_COLUMNS) for path in translation_paths]
    report["translation"] = {"original": accuracy(original, "gold_isco_4digit", "pred_isco_4digit"),
        "translated": accuracy(translated, "gold_isco_4digit", "pred_isco_4digit"),
        "paired": paired(original, translated)}
    assert (report["translation"]["original"]["correct"], report["translation"]["translated"]["correct"]) == (20, 32)

    synthetic_path = "eval/results/synthetic_isic_iscedf_benchmark/synthetic_eval_results_20260827T031325Z.csv"
    quality_path = "eval/results/synthetic_isic_iscedf_benchmark/quality_check_20260827T164253Z.csv"
    sources.update((synthetic_path, quality_path))
    synthetic = load_csv(synthetic_path)
    b = sum(row["pred_legacy"] == row["gold_code"] and row["pred_flat"] != row["gold_code"] for row in synthetic)
    c = sum(row["pred_flat"] == row["gold_code"] and row["pred_legacy"] != row["gold_code"] for row in synthetic)
    report["synthetic_original_449"] = {"legacy": accuracy(synthetic, "gold_code", "pred_legacy"),
        "flat": accuracy(synthetic, "gold_code", "pred_flat"), "b": b, "c": c,
        "exact_mcnemar_p": exact_mcnemar(b, c),
        "by_standard": {standard: {"legacy": accuracy([row for row in synthetic if row["standard"] == standard], "gold_code", "pred_legacy"),
                                   "flat": accuracy([row for row in synthetic if row["standard"] == standard], "gold_code", "pred_flat")}
                        for standard in ("isic", "iscedf")},
        # Added 2026-10-09: Chapter 6 cites the per-language range of flat
        # accuracy on this benchmark, which nothing in this report previously
        # substantiated. Languages are read from the data rather than hardcoded
        # so a regenerated benchmark cannot silently drop one.
        "by_language": {language: {"legacy": accuracy([row for row in synthetic if row["language"] == language], "gold_code", "pred_legacy"),
                                   "flat": accuracy([row for row in synthetic if row["language"] == language], "gold_code", "pred_flat")}
                        for language in sorted({row["language"] for row in synthetic})},
        "refusal_case_ids": [row["case_id"] for row in load_csv(quality_path) if row.get("refusal_pattern") == "True"]}
    assert (len(synthetic), report["synthetic_original_449"]["legacy"]["correct"],
            report["synthetic_original_449"]["flat"]["correct"], b, c) == (449, 59, 372, 9, 322)
    assert set(report["synthetic_original_449"]["refusal_case_ids"]) == {"SYN-ISIC-9609-hi-1", "SYN-ISIC-9609-tl-1"}

    pilot_path = "eval/results/synthetic_pilot_n30/live_results.csv"
    sources.add(pilot_path)
    pilot = load_csv(pilot_path)
    report["synthetic_operational_30"] = {"n": len(pilot),
        "completed": sum(row["completed"] == "True" for row in pilot),
        "mean_scripted_seconds": statistics.mean(float(row["wall_clock_seconds"]) for row in pilot),
        "mean_turns": statistics.mean(int(row["turns"]) for row in pilot),
        "sessions_with_clarification": sum(int(row["clarifying_turns"]) > 0 for row in pilot),
        "high_sre": sum(row["sre_high_severity_seen"] == "True" for row in pilot),
        "incoherent_sre": sum(row["sre_is_coherent"] == "False" for row in pilot),
        "occupation_correct": sum(row["isco_code_assigned"] == row["gold_isco_4digit"] for row in pilot)}
    assert (len(pilot), report["synthetic_operational_30"]["completed"],
            report["synthetic_operational_30"]["occupation_correct"]) == (30, 30, 5)

    pool_path = "eval/results/raw_runs/20260805T055741Z_full130_leafvote_beam3_llama3b_pooled.csv"
    sources.add(pool_path)
    eligible = [row for row in load_csv(pool_path) if row["gold_isco_4digit"] and json.loads(row["stage4_pool"])]
    hits = sum(bool(row["gold_rank_in_pool"]) and int(row["gold_rank_in_pool"]) <= 3 for row in eligible)
    ranked = sum(bool(row["gold_rank_in_pool"]) for row in eligible)
    report["historical_top3_denominator_check"] = {"eligible_with_pool": len(eligible), "gold_found_at_any_rank": ranked,
        "top3_hits": hits, "correct_accuracy_percent": 100 * hits / len(eligible),
        "incorrect_conditional_accuracy_percent": 100 * hits / ranked,
        "scope": "Historical 130-case pool experiment; not a main thesis headline result"}
    assert (len(eligible), ranked, hits) == (130, 108, 61)

    reranking_pairs = {
        "large_dev500": (
            "eval/local_runs/e5large_vs_e5small_dev500_20260824/results.csv",
            "eval/local_runs/e5large_with_groq_reranker_dev500_20260824/results.csv"),
        "small_validation642": (
            "eval/results/dev_selection/validation_reranking_check/20260824T084635Z_flat.csv",
            "eval/results/dev_selection/validation_reranking_check/20260824T084818Z_flat.csv"),
        "enriched_validation642": (
            "eval/results/dev_selection/enriched_catalogue_validation_check/20260824T113519Z_flat.csv",
            "eval/results/dev_selection/enriched_catalogue_with_rerank_check/20260824T120417Z_flat.csv"),
        "corrective60": (
            "eval/results/dev_selection/corrective_retry_check_ollama_off/20260824T172619Z_flat.csv",
            "eval/results/dev_selection/corrective_retry_check_ollama/20260824T142821Z_flat.csv"),
    }
    report["reranking_and_corrective_checks"] = {}
    for name, paths in reranking_pairs.items():
        sources.update(paths)
        first, second = [load_csv(path, ISCO_COLUMNS) for path in paths]
        if name == "large_dev500":
            # This source combines 500 small and 500 large rows with repeated
            # case IDs. Select the recorded large arm before pairing.
            first = [row for row in first if row.get("config") == "e5_large"]
            assert len(first) == 500
        second_by_id = indexed(second)
        report["reranking_and_corrective_checks"][name] = {
            "first": accuracy(first, "gold_isco_4digit", "pred_isco_4digit"),
            "second": accuracy(second, "gold_isco_4digit", "pred_isco_4digit"),
            "paired": paired(first, second),
            "changed_codes": sum(row["pred_isco_4digit"] != second_by_id[row["case_id"]]["pred_isco_4digit"]
                                 for row in first),
        }
    expected_interventions = {
        "large_dev500": (146, 146, 0, 0, 1),
        "small_validation642": (117, 120, 0, 3, 4),
        "enriched_validation642": (209, 211, 0, 2, 3),
        "corrective60": (31, 31, 2, 2, 9),
    }
    for name, expected in expected_interventions.items():
        result = report["reranking_and_corrective_checks"][name]
        assert (result["first"]["correct"], result["second"]["correct"],
                result["paired"]["first_only_correct_b"],
                result["paired"]["second_only_correct_c"],
                result["changed_codes"]) == expected, name

    coordination_paths = {
        "legacy": "eval/results/synthetic_coordination_benchmark/eval_results.csv",
        "enriched_large": "eval/results/synthetic_coordination_benchmark/eval_results_enriched_e5large_final.csv",
        "query_planning": "eval/results/synthetic_query_planning_eval/isco_results.csv",
    }
    report["synthetic_coordination"] = {}
    for name, path in coordination_paths.items():
        sources.add(path)
        rows = load_csv(path)
        prediction = "planned_code" if name == "query_planning" else "coordinated_isco_code"
        baseline_prediction = "baseline_code" if name == "query_planning" else "baseline_isco_code"
        b = sum(row[baseline_prediction] == row["gold_isco_4digit"] and row[prediction] != row["gold_isco_4digit"] for row in rows)
        c = sum(row[prediction] == row["gold_isco_4digit"] and row[baseline_prediction] != row["gold_isco_4digit"] for row in rows)
        report["synthetic_coordination"][name] = {
            "baseline": accuracy(rows, "gold_isco_4digit", baseline_prediction),
            "intervention": accuracy(rows, "gold_isco_4digit", prediction),
            "b": b, "c": c, "exact_mcnemar_p": exact_mcnemar(b, c),
            "changed_codes": sum(row[prediction] != row[baseline_prediction] for row in rows),
        }
    expected_coordination = {
        "legacy": (16, 16, 0, 0, 0),
        "enriched_large": (29, 29, 0, 0, 0),
        "query_planning": (16, 17, 1, 2, 21),
    }
    for name, expected in expected_coordination.items():
        result = report["synthetic_coordination"][name]
        assert (result["baseline"]["correct"], result["intervention"]["correct"],
                result["b"], result["c"], result["changed_codes"]) == expected, name
    for standard in ("isic", "isced"):
        path = f"eval/results/synthetic_query_planning_eval/{standard}_results.csv"
        sources.add(path)
        rows = load_csv(path)
        report["synthetic_coordination"][standard + "_query_planning"] = {
            "n": len(rows), "changed_predictions": sum(row["baseline"] != row["planned"] for row in rows)}

    json_paths = {
        "sre_constructed_cases": "eval/results/legacy_thesis_ch6/sre_expanded_validation.json",
        "sre_endpoint_repetitions": "eval/results/legacy_thesis_ch6/sre_hitl_enforcement_endpoint_verification.json",
        "encoder_microbenchmark": "eval/results/legacy_thesis_ch6/embedding_timing_benchmark.json",
        "load_sequences": "eval/results/legacy_thesis_ch6/load_test_results.json",
        "qdrant_memory": "eval/results/legacy_thesis_ch6/qdrant_collection_memory_audit.json",
        "enriched_large_manifest": "eval/results/raw_runs/enriched_e5large_heldout_20260824/manifests/manifest_enriched_e5large_heldout_20260824.jsonl",
    }
    summaries = {}
    for name, path in json_paths.items():
        sources.add(path)
        summaries[name] = json.loads((ROOT / path).read_text(encoding="utf-8"))
    sre = summaries["sre_constructed_cases"]
    assert sre["n_cases"] == 61 and sre["n_mismatches"] == 0
    assert len(summaries["sre_endpoint_repetitions"]["runs"]) == 3
    report["sre_rule_checks"] = {"n": sre["n_cases"], "mismatches": sre["n_mismatches"],
        "severity_distribution": sre["severity_distribution"],
        "repeated_endpoint_runs": summaries["sre_endpoint_repetitions"]["runs"],
        "scope": "Constructed rule outcomes; same cases repeated, not independent adjudication"}
    load = summaries["load_sequences"]
    report["scripted_load_sequences"] = {"n": len(load), "successes": sum(row["success"] for row in load),
        "final_state_counts": dict(Counter(row["final_state"] for row in load))}
    report["encoder_microbenchmark"] = {key: summaries["encoder_microbenchmark"][key]
                                        for key in ("model", "hardware_disclosure", "n_calls", "mean_ms")}
    memory_bytes = summaries["qdrant_memory"]["process_wide_memory_real_measured"]["memory_resident_bytes"]
    report["qdrant_memory_snapshot"] = {"rss_bytes": memory_bytes, "rss_mib": memory_bytes / 1024**2}
    manifest = summaries["enriched_large_manifest"]
    report["recorded_enriched_large_environment"] = {key: manifest[key]
        for key in ("hardware", "os_name", "python_version", "retrieval_params", "llm_params")}
    report["recorded_enriched_large_environment"]["relevant_dependency_versions"] = {
        key: manifest["dependency_versions"].get(key)
        for key in ("qdrant-client", "sentence-transformers", "torch", "transformers")}
    report["source_file_identities"] = [file_identity(path) for path in sorted(sources)]
    report["unverified_from_saved_per_case_outputs"] = [
        "Normalised Gulf Arabic arm; only experiment summary retained",
        "Expanded 1091-record classification accuracy",
        "Participant outcomes, ethics approval, or deployed best-profile accuracy",
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON audit report path")
    args = parser.parse_args()
    payload = json.dumps(audit(), indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Verified arithmetic; wrote {args.output}")


if __name__ == "__main__":
    main()
