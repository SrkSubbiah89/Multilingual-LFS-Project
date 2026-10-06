"""Reproducible local-only ISCO retrieval comparison with dev-only selection.

Commands run separately: snapshot authoritative catalogue vectors from local
Qdrant; select parameters on development labels; evaluate the frozen choice on
another split. No encoder, LLM, training, catalogue writes or paid API is used.
Historical WISCO holdout and validation have previously been examined. Enriched
catalogue design was informed by historical heldout errors: none of these
reused-split results establishes performance on a pristine unseen population.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

LEVELS = ("major", "submajor", "minor", "unit")
COUNTS = dict(zip(LEVELS, (10, 43, 130, 436)))
MODEL = "intfloat/multilingual-e5-small"
PROFILES = ("official_ilo2021_v1", "official_ilo2021_v1_enriched")
ROOT = Path(__file__).resolve().parents[1]
NORMALIZED = ROOT / "eval/local_catalogues/ilo_isco08_2021/normalized"
LEXICAL_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
HIERARCHY_GRID = (0.0, 0.2, 0.5, 1.0)
BEAM_GRID = (2, 4, 8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True)
class Query:
    case_id: str
    input_text: str
    input_language: str

    def digest(self) -> str:
        return object_hash({"case_id": self.case_id, "input_text": self.input_text,
                            "input_language": self.input_language})


def read_cases(path: Path) -> tuple[list[Query], list[str]]:
    queries, labels, identifiers = [], [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"case_id", "input_text", "input_language", "gold_isco_4digit"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Case CSV missing required columns")
        for row in reader:
            query = Query(*(row[key] for key in ("case_id", "input_text", "input_language")))
            gold = row["gold_isco_4digit"]
            if not query.case_id or query.case_id in identifiers or not query.input_text.strip():
                raise ValueError("Blank input or duplicate case_id")
            if len(gold) != 4 or not gold.isascii() or not gold.isdigit():
                raise ValueError("Gold label must be an exact four-digit code")
            identifiers.add(query.case_id)
            queries.append(query)
            labels.append(gold)
    if not queries:
        raise ValueError("No cases")
    return queries, labels


def valid_vectors(vectors, *, rows: int, dimension: int = 384) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (rows, dimension) or not np.isfinite(array).all():
        raise ValueError("Invalid vector dimensions or nonfinite vectors")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, atol=0.002, rtol=0):
        raise ValueError("Vectors must already be normalized")
    return array


def load_query_vectors(path: Path, queries: list[Query]) -> tuple[np.ndarray, dict]:
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if (metadata.get("encoder_id") != MODEL or metadata.get("dimension") != 384
            or metadata.get("query_prefix") != "query: "
            or metadata.get("normalize_embeddings") is not True
            or metadata.get("gold_labels_used") is not False
            or not metadata.get("encoder_revision") or not metadata.get("weights_sha256")):
        raise ValueError("Query cache has incomplete or incompatible encoder provenance")
    if metadata.get("cache_sha256") != sha256(path):
        raise ValueError("Query cache digest mismatch")
    with np.load(path, allow_pickle=False) as cache:
        identifiers = cache["case_ids"].tolist()
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Duplicate cached case_id")
        vectors = valid_vectors(cache["vectors"], rows=len(identifiers))
    by_identifier = {identifier: i for i, identifier in enumerate(identifiers)}
    selected = []
    for query in queries:
        if query.case_id not in by_identifier:
            raise ValueError("Case missing from query cache")
        if metadata.get("case_sha256", {}).get(query.case_id) != query.digest():
            raise ValueError("Cached query input digest mismatch")
        selected.append(by_identifier[query.case_id])
    return vectors[selected].copy(), metadata


def authoritative_records(profile: str):
    from backend.rag.official_isco08_catalogue import load_enriched_catalogue, load_official_catalogue
    filename = "isco08_official_normalized_enriched.csv" if profile == PROFILES[1] else "isco08_official_normalized.csv"
    loader = load_enriched_catalogue if profile == PROFILES[1] else load_official_catalogue
    return loader(NORMALIZED / filename, profile=profile)


def snapshot_catalogue(profile: str, output: Path, url: str = "http://127.0.0.1:6333") -> dict:
    from backend.rag.official_isco08_catalogue import PROFILE_COLLECTION_NAMES
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.username or parsed.password:
        raise ValueError("Snapshot endpoint must be local unauthenticated HTTP")
    if output.exists() or output.with_suffix(".meta.json").exists():
        raise ValueError("Catalogue snapshot already exists")
    records = authoritative_records(profile)
    expected = {record.code: record for record in records}
    arrays, collection_digests = {}, {}
    for level in LEVELS + ("flat",):
        collection = PROFILE_COLLECTION_NAMES[profile][level]
        points, offset = [], None
        while True:
            body = {"limit": 256, "with_vector": True, "with_payload": True}
            if offset is not None:
                body["offset"] = offset
            request = Request(url.rstrip("/") + "/collections/" + collection + "/points/scroll",
                              data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=15) as response:
                result = json.load(response)["result"]
            points.extend(result["points"])
            offset = result.get("next_page_offset")
            if offset is None:
                break
            if len(points) > 1000:
                raise ValueError("Unexpected catalogue size")
        actual_level = "unit" if level == "flat" else level
        points.sort(key=lambda point: point["payload"].get("code", ""))
        codes = [point["payload"].get("code", "") for point in points]
        official_codes = sorted(record.code for record in records if record.level == actual_level)
        if codes != official_codes or len(codes) != COUNTS[actual_level]:
            raise ValueError("Snapshot code set differs from official catalogue")
        for point in points:
            payload = point["payload"]
            record = expected[payload["code"]]
            for key in ("level", "parent_code", "title_en", "profile", "source_catalogue_sha256", "embedding_text"):
                if payload.get(key) != getattr(record, key):
                    raise ValueError("Stored catalogue payload differs from authoritative source: " + key)
        vectors = valid_vectors([point["vector"] for point in points], rows=len(points))
        collection_digests[level] = {"collection": collection, "count": len(points),
                                     "payload_sha256": object_hash([point["payload"] for point in points]),
                                     "vector_bytes_sha256": hashlib.sha256(vectors.tobytes()).hexdigest()}
        if level == "flat":
            if not np.allclose(vectors, arrays["unit_vectors"], atol=1e-7, rtol=0):
                raise ValueError("Flat and hierarchical unit collections differ; cannot conflate profiles")
        else:
            arrays[level + "_vectors"] = vectors
            arrays[level + "_codes"] = np.asarray(codes)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "profile": profile,
                "dimension": 384, "encoder_id": MODEL,
                "encoder_revision": None, "weights_sha256": None,
                "catalogue_encoder_identity_status": "Model name/dimension declared by profile; historical weights revision unavailable",
                "source_catalogue_sha256": records[0].source_catalogue_sha256,
                "collections": collection_digests, "cache_sha256": sha256(output),
                "operation": "read-only local Qdrant scroll; no embedding or catalogue write",
                "catalogue_text_source": "Official ILO title" + (", definition and included occupations" if profile == PROFILES[1] else ""),
                "prior_heldout_informed_enrichment": profile == PROFILES[1]}
    write_json_new(output.with_suffix(".meta.json"), metadata)
    return metadata


def load_catalogue(path: Path, profile: str) -> tuple[dict, dict, list]:
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    records = authoritative_records(profile)
    if (metadata.get("profile") != profile or metadata.get("encoder_id") != MODEL
            or metadata.get("dimension") != 384 or metadata.get("cache_sha256") != sha256(path)
            or metadata.get("source_catalogue_sha256") != records[0].source_catalogue_sha256):
        raise ValueError("Catalogue snapshot provenance mismatch")
    catalogue = {}
    with np.load(path, allow_pickle=False) as cache:
        for level in LEVELS:
            codes = cache[level + "_codes"].tolist()
            expected = sorted(record.code for record in records if record.level == level)
            if codes != expected:
                raise ValueError("Catalogue snapshot code set mismatch")
            catalogue[level] = (codes, valid_vectors(cache[level + "_vectors"], rows=COUNTS[level]))
    return catalogue, metadata, records


def dense_scores(vectors: np.ndarray, catalogue: dict) -> list[dict]:
    matrices = {level: vectors @ values[1].T for level, values in catalogue.items()}
    return [{level: sorted(zip(catalogue[level][0], map(float, scores[index])),
                           key=lambda pair: (-pair[1], pair[0]))
             for level, scores in matrices.items()} for index in range(len(vectors))]


def hierarchy_candidates(scores: dict, records: list, width: int, top_k: int = 5) -> list[str]:
    """Offline beam: cumulative ancestor cosine for paths; leaf cosine at end.

    This is explicitly a new baseline, not a reproduction of historical strict
    hierarchy with its prompts, keyword anchors, staged filters and budgets.
    """
    parents = {record.code: record.parent_code for record in records}
    active = {"": 0.0}
    for level in LEVELS[:-1]:
        children = [(code, active[parents[code]] + score)
                    for code, score in scores[level] if parents[code] in active]
        active = dict(sorted(children, key=lambda pair: (-pair[1], pair[0]))[:width])
    return [code for code, _ in scores["unit"] if parents[code] in active][:top_k]


def metrics(predictions: list[list[str]], labels: list[str]) -> dict:
    if len(predictions) != len(labels) or not labels:
        raise ValueError("Metric rows must be nonempty and aligned")
    n = len(labels)
    counts = {"top1": sum(bool(p) and p[0] == g for p, g in zip(predictions, labels)),
              "top3": sum(g in p[:3] for p, g in zip(predictions, labels)),
              "top5": sum(g in p[:5] for p, g in zip(predictions, labels))}
    z = 1.959963984540054
    rate = counts["top1"] / n
    center = (rate + z*z/(2*n)) / (1+z*z/n)
    radius = z * math.sqrt(rate*(1-rate)/n+z*z/(4*n*n)) / (1+z*z/n)
    return {"n": n, **{key + "_correct": value for key, value in counts.items()},
            **{key + "_accuracy": value / n for key, value in counts.items()},
            "top1_wilson95_case_level": [center-radius, center+radius],
            "abstentions": sum(not p for p in predictions)}


def parameter_configs() -> list[dict]:
    return [{"lexical_weight": lexical, "hierarchy_weight": hierarchy,
             "dense_weight": 1.0, "title_weight": 2.0, "body_weight": 1.0,
             "rrf_k": 60, "candidate_k": 50}
            for lexical in LEXICAL_GRID for hierarchy in HIERARCHY_GRID]


def config_key(config: dict) -> str:
    return f"lex={config['lexical_weight']:g};hier={config['hierarchy_weight']:g}"


def select_best(scored: list[dict], *, soft: bool) -> dict:
    eligible = [row for row in scored if (row["config"]["hierarchy_weight"] > 0) == soft]
    if not eligible:
        raise ValueError("No eligible configuration")
    return min(eligible, key=lambda row: (-row["metrics"]["top1_correct"],
               -row["metrics"]["top3_correct"], row["config"]["hierarchy_weight"],
               row["config"]["lexical_weight"]))["config"]


def make_retriever(records: list, profile: str, config: dict):
    from backend.rag.hybrid_isco import HybridISCORetriever
    return HybridISCORetriever(records, profile=profile, **config)


def run_fixed(queries: list[Query], scores: list[dict], records: list, profile: str, selected: dict) -> tuple[dict, dict]:
    started = time.perf_counter()
    lexical = make_retriever(records, profile, selected["rrf"])
    predictions = {"dense_flat": [[code for code, _ in row["unit"][:5]] for row in scores],
                   "bm25": [[code for code, _ in lexical.lexical_scores(query.input_text)[:5]] for query in queries],
                   "offline_greedy_hierarchy": [hierarchy_candidates(row, records, 1) for row in scores],
                   "offline_beam_hierarchy": [hierarchy_candidates(row, records, selected["beam_width"]) for row in scores]}
    components = {}
    for name, config in (("hybrid_rrf", selected["rrf"]), ("hybrid_soft_hierarchy_rrf", selected["soft"])):
        retriever = make_retriever(records, profile, config)
        results = [retriever.rank(query.input_text, dense_scores_by_level=row, top_k=5)
                   for query, row in zip(queries, scores)]
        predictions[name] = [[candidate.code for candidate in result.top_candidates] for result in results]
        components[name] = [[{"code": candidate.code, "dense_score": candidate.dense_score,
                              "lexical_score": candidate.lexical_score, "fusion_score": candidate.fusion_score,
                              "dense_rank": candidate.dense_rank, "lexical_rank": candidate.lexical_rank,
                              "hierarchy_support": candidate.hierarchy_support}
                             for candidate in result.top_candidates] for result in results]
    return predictions, {"fixed_methods_ranking_seconds": round(time.perf_counter()-started, 4), "components": components}


def summaries(predictions: dict, queries: list[Query], labels: list[str]) -> dict:
    report = {}
    baseline = predictions["dense_flat"]
    for name, output in predictions.items():
        language = {}
        for lang in sorted({query.input_language for query in queries}):
            indices = [i for i, query in enumerate(queries) if query.input_language == lang]
            language[lang] = metrics([output[i] for i in indices], [labels[i] for i in indices])
        fixed = [bool(p) and p[0] == gold for p, gold in zip(output, labels)]
        base = [bool(p) and p[0] == gold for p, gold in zip(baseline, labels)]
        report[name] = {**metrics(output, labels), "by_language": language,
                        "paired_vs_same_profile_dense_flat": {
                            "changed_wrong_to_correct": sum(a and not b for a, b in zip(fixed, base)),
                            "changed_correct_to_wrong": sum(b and not a for a, b in zip(fixed, base)),
                            "delta_accuracy": (sum(fixed)-sum(base))/len(labels)}}
    return report


def prediction_csv(path: Path, queries: list[Query], labels: list[str], predictions: dict, components: dict) -> None:
    columns = ["case_id", "input_language", "input_text", "gold_isco_4digit", "method", "prediction", "top5", "components"]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for i, (query, gold) in enumerate(zip(queries, labels)):
            for method, output in predictions.items():
                writer.writerow({"case_id": query.case_id, "input_language": query.input_language,
                                 "input_text": query.input_text, "gold_isco_4digit": gold,
                                 "method": method, "prediction": output[i][0] if output[i] else "",
                                 "top5": json.dumps(output[i]),
                                 "components": json.dumps(components.get(method, [[]]*len(queries))[i], ensure_ascii=False)})


def common_provenance(cases: Path, queries: list[Query], query_meta: dict, catalogue_meta: dict) -> dict:
    return {"dataset": "WISCO ISCO-08 v3 grouped split", "cases_csv": str(cases), "cases_csv_sha256": sha256(cases),
            "n": len(queries), "languages": {lang: sum(q.input_language == lang for q in queries)
                                            for lang in sorted({q.input_language for q in queries})},
            "query_encoder": {key: query_meta[key] for key in ("encoder_id", "encoder_revision", "weights_sha256", "query_prefix", "normalize_embeddings")},
            "query_cache_sha256": query_meta["cache_sha256"], "catalogue": catalogue_meta,
            "embedding_model": MODEL, "reranker": "off", "llm_calls": 0,
            "paid_api_calls": 0, "catalogue_benchmark_titles_added": False,
            "rrf_rank_indexing": "one-based", "rrf_k": 60,
            "confidence_calibrated": False, "runtime_mode": "offline cached cosine + official catalogue BM25",
            "statistical_note": "Case-level Wilson interval is descriptive; multilingual titles share source groups and are not independent respondents",
            "evaluation_limitation": "Historical validation/holdout reused; enriched catalogue choice was informed by historical heldout errors; no pristine unseen-test claim",
            "historical_strict_hierarchy_reproduced": False}


def select_development(cases: Path, query_cache: Path, catalogue_path: Path, profile: str, output: Path) -> dict:
    if output.exists():
        raise ValueError("Selection output directory already exists")
    queries, labels = read_cases(cases)
    vectors, query_meta = load_query_vectors(query_cache, queries)
    catalogue, catalogue_meta, records = load_catalogue(catalogue_path, profile)
    if any(label not in set(catalogue["unit"][0]) for label in labels):
        raise ValueError("Gold outside official catalogue")
    started = time.perf_counter()
    scores = dense_scores(vectors, catalogue)
    scored = []
    for config in parameter_configs():
        retriever = make_retriever(records, profile, config)
        predictions = [[candidate.code for candidate in retriever.rank(query.input_text, dense_scores_by_level=row, top_k=5).top_candidates]
                       for query, row in zip(queries, scores)]
        scored.append({"config": config, "metrics": metrics(predictions, labels)})
    beams = [{"width": width, "metrics": metrics([hierarchy_candidates(row, records, width) for row in scores], labels)} for width in BEAM_GRID]
    best_beam = min(beams, key=lambda row: (-row["metrics"]["top1_correct"], -row["metrics"]["top3_correct"], row["width"]))["width"]
    selected = {"rrf": select_best(scored, soft=False), "soft": select_best(scored, soft=True), "beam_width": best_beam}
    predictions, timing = run_fixed(queries, scores, records, profile, selected)
    output.mkdir(parents=True)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "split": "development", "purpose": "parameter selection only",
              **common_provenance(cases, queries, query_meta, catalogue_meta), "grid": scored, "beam_grid": beams,
              "selected": selected, "methods": summaries(predictions, queries, labels),
              "selection_seconds": round(time.perf_counter()-started, 4),
              "fixed_methods_ranking_seconds": timing["fixed_methods_ranking_seconds"]}
    prediction_csv(output / "development_predictions.csv", queries, labels, predictions, timing["components"])
    write_json_new(output / "development_report.json", report)
    selection = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "profile": profile,
                 "selected": selected, "selection_objective": "development top1; ties top3 then lower hierarchy/lexical weight or beam width",
                 "development_report": str(output / "development_report.json"),
                 "development_report_sha256": sha256(output / "development_report.json"),
                 "development_csv_sha256": sha256(cases), "development_case_ids": [query.case_id for query in queries],
                 "catalogue_cache_sha256": catalogue_meta["cache_sha256"],
                 "source_catalogue_sha256": catalogue_meta["source_catalogue_sha256"],
                 "query_encoder": report["query_encoder"], "validation_labels_accessed_for_selection": False}
    selection["selection_sha256"] = object_hash(selection)
    write_json_new(output / "selected_config.json", selection)
    return report


def load_frozen_selection(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value.pop("selection_sha256", None)
    if not digest or object_hash(value) != digest or value.get("validation_labels_accessed_for_selection") is not False:
        raise ValueError("Frozen selection integrity failure")
    if sha256(Path(value["development_report"])) != value.get("development_report_sha256"):
        raise ValueError("Development report changed after selection")
    value["selection_sha256"] = digest
    return value


def evaluate_frozen(cases: Path, query_cache: Path, catalogue_path: Path, selection_path: Path, split: str, output: Path) -> dict:
    if output.exists():
        raise ValueError("Evaluation output directory already exists")
    selection = load_frozen_selection(selection_path)
    queries, labels = read_cases(cases)
    if set(query.case_id for query in queries) & set(selection["development_case_ids"]):
        raise ValueError("Evaluation split overlaps development case IDs")
    vectors, query_meta = load_query_vectors(query_cache, queries)
    catalogue, catalogue_meta, records = load_catalogue(catalogue_path, selection["profile"])
    identity = {key: query_meta[key] for key in selection["query_encoder"]}
    if (identity != selection["query_encoder"] or catalogue_meta["cache_sha256"] != selection["catalogue_cache_sha256"]
            or catalogue_meta["source_catalogue_sha256"] != selection["source_catalogue_sha256"]):
        raise ValueError("Evaluation inputs differ from frozen encoder/catalogue")
    started = time.perf_counter()
    scores = dense_scores(vectors, catalogue)
    predictions, timing = run_fixed(queries, scores, records, selection["profile"], selection["selected"])
    output.mkdir(parents=True)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "split": split,
              "purpose": "frozen new-method evaluation on historically reused split",
              **common_provenance(cases, queries, query_meta, catalogue_meta),
              "selected": selection["selected"], "selection_sha256": selection["selection_sha256"],
              "development_report_sha256": selection["development_report_sha256"],
              "methods": summaries(predictions, queries, labels),
              "evaluation_seconds": round(time.perf_counter()-started, 4),
              "fixed_methods_ranking_seconds": timing["fixed_methods_ranking_seconds"]}
    prediction_csv(output / (split + "_predictions.csv"), queries, labels, predictions, timing["components"])
    write_json_new(output / (split + "_report.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--profile", choices=PROFILES, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    for name in ("select", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--cases", type=Path, required=True)
        command.add_argument("--queries", type=Path, required=True)
        command.add_argument("--catalogue", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "select":
            command.add_argument("--profile", choices=PROFILES, required=True)
        else:
            command.add_argument("--selection", type=Path, required=True)
            command.add_argument("--split", choices=("validation", "heldout"), required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        report = snapshot_catalogue(args.profile, args.output, args.qdrant_url)
        print(json.dumps({"profile": report["profile"], "status": "snapshot complete"}))
    else:
        if args.command == "select":
            report = select_development(args.cases, args.queries, args.catalogue, args.profile, args.output)
        else:
            report = evaluate_frozen(args.cases, args.queries, args.catalogue, args.selection, args.split, args.output)
        print(json.dumps({"split": report["split"], "n": report["n"], "methods": {name: {key: row[key] for key in ("top1_correct", "top1_accuracy", "top3_accuracy")} for name, row in report["methods"].items()}}, indent=2))


if __name__ == "__main__":
    main()
