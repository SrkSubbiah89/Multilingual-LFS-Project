"""Build independent official child vectors, select on dev, evaluate frozen RAG.

Historical splits have been used before; this is a controlled reused-benchmark
comparison, not pristine population validation. No benchmark example is indexed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'USE_TF': '0',
                   'USE_FLAX': '0', 'CREWAI_TRACING_ENABLED': 'false', 'OTEL_SDK_DISABLED': 'true'})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from eval import compare_hybrid_isco as baseline
from backend.rag.parent_document_isco import catalogue_fragments, aggregate_children, combine_parent_evidence

SOURCE = baseline.NORMALIZED / 'isco08_official_normalized_enriched.csv'
PROFILE = baseline.PROFILES[1]
SOURCE_FILES = [Path(__file__), Path(baseline.__file__),
                baseline.ROOT / 'backend/rag/parent_document_isco.py',
                baseline.ROOT / 'backend/rag/hybrid_isco.py',
                baseline.ROOT / 'backend/rag/official_isco08_catalogue.py',
                baseline.ROOT / 'scripts/cache_rag_query_embeddings.py']


def load_frozen_selection(path: Path):
    """Verify the dev choice and its evidence before reading evaluation labels."""
    selection = json.loads(path.read_text(encoding='utf-8'))
    digest = selection.pop('selection_sha256')
    if (baseline.object_hash(selection) != digest
            or selection['source_hashes'] != {str(source): baseline.sha256(source) for source in SOURCE_FILES}):
        raise ValueError('Frozen parameters or retrieval source changed')
    if selection.get('validation_labels_accessed_for_selection') is not False:
        raise ValueError('Frozen choice must record development-only label selection')
    report_path = Path(selection['development_report'])
    if baseline.sha256(report_path) != selection['development_report_sha256']:
        raise ValueError('Frozen development report changed')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('split') != 'development' or report.get('selected') != selection['config']
            or report.get('source_hashes') != selection['source_hashes']):
        raise ValueError('Frozen choice differs from its development evidence')
    selection['selection_sha256'] = digest
    return selection


def build_cache(output: Path, encoder_metadata: Path, catalogue_cache: Path):
    if output.exists() or output.with_suffix('.meta.json').exists():
        raise ValueError('Existing fragment cache will not be overwritten')
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    metadata = json.loads(encoder_metadata.read_text(encoding='utf-8'))
    snapshot = Path(metadata['local_snapshot_path'])
    if snapshot.name != metadata['encoder_revision'] or baseline.sha256(snapshot / metadata['weights_file']) != metadata['weights_sha256']:
        raise ValueError('Encoder snapshot differs from the recorded query model')
    model = SentenceTransformer(str(snapshot), device='cpu', local_files_only=True)
    records, fragments = catalogue_fragments(SOURCE)
    print(json.dumps({'fragments': len(fragments), 'parents': sum(record.level == 'unit' for record in records)}), flush=True)
    started = time.perf_counter()
    vectors = model.encode(['passage: ' + fragment.text for fragment in fragments], normalize_embeddings=True,
                           show_progress_bar=False, batch_size=32, convert_to_numpy=True)
    vectors = baseline.valid_vectors(vectors, rows=len(fragments))
    catalogue, catalogue_metadata, _ = baseline.load_catalogue(catalogue_cache, PROFILE)
    units = {record.code: record for record in records if record.level == 'unit'}
    codes, stored = catalogue['unit']
    reproduced = model.encode(['passage: ' + units[code].embedding_text for code in codes],
                normalize_embeddings=True, show_progress_bar=False, batch_size=32, convert_to_numpy=True)
    cosines = np.sum(reproduced * stored, axis=1)
    if float(cosines.min()) < 0.9999:
        raise ValueError('Stored parent vectors do not match the current encoder; controlled comparison aborted')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, fragment_vectors=vectors, fragment_codes=np.asarray([fragment.code for fragment in fragments]),
                        fragment_ids=np.asarray([fragment.fragment_id for fragment in fragments]))
    baseline.write_json_new(output.with_suffix('.meta.json'), {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'encoder_id': metadata['encoder_id'],
        'encoder_revision': metadata['encoder_revision'], 'weights_sha256': metadata['weights_sha256'],
        'dimension': 384, 'normalize_embeddings': True, 'passage_prefix': 'passage: ',
        'source_catalogue_sha256': records[0].source_catalogue_sha256,
        'fragment_map_sha256': baseline.object_hash([vars(fragment) for fragment in fragments]),
        'cache_sha256': baseline.sha256(output), 'fragments': len(fragments), 'parent_codes': len(units),
        'source': 'Independent official ILO titles, definitions and included occupations', 'benchmark_text_indexed': False,
        'stored_parent_vectors_reencoded': len(codes), 'minimum_reencoded_parent_cosine': float(cosines.min()),
        'catalogue_cache_sha256': catalogue_metadata['cache_sha256'], 'build_seconds': round(time.perf_counter() - started, 3)})


def rank_codes(scores, codes):
    # Stable sort preserves the canonical ascending-code tie order.
    return [[codes[index] for index in order[:5]] for order in np.argsort(-scores, axis=1, kind='stable')]


def inputs(cases, queries_path, fragments_path, catalogue_path):
    queries, labels = baseline.read_cases(cases)
    query_vectors, query_meta = baseline.load_query_vectors(queries_path, queries)
    catalogue, catalogue_meta, records = baseline.load_catalogue(catalogue_path, PROFILE)
    _, fragments = catalogue_fragments(SOURCE)
    fragment_meta = json.loads(fragments_path.with_suffix('.meta.json').read_text(encoding='utf-8'))
    if (fragment_meta['cache_sha256'] != baseline.sha256(fragments_path)
            or fragment_meta['fragment_map_sha256'] != baseline.object_hash([vars(fragment) for fragment in fragments])
            or fragment_meta['source_catalogue_sha256'] != catalogue_meta['source_catalogue_sha256']
            or fragment_meta['catalogue_cache_sha256'] != catalogue_meta['cache_sha256']
            or any(fragment_meta[key] != query_meta[key] for key in ('encoder_id', 'encoder_revision', 'weights_sha256'))):
        raise ValueError('Fragment provenance does not match the controlled comparison')
    with np.load(fragments_path, allow_pickle=False) as cache:
        if cache['fragment_ids'].tolist() != [fragment.fragment_id for fragment in fragments]:
            raise ValueError('Fragment vector order differs from the authoritative source')
        fragment_vectors = baseline.valid_vectors(cache['fragment_vectors'], rows=len(fragments))
    codes, parent_vectors = catalogue['unit']
    parent_scores = query_vectors @ parent_vectors.T
    aggregated = {mode: np.empty((len(queries), len(codes)), dtype=np.float32) for mode in ('max', 'mean_top2')}
    for offset in range(0, len(queries), 128):
        similarities = query_vectors[offset:offset + 128] @ fragment_vectors.T
        for mode in aggregated:
            aggregated[mode][offset:offset + len(similarities)] = aggregate_children(
                similarities, [fragment.code for fragment in fragments], codes, mode=mode)
    return queries, labels, codes, parent_scores, aggregated, query_meta, catalogue_meta, fragment_meta


def run(args):
    if args.output.exists():
        raise ValueError('Existing experiment output will not be overwritten')
    selection = None
    if args.command == 'evaluate':
        selection = load_frozen_selection(args.selection)
    started = time.perf_counter()
    queries, labels, codes, parent_scores, aggregated, query_meta, catalogue_meta, fragment_meta = inputs(
        args.cases, args.queries, args.fragments, args.catalogue)
    if selection and set(selection['development_case_ids']) & {query.case_id for query in queries}:
        raise ValueError('Evaluation overlaps development cases')
    if selection and (selection['fragment_cache_sha256'] != fragment_meta['cache_sha256'] or selection['catalogue_cache_sha256'] != catalogue_meta['cache_sha256']
                      or any(query_meta[key] != value for key, value in selection['query_encoder'].items())):
        raise ValueError('Evaluation encoder/catalogue differs from the frozen choice')
    results = []
    configurations = ([selection['config']] if selection else
                      [{'child_weight': weight, 'aggregation': mode} for weight in (0.25, 0.5, 0.75, 1.0) for mode in ('max', 'mean_top2')])
    for config in configurations:
        score = combine_parent_evidence(aggregated[config['aggregation']], parent_scores, child_weight=config['child_weight'])
        predictions = rank_codes(score, codes)
        results.append({'config': config, 'metrics': baseline.metrics(predictions, labels), 'predictions': predictions})
    chosen = min(results, key=lambda row: (-row['metrics']['top1_correct'], -row['metrics']['top3_correct'], row['config']['child_weight'], row['config']['aggregation']))
    dense = rank_codes(parent_scores, codes)
    summaries = {'dense_flat': baseline.metrics(dense, labels), 'parent_document_rag': chosen['metrics']}
    provenance = baseline.common_provenance(args.cases, queries, query_meta, catalogue_meta)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'split': 'development' if selection is None else args.split,
              **provenance, 'methods': summaries, 'selected': chosen['config'], 'fragment_metadata': fragment_meta,
              'ranking_seconds': round(time.perf_counter() - started, 3),
              'per_language': {language: {method: baseline.metrics([prediction for query, prediction in zip(queries, predictions) if query.input_language == language],
                   [gold for query, gold in zip(queries, labels) if query.input_language == language])
                   for method, predictions in [('dense_flat', dense), ('parent_document_rag', chosen['predictions'])]}
                   for language in sorted({query.input_language for query in queries})},
              'parameter_search': [{'config': row['config'], 'metrics': row['metrics']} for row in results] if selection is None else None,
              'source_hashes': {str(path): baseline.sha256(path) for path in SOURCE_FILES},
              'selection_sha256': selection['selection_sha256'] if selection else None}
    args.output.mkdir(parents=True)
    baseline.write_json_new(args.output / 'report.json', report)
    baseline.prediction_csv(args.output / 'predictions.csv', queries, labels,
                           {'dense_flat': dense, 'parent_document_rag': chosen['predictions']}, {})
    if selection is None:
        selected = {'config': chosen['config'], 'development_case_ids': [query.case_id for query in queries],
            'source_hashes': report['source_hashes'], 'development_report': str((args.output / 'report.json').resolve()),
            'development_report_sha256': baseline.sha256(args.output / 'report.json'),
            'fragment_cache_sha256': fragment_meta['cache_sha256'], 'catalogue_cache_sha256': catalogue_meta['cache_sha256'],
            'query_encoder': {key: query_meta[key] for key in ('encoder_id', 'encoder_revision', 'weights_sha256')},
            'validation_labels_accessed_for_selection': False}
        selected['selection_sha256'] = baseline.object_hash(selected)
        baseline.write_json_new(args.output / 'selected_config.json', selected)
    print(json.dumps({'split': report['split'], 'n': len(queries), 'selected': chosen['config'], 'methods': summaries}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--encoder-metadata', type=Path, required=True)
    build.add_argument('--catalogue', type=Path, required=True)
    for name in ('select', 'evaluate'):
        command = commands.add_parser(name)
        for field in ('cases', 'queries', 'fragments', 'catalogue', 'output'):
            command.add_argument('--' + field, type=Path, required=True)
        if name == 'evaluate':
            command.add_argument('--selection', type=Path, required=True)
            command.add_argument('--split', choices=['validation', 'heldout'], required=True)
    args = parser.parse_args()
    if args.command == 'build':
        build_cache(args.output, args.encoder_metadata, args.catalogue)
    else:
        run(args)


if __name__ == '__main__':
    main()
