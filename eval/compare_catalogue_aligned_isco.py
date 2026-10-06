"""Seven global catalogue-alignment configurations; no large model inference.

The fitted transform uses independent official parent vectors only. Queries
use cached verified E5-small vectors. Historical large-vector encoder weights
remain unverified; no benchmark query/title/label enters alignment fitting.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from eval import compare_parent_document_isco as parent
from eval import compare_hybrid_isco as common
from eval import compare_balanced_pooling_isco as pooling
from backend.rag.catalogue_alignment import fit_catalogue_alignment, projected_parent_scores, combine_aligned_evidence
from backend.rag.official_isco08_catalogue import ENRICHED_E5LARGE_PROFILE, PROFILE_COLLECTION_NAMES

SOURCE_FILES = list(dict.fromkeys([Path(__file__), common.ROOT / 'backend/rag/catalogue_alignment.py',
                                 Path(pooling.__file__), common.ROOT / 'backend/rag/isco_balanced_pooling.py'] + parent.SOURCE_FILES))
GRID = [{'ridge': None, 'large_parent_weight': 0.0}] + [
    {'ridge': ridge, 'large_parent_weight': weight} for ridge in (0.01, 0.1, 1.0) for weight in (0.5, 1.0)]
FIELDS = ('code', 'level', 'parent_code', 'title_en', 'source_catalogue_sha256', 'embedding_text')


def validate_alignment_config(config):
    if not isinstance(config, dict) or set(config) != {'ridge', 'large_parent_weight'}:
        raise ValueError('Invalid alignment configuration schema')
    ridge, weight = config['ridge'], config['large_parent_weight']
    if (isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight)
            or ridge is not None and (isinstance(ridge, bool) or not isinstance(ridge, (int, float))
                                     or not math.isfinite(ridge) or ridge <= 0)
            or config not in GRID):
        raise ValueError('Invalid predeclared alignment configuration')


def snapshot_large(output, small_cache):
    if output.exists() or output.with_suffix('.meta.json').exists():
        raise ValueError('Existing aligned catalogue snapshot cannot be overwritten')
    catalogue, small_metadata, records = common.load_catalogue(small_cache, parent.PROFILE)
    expected = {record.code: record for record in records if record.level == 'unit'}
    name = PROFILE_COLLECTION_NAMES[ENRICHED_E5LARGE_PROFILE]['unit']
    endpoint = 'http://127.0.0.1:6333/collections/' + name
    with urlopen(endpoint, timeout=15) as response:
        info = json.load(response)['result']
    if info['config']['params']['vectors'] != {'size': 1024, 'distance': 'Cosine'} or info['points_count'] != 436:
        raise ValueError('Stored large catalogue has incompatible dimension/count')
    points, offset = [], None
    while True:
        body = {'limit': 256, 'with_payload': True, 'with_vector': True}
        if offset is not None:
            body['offset'] = offset
        request = Request(endpoint + '/points/scroll', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=15) as response:
            result = json.load(response)['result']
        points.extend(result['points'])
        if len(points) > 436:
            raise ValueError('Unexpected large catalogue size')
        offset = result.get('next_page_offset')
        if offset is None:
            break
    points.sort(key=lambda point: point['payload'].get('code', ''))
    codes = [point['payload'].get('code', '') for point in points]
    if codes != catalogue['unit'][0] or codes != sorted(expected):
        raise ValueError('Paired official parent code sets differ')
    for point in points:
        payload, record = point['payload'], expected[point['payload']['code']]
        if payload.get('profile') != ENRICHED_E5LARGE_PROFILE or any(payload.get(key) != getattr(record, key) for key in FIELDS):
            raise ValueError('Small/large parent payload text or source differs')
    vectors = common.valid_vectors([point['vector'] for point in points], rows=436, dimension=1024)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vectors=vectors, unit_codes=np.asarray(codes))
    common.write_json_new(output.with_suffix('.meta.json'), {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'operation': 'read-only local Qdrant official unit snapshot',
        'collection': name, 'declared_profile': ENRICHED_E5LARGE_PROFILE, 'dimension': 1024, 'records': 436,
        'source_catalogue_sha256': small_metadata['source_catalogue_sha256'],
        'paired_small_cache_sha256': small_metadata['cache_sha256'], 'cache_sha256': common.sha256(output),
        'vector_bytes_sha256': hashlib.sha256(vectors.tobytes()).hexdigest(),
        'payload_sha256': common.object_hash([point['payload'] for point in points]),
        'encoder_execution_identity': 'Historical stored vectors; encoder revision/weights unverified',
        'encoder_revision': None, 'encoder_weights_sha256': None,
        'benchmark_queries_or_labels_used': False})


def load_large(path, small_metadata):
    metadata = json.loads(path.with_suffix('.meta.json').read_text(encoding='utf-8'))
    if (metadata['cache_sha256'] != common.sha256(path) or metadata['dimension'] != 1024 or metadata['records'] != 436
            or metadata['declared_profile'] != ENRICHED_E5LARGE_PROFILE
            or metadata['source_catalogue_sha256'] != small_metadata['source_catalogue_sha256']
            or metadata['paired_small_cache_sha256'] != small_metadata['cache_sha256']
            or metadata['encoder_revision'] is not None or metadata['encoder_weights_sha256'] is not None
            or metadata['encoder_execution_identity'] != 'Historical stored vectors; encoder revision/weights unverified'
            or metadata['benchmark_queries_or_labels_used'] is not False):
        raise ValueError('Large stored-vector snapshot provenance differs')
    with np.load(path, allow_pickle=False) as cache:
        vectors = common.valid_vectors(cache['vectors'], rows=436, dimension=1024)
        codes = cache['unit_codes'].tolist()
    if hashlib.sha256(vectors.tobytes()).hexdigest() != metadata['vector_bytes_sha256']:
        raise ValueError('Stored large-vector digest differs')
    return vectors, codes, metadata


def load_frozen_selection(path, parent_choice):
    choice = json.loads(path.read_text(encoding='utf-8'))
    digest = choice.pop('selection_sha256')
    validate_alignment_config(choice['config'])
    if (common.object_hash(choice) != digest or choice['source_hashes'] != {str(source): common.sha256(source) for source in SOURCE_FILES}
            or choice['parent_selection_sha256'] != parent_choice['selection_sha256']
            or choice['parent_config'] != parent_choice['config'] or choice['config'] not in GRID
            or choice['validation_labels_accessed_for_selection'] is not False):
        raise ValueError('Frozen alignment source/config/parent binding changed')
    report_path = Path(choice['development_report'])
    if common.sha256(report_path) != choice['development_report_sha256']:
        raise ValueError('Frozen alignment development report changed')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('selected') != choice['config'] or report.get('split') != 'development'
            or any(report.get(key) != choice[key] for key in ('source_hashes', 'parent_selection_sha256', 'parent_config'))
            or any(report.get('query_encoder', {}).get(key) != value for key, value in choice['query_encoder'].items())
            or report.get('large_encoder_inference') is not False
            or report.get('benchmark_queries_or_labels_used_for_fit') is not False
            or report.get('large_snapshot_metadata', {}).get('cache_sha256') != choice['large_cache_sha256']
            or report.get('large_snapshot_metadata_sha256') != choice['large_snapshot_metadata_sha256']
            or report.get('fragment_metadata', {}).get('cache_sha256') != choice['fragment_cache_sha256']
            or report.get('catalogue', {}).get('cache_sha256') != choice['catalogue_cache_sha256']):
        raise ValueError('Alignment choice differs from development evidence')
    choice['selection_sha256'] = digest
    return choice


def run(args):
    if args.output.exists():
        raise ValueError('Existing alignment experiment cannot be overwritten')
    parent_choice = parent.load_frozen_selection(args.parent_selection)
    if parent_choice['config'] != {'child_weight': 0.5, 'aggregation': 'max'}:
        raise ValueError('Alignment experiment requires frozen 0.5/max parent control')
    choice = load_frozen_selection(args.selection, parent_choice) if args.command == 'evaluate' else None
    query_metadata, small_metadata, fragment_metadata = pooling.cache_provenance(args, parent_choice)
    large, large_codes, large_metadata = load_large(args.large, small_metadata)
    catalogue, _, _ = common.load_catalogue(args.catalogue, parent.PROFILE)
    codes, small = catalogue['unit']
    if large_codes != codes:
        raise ValueError('Large/small vector rows are not paired by official code')
    if choice and (choice['large_cache_sha256'] != large_metadata['cache_sha256']
                   or choice['large_snapshot_metadata_sha256'] != common.sha256(args.large.with_suffix('.meta.json'))
                   or choice['fragment_cache_sha256'] != fragment_metadata['cache_sha256']
                   or choice['catalogue_cache_sha256'] != small_metadata['cache_sha256']
                   or choice['query_encoder'] != parent_choice['query_encoder']):
        raise ValueError('Alignment cache/encoder differs from frozen choice')
    configs = [choice['config']] if choice else GRID
    # Fit from official catalogue pairs before the first query or label load.
    matrices = {ridge: fit_catalogue_alignment(small, large, ridge=ridge)
                for ridge in sorted({config['ridge'] for config in configs if config['ridge'] is not None})}
    queries, labels, input_codes, parent_scores, aggregates, *_ = parent.inputs(args.cases, args.queries, args.fragments, args.catalogue)
    if input_codes != codes:
        raise ValueError('Query parent ordering differs from catalogue fit')
    if choice and (set(choice['development_case_ids']) | set(parent_choice['development_case_ids'])) & {query.case_id for query in queries}:
        raise ValueError('Alignment evaluation overlaps development selection')
    vectors, _ = common.load_query_vectors(args.queries, queries)
    control_scores = combine_aligned_evidence(aggregates['max'], parent_scores, parent_scores, large_parent_weight=0)
    control_predictions = parent.rank_codes(control_scores, codes)
    projected = {ridge: projected_parent_scores(vectors, matrix, large) for ridge, matrix in matrices.items()}
    results = []
    for config in configs:
        scores = combine_aligned_evidence(aggregates['max'], parent_scores,
                    parent_scores if config['ridge'] is None else projected[config['ridge']],
                    large_parent_weight=config['large_parent_weight'])
        predictions = parent.rank_codes(scores, codes)
        results.append({'config': config, 'metrics': common.metrics(predictions, labels), 'predictions': predictions})
    chosen = max(results, key=lambda row: (row['metrics']['top1_correct'], row['metrics']['top3_correct']))
    predictions = {'parent_document_rag': control_predictions, 'parent_document_catalogue_aligned': chosen['predictions']}
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              **common.common_provenance(args.cases, queries, query_metadata, small_metadata),
              'split': 'development' if choice is None else args.split, 'runtime_mode': 'cached small queries projected through catalogue-only ridge fit',
              'rrf_rank_indexing': None, 'rrf_k': None, 'large_encoder_inference': False,
              'large_snapshot_metadata': large_metadata, 'fragment_metadata': fragment_metadata,
              'large_snapshot_metadata_sha256': common.sha256(args.large.with_suffix('.meta.json')),
              'alignment_fit_source': '436 paired independently sourced official parent vectors only',
              'benchmark_queries_or_labels_used_for_fit': False, 'predeclared_grid': GRID,
              'benchmark_text_indexed': False, 'per_language_parameters': False,
              'numpy_version': np.__version__,
              'selected': chosen['config'], 'parent_config': parent_choice['config'],
              'parent_selection_sha256': parent_choice['selection_sha256'],
              'source_hashes': {str(source): common.sha256(source) for source in SOURCE_FILES},
              'methods': {method: common.metrics(rows, labels) for method, rows in predictions.items()},
              'parameter_search': [{'config': row['config'], 'metrics': row['metrics']} for row in results] if choice is None else None,
              'selection_sha256': choice['selection_sha256'] if choice else None}
    args.output.mkdir(parents=True)
    common.write_json_new(args.output / 'report.json', report)
    common.prediction_csv(args.output / 'predictions.csv', queries, labels, predictions, {})
    if choice is None:
        selected = {'config': chosen['config'], 'parent_config': parent_choice['config'],
                    'parent_selection_sha256': parent_choice['selection_sha256'], 'source_hashes': report['source_hashes'],
                    'development_case_ids': [query.case_id for query in queries], 'large_cache_sha256': large_metadata['cache_sha256'],
                    'large_snapshot_metadata_sha256': report['large_snapshot_metadata_sha256'],
                    'fragment_cache_sha256': fragment_metadata['cache_sha256'], 'catalogue_cache_sha256': small_metadata['cache_sha256'],
                    'query_encoder': parent_choice['query_encoder'], 'development_report': str((args.output / 'report.json').resolve()),
                    'development_report_sha256': common.sha256(args.output / 'report.json'),
                    'validation_labels_accessed_for_selection': False}
        selected['selection_sha256'] = common.object_hash(selected)
        common.write_json_new(args.output / 'selected_config.json', selected)
    print(json.dumps({'split': report['split'], 'n': len(queries), 'selected': chosen['config'], 'methods': report['methods']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    snapshot = commands.add_parser('snapshot')
    snapshot.add_argument('--output', type=Path, required=True)
    snapshot.add_argument('--catalogue', type=Path, required=True)
    for name in ('select', 'evaluate'):
        command = commands.add_parser(name)
        for field in ('cases', 'queries', 'fragments', 'catalogue', 'large', 'parent-selection', 'output'):
            command.add_argument('--' + field, type=Path, required=True)
        if name == 'evaluate':
            command.add_argument('--selection', type=Path, required=True)
            command.add_argument('--split', choices=['validation', 'heldout'], required=True)
    args = parser.parse_args()
    if args.command == 'snapshot':
        snapshot_large(args.output, args.catalogue)
    else:
        run(args)


if __name__ == '__main__':
    main()
