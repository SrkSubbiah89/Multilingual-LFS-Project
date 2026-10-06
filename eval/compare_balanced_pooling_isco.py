"""Bounded development-only pooling selection with existing official caches.

Fifteen fixed configurations retain the original maximum-pooling control.
The parent blend stays frozen at its earlier development-selected weight.
No encoder, catalogue write, benchmark indexing or per-language tuning occurs.
Evaluation requires a new frozen choice verified before evaluation labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from eval import compare_parent_document_isco as parent
from eval import compare_hybrid_isco as common
from backend.rag.parent_document_isco import combine_parent_evidence
from backend.rag.isco_balanced_pooling import aggregate_balanced_children, validate_pooling_config

SOURCE_FILES = list(dict.fromkeys([Path(__file__), common.ROOT / 'backend/rag/isco_balanced_pooling.py'] + parent.SOURCE_FILES))
GRID = [{'kind': 'max'}] + [
    {'kind': 'kind_balanced', 'weights': list(weights)} for weights in (
        (1/3, 1/3, 1/3), (0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5),
        (0.5, 0.5, 0), (0.5, 0, 0.5), (0.75, 0.125, 0.125), (0.75, 0.25, 0),
        (0.75, 0, 0.25), (1, 0, 0))] + [
    {'kind': 'logmeanexp', 'temperature': temperature} for temperature in (0.02, 0.05, 0.1, 0.2)]


def cache_provenance(args, parent_selection):
    """Validate exact cache files and frozen encoder before any label loader."""
    query = json.loads(args.queries.with_suffix('.meta.json').read_text(encoding='utf-8'))
    fragment = json.loads(args.fragments.with_suffix('.meta.json').read_text(encoding='utf-8'))
    catalogue = json.loads(args.catalogue.with_suffix('.meta.json').read_text(encoding='utf-8'))
    for path, metadata in ((args.queries, query), (args.fragments, fragment), (args.catalogue, catalogue)):
        if common.sha256(path) != metadata['cache_sha256']:
            raise ValueError('Pooling cache content hash changed')
    compatibility = fragment.get('minimum_reencoded_parent_cosine')
    if (isinstance(compatibility, bool) or not isinstance(compatibility, (int, float))
            or not math.isfinite(compatibility) or not 0.9999 <= compatibility <= 1.00001
            or fragment.get('stored_parent_vectors_reencoded') != 436):
        raise ValueError('Parent encoder compatibility evidence is incomplete')
    if (fragment['cache_sha256'] != parent_selection['fragment_cache_sha256']
            or catalogue['cache_sha256'] != parent_selection['catalogue_cache_sha256']
            or fragment['catalogue_cache_sha256'] != catalogue['cache_sha256']
            or fragment['source_catalogue_sha256'] != catalogue['source_catalogue_sha256']
            or catalogue['profile'] != parent.PROFILE
            or any(query.get(key) != value or fragment.get(key) != value
                   for key, value in parent_selection['query_encoder'].items())
            or query.get('gold_labels_used') is not False or query.get('query_prefix') != 'query: '
            or query.get('normalize_embeddings') is not True or query.get('dimension') != 384
            or fragment.get('benchmark_text_indexed') is not False
            or fragment.get('normalize_embeddings') is not True or fragment.get('dimension') != 384
            or fragment.get('passage_prefix') != 'passage: '):
        raise ValueError('Pooling caches differ from the frozen parent comparison')
    return query, catalogue, fragment


def load_frozen_selection(path, parent_selection):
    choice = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = choice.pop('selection_sha256')
    if (common.object_hash(choice) != digest
            or choice['source_hashes'] != {str(source): common.sha256(source) for source in SOURCE_FILES}
            or choice['parent_selection_sha256'] != parent_selection['selection_sha256']
            or choice['parent_config'] != parent_selection['config']
            or choice['validation_labels_accessed_for_selection'] is not False
            or choice['config'] not in GRID):
        raise ValueError('Frozen pooling parameters or source changed')
    validate_pooling_config(choice['config'])
    report_path = Path(choice['development_report'])
    if common.sha256(report_path) != choice['development_report_sha256']:
        raise ValueError('Frozen pooling development report changed')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('split') != 'development' or report.get('selected') != choice['config']
            or report.get('source_hashes') != choice['source_hashes']
            or report.get('parent_selection_sha256') != choice['parent_selection_sha256']
            or report.get('parent_config') != choice['parent_config']):
        raise ValueError('Pooling choice differs from development evidence')
    choice['selection_sha256'] = digest
    return choice


def inputs(args, query_meta, catalogue_meta, fragment_meta):
    queries, labels = common.read_cases(args.cases)
    query_vectors, _ = common.load_query_vectors(args.queries, queries)
    catalogue, _, _ = common.load_catalogue(args.catalogue, parent.PROFILE)
    _, fragments = parent.catalogue_fragments(parent.SOURCE)
    if fragment_meta['fragment_map_sha256'] != common.object_hash([vars(fragment) for fragment in fragments]):
        raise ValueError('Fragment map differs from official source')
    with np.load(args.fragments, allow_pickle=False) as cache:
        if (cache['fragment_ids'].tolist() != [fragment.fragment_id for fragment in fragments]
                or cache['fragment_codes'].tolist() != [fragment.code for fragment in fragments]):
            raise ValueError('Fragment vectors differ from official ordering')
        fragment_vectors = common.valid_vectors(cache['fragment_vectors'], rows=len(fragments))
    return queries, labels, catalogue['unit'][0], query_vectors, catalogue['unit'][1], fragment_vectors, fragments


def run(args):
    if args.output.exists():
        raise ValueError('Existing pooling evidence cannot be overwritten')
    frozen_parent = parent.load_frozen_selection(args.parent_selection)
    if frozen_parent['config'] != {'child_weight': 0.5, 'aggregation': 'max'}:
        raise ValueError('This experiment requires the original frozen 0.5/max parent control')
    choice = load_frozen_selection(args.selection, frozen_parent) if args.command == 'evaluate' else None
    query_meta, catalogue_meta, fragment_meta = cache_provenance(args, frozen_parent)
    if choice and (choice['fragment_cache_sha256'] != fragment_meta['cache_sha256']
                   or choice['catalogue_cache_sha256'] != catalogue_meta['cache_sha256']
                   or choice['query_encoder'] != frozen_parent['query_encoder']):
        raise ValueError('Pooling cache/encoder differs from its frozen choice')
    started = time.perf_counter()
    queries, labels, codes, query_vectors, parent_vectors, fragment_vectors, fragments = inputs(
        args, query_meta, catalogue_meta, fragment_meta)
    selected_cases = set(frozen_parent['development_case_ids']) | (set(choice['development_case_ids']) if choice else set())
    if choice and selected_cases & {query.case_id for query in queries}:
        raise ValueError('Pooling evaluation overlaps development')
    configs = [choice['config']] if choice else GRID
    # Always preserve the same original control even if only another selected
    # config is evaluated. Control does not participate in evaluation selection.
    requested = [GRID[0]] + [config for config in configs if config != GRID[0]]
    scores = [np.empty((len(queries), len(codes)), dtype=np.float32) for _ in requested]
    fragment_codes = [fragment.code for fragment in fragments]
    fragment_kinds = [fragment.kind for fragment in fragments]
    for offset in range(0, len(queries), 128):
        vector = query_vectors[offset:offset + 128]
        child_cosine, parent_cosine = vector @ fragment_vectors.T, vector @ parent_vectors.T
        for index, config in enumerate(requested):
            pooled = aggregate_balanced_children(child_cosine, fragment_codes, fragment_kinds, codes, config=config)
            scores[index][offset:offset + len(vector)] = combine_parent_evidence(pooled, parent_cosine, child_weight=0.5)
    results = []
    for config, matrix in zip(requested, scores):
        ranked = parent.rank_codes(matrix, codes)
        results.append({'config': config, 'metrics': common.metrics(ranked, labels), 'predictions': ranked})
    if choice:
        chosen = next(row for row in results if row['config'] == choice['config'])
    else:
        # Top-one, then top-three; exact ties retain the predeclared control-first
        # grid order, so no-op configurations are never sold as improvements.
        chosen = max(results, key=lambda row: (row['metrics']['top1_correct'], row['metrics']['top3_correct']))
    predictions = {'parent_document_rag': results[0]['predictions'], 'parent_document_balanced_pooling': chosen['predictions']}
    provenance = common.common_provenance(args.cases, queries, query_meta, catalogue_meta)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), **provenance,
              'split': 'development' if choice is None else args.split,
              'runtime_mode': 'offline cached official child pooling and frozen parent cosine blend',
              'rrf_rank_indexing': None, 'rrf_k': None,
              'source_hashes': {str(source): common.sha256(source) for source in SOURCE_FILES},
              'parent_selection_sha256': frozen_parent['selection_sha256'], 'parent_config': frozen_parent['config'],
              'fragment_metadata': fragment_meta,
              'selected': chosen['config'], 'predeclared_grid': GRID,
              'methods': {method: common.metrics(rows, labels) for method, rows in predictions.items()},
              'parameter_search': [{'config': row['config'], 'metrics': row['metrics']} for row in results] if choice is None else None,
              'selection_sha256': choice['selection_sha256'] if choice else None,
              'ranking_seconds': round(time.perf_counter() - started, 3),
              'benchmark_text_indexed': False, 'per_language_parameters': False,
              'per_language': {language: {method: common.metrics(
                  [prediction for query, prediction in zip(queries, rows) if query.input_language == language],
                  [label for query, label in zip(queries, labels) if query.input_language == language])
                  for method, rows in predictions.items()} for language in sorted({query.input_language for query in queries})}}
    args.output.mkdir(parents=True)
    common.write_json_new(args.output / 'report.json', report)
    common.prediction_csv(args.output / 'predictions.csv', queries, labels, predictions, {})
    if choice is None:
        selected = {'config': chosen['config'], 'parent_selection_sha256': frozen_parent['selection_sha256'],
                    'parent_config': frozen_parent['config'], 'source_hashes': report['source_hashes'],
                    'development_case_ids': [query.case_id for query in queries],
                    'development_report': str((args.output / 'report.json').resolve()),
                    'development_report_sha256': common.sha256(args.output / 'report.json'),
                    'fragment_cache_sha256': fragment_meta['cache_sha256'],
                    'catalogue_cache_sha256': catalogue_meta['cache_sha256'],
                    'query_encoder': frozen_parent['query_encoder'], 'validation_labels_accessed_for_selection': False}
        selected['selection_sha256'] = common.object_hash(selected)
        common.write_json_new(args.output / 'selected_config.json', selected)
    print(json.dumps({'split': report['split'], 'n': len(queries), 'selected': chosen['config'], 'methods': report['methods']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['select', 'evaluate'])
    for name in ('cases', 'queries', 'fragments', 'catalogue', 'parent-selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--split', choices=['validation', 'heldout'])
    args = parser.parse_args()
    if args.command == 'evaluate' and (args.selection is None or args.split is None):
        parser.error('evaluate requires --selection and --split')
    run(args)


if __name__ == '__main__':
    main()
