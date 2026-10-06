"""Select catalogue-only density correction on development, then freeze it.

This is an exploratory follow-up on historically reused WISCO splits. Never
index benchmark text, tune on validation/heldout, or imply fresh field evidence.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from eval import compare_parent_document_isco as parent
from eval import compare_hybrid_isco as common
from backend.rag.parent_document_isco import combine_parent_evidence
from backend.rag.isco_density import reference_density, correct_density

SOURCE_FILES = [Path(__file__), common.ROOT / 'backend/rag/isco_density.py'] + parent.SOURCE_FILES


def load_frozen_selection(path, parent_selection):
    """Bind the choice to its development evidence before reading labels."""
    selection = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = selection.pop('selection_sha256')
    source_hashes = {str(source): common.sha256(source) for source in SOURCE_FILES}
    if (common.object_hash(selection) != digest or selection['source_hashes'] != source_hashes
            or selection['parent_selection_sha256'] != parent_selection['selection_sha256']
            or selection['validation_labels_accessed_for_selection'] is not False):
        raise ValueError('Density selection integrity failure')
    report_path = Path(selection['development_report'])
    if common.sha256(report_path) != selection['development_report_sha256']:
        raise ValueError('Frozen density development report changed')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('split') != 'development' or report.get('selected') != selection['config']
            or report.get('source_hashes') != source_hashes
            or report.get('parent_selection_sha256') != parent_selection['selection_sha256']):
        raise ValueError('Density choice differs from its development evidence')
    config = selection['config']
    if (set(config) != {'neighbors', 'strength'}
            or type(config['neighbors']) is not int or config['neighbors'] not in (5, 10, 20, 50)
            or isinstance(config['strength'], bool) or config['strength'] not in (0, 0.25, 0.5, 0.75, 1)):
        raise ValueError('Density choice is outside the development grid')
    selection['selection_sha256'] = digest
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['select', 'evaluate'])
    for name in ['cases', 'queries', 'fragments', 'catalogue', 'parent-selection', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--split', choices=['validation', 'heldout'])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Existing experiments cannot be overwritten')
    source_hashes = {str(path): common.sha256(path) for path in SOURCE_FILES}
    parent_selection = parent.load_frozen_selection(args.parent_selection)
    if parent_selection['config']['aggregation'] != 'max':
        raise ValueError('This experiment requires the frozen maximum-child parent method')
    selection = None
    if args.command == 'evaluate':
        if args.selection is None or args.split is None:
            raise ValueError('Frozen selection and evaluation split required')
        selection = load_frozen_selection(args.selection, parent_selection)
    queries, labels, codes, parent_scores, aggregated, query_meta, catalogue_meta, fragment_meta = parent.inputs(
        args.cases, args.queries, args.fragments, args.catalogue)
    if (fragment_meta['cache_sha256'] != parent_selection['fragment_cache_sha256']
            or catalogue_meta['cache_sha256'] != parent_selection['catalogue_cache_sha256']
            or any(query_meta[key] != value for key, value in parent_selection['query_encoder'].items())):
        raise ValueError('Catalogue/encoder differs from the frozen parent method')
    if selection and (set(selection['development_case_ids']) | set(parent_selection['development_case_ids'])) & {query.case_id for query in queries}:
        raise ValueError('Evaluation overlaps development')
    weight = parent_selection['config']['child_weight']
    baseline_scores = combine_parent_evidence(aggregated['max'], parent_scores, child_weight=weight)
    base_predictions = parent.rank_codes(baseline_scores, codes)
    catalogue, _, _ = common.load_catalogue(args.catalogue, parent.PROFILE)
    _, fragments = parent.catalogue_fragments(parent.SOURCE)
    with np.load(args.fragments, allow_pickle=False) as cache:
        vectors = cache['fragment_vectors'].copy()
    grid = [selection['config']] if selection else [
        {'neighbors': neighbors, 'strength': strength} for neighbors in (5, 10, 20, 50)
        for strength in (0, 0.25, 0.5, 0.75, 1)]
    results = []
    densities = {}
    for config in grid:
        neighbors = config['neighbors']
        if neighbors not in densities:
            densities[neighbors] = reference_density(parent_vectors=catalogue['unit'][1], unit_codes=codes,
                fragment_vectors=vectors, fragments=fragments, neighbors=neighbors, child_weight=weight)
        scores = correct_density(baseline_scores, densities[neighbors], config['strength'])
        predictions = parent.rank_codes(scores, codes)
        results.append({'config': config, 'metrics': common.metrics(predictions, labels), 'predictions': predictions})
    chosen = min(results, key=lambda row: (-row['metrics']['top1_correct'], -row['metrics']['top3_correct'],
                                           row['config']['strength'], row['config']['neighbors']))
    methods = {'parent_document_rag': base_predictions, 'parent_document_density': chosen['predictions']}
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'development' if selection is None else args.split,
        **common.common_provenance(args.cases, queries, query_meta, catalogue_meta),
        'runtime_mode': 'cached parent/child cosine blend minus official-title density correction',
        'rrf_rank_indexing': None, 'rrf_k': None, 'source_hashes': source_hashes,
        'parent_selection_sha256': parent_selection['selection_sha256'], 'selected': chosen['config'],
        'parent_config': parent_selection['config'], 'fragment_metadata': fragment_meta,
        'methods': {method: common.metrics(predictions, labels) for method, predictions in methods.items()},
        'parameter_search': [{'config': row['config'], 'metrics': row['metrics']} for row in results] if selection is None else None,
        'density_source': 'independent official catalogue title vectors; exclude same-parent self reference',
        'density_reference_prefix': 'passage: ',
        'density_note': 'Catalogue-only reference approximation; not query-prefixed reference embeddings or standard CSLS',
        'benchmark_text_used_for_density': False,
        'selection_sha256': selection['selection_sha256'] if selection else None,
        'per_language': {language: {method: common.metrics(
            [prediction for query, prediction in zip(queries, predictions) if query.input_language == language],
            [label for query, label in zip(queries, labels) if query.input_language == language])
            for method, predictions in methods.items()} for language in sorted({query.input_language for query in queries})}}
    args.output.mkdir(parents=True)
    common.write_json_new(args.output / 'report.json', report)
    common.prediction_csv(args.output / 'predictions.csv', queries, labels, methods, {})
    if selection is None:
        choice = {'config': chosen['config'], 'parent_selection_sha256': parent_selection['selection_sha256'],
            'development_case_ids': [query.case_id for query in queries], 'source_hashes': source_hashes,
            'development_report': str((args.output / 'report.json').resolve()),
            'development_report_sha256': common.sha256(args.output / 'report.json'),
            'validation_labels_accessed_for_selection': False}
        choice['selection_sha256'] = common.object_hash(choice)
        common.write_json_new(args.output / 'selected_config.json', choice)
    print(json.dumps({'split': report['split'], 'selected': chosen['config'], 'methods': report['methods']}))


if __name__ == '__main__':
    main()
