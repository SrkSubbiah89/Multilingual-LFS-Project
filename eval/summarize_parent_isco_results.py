"""Paired descriptive uncertainty on reused benchmark prediction rows.

Bootstrap clusters combine WISCO occupation families and families sharing an
identical normalized retained title in any language. Multilingual variants
are never treated as independent resampling units.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def summarize(path, *, replicates=2000, seed=20261006):
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise ValueError('At least 100 integer bootstrap replicates required')
    cases = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            case = cases.setdefault(row['case_id'], {})
            if row['method'] in case:
                raise ValueError('Duplicate method/case prediction')
            case[row['method']] = row
    families, text_family = {}, {}

    def find(key):
        while families[key] != key:
            families[key] = families[families[key]]
            key = families[key]
        return key

    for case_id, methods in cases.items():
        if set(methods) != {'dense_flat', 'parent_document_rag'}:
            raise ValueError('Paired methods must cover the same cases')
        dense, parent = methods['dense_flat'], methods['parent_document_rag']
        if any(dense[field] != parent[field] for field in ('input_language', 'input_text', 'gold_isco_4digit')):
            raise ValueError('Paired query/label differs')
        parts = case_id.split('-')
        if (len(parts) != 3 or parts[0] != 'WISCO' or not parts[1].isdigit()
                or parts[2] != dense['input_language'] or not dense['input_text'].strip()
                or re.fullmatch(r'[0-9]{4}', dense['gold_isco_4digit']) is None):
            raise ValueError('Invalid WISCO family identifier')
        key = parts[1]
        families.setdefault(key, key)
        text = (dense['input_language'], re.sub(r'\s+', ' ', dense['input_text'].strip().lower()))
        previous = text_family.setdefault(text, key)
        families[find(key)] = find(previous)
    clusters = defaultdict(lambda: [0, 0])
    wins = losses = both = neither = 0
    for case_id, methods in cases.items():
        dense, parent = methods['dense_flat'], methods['parent_document_rag']
        correct_dense = dense['prediction'] == dense['gold_isco_4digit']
        correct_parent = parent['prediction'] == parent['gold_isco_4digit']
        delta = int(correct_parent) - int(correct_dense)
        cluster = clusters[find(case_id.split('-')[1])]
        cluster[0] += delta
        cluster[1] += 1
        wins += delta == 1
        losses += delta == -1
        both += correct_dense and correct_parent
        neither += not correct_dense and not correct_parent
    if not clusters or replicates < 100:
        raise ValueError('Nonempty cases and at least 100 replicates required')
    values = np.asarray(list(clusters.values()))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        selected = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        samples.append(selected[0] / selected[1])
    return {'n': len(cases), 'parent_only_correct': wins, 'dense_only_correct': losses,
        'both_correct': both, 'both_wrong': neither, 'accuracy_difference': (wins - losses) / len(cases),
        'cluster_bootstrap_percentile95_difference': np.quantile(samples, [.025, .975]).tolist(),
        'clusters': len(clusters), 'replicates': replicates, 'seed': seed,
        'cluster_definition': 'WISCO key families unioned by identical normalized retained title within a language',
        'prediction_csv_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'interpretation': 'Descriptive uncertainty on a reused reference benchmark; no population or LFS field inference'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Existing evidence must not be overwritten')
    result = summarize(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
