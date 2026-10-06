"""Read-only comparison of live Qdrant retrieval with frozen offline predictions.

Uses cached query embeddings: this verifies serving/index equivalence rather
than repeating encoder validation or measuring population accuracy.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from backend.agents.parent_document_isco_classifier import ParentDocumentISCOClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-language', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or args.per_language < 1:
        raise ValueError('New output and positive sample size required')
    by_language = {}
    with args.predictions.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['method'] == 'parent_document_rag':
                by_language.setdefault(row['input_language'], []).append(row)
    selected = []
    for language, rows in sorted(by_language.items()):
        indices = np.linspace(0, len(rows) - 1, min(args.per_language, len(rows)), dtype=int)
        selected.extend(rows[index] for index in indices)
    with np.load(args.queries, allow_pickle=False) as cache:
        vectors, case_ids = cache['vectors'], cache['case_ids'].tolist()
    by_id = {case_id: index for index, case_id in enumerate(case_ids)}
    current = [None]
    clf = ParentDocumentISCOClassifier(embed_query=lambda _: current[0])
    results = []
    try:
        for row in selected:
            current[0] = vectors[by_id[row['case_id']]]
            started = time.perf_counter()
            trace = {}
            result = clf.classify(row['input_text'], language=row['input_language'], top_k=5, use_llm=False, trace=trace)
            top5 = [result.primary.code] + [match.code for match in result.alternatives]
            results.append({'case_id': row['case_id'], 'language': row['input_language'],
                'top5_matches_offline': top5 == json.loads(row['top5']),
                'candidates_scored': trace['candidates_scored'], 'method': result.method,
                'retrieval_seconds': round(time.perf_counter() - started, 4)})
    finally:
        clf.client.close()
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'operation': 'read-only exact Qdrant queries with cached embeddings; no survey/auth/model calls',
        'config_sha256': hashlib.sha256(Path('backend/rag/parent_isco_config.json').read_bytes()).hexdigest(),
        'selection_sha256': clf.config['selection_sha256'],
        'n': len(results), 'top5_matches': sum(row['top5_matches_offline'] for row in results),
        'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'n': report['n'], 'top5_matches': report['top5_matches']}))
    if report['top5_matches'] != report['n']:
        raise RuntimeError('Live retrieval differs from frozen offline results')


if __name__ == '__main__':
    main()
