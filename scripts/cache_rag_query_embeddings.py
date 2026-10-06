"""Cache local E5 query vectors without classifying or using gold labels.

Only case_id, input_text and input_language enter the encoder/cache. The
existing cached model is required; no model download or inference API is used.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_queries(paths: list[Path]) -> list[dict]:
    records = []
    identifiers = set()
    for path in paths:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'case_id', 'input_text', 'input_language'} <= set(reader.fieldnames or []):
                raise ValueError('Input CSV must contain case_id, input_text and input_language')
            for row in reader:
                query = {key: row[key] for key in ('case_id', 'input_text', 'input_language')}
                if not query['case_id'] or not query['input_text'].strip() or query['case_id'] in identifiers:
                    raise ValueError('Blank query or duplicate case_id in input CSV')
                identifiers.add(query['case_id'])
                records.append(query)
    if not records:
        raise ValueError('No query records')
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 128 or not 1 <= args.threads <= 16:
        parser.error('batch-size must be 1–128 and threads 1–16')
    if args.output.suffix != '.npz':
        parser.error('output must be an .npz file')
    metadata_path = args.output.with_suffix('.meta.json')
    if args.output.exists() or metadata_path.exists():
        parser.error('Existing cache will not be overwritten')
    records = load_queries(args.input)
    os.environ.update({
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'USE_TF': '0', 'USE_FLAX': '0', 'TF_CPP_MIN_LOG_LEVEL': '3',
        'CREWAI_TRACING_ENABLED': 'false', 'OTEL_SDK_DISABLED': 'true',
    })
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    encoder_id = 'intfloat/multilingual-e5-small'
    snapshot = Path(snapshot_download(encoder_id, local_files_only=True))
    weights = next((snapshot / name for name in ('model.safetensors', 'pytorch_model.bin')
                    if (snapshot / name).is_file()), None)
    if weights is None:
        raise RuntimeError('The local E5 snapshot has no supported weights file')
    weights_hash = file_sha256(weights)
    started = time.perf_counter()
    model = SentenceTransformer(str(snapshot), device='cpu', local_files_only=True)
    load_seconds = time.perf_counter() - started
    vectors = np.empty((len(records), 384), dtype=np.float32)
    encode_started = time.perf_counter()
    for offset in range(0, len(records), args.batch_size):
        batch = records[offset:offset + args.batch_size]
        encoded = model.encode(['query: ' + item['input_text'].strip() for item in batch],
                               normalize_embeddings=True, show_progress_bar=False,
                               batch_size=args.batch_size, convert_to_numpy=True)
        if encoded.shape != (len(batch), 384) or not np.isfinite(encoded).all():
            raise RuntimeError('Encoder produced invalid query vectors')
        vectors[offset:offset + len(batch)] = encoded
        if offset == 0 or (offset // args.batch_size + 1) % 10 == 0 or offset + len(batch) == len(records):
            print(json.dumps({'encoded': offset + len(batch), 'total': len(records),
                              'elapsed_seconds': round(time.perf_counter() - encode_started, 2)}), flush=True)
    encode_seconds = time.perf_counter() - encode_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.pending.npz')
    np.savez_compressed(temp, vectors=vectors, case_ids=np.asarray([item['case_id'] for item in records]))
    metadata = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'encoder_id': encoder_id, 'encoder_revision': snapshot.name,
        'local_snapshot_path': str(snapshot), 'weights_file': weights.name,
        'weights_sha256': weights_hash, 'query_prefix': 'query: ',
        'normalize_embeddings': True, 'dimension': 384, 'records': len(records),
        'strip_outer_query_whitespace': True,
        'input_csv_sha256': {str(path): file_sha256(path) for path in args.input},
        'case_sha256': {item['case_id']: hashlib.sha256(json.dumps(item, sort_keys=True,
                        ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest() for item in records},
        'batch_size': args.batch_size, 'torch_threads': args.threads,
        'max_sequence_length': model.max_seq_length, 'load_seconds': round(load_seconds, 3),
        'encode_seconds': round(encode_seconds, 3),
        'numpy_version': np.__version__, 'torch_version': torch.__version__,
        'gold_labels_used': False, 'model_downloaded': False,
        'cache_sha256': file_sha256(temp),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    temp.replace(args.output)
    print(json.dumps({'status': 'completed', 'records': len(records), 'dimension': 384,
                      'load_seconds': metadata['load_seconds'], 'encode_seconds': metadata['encode_seconds']}), flush=True)


if __name__ == '__main__':
    main()
