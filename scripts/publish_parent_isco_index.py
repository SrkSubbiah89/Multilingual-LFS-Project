"""Publish a new, content-addressed local Qdrant child index from official text.

Existing collections are never overwritten. The source and cached vectors are
verified before writing any point; this operation touches no respondent data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from uuid import UUID, uuid5

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from qdrant_client import QdrantClient, models
from backend.rag.parent_document_isco import catalogue_fragments
from scripts.cache_rag_query_embeddings import file_sha256

NAMESPACE = UUID('a67783cb-186a-4147-aa10-231b6d61f827')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists():
        raise ValueError('Existing publish manifest will not be overwritten')
    metadata = json.loads(args.cache.with_suffix('.meta.json').read_text(encoding='utf-8'))
    records, fragments = catalogue_fragments(args.source)
    fragment_hash = hashlib.sha256(json.dumps([vars(fragment) for fragment in fragments], sort_keys=True,
                                  ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (metadata['cache_sha256'] != file_sha256(args.cache) or metadata['fragment_map_sha256'] != fragment_hash
            or metadata['source_catalogue_sha256'] != records[0].source_catalogue_sha256
            or metadata.get('benchmark_text_indexed') is not False or metadata.get('minimum_reencoded_parent_cosine', 0) < 0.9999):
        raise ValueError('Official child cache provenance validation failed')
    with np.load(args.cache, allow_pickle=False) as cache:
        vectors = cache['fragment_vectors'].copy()
        if cache['fragment_ids'].tolist() != [fragment.fragment_id for fragment in fragments]:
            raise ValueError('Fragment vector order mismatch')
    if vectors.shape != (len(fragments), 384) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
        raise ValueError('Invalid child vectors')
    name = 'isco08_parent_children_e5small_' + metadata['cache_sha256'][:16]
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'collection': name,
                'source_catalogue_sha256': metadata['source_catalogue_sha256'],
                'fragment_map_sha256': fragment_hash, 'fragment_cache_sha256': metadata['cache_sha256'],
                'encoder_id': metadata['encoder_id'], 'encoder_revision': metadata['encoder_revision'],
                'encoder_weights_sha256': metadata['weights_sha256'], 'points': len(fragments),
                'parents': 436, 'operation': 'new local official-source child index', 'existing_collections_modified': False,
                'status': 'starting'}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(url='http://127.0.0.1:6333', timeout=10, check_compatibility=False)
    try:
        if client.collection_exists(name):
            raise ValueError('Content-addressed child collection already exists; refusing overwrite')
        args.manifest.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        client.create_collection(name, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
        for offset in range(0, len(fragments), 256):
            points = [models.PointStruct(id=str(uuid5(NAMESPACE, fragment.fragment_id)), vector=vectors[index].tolist(),
                payload={'unit_code': fragment.code, 'fragment_id': fragment.fragment_id, 'kind': fragment.kind,
                         'text': fragment.text, 'source_catalogue_sha256': metadata['source_catalogue_sha256'],
                         'encoder_weights_sha256': metadata['weights_sha256'], 'index_owner': 'Multilingual-LFS/parent-document/v1'})
                for index, fragment in enumerate(fragments[offset:offset + 256], start=offset)]
            client.upsert(name, points=points, wait=True)
        count = client.count(name, exact=True).count
        if count != len(fragments):
            raise RuntimeError('Published child count mismatch')
        manifest.update(status='ready', verified_point_count=count)
        args.manifest.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    finally:
        client.close()
    print(json.dumps({'status': 'ready', 'collection': name, 'points': count, 'parents': 436}))


if __name__ == '__main__':
    main()
