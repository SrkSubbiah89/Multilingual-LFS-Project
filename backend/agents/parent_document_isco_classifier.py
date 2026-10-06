"""Live, provenance-checked parent-document ISCO retrieval through Qdrant.

Retrieves the best official child for every unit group and blends its cosine
with the full parent-definition cosine. No major-group filter removes codes.
The configuration is selected on development cases; scores remain uncalibrated.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from backend.agents.isco_classifier import ISCOClassification, ISCOMatch, _detect_script
from backend.rag import make_qdrant_client
from backend.rag.official_isco08_catalogue import ENRICHED_PROFILE, PROFILE_COLLECTION_NAMES
from backend.rag.parent_document_isco import catalogue_fragments

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'backend/rag/parent_isco_config.json'
SELECTION = ROOT / 'backend/rag/parent_isco_selection.json'
METHOD = 'isco_parent_document_rag'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def _validate_selection(config, selection_path=SELECTION):
    selection = json.loads(Path(selection_path).read_text(encoding='utf-8'))
    digest = selection.pop('selection_sha256')
    if (_digest(selection) != digest or digest != config['selection_sha256']
            or selection['config'] != {'child_weight': config['child_weight'], 'aggregation': config['aggregation']}
            or selection['validation_labels_accessed_for_selection'] is not False):
        raise ValueError('Live parameters differ from the frozen development selection')
    if any(selection['query_encoder'][key] != config[config_key] for key, config_key in
           [('encoder_id', 'encoder_id'), ('encoder_revision', 'encoder_revision'),
            ('weights_sha256', 'encoder_weights_sha256')]):
        raise ValueError('Live encoder differs from the frozen development selection')


class ParentDocumentISCOClassifier:
    def __init__(self, *, client=None, embed_query=None, config_path=CONFIG,
                 translate_before_retrieval=False):
        """translate_before_retrieval (default False) translates a non-English
        query to English before embedding, via the local model used elsewhere
        in this codebase. Default False reproduces the measured 38.83%
        configuration byte for byte.

        UNMEASURED ON THIS PATH. The +20pp dev-sample gain recorded for the
        legacy classifier used a different retrieval method, and no
        development, validation or heldout run exists for this combination.
        Enabling it changes the served configuration away from the one bound
        to the recorded serving identity, so it must not be enabled in a run
        reported as the measured parent-document result until it has its own
        dev-selected, heldout-confirmed evidence.
        """
        self.translate_before_retrieval = bool(translate_before_retrieval)
        self.config = json.loads(Path(config_path).read_text(encoding='utf-8'))
        config = self.config
        if (config['profile'] != ENRICHED_PROFILE or config['aggregation'] != 'max'
                or isinstance(config['child_weight'], bool)
                or not isinstance(config['child_weight'], (int, float))
                or not math.isfinite(config['child_weight']) or not 0 <= config['child_weight'] <= 1
                or config['parents'] != 436 or config['encoder_id'] != 'intfloat/multilingual-e5-small'):
            raise ValueError('Unsupported parent-document retrieval configuration')
        _validate_selection(config)
        records, fragments = catalogue_fragments(ROOT / config['source_csv'])
        self.units = {record.code: record for record in records if record.level == 'unit'}
        self.fragments = {fragment.fragment_id: fragment for fragment in fragments}
        if (records[0].source_catalogue_sha256 != config['source_catalogue_sha256']
                or _digest([vars(fragment) for fragment in fragments]) != config['fragment_map_sha256']
                or len(self.fragments) != config['points']):
            raise ValueError('Official source differs from the measured parent-document index')
        self._owns_client = client is None
        self.client = client or make_qdrant_client(host=os.getenv('QDRANT_HOST', 'localhost'),
                            port=int(os.getenv('QDRANT_PORT', '6333')), client_cls=QdrantClient, timeout=10)
        self.parent_collection = PROFILE_COLLECTION_NAMES[ENRICHED_PROFILE]['unit']
        try:
            self._validate_index()
            self.embed_query = embed_query or self._load_encoder()
        except Exception:
            if self._owns_client:
                self.client.close()
            raise
        self.ready = True

    def _scroll(self, collection, maximum):
        points, offset = [], None
        while True:
            batch, offset = self.client.scroll(collection_name=collection, offset=offset,
                                    limit=256, with_payload=True, with_vectors=True)
            points.extend(batch)
            if len(points) > maximum:
                raise ValueError('Unexpected parent-document index size')
            if offset is None:
                return points

    def _validate_child(self, point, expected_code=None):
        payload = point.payload or {}
        fragment = self.fragments.get(payload.get('fragment_id'))
        if (fragment is None or payload.get('unit_code') != fragment.code
                or (expected_code is not None and fragment.code != expected_code)
                or payload.get('kind') != fragment.kind or payload.get('text') != fragment.text
                or payload.get('source_catalogue_sha256') != self.config['source_catalogue_sha256']
                or payload.get('encoder_weights_sha256') != self.config['encoder_weights_sha256']
                or payload.get('index_owner') != 'Multilingual-LFS/parent-document/v1'):
            raise ValueError('Unverified child occupation evidence')
        return fragment

    def _validate_parent(self, point):
        payload = point.payload or {}
        record = self.units.get(payload.get('code'))
        if record is None or any(payload.get(field) != getattr(record, field) for field in
                  ('level', 'parent_code', 'title_en', 'profile', 'source_catalogue_sha256', 'embedding_text')):
            raise ValueError('Unverified parent occupation definition')
        return record

    def _validate_index(self):
        children = self._scroll(self.config['collection'], self.config['points'])
        for point in children:
            self._validate_child(point)
        children.sort(key=lambda point: point.payload['fragment_id'])
        if [point.payload['fragment_id'] for point in children] != sorted(self.fragments):
            raise ValueError('Child index is incomplete or contains duplicate fragments')
        parents = self._scroll(self.parent_collection, 436)
        for point in parents:
            self._validate_parent(point)
        parents.sort(key=lambda point: point.payload['code'])
        if [point.payload['code'] for point in parents] != sorted(self.units):
            raise ValueError('Parent index is incomplete or contains duplicate codes')
        for points, key in ((children, 'child_vector_bytes_sha256'), (parents, 'parent_vector_bytes_sha256')):
            vectors = np.asarray([point.vector for point in points], dtype=np.float32)
            if (vectors.shape != (len(points), 384) or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)
                    or hashlib.sha256(vectors.tobytes()).hexdigest() != self.config[key]):
                raise ValueError('Live vectors differ from the verified measured index')

    def _load_encoder(self):
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(4)
        snapshot = Path(snapshot_download(self.config['encoder_id'], revision=self.config['encoder_revision'],
                                         local_files_only=True))
        weights = snapshot / self.config['weights_file']
        if _file_digest(weights) != self.config['encoder_weights_sha256']:
            raise ValueError('Local query model differs from the measured encoder')
        self.model = SentenceTransformer(str(snapshot), device='cpu', local_files_only=True)

        def embed(text):
            return self.model.encode(['query: ' + text.strip()], normalize_embeddings=True,
                                     show_progress_bar=False, batch_size=1)[0].tolist()
        return embed

    def _translate_to_english(self, text, language):
        """Translate *text* to English with the same local model and fallback
        contract as ISCOClassifier._translate_to_english. Any failure returns
        the original text, so a translation problem degrades to untranslated
        behaviour instead of blocking classification.

        That method uses no instance state, so it is reused here rather than
        duplicated; ``None`` is passed as its ``self`` deliberately, so a
        future change that does rely on instance state raises immediately
        instead of silently reading this class's attributes.
        """
        from backend.agents.isco_classifier import ISCOClassifier
        return ISCOClassifier._translate_to_english(None, text, language)

    def classify(self, job_title, context='', language='', top_k=3, use_llm=True, trace=None,
                 max_stage_latency_ms=None):
        if not isinstance(job_title, str) or not job_title.strip():
            raise ValueError('A nonblank occupation description is required')
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 436:
            raise ValueError('top_k must be between 1 and 436')
        query = job_title.strip()
        # Only the retrieval text is ever translated: `query` keeps the
        # respondent's own words, so a stored result or report never presents
        # a machine translation as something the respondent wrote.
        retrieval_text = query
        # getattr, not attribute access: several tests build instances with
        # object.__new__, and "absent" must mean the measured configuration.
        if getattr(self, 'translate_before_retrieval', False):
            detected = (language or '').strip().lower() or _detect_script(query)
            if detected != 'en':
                translated = self._translate_to_english(query, detected).strip()
                retrieval_text = translated or query
                if trace is not None:
                    trace['translated_before_retrieval'] = retrieval_text != query
                    trace['translation_source_language'] = detected
                    trace['retrieval_text'] = retrieval_text
        vector = np.asarray(self.embed_query(retrieval_text), dtype=np.float32)
        if vector.shape != (384,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-3):
            raise ValueError('Invalid query embedding')
        groups = self.client.query_points_groups(collection_name=self.config['collection'],
                        query=vector.tolist(), group_by='unit_code', group_size=1, limit=436,
                        search_params=models.SearchParams(exact=True), with_payload=True, timeout=10).groups
        parent_points = self.client.query_points(collection_name=self.parent_collection, query=vector.tolist(),
                        limit=436, search_params=models.SearchParams(exact=True), with_payload=True, timeout=10).points
        children, parents = {}, {}
        for group in groups:
            code = str(group.id)
            if code in children or len(group.hits) != 1:
                raise ValueError('Duplicate or missing child result')
            point = group.hits[0]
            fragment = self._validate_child(point, code)
            children[code] = (float(point.score), fragment)
        for point in parent_points:
            record = self._validate_parent(point)
            if record.code in parents:
                raise ValueError('Duplicate parent result')
            parents[record.code] = float(point.score)
        if set(children) != set(self.units) or set(parents) != set(self.units):
            raise RuntimeError('Parent-document retrieval did not cover all 436 official codes')
        weight = self.config['child_weight']
        ranked = []
        for code in self.units:
            child_score, fragment = children[code]
            score = weight * child_score + (1 - weight) * parents[code]
            if not all(math.isfinite(value) for value in (score, child_score, parents[code])):
                raise ValueError('Nonfinite occupation evidence score')
            ranked.append((code, score, child_score, parents[code], fragment))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        selected = ranked[:top_k]

        def match(item):
            return ISCOMatch(code=item[0], title_en=self.units[item[0]].title_en,
                             title_ar='', confidence=item[1])
        primary = selected[0]
        if trace is not None:
            trace.update(method=METHOD, profile=ENRICHED_PROFILE, candidates_scored=436,
                child_weight=weight, aggregation='max', child_collection=self.config['collection'],
                selection_sha256=self.config['selection_sha256'], query_encoder=self.config['encoder_id'],
                source_catalogue_sha256=self.config['source_catalogue_sha256'], reranker_fired=False,
                score_calibrated=False, context_used=False,
                evidence=[{'code': item[0], 'score': item[1], 'child_score': item[2], 'parent_score': item[3],
                           'fragment_id': item[4].fragment_id, 'fragment_kind': item[4].kind} for item in selected])
        return ISCOClassification(query=query, language=language.strip() or _detect_script(query),
            primary=match(primary), alternatives=[match(item) for item in selected[1:]], method=METHOD,
            hierarchy_path=[primary[0][:length] for length in (1, 2, 3, 4)], stage_confidences=None,
            hitl_required=True, reasoning='Official ' + primary[4].kind + ' evidence: ' + primary[4].text
                + '. Matched to its full occupation definition; similarity scores require human review.')
