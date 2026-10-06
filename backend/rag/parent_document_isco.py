"""Multi-vector child retrieval mapped to authoritative ISCO parent records.

Each occupation retains separate title, definition and included-example
representations. Similar child passages vote for their parent unit code;
no early taxonomy decision eliminates a leaf. Scores are similarity signals,
not calibrated probabilities. Catalogue fragments never use benchmark text.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re

import numpy as np

from backend.rag.official_isco08_catalogue import ENRICHED_PROFILE, load_enriched_catalogue


@dataclass(frozen=True)
class ISCOFragment:
    fragment_id: str
    code: str
    kind: str
    text: str


def catalogue_fragments(path: Path):
    """Validate the official source and preserve independently sourced examples."""
    records = load_enriched_catalogue(path, profile=ENRICHED_PROFILE)
    units = {record.code: record for record in records if record.level == 'unit'}
    fragments = []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = {row['code']: row for row in csv.DictReader(stream) if row['level'] == 'unit'}
    for code, record in sorted(units.items()):
        row = rows[code]
        texts = [('title', record.title_en)]
        definition = row['definition'].strip()
        if definition:
            words = definition.split()
            texts.extend(('definition', ' '.join(words[start:start + 120]))
                         for start in range(0, len(words), 120))
        examples = re.findall(r'(?:^|\n)\s*[-•]\s*([^\n]+)', row['included_occupations'])
        texts.extend(('example', example.strip()) for example in examples if example.strip())
        seen = set()
        for kind, text in texts:
            normalized = ' '.join(text.split())
            if not normalized or (kind, normalized.casefold()) in seen:
                continue
            seen.add((kind, normalized.casefold()))
            digest = hashlib.sha256(f'{code}\x1f{kind}\x1f{normalized}'.encode('utf-8')).hexdigest()
            fragments.append(ISCOFragment(digest, code, kind, normalized))
    return records, fragments


def aggregate_children(scores: np.ndarray, fragment_codes: list[str], unit_codes: list[str], *, mode='max') -> np.ndarray:
    """Aggregate child similarities without a hard major/submajor filter."""
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(fragment_codes) or not np.isfinite(values).all():
        raise ValueError('Invalid child similarity matrix')
    if mode not in ('max', 'mean_top2') or len(set(unit_codes)) != len(unit_codes):
        raise ValueError('Invalid aggregation mode or duplicate parent code')
    if set(fragment_codes) != set(unit_codes):
        raise ValueError('Child and parent code sets differ')
    result = np.empty((len(values), len(unit_codes)), dtype=np.float32)
    indices = {code: [] for code in unit_codes}
    for index, code in enumerate(fragment_codes):
        indices[code].append(index)
    for index, code in enumerate(unit_codes):
        children = values[:, indices[code]]
        if mode == 'max' or children.shape[1] == 1:
            result[:, index] = children.max(axis=1)
        else:
            result[:, index] = np.partition(children, -2, axis=1)[:, -2:].mean(axis=1)
    return result


def combine_parent_evidence(child_scores: np.ndarray, definition_scores: np.ndarray, *, child_weight: float) -> np.ndarray:
    if isinstance(child_weight, bool) or not isinstance(child_weight, (int, float)) or not math.isfinite(child_weight) or not 0 <= child_weight <= 1:
        raise ValueError('child_weight must be a finite number in [0,1]')
    children, definitions = np.asarray(child_scores), np.asarray(definition_scores)
    if children.shape != definitions.shape or not np.isfinite(children).all() or not np.isfinite(definitions).all():
        raise ValueError('Parent evidence matrices must be finite and aligned')
    return child_weight * children + (1 - child_weight) * definitions


class ParentDocumentISCORetriever:
    """Lightweight local search over cached child vectors and full parent records."""
    def __init__(self, *, records, fragments, fragment_vectors, definition_vectors, unit_codes,
                 embed_query, child_weight, aggregation):
        self.records = {record.code: record for record in records if record.level == 'unit'}
        self.fragments = list(fragments)
        self.fragment_vectors = np.asarray(fragment_vectors, dtype=np.float32)
        self.definition_vectors = np.asarray(definition_vectors, dtype=np.float32)
        self.unit_codes = list(unit_codes)
        self.embed_query = embed_query
        self.child_weight, self.aggregation = child_weight, aggregation
        if set(self.records) != set(self.unit_codes):
            raise ValueError('Parent records and unit vectors differ')
        if self.fragment_vectors.shape != (len(self.fragments), 384) or self.definition_vectors.shape != (len(self.unit_codes), 384):
            raise ValueError('Parent retrieval vectors must match the 384-dimensional encoder')
        if not np.isfinite(self.fragment_vectors).all() or not np.isfinite(self.definition_vectors).all():
            raise ValueError('Nonfinite retrieval vectors')
        if not np.allclose(np.linalg.norm(self.fragment_vectors, axis=1), 1, atol=1e-3) or not np.allclose(np.linalg.norm(self.definition_vectors, axis=1), 1, atol=1e-3):
            raise ValueError('Retrieval vectors must be normalized')
        combine_parent_evidence(np.zeros((1, 1)), np.zeros((1, 1)), child_weight=child_weight)
        aggregate_children(np.zeros((1, len(self.fragments))), [fragment.code for fragment in self.fragments], self.unit_codes, mode=aggregation)

    def search(self, query: str, *, top_k=5):
        if not isinstance(query, str) or not query.strip() or isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 436:
            raise ValueError('Nonblank query and valid top_k required')
        vector = np.asarray(self.embed_query(query.strip()), dtype=np.float32)
        if vector.shape != (384,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-3):
            raise ValueError('Query embedding must be a finite normalized 384-vector')
        child_similarities = vector[None, :] @ self.fragment_vectors.T
        aggregated = aggregate_children(child_similarities, [fragment.code for fragment in self.fragments], self.unit_codes, mode=self.aggregation)
        parent_similarities = vector[None, :] @ self.definition_vectors.T
        scores = combine_parent_evidence(aggregated, parent_similarities, child_weight=self.child_weight)[0]
        ordering = sorted(range(len(self.unit_codes)), key=lambda index: (-float(scores[index]), self.unit_codes[index]))[:top_k]
        output = []
        for index in ordering:
            code = self.unit_codes[index]
            evidence_indices = [i for i, fragment in enumerate(self.fragments) if fragment.code == code]
            best_child = max(evidence_indices, key=lambda i: float(child_similarities[0, i]))
            record = self.records[code]
            output.append({'code': code, 'title_en': record.title_en,
                'score': float(scores[index]), 'definition_score': float(parent_similarities[0, index]),
                'child_score': float(aggregated[0, index]), 'fragment_id': self.fragments[best_child].fragment_id,
                'fragment_kind': self.fragments[best_child].kind, 'fragment_text': self.fragments[best_child].text,
                'parent_definition': record.embedding_text, 'hierarchy_path': [code[:length] for length in (1, 2, 3, 4)]})
        return output
