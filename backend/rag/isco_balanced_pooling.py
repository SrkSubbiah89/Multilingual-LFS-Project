"""Global parent pooling from independent official occupation fragments.

Kind balancing gives title, definition and example evidence separate votes.
Log-mean-exp normalizes for child count rather than summing extra votes. These
are uncalibrated cosine aggregations; no early taxonomy filter removes codes.
"""
from __future__ import annotations

import math
import numpy as np

KINDS = ('title', 'definition', 'example')


def validate_pooling_config(config):
    if not isinstance(config, dict) or config.get('kind') not in ('max', 'kind_balanced', 'logmeanexp'):
        raise ValueError('Unsupported child pooling configuration')
    kind = config['kind']
    if kind == 'max':
        if set(config) != {'kind'}:
            raise ValueError('Unexpected maximum-pooling parameters')
    elif kind == 'kind_balanced':
        if set(config) != {'kind', 'weights'} or not isinstance(config['weights'], (list, tuple)) or len(config['weights']) != 3:
            raise ValueError('Three title/definition/example weights required')
        weights = config['weights']
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 for value in weights)
                or not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-8)):
            raise ValueError('Kind weights must be finite, nonnegative and sum to one')
    else:
        temperature = config.get('temperature')
        if (set(config) != {'kind', 'temperature'} or isinstance(temperature, bool)
                or not isinstance(temperature, (int, float)) or not math.isfinite(temperature)
                or temperature <= 0):
            raise ValueError('Pooling temperature must be finite and positive')


def aggregate_balanced_children(scores, fragment_codes, fragment_kinds, unit_codes, *, config):
    """Return one finite similarity per query and parent in caller code order."""
    validate_pooling_config(config)
    values = np.asarray(scores, dtype=np.float32)
    codes, kinds, parents = list(fragment_codes), list(fragment_kinds), list(unit_codes)
    if (values.ndim != 2 or values.shape[1] != len(codes) or len(kinds) != len(codes)
            or not np.isfinite(values).all() or not parents or len(set(parents)) != len(parents)
            or set(codes) != set(parents) or any(kind not in KINDS for kind in kinds)):
        raise ValueError('Child similarities, kinds and parent codes must be finite and aligned')
    indices = {code: [] for code in parents}
    for index, code in enumerate(codes):
        indices[code].append(index)
    output = np.empty((values.shape[0], len(parents)), dtype=np.float32)
    for parent_index, code in enumerate(parents):
        positions = indices[code]
        child = values[:, positions]
        if config['kind'] == 'max':
            pooled = child.max(axis=1)
        elif config['kind'] == 'kind_balanced':
            weighted = np.zeros(values.shape[0], dtype=np.float64)
            denominator = 0.0
            for kind, weight in zip(KINDS, config['weights']):
                selected = [index for index in positions if kinds[index] == kind]
                if selected and weight > 0:
                    weighted += weight * values[:, selected].max(axis=1)
                    denominator += weight
            if denominator <= 0:
                raise ValueError('Every parent must retain positive-weight evidence')
            pooled = weighted / denominator
        else:
            # Subtract the maximum before exponentiation: safe even when
            # temperature is small, and invariant to duplicated identical
            # children when all of a parent's children are duplicated.
            temperature = config['temperature']
            child64 = child.astype(np.float64)
            maximum = child64.max(axis=1)
            pooled = maximum + temperature * np.log(np.exp((child64 - maximum[:, None]) / temperature).mean(axis=1))
        if not np.isfinite(pooled).all():
            raise ValueError('Nonfinite aggregated occupation evidence')
        output[:, parent_index] = pooled
    return output
