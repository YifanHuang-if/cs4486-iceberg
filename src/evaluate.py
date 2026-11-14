"""Shared unweighted probability metrics, using the frozen 1e-7 clipping rule."""
import numpy as np

EPSILON = 1e-7


def log_loss(labels, probabilities):
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError('Expected equal non-empty one-dimensional arrays')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid labels or probabilities')
    p = np.clip(p, EPSILON, 1 - EPSILON)
    return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())


def metrics(labels, probabilities, angles):
    y, p = np.asarray(labels), np.asarray(probabilities)
    missing = ~np.isfinite(angles)
    result = {'samples': len(y), 'log_loss': log_loss(y, p),
              'accuracy': float(((p >= .5) == y).mean())}
    for name, mask in [('observed_angle', ~missing), ('missing_angle', missing)]:
        result[f'{name}_samples'] = int(mask.sum())
        result[f'{name}_log_loss'] = log_loss(y[mask], p[mask]) if mask.any() else None
    return result
