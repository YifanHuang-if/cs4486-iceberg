"""The frozen unweighted binary Log Loss and subgroup metric contract."""
import numpy as np


def log_loss(labels, probabilities):
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError('Expected equal non-empty vectors')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid labels/probabilities')
    p = np.clip(p, 1e-7, 1-1e-7)
    return float(-np.mean(y * np.log(p) + (1-y) * np.log1p(-p)))


def metrics(labels, probabilities, missing):
    y, p = np.asarray(labels), np.asarray(probabilities)
    missing = np.asarray(missing, dtype=bool)
    result = {'samples': len(y), 'oof_log_loss': log_loss(y, p), 'accuracy': float(np.mean((p >= .5) == y))}
    for name, mask in [('observed_angle', ~missing), ('missing_angle', missing)]:
        result[name+'_samples'] = int(mask.sum())
        result[name+'_log_loss'] = log_loss(y[mask], p[mask]) if mask.any() else None
    return result
