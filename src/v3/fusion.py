"""Fold-isolated angle evidence. No query labels are accepted by this API."""
from __future__ import annotations
import numpy as np

def logit(p):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1-1e-6)
    return np.log(q) - np.log1p(-q)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))

def loss(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1-1e-7)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))

def prior(reference_angles, reference_labels, query_angles, sigma=.02, alpha=.1):
    if sigma <= 0 or alpha <= 0:
        raise ValueError("sigma and alpha must be positive")
    a = np.asarray(reference_angles, dtype=np.float64)
    y = np.asarray(reference_labels, dtype=np.float64)
    q = np.asarray(query_angles, dtype=np.float64)
    if len(a) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("reference labels must be aligned binary labels")
    valid = np.isfinite(a)
    a, y = a[valid], y[valid]
    out, coverage = np.full(len(q), .5), np.zeros(len(q), bool)
    if not len(a):
        return out, coverage
    for start in range(0, len(q), 512):
        stop = min(start+512, len(q))
        distance = q[start:stop, None] - a[None, :]
        exact = (distance == 0).astype(np.float64)
        kernel = np.exp(-.5*(distance/sigma)**2)
        kernel[~np.isfinite(kernel)] = 0
        weights = np.where((exact.sum(1)>0)[:, None], exact, kernel)
        mass = weights.sum(1)
        out[start:stop] = (weights@y+alpha)/(mass+2*alpha)
        coverage[start:stop] = (mass>1e-3) & np.isfinite(q[start:stop])
    return out, coverage

def fuse(model_p, angle_p, covered, model_weight=.75, prior_weight=1.):
    out = np.array(model_p, dtype=np.float64, copy=True)
    out[covered] = sigmoid(model_weight*logit(out[covered])+prior_weight*logit(angle_p[covered]))
    return np.clip(out, 1e-6, 1-1e-6)

def fold_fusion(model_p, angles, labels, folds, **kwargs):
    out = np.full(len(labels), np.nan)
    covered = np.zeros(len(labels), bool)
    for fold in np.unique(folds):
        heldout = folds == fold
        p, mask = prior(angles[~heldout], labels[~heldout], angles[heldout])
        out[heldout] = fuse(model_p[heldout], p, mask, **kwargs)
        covered[heldout] = mask
    return out, covered

def soft_targets(probability, floor=.02):
    if not 0 < floor < .5:
        raise ValueError("floor must be in (0, .5)")
    p = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("invalid probabilities")
    return np.clip(p, floor, 1-floor).astype(np.float32)

def pool_queries(probability, angles, weight=.5):
    """Label-free, batch-dependent exact-angle pooling; missing angles stay separate."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must be in [0, 1]")
    p=np.asarray(probability,dtype=np.float64)
    a=np.asarray(angles,dtype=np.float64)
    out=p.copy()
    for key in np.unique(a[np.isfinite(a)]):
        rows=a==key
        if rows.sum()>1:
            out[rows]=(1-weight)*p[rows]+weight*p[rows].mean()
    return out

def pooled_fold_fusion(model_p, angles, labels, folds, pool_weight=.5):
    out=np.full(len(labels),np.nan)
    for fold in np.unique(folds):
        heldout=folds==fold
        pooled=pool_queries(model_p[heldout],angles[heldout],pool_weight)
        ap,covered=prior(angles[~heldout],labels[~heldout],angles[heldout])
        out[heldout]=fuse(pooled,ap,covered)
    return out

