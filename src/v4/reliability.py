"""Fixed, fold-isolated reliability gating for angle priors.

Experimental alternative to V3 fusion. Never uses held-out query labels.
"""
from __future__ import annotations
import numpy as np
from src.v3.fusion import logit, sigmoid

def gated_prior(reference_angles, reference_labels, query_angles, *, sigma=.02,
                alpha=.1, tau=8., distance_scale=.02, preserve_exact=False):
    """Return smoothed angle prior and evidence-dependent confidence."""
    if min(sigma, alpha, tau, distance_scale) <= 0:
        raise ValueError("All hyperparameters must be positive")
    ref = np.asarray(reference_angles, dtype=np.float64).reshape(-1)
    labels = np.asarray(reference_labels, dtype=np.float64).reshape(-1)
    query = np.asarray(query_angles, dtype=np.float64).reshape(-1)
    if len(ref) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Reference angles and labels must align and be binary")
    valid = np.isfinite(ref)
    ref, labels = ref[valid], labels[valid]
    probabilities = np.full(len(query), .5, dtype=np.float64)
    confidence = np.zeros(len(query), dtype=np.float64)
    if not len(ref):
        return probabilities, confidence
    for start in range(0, len(query), 256):
        q = query[start:start+256]
        present = np.isfinite(q)
        if not np.any(present):
            continue
        indices = start + np.flatnonzero(present)
        distance = np.abs(q[present, None] - ref[None, :])
        exact = distance == 0
        exact_available = exact.any(axis=1)
        kernel = np.exp(-.5 * (distance / sigma)**2)
        weights = np.where(exact_available[:, None], exact.astype(float), kernel)
        mass = weights.sum(axis=1)
        effective = mass**2 / np.maximum((weights**2).sum(axis=1), 1e-14)
        probabilities[indices] = (weights @ labels + alpha) / (mass + 2 * alpha)
        nearest = distance.min(axis=1)
        reliability = effective / (effective + tau)
        reliability *= np.where(exact_available, 1.,
                                np.exp(-.5 * (nearest / distance_scale)**2))
        if preserve_exact:
            reliability = np.where(exact_available, 1., reliability)
        confidence[indices] = np.where(mass > 1e-3, reliability, 0.)
    return probabilities, np.clip(confidence, 0., 1.)

def gated_fuse(model_prob, angle_prob, confidence, *,
               model_weight=.75, angle_weight=1.):
    p = np.asarray(model_prob, dtype=float)
    ap = np.asarray(angle_prob, dtype=float)
    gate = np.asarray(confidence, dtype=float)
    if p.shape != ap.shape or p.shape != gate.shape:
        raise ValueError("Inputs must have identical shapes")
    if not np.isfinite(gate).all() or np.any((gate < 0) | (gate > 1)):
        raise ValueError("Invalid reliability")
    # Gate the full V3 log-odds adjustment, not only the angle term.
    # As the gate approaches zero, the output must approach model_prob.
    out = p.copy()
    use = gate > 0
    if np.any(use):
        z = logit(p[use])
        full_v3 = model_weight * z + angle_weight * logit(ap[use])
        out[use] = sigmoid(z + gate[use] * (full_v3 - z))
    return np.clip(out, 1e-6, 1-1e-6)

def fold_gated_fusion(model_prob, angles, labels, folds, **kwargs):
    """Each held-out fold sees only other folds' labels."""
    p = np.asarray(model_prob, dtype=float)
    a = np.asarray(angles, dtype=float)
    y = np.asarray(labels, dtype=float)
    f = np.asarray(folds)
    if any(len(v) != len(y) for v in (p, a, f)):
        raise ValueError("Inconsistent row counts")
    out, conf = np.empty_like(p), np.empty_like(p)
    for fold in np.unique(f):
        valid = f == fold
        ap, g = gated_prior(a[~valid], y[~valid], a[valid], **kwargs)
        out[valid] = gated_fuse(p[valid], ap, g)
        conf[valid] = g
    return out, conf
