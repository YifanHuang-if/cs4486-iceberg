"""Nested training-only selection of Gaussian angle evidence.

The outer query labels are never provided. Hyperparameters minimize angle-only
inner-CV loss, independently of outer CNN predictions and labels.
"""
import numpy as np
from src.v3.fusion import prior, fuse, loss, pool_queries

GRID=[(sigma,alpha) for sigma in (.005,.01,.02,.04) for alpha in (.03,.1,.3)]

def select_parameters(angles, labels, seed=2026):
    angles=np.asarray(angles); labels=np.asarray(labels)
    order=np.random.default_rng(seed).permutation(len(labels))
    inner=np.empty(len(labels),dtype=int);inner[order]=np.arange(len(labels))%3
    observed=np.isfinite(angles)
    candidates=[]
    for sigma,alpha in GRID:
        p=np.full(len(labels),.5)
        for fold in range(3):
            query=inner==fold
            p[query],_=prior(angles[~query],labels[~query],angles[query],sigma,alpha)
        candidates.append(dict(sigma=sigma,alpha=alpha,
                               inner_angle_log_loss=loss(labels[observed],p[observed])))
    selected=min(candidates,key=lambda x:x["inner_angle_log_loss"])
    return selected,candidates

def nested_predict(model_p,reference_angles,reference_labels,query_angles,pool=False):
    selected,candidates=select_parameters(reference_angles,reference_labels)
    ap,covered=prior(reference_angles,reference_labels,query_angles,
                     selected["sigma"],selected["alpha"])
    image=pool_queries(model_p,query_angles) if pool else model_p
    return fuse(image,ap,covered),dict(selected=selected,candidates=candidates)

def nested_oof(model_p,angles,labels,folds,pool=False):
    out=np.full(len(labels),np.nan);records=[]
    for fold in np.unique(folds):
        query=folds==fold
        out[query],report=nested_predict(model_p[query],angles[~query],labels[~query],angles[query],pool)
        report["outer_fold"]=int(fold)
        records.append(report)
    return out,records
