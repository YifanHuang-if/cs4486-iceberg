"""Training-only group evidence with optional unlabeled-query prediction pooling."""
import numpy as np
from scipy.special import expit,logit
from src.refinement.context import AngleContext,EmpiricalAngleContext


def query_group_pool(angles,probabilities):
    result=np.asarray(probabilities,dtype=np.float64).copy();groups={}
    for index,angle in enumerate(angles):
        if np.isfinite(angle):groups.setdefault(float(angle).hex(),[]).append(index)
    for indices in groups.values():result[indices]=.5*result[indices]+.5*result[indices].mean()
    return result


class OddsContext:
    def __init__(self,weight=.5,empirical=False,pool=False):
        self.weight=weight;self.empirical=empirical;self.pool=pool

    def fit(self,ids,angles,labels):
        self.context=(EmpiricalAngleContext() if self.empirical else AngleContext()).fit(ids,angles,labels)
        self.prior=float(np.clip(np.mean(labels),1e-6,1-1e-6));return self

    def transform(self,ids,angles,probabilities):
        if set(ids)&self.context.training_ids:raise ValueError('Cannot predict fitted records')
        p=query_group_pool(angles,probabilities) if self.pool else np.asarray(probabilities,dtype=np.float64).copy()
        counts=np.zeros(len(p),dtype=int)
        for index,angle in enumerate(angles):
            if not np.isfinite(angle):continue
            n,positive=self.context.groups.get(float(angle).hex(),(0,0.))
            if n==0:continue
            group_prior=(positive+self.context.strength*self.prior)/(n+self.context.strength)
            evidence=logit(group_prior)-logit(self.prior)
            p[index]=expit(logit(np.clip(p[index],1e-7,1-1e-7))+self.weight*evidence);counts[index]=n
        return p,counts
