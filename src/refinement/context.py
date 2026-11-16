"""A declared exact-angle context prior fitted exclusively to training-fold labels."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln


class AngleContext:
    def __init__(self, strength=1.):
        if strength<=0:raise ValueError('Prior strength must be positive')
        self.strength=float(strength);self.groups={};self.training_ids=set()

    def fit(self, ids, angles, labels):
        if len(ids)!=len(angles) or len(ids)!=len(labels) or len(set(ids))!=len(ids):raise ValueError('Invalid training records')
        if not np.isin(labels,[0,1]).all():raise ValueError('Invalid labels')
        self.training_ids=set(ids);self.groups={}
        for angle,label in zip(angles,labels):
            if np.isfinite(angle):
                key=float(angle).hex();count,positive=self.groups.get(key,(0,0.))
                self.groups[key]=(count+1,positive+float(label))
        return self

    def transform(self, ids, angles, probabilities):
        if set(ids)&self.training_ids:raise ValueError('Context predictions must exclude all fitting IDs')
        p=np.asarray(probabilities,dtype=np.float64).copy();counts=np.zeros(len(p),dtype=int)
        if len(ids)!=len(p) or len(angles)!=len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid query')
        for index,angle in enumerate(angles):
            if not np.isfinite(angle):continue
            count,positive=self.groups.get(float(angle).hex(),(0,0.))
            if count==0:continue
            p[index]=(self.strength*p[index]+positive)/(self.strength+count);counts[index]=count
        return p,counts


class EmpiricalAngleContext(AngleContext):
    """Estimate group-prior concentration from training labels, never query labels."""
    def fit(self,ids,angles,labels):
        super().fit(ids,angles,labels)
        repeated=[v for v in self.groups.values() if v[0]>1]
        if not repeated:
            self.strength=1.;return self
        counts=np.array([v[0] for v in repeated],dtype=np.float64)
        positive=np.array([v[1] for v in repeated],dtype=np.float64)
        mean=float(np.clip(np.mean(labels),.01,.99))
        def objective(log_strength):
            strength=np.exp(log_strength);a=strength*mean;b=strength*(1-mean)
            return -float(np.sum(betaln(a+positive,b+counts-positive)-betaln(a,b)))
        fit=minimize_scalar(objective,bounds=(-6.,6.),method='bounded')
        if not fit.success:raise ValueError('Training-only concentration estimation failed')
        self.strength=float(np.exp(fit.x));return self
