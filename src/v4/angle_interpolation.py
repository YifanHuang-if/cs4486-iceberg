"""Nearest angle-group interpolation, selected only inside the training fold.

Unlike a fixed Gaussian bandwidth, interpolation uses the adjacent observed
angle groups. Long gaps attenuate towards 0.5. Exact groups keep their posterior.
"""
import numpy as np

def interpolated_prior(reference_angles,reference_labels,query_angles,sigma=.02,alpha=.1):
    a=np.asarray(reference_angles,float);y=np.asarray(reference_labels,float)
    q=np.asarray(query_angles,float)
    if sigma<=0 or alpha<=0: raise ValueError('positive parameters required')
    if len(a)!=len(y) or not np.isin(y,[0,1]).all():raise ValueError('binary aligned reference labels required')
    use=np.isfinite(a);a=a[use];y=y[use]
    out=np.full(len(q),.5);covered=np.zeros(len(q),bool)
    if not len(a):return out,covered
    keys,index=np.unique(a,return_inverse=True)
    count=np.bincount(index);positives=np.bincount(index,weights=y)
    posterior=(positives+alpha)/(count+2*alpha)
    for i,angle in enumerate(q):
        if not np.isfinite(angle):continue
        k=np.searchsorted(keys,angle)
        if k<len(keys) and keys[k]==angle:
            out[i]=posterior[k];covered[i]=True;continue
        left=max(k-1,0);right=min(k,len(keys)-1)
        nearest=min(abs(angle-keys[left]),abs(angle-keys[right]))
        if left==right:p=posterior[left]
        else:
            t=(angle-keys[left])/(keys[right]-keys[left]);p=(1-t)*posterior[left]+t*posterior[right]
        confidence=np.exp(-.5*(nearest/sigma)**2)
        out[i]=.5+confidence*(p-.5);covered[i]=confidence>1e-3
    return out,covered
