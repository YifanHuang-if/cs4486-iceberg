"""Angle neighbors weighted by radar-feature similarity, fitted inside each fold.

Exact matches retain the original prior. Nonexact neighbors are reweighted by
per-image shape/statistics distance; the original angle kernel mass is preserved.
All feature scaling, inner proxies and fusion selection exclude the outer fold.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from src.v3.train import ROOT,OUT,fit_meta,meta_transform,digest_ids,save_csv
from src.v3.shape_model import get_features
from src.v3.evaluate import load_metadata,load_tag
from src.v3.fusion import logit,sigmoid,loss,fold_fusion
from src.v4.nested_fusion import get_fit,design,ANCHOR,RIDGE

GRID=[(s,a,b) for s in (.01,.02,.04) for a in (.03,.1) for b in (.5,1.,2.)]

def distances(ref,query):
    fit=fit_meta(ref[:,2:])
    x=np.clip(meta_transform(ref[:,2:],fit),-8,8)
    q=np.clip(meta_transform(query[:,2:],fit),-8,8)
    return cdist(q,x,'sqeuclidean')/x.shape[1]

def joint_prior(ra,ry,qa,d2,sigma=.02,alpha=.1,bandwidth=1.):
    a=np.asarray(ra,float);y=np.asarray(ry,float);q=np.asarray(qa,float)
    observed=np.isfinite(a)
    if d2.shape!=(len(q),len(a)):raise ValueError('unaligned feature distances')
    a=a[observed];y=y[observed];d2=d2[:,observed]
    if not len(a):return np.full(len(q),.5),np.zeros(len(q),bool)
    delta=q[:,None]-a[None,:];exact=delta==0;has_exact=exact.any(1)
    kernel=np.exp(-.5*(delta/sigma)**2);kernel[~np.isfinite(kernel)]=0
    mass=kernel.sum(1)
    # Subtract the row minimum for numerical stability; it cancels in normalization.
    similarity=np.exp(-.5*(d2-d2.min(1,keepdims=True))/bandwidth**2)
    weighted=kernel*similarity
    weighted*=np.divide(mass,weighted.sum(1),out=np.zeros_like(mass),where=weighted.sum(1)>0)[:,None]
    weights=np.where(has_exact[:,None],exact,weighted)
    mass=weights.sum(1)
    p=(weights@y+alpha)/(mass+2*alpha)
    return p,(mass>1e-3)&np.isfinite(q)

def fit_outer(fold,data,features):
    tr=data['folds']!=fold;source=get_fit(fold,data,features)
    dest=OUT/'linux_joint_proxy';dest.mkdir(parents=True,exist_ok=True)
    path=dest/f'fold{fold}_fit.json';trainhash=digest_ids(data['ids'][tr])
    if path.exists():
        record=json.loads(path.read_text());assert record['train_ids_sha256']==trainhash;return record
    saved=np.load(OUT/'linux_nested_proxy'/f'fold{fold}_inner_oof.npz')
    assert np.array_equal(saved['ids'],data['ids'][tr])
    proxy=saved['probability'];inner=saved['inner_folds'];a=data['angles'][tr];y=data['labels'][tr];x=features[tr]
    cached={k:distances(x[inner!=k],x[inner==k]) for k in range(3)}
    records=[]
    for sigma,alpha,bandwidth in GRID:
        p=np.zeros(len(y));cover=np.zeros(len(y),bool);exact=np.zeros(len(y),bool)
        for k in range(3):
            va=inner==k;rr=~va
            p[va],cover[va]=joint_prior(a[rr],y[rr],a[va],cached[k],sigma,alpha,bandwidth)
            exact[va]=np.isin(a[va],a[rr][np.isfinite(a[rr])])&np.isfinite(a[va])
        matrix,_=design(proxy,p,cover,exact);use=cover&np.isfinite(a);xx=matrix[use];yy=y[use]
        def objective(w):
            z=xx@w
            return float(np.mean(np.logaddexp(0,z)-yy*z)+RIDGE*np.square(w-ANCHOR).sum()),xx.T@(sigmoid(z)-yy)/len(yy)+2*RIDGE*(w-ANCHOR)
        fit=minimize(objective,ANCHOR.copy(),jac=True,bounds=[(.1,2.5),(0,2.5),(.1,2.5),(0,2.5)],method='L-BFGS-B')
        probability=proxy.copy();probability[cover]=sigmoid(matrix[cover]@fit.x)
        records.append(dict(sigma=sigma,alpha=alpha,bandwidth=bandwidth,weights=fit.x.tolist(),
            inner_combined_loss=loss(y[np.isfinite(a)],probability[np.isfinite(a)]),converged=bool(fit.success)))
    record=dict(fold=fold,train_ids_sha256=trainhash,selected=min(records,key=lambda r:r['inner_combined_loss']),candidates=records)
    path.write_text(json.dumps(record,indent=2));print('JOINT FIT',fold,record['selected'],flush=True);return record

def apply(model,ra,ry,qa,ref_features,query_features,fit):
    out=np.array(model,dtype=float,copy=True)
    for lo in range(0,len(qa),512):
        q=qa[lo:lo+512];d2=distances(ref_features,query_features[lo:lo+512])
        p,cover=joint_prior(ra,ry,q,d2,fit['sigma'],fit['alpha'],fit['bandwidth'])
        exact=np.isin(q,ra[np.isfinite(ra)])&np.isfinite(q)
        x,_=design(out[lo:lo+512],p,cover,exact)
        out[lo:lo+512][cover]=sigmoid(x[cover]@np.asarray(fit['weights']))
    return np.clip(out,1e-6,1-1e-6)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tags',nargs='+');parser.add_argument('--name',default='linux_joint')
    parser.add_argument('--fit-only',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    data=load_metadata();features,test_features=get_features()
    fits=[fit_outer(f,data,features) for f in range(5)]
    if args.fit_only:return
    base=np.mean([load_tag(t,data)[0] for t in args.tags],axis=0)
    control,_=fold_fusion(base,data['angles'],data['labels'],data['folds'])
    pred=np.empty(len(base));tests=[];rows=[]
    for fold,record in enumerate(fits):
        tr=data['folds']!=fold;va=~tr;fit=record['selected']
        pred[va]=apply(base[va],data['angles'][tr],data['labels'][tr],data['angles'][va],features[tr],features[va],fit)
        tb=np.mean([np.load(OUT/t/f'fold{fold}_predictions.npz')['test_probability'] for t in args.tags],axis=0)
        tests.append(apply(tb,data['angles'][tr],data['labels'][tr],data['test_angles'],features[tr],test_features,fit))
        rows.append(dict(fold=fold,baseline=loss(data['labels'][va],control[va]),candidate=loss(data['labels'][va],pred[va])))
    report=dict(name=args.name,tags=args.tags,baseline=loss(data['labels'],control),candidate=loss(data['labels'],pred),
        metric='pooled development five-fold OOF log loss',official_kaggle_score=None,by_fold=rows,fits=fits,
        target=.083,target_met=bool(loss(data['labels'],pred)<.083),
        limitation='Inner fusion is learned on shape-MLP proxies; CNN transfer and outer model selection remain development experiments.')
    out=OUT/'submissions';out.mkdir(parents=True,exist_ok=True);test=np.mean(tests,axis=0)
    np.savez(out/f'{args.name}_predictions.npz',ids=data['ids'],oof=pred,test_ids=data['test_ids'],test_probability=test)
    save_csv(out/f'{args.name}.csv',['id','is_iceberg'],zip(data['test_ids'],test))
    report['submission_sha256']=hashlib.sha256((out/f'{args.name}.csv').read_bytes()).hexdigest()
    dest=ROOT/'results/v4';dest.mkdir(parents=True,exist_ok=True);(dest/f'{args.name}.json').write_text(json.dumps(report,indent=2))
    print('RESULT',args.name,report['candidate'],flush=True)

if __name__=='__main__':main()
