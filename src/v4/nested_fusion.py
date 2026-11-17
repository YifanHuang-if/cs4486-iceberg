"""Train fusion weights inside each outer training fold using shape-MLP proxies.

The proxy enables affordable inner CV. Transfer to CNN probabilities is an
experimental assumption, assessed only on the outer folds. No outer labels enter
parameter selection. Every inner model is trained for a fixed 160 epochs.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from src.v3.train import ROOT, OUT, fit_meta, meta_transform, digest_ids, save_csv
from src.v3.shape_model import get_features, network, probabilities
from src.v3.evaluate import load_metadata, load_tag
from src.v3.fusion import prior, logit, sigmoid, loss, fold_fusion

# Declared before any new outer-fold scoring. Exact and nonexact evidence have
# different coefficients; weak L2 shrinkage anchors them to the V3 rule.
GRID=[(s,a) for s in (.005,.01,.02,.04) for a in (.03,.1,.3)]
ANCHOR=np.array([.75,1.,.75,1.])
RIDGE=.002

def design(model, ap, covered, exact):
    z=logit(model);az=logit(ap)
    x=np.column_stack([z*exact,az*exact,z*~exact,az*~exact])
    return x,covered

def predict_with_fit(model,ra,ry,qa,fit):
    ap,covered=prior(ra,ry,qa,fit['sigma'],fit['alpha'])
    exact=np.isin(qa,np.asarray(ra)[np.isfinite(ra)]) & np.isfinite(qa)
    x,_=design(model,ap,covered,exact)
    result=np.asarray(model,dtype=float).copy()
    result[covered]=sigmoid(x[covered]@np.asarray(fit['weights']))
    return np.clip(result,1e-6,1-1e-6)

def proxy_predictions(features,labels,seed=7301):
    order=np.random.default_rng(seed).permutation(len(labels))
    folds=np.empty(len(labels),int);folds[order]=np.arange(len(labels))%3
    pred=np.zeros(len(labels));states=[]
    for k in range(3):
        tr=folds!=k;va=~tr;fitted=fit_meta(features[tr])
        tx=np.clip(meta_transform(features[tr],fitted),-8,8)
        vx=np.clip(meta_transform(features[va],fitted),-8,8)
        torch.manual_seed(seed+k)
        model=network(features.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=160,eta_min=.00001)
        generator=torch.Generator().manual_seed(seed+k)
        x=torch.from_numpy(tx);y=torch.from_numpy(labels[tr].astype(np.float32))
        for epoch in range(160):
            model.train();permutation=torch.randperm(len(x),generator=generator)
            for start in range(0,len(x),128):
                ix=permutation[start:start+128];opt.zero_grad(set_to_none=True)
                objective=torch.nn.functional.binary_cross_entropy_with_logits(model(x[ix]).squeeze(1),y[ix])
                objective.backward();opt.step()
            schedule.step()
        pred[va]=probabilities(model,vx)
        states.append(dict(state=model.state_dict(),fit=fitted,inner_fold=k))
    return pred,folds,states

def select_fit(angles,labels,proxy,inner):
    records=[];observed=np.isfinite(angles)
    for sigma,alpha in GRID:
        ap=np.empty(len(labels));covered=np.zeros(len(labels),bool);exact=np.zeros(len(labels),bool)
        for k in range(3):
            va=inner==k;tr=~va
            ap[va],covered[va]=prior(angles[tr],labels[tr],angles[va],sigma,alpha)
            exact[va]=np.isin(angles[va],angles[tr][np.isfinite(angles[tr])]) & np.isfinite(angles[va])
        x,_=design(proxy,ap,covered,exact);use=observed & covered
        xx=x[use];yy=labels[use]
        def objective(w):
            z=xx@w
            v=np.mean(np.logaddexp(0,z)-yy*z)+RIDGE*np.square(w-ANCHOR).sum()
            g=xx.T@(sigmoid(z)-yy)/len(yy)+2*RIDGE*(w-ANCHOR)
            return float(v),g
        fit=minimize(objective,ANCHOR.copy(),jac=True,bounds=[(.1,2.5),(0.,2.5),(.1,2.5),(0.,2.5)],method='L-BFGS-B')
        p=np.array(proxy,copy=True);p[covered]=sigmoid(x[covered]@fit.x)
        records.append(dict(sigma=sigma,alpha=alpha,weights=fit.x.tolist(),
            inner_combined_loss=loss(labels[observed],p[observed]),converged=bool(fit.success)))
    return min(records,key=lambda r:r['inner_combined_loss']),records

def get_fit(fold,data,features):
    out=OUT/'linux_nested_proxy';out.mkdir(parents=True,exist_ok=True)
    tr=data['folds']!=fold;ids_hash=digest_ids(data['ids'][tr]);path=out/f'fold{fold}_fit.json'
    if path.exists():
        record=json.loads(path.read_text());assert record['train_ids_sha256']==ids_hash
        return record
    proxy,inner,states=proxy_predictions(features[tr],data['labels'][tr])
    selected,grid=select_fit(data['angles'][tr],data['labels'][tr],proxy,inner)
    record=dict(fold=fold,train_ids_sha256=ids_hash,selected=selected,candidates=grid,
        inner_proxy_loss=loss(data['labels'][tr],proxy),seed=7301,epochs=160,ridge=RIDGE)
    torch.save(states,out/f'fold{fold}_inner_models.pt')
    np.savez(out/f'fold{fold}_inner_oof.npz',ids=data['ids'][tr],probability=proxy,inner_folds=inner)
    path.write_text(json.dumps(record,indent=2));print('FIT',fold,selected,flush=True)
    return record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tags',nargs='+')
    parser.add_argument('--name',default='linux_nested_fusion');parser.add_argument('--fit-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    data=load_metadata();features,_=get_features()
    records=[get_fit(fold,data,features) for fold in range(5)]
    if args.fit_only:return
    if not args.tags:raise ValueError('Need complete outer prediction tags')
    base=np.mean([load_tag(t,data)[0] for t in args.tags],axis=0)
    fused=np.empty(len(base));test_predictions=[];by_fold=[]
    control,_=fold_fusion(base,data['angles'],data['labels'],data['folds'])
    for fold,record in enumerate(records):
        tr=data['folds']!=fold;va=~tr
        fused[va]=predict_with_fit(base[va],data['angles'][tr],data['labels'][tr],data['angles'][va],record['selected'])
        test_base=np.mean([np.load(OUT/t/f'fold{fold}_predictions.npz')['test_probability'] for t in args.tags],axis=0)
        test_predictions.append(predict_with_fit(test_base,data['angles'][tr],data['labels'][tr],data['test_angles'],record['selected']))
        by_fold.append(dict(fold=fold,baseline=loss(data['labels'][va],control[va]),candidate=loss(data['labels'][va],fused[va])))
    report=dict(name=args.name,tags=args.tags,baseline=loss(data['labels'],control),candidate=loss(data['labels'],fused),
        metric='pooled development five-fold OOF log loss',official_kaggle_score=None,target=.083,
        target_met=bool(loss(data['labels'],fused)<.083),by_fold=by_fold,fits=records,
        limitation='Fusion learned on inner shape-MLP proxies, then transferred to outer ensemble probabilities. Development selection; no independent test.')
    dest=ROOT/'results/v4';dest.mkdir(parents=True,exist_ok=True)
    out=OUT/'submissions';out.mkdir(parents=True,exist_ok=True)
    test=np.mean(test_predictions,axis=0)
    np.savez(out/f'{args.name}_predictions.npz',ids=data['ids'],oof=fused,test_ids=data['test_ids'],test_probability=test)
    save_csv(out/f'{args.name}.csv',['id','is_iceberg'],zip(data['test_ids'],test))
    report['submission_sha256']=hashlib.sha256((out/f'{args.name}.csv').read_bytes()).hexdigest()
    (dest/f'{args.name}.json').write_text(json.dumps(report,indent=2));print('RESULT',args.name,report['candidate'],flush=True)

if __name__=='__main__':main()
