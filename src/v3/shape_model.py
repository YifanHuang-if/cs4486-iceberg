"""Complementary CPU classifier from bright-object geometry and radar statistics."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from src.v3.train import ROOT,OUT,fit_meta,meta_transform,digest_ids
from src.v3.evaluate import load_metadata
from src.v3.fusion import loss,prior,fuse

def features(images,angles):
    # Per-image computations only. No dataset-level fitted values enter here.
    n=len(images);out=[np.asarray(angles),~np.isfinite(angles)]
    yy,xx=np.mgrid[-1:1:75j,-1:1:75j]
    xx=xx.reshape(1,-1);yy=yy.reshape(1,-1)
    channels=[images[:,0],images[:,1],images[:,0]-images[:,1]]
    for image in channels:
        flat=image.astype(np.float64).reshape(n,-1)
        q=np.percentile(flat,[1,10,25,50,75,90,95,98,99,99.5,99.9,100],axis=1)
        out.extend(q)
        mean=flat.mean(1);std=np.maximum(flat.std(1),1e-6)
        z=(flat-mean[:,None])/std[:,None]
        out.extend([mean,std,(z**3).mean(1),np.minimum(z**4,1e5).mean(1)])
        for k in [1,2,3,5]:
            out.append((z>k).mean(1))
        weight=np.maximum(flat-q[6,:,None],0)
        mass=weight.sum(1)+1e-9
        mx=(weight*xx).sum(1)/mass;my=(weight*yy).sum(1)/mass
        dx=xx-mx[:,None];dy=yy-my[:,None]
        cxx=(weight*dx**2).sum(1)/mass;cyy=(weight*dy**2).sum(1)/mass
        cxy=(weight*dx*dy).sum(1)/mass
        trace=cxx+cyy
        anisotropy=np.sqrt((cxx-cyy)**2+4*cxy**2)/np.maximum(trace,1e-8)
        out.extend([np.sqrt(mx*mx+my*my),trace,anisotropy,
                    (weight*(dx*dx+dy*dy)**2).sum(1)/mass])
        crop=image[:,25:50,25:50].reshape(n,-1)
        out.extend([crop.mean(1),crop.std(1),crop.max(1)])
        grad=np.concatenate([np.abs(np.diff(image,axis=1)).reshape(n,-1),
                             np.abs(np.diff(image,axis=2)).reshape(n,-1)],axis=1)
        out.extend([grad.mean(1),np.percentile(grad,90,axis=1)])
    a=images[:,0].reshape(n,-1).astype(np.float64);b=images[:,1].reshape(n,-1).astype(np.float64)
    out.append(((a-a.mean(1,keepdims=True))*(b-b.mean(1,keepdims=True))).mean(1)/(a.std(1)*b.std(1)+1e-9))
    return np.column_stack(out).astype(np.float32)

def network(width):
    return nn.Sequential(nn.Linear(width,96),nn.SiLU(),nn.Dropout(.25),
                         nn.Linear(96,32),nn.SiLU(),nn.Dropout(.15),nn.Linear(32,1))

def get_features():
    target=ROOT/"work/cache/v3_shape_features.npz"
    if target.exists():
        with np.load(target,allow_pickle=False) as f: return f["train"],f["test"]
    pieces=[]
    for split in ["train","test"]:
        raw=np.load(ROOT/"work/cache"/f"{split}.npz",allow_pickle=False)
        images=raw["images"];angles=raw["angles"];parts=[]
        for start in range(0,len(images),128):
            parts.append(features(images[start:start+128],angles[start:start+128]))
        pieces.append(np.concatenate(parts))
        print(f"Extracted {split}: {pieces[-1].shape}",flush=True)
    np.savez(target,train=pieces[0],test=pieces[1])
    return pieces

def probabilities(model,x):
    model.eval()
    with torch.no_grad(): return torch.sigmoid(model(torch.from_numpy(x)).squeeze(1)).numpy().astype(np.float64)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tag",default="v3_shape_mlp")
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    data=load_metadata();a,b=get_features()
    out=OUT/args.tag;out.mkdir(parents=True,exist_ok=True)
    y=data["labels"];folds=data["folds"]
    for fold in range(5):
        train=folds!=fold;valid=~train
        fitted=fit_meta(a[train])
        tx=np.clip(meta_transform(a[train],fitted),-8,8)
        vx=np.clip(meta_transform(a[valid],fitted),-8,8)
        testx=np.clip(meta_transform(b,fitted),-8,8)
        if args.verify:
            checkpoint=torch.load(out/f"fold{fold}.pt",weights_only=True)
            if checkpoint["metadata_fit"]!=fitted: raise ValueError("Feature preprocessing mismatch")
            pv=[];pt=[]
            for state in checkpoint["states"]:
                model=network(a.shape[1]);model.load_state_dict(state)
                pv.append(probabilities(model,vx));pt.append(probabilities(model,testx))
            saved=np.load(out/f"fold{fold}_predictions.npz")
            np.testing.assert_allclose(np.mean(pv,axis=0),saved["probability"],atol=1e-7)
            np.testing.assert_allclose(np.mean(pt,axis=0),saved["test_probability"],atol=1e-7)
            print(f"REPLAY PASS fold {fold}",flush=True)
            continue
        if (out/f"fold{fold}_metrics.json").exists():
            print(f"SKIP shape fold {fold}",flush=True);continue
        start=time.perf_counter();pv=[];pt=[];states=[]
        for seed in [71,83,97]:
            torch.manual_seed(seed*10+fold)
            model=network(a.shape[1])
            loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(y[train])),
                              batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed+fold))
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=160,eta_min=.00001)
            for epoch in range(160):
                model.train()
                for x,target in loader:
                    opt.zero_grad(set_to_none=True)
                    objective=nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1),target)
                    objective.backward();opt.step()
                scheduler.step()
            pv.append(probabilities(model,vx));pt.append(probabilities(model,testx))
            states.append({k:v.clone() for k,v in model.state_dict().items()})
        pred=np.mean(pv,axis=0);test=np.mean(pt,axis=0)
        ap,covered=prior(data["angles"][train],y[train],data["angles"][valid])
        fused=fuse(pred,ap,covered)
        np.savez(out/f"fold{fold}_predictions.npz",valid_ids=data["ids"][valid],probability=pred,
                 plain_probability=pred,fused_probability=fused,test_ids=data["test_ids"],test_probability=test)
        torch.save(dict(states=states,metadata_fit=fitted,feature_width=a.shape[1],
                        train_ids_sha256=digest_ids(data["ids"][train]),architecture="shape_mlp",
                        seeds=[71,83,97],epochs=160),out/f"fold{fold}.pt")
        report=dict(fold=fold,config=dict(tag=args.tag,architecture="shape_mlp",epochs=160,seeds=[71,83,97]),
                    train_ids_sha256=digest_ids(data["ids"][train]),
                    val_log_loss=loss(y[valid],pred),val_fused_log_loss=loss(y[valid],fused),
                    valid_rows=int(valid.sum()),seconds=time.perf_counter()-start)
        (out/f"fold{fold}_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print("DONE "+json.dumps(report),flush=True)
if __name__=="__main__": main()
