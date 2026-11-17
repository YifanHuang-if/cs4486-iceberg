"""Fixed-budget fold training with complete inference artifacts and safe pseudo-labels.

A teacher and its angle prior are fitted only on the corresponding outer training
fold. Fixed epoch counts prevent outer validation labels influencing the teacher.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.v2.model import RadarV1, image_statistics, augment, predict
from src.v2.train import read_folds, fit_normalisation, transform_images
from src.v3.fusion import loss, prior, fuse, soft_targets

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"outputs"/"v3"

def digest_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids.tolist())).encode()).hexdigest()

def load_data(folds_path=None, include_angle=True):
    """Build the training tensors.

    When ``include_angle`` is False the incidence angle and its missingness flag
    are kept out of the network's metadata branch, so the image model and the
    angle evidence stay statistically independent. The angle is still carried in
    ``angles`` for the fusion layer.
    """
    a=np.load(ROOT/"work/cache/train.npz", allow_pickle=False)
    b=np.load(ROOT/"work/cache/test.npz", allow_pickle=False)
    stats_path=ROOT/"work/cache/v3_image_stats.npz"
    if stats_path.exists():
        st=np.load(stats_path, allow_pickle=False)
        sa,sb=st["train"],st["test"]
    else:
        sa,sb=image_statistics(a["images"]),image_statistics(b["images"])
        np.savez(stats_path,train=sa,test=sb)
    if include_angle:
        train_stats=np.column_stack([a["angles"],a["angle_missing"],sa]).astype(np.float32)
        test_stats=np.column_stack([b["angles"],b["angle_missing"],sb]).astype(np.float32)
    else:
        train_stats=np.asarray(sa,dtype=np.float32)
        test_stats=np.asarray(sb,dtype=np.float32)
    return dict(images=a["images"],labels=a["labels"],ids=a["ids"],angles=a["angles"],
                test_images=b["images"],test_ids=b["ids"],test_angles=b["angles"],
                stats=train_stats,test_stats=test_stats,
                stats_dim=train_stats.shape[1],
                folds=read_folds(a["ids"],a["labels"],folds_path)["fold"])

def fit_meta(raw):
    med=np.nanmedian(raw,axis=0)
    med=np.where(np.isfinite(med),med,0)
    clean=np.where(np.isfinite(raw),raw,med)
    mean=clean.mean(0)
    std=np.maximum(clean.std(0),1e-6)
    return dict(median=med.tolist(),mean=mean.tolist(),std=std.tolist())

def meta_transform(raw,fit):
    x=np.where(np.isfinite(raw),raw,np.asarray(fit["median"]))
    return np.asarray((x-fit["mean"])/fit["std"],dtype=np.float32)

def mild_augment(x,generator,noise=0.02):
    for axis in [-1,-2]:
        flag=(torch.rand((len(x),1,1,1),generator=generator)<.5).to(x.device)
        x=torch.where(flag,x.flip(axis),x)
    flag=(torch.rand((len(x),1,1,1),generator=generator)<.5).to(x.device)
    x=torch.where(flag,x.transpose(-1,-2),x)
    if noise<=0.:
        return x
    return x+torch.randn(x.shape,generator=generator).to(x.device)*noise

def save_csv(path, fields, rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f); writer.writerow(fields); writer.writerows(rows)

def run_fold(args,data,fold):
    out=OUT/args.tag; out.mkdir(parents=True,exist_ok=True)
    marker=out/f"fold{fold}_metrics.json"
    config=vars(args).copy()
    config["folds_file"]=str(config["folds_file"]) if config["folds_file"] else None
    config["folds"]=list(args.folds)
    train=data["folds"]!=fold
    valid=~train
    train_hash=digest_ids(data["ids"][train])
    if marker.exists():
        previous=json.loads(marker.read_text())
        old_config=previous["config"].copy()
        old_config.setdefault("initialize_teacher",False)
        old_config.setdefault("validation_only",False)
        comparison=config.copy()
        old_config.pop("folds",None);comparison.pop("folds",None)
        if old_config!=comparison or previous["train_ids_sha256"]!=train_hash:
            raise ValueError("Existing run differs; choose a new tag")
        print(f"SKIP completed {args.tag} fold {fold}",flush=True)
        return previous
    seed=args.seed*100+fold
    torch.manual_seed(seed)
    fitted=fit_normalisation(data["images"][train])
    meta=fit_meta(data["stats"][train])
    tx=transform_images(data["images"][train],args.normalisation,fitted)
    vx=transform_images(data["images"][valid],args.normalisation,fitted)
    ts=meta_transform(data["stats"][train],meta)
    vs=meta_transform(data["stats"][valid],meta)
    y=data["labels"][train]
    train_angles=data["angles"][train]
    weights=np.ones(len(y),np.float32)
    pseudo_count=0
    if args.subset=="no_pure_iceberg":
        # Angle groups that are entirely iceberg within this training fold. The
        # grouping uses training labels only, so no validation label is involved.
        fold_angles=data["angles"][train]
        keep=np.ones(len(y),bool)
        for value in np.unique(fold_angles[np.isfinite(fold_angles)]):
            members=np.flatnonzero(fold_angles==value)
            if len(members)>1 and y[members].min()>0.5:
                keep[members]=False
        dropped=int((~keep).sum())
        if dropped:
            tx,ts,y,weights=tx[keep],ts[keep],y[keep],weights[keep]
            train_angles=train_angles[keep]
        print(f"{args.tag} fold={fold} dropped {dropped} pure-iceberg rows",flush=True)
    if args.teacher:
        teacher_dir=OUT/args.teacher
        teacher_info=json.loads((teacher_dir/f"fold{fold}_metrics.json").read_text())
        if teacher_info["train_ids_sha256"]!=train_hash or teacher_info["config"]["teacher"]:
            raise ValueError("Teacher must be supervised and use exactly this training fold")
        source=np.load(teacher_dir/f"fold{fold}_predictions.npz",allow_pickle=False)
        if not np.array_equal(source["test_ids"],data["test_ids"]):
            raise ValueError("Teacher test IDs mismatch")
        ap,coverage=prior(train_angles,y,data["test_angles"])
        probability=fuse(source["test_probability"],ap,coverage)
        threshold=args.pseudo_threshold
        selected=(probability<threshold)|(probability>1.-threshold)
        # Agreement prevents the angle prior alone overriding an uncertain image teacher.
        image_floor=min(0.5,max(0.1,threshold))
        selected &= (source["test_probability"]<1.-image_floor)|(source["test_probability"]>image_floor)
        selected &= ((probability>.5)==(source["test_probability"]>.5))
        rows=np.flatnonzero(selected)
        pseudo_count=len(rows)
        if pseudo_count:
            tx=np.concatenate([tx,transform_images(data["test_images"][rows],args.normalisation,fitted)])
            ts=np.concatenate([ts,meta_transform(data["test_stats"][rows],meta)])
            y=np.concatenate([y,soft_targets(probability[rows])])
            weights=np.concatenate([weights,np.full(len(rows),args.pseudo_weight,np.float32)])
            np.savez(out/f"fold{fold}_pseudo.npz",test_ids=data["test_ids"][rows],
                     labels=soft_targets(probability[rows]),teacher_train_ids_sha256=train_hash)
    g=torch.Generator().manual_seed(seed)
    aug_g=torch.Generator().manual_seed(seed+7919)
    loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ts),
                                   torch.from_numpy(y),torch.from_numpy(weights)),
                      batch_size=args.batch,shuffle=True,generator=g,num_workers=0)
    model=RadarV1(dropout=args.dropout, stats_dim=data["stats_dim"],
                  widths=tuple(args.widths)).to(args.device)
    if args.initialize_teacher:
        if not args.teacher:
            raise ValueError("--initialize-teacher requires --teacher")
        checkpoint=torch.load(OUT/args.teacher/f"fold{fold}.pt",map_location=args.device,weights_only=True)
        if checkpoint["image_normalisation"]!=args.normalisation:
            raise ValueError("Teacher fine-tuning requires identical image normalization")
        if checkpoint["metadata_fit"]!=meta or checkpoint["image_fit"]!=fitted:
            raise ValueError("Teacher preprocessing does not match the training fold")
        model.load_state_dict(checkpoint["state_dict"])
    if args.optimizer=="sgd":
        opt=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,nesterov=True,
                            weight_decay=args.weight_decay)
    else:
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.lr*.01)
    bce=nn.BCEWithLogitsLoss(reduction="none")
    history=[]
    # Exponential moving average of the weights. Small datasets give noisy final
    # weights, and the averaged trajectory usually generalises better.
    ema_state=None
    if args.ema>0.:
        ema_state={k:(v.detach().clone().float() if v.dtype.is_floating_point else v.detach().clone())
                   for k,v in model.state_dict().items()}
    start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.
        for x,s,target,weight in loader:
            x=x.to(args.device); s=s.to(args.device); target=target.to(args.device); weight=weight.to(args.device)
            x=augment(x,aug_g) if args.augmentation=="affine" else mild_augment(x,aug_g,args.noise)
            if args.mixup>0. and len(x)>1:
                # Mixup on both channels and metadata; targets are already soft
                # because pseudo-labelled rows carry a probability, not a class.
                lam=float(torch.distributions.Beta(args.mixup,args.mixup).sample())
                lam=max(lam,1.-lam)
                order=torch.randperm(len(x),generator=aug_g,device=x.device)
                x=lam*x+(1.-lam)*x[order]
                s=lam*s+(1.-lam)*s[order]
                target=lam*target+(1.-lam)*target[order]
                weight=lam*weight+(1.-lam)*weight[order]
            opt.zero_grad(set_to_none=True)
            if args.label_smoothing>0.:
                # Soften the target toward 0.5; the pooled loss is dominated by a
                # handful of overconfident mistakes, so this can help the tail.
                target=target*(1.-args.label_smoothing)+0.5*args.label_smoothing
            objective=(bce(model(x,s),target)*weight).sum()/weight.sum()
            if not torch.isfinite(objective): raise ValueError("Nonfinite loss")
            objective.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
            if ema_state is not None:
                with torch.no_grad():
                    for key,value in model.state_dict().items():
                        if value.dtype.is_floating_point:
                            ema_state[key].mul_(args.ema).add_(value.detach().float(),alpha=1.-args.ema)
                        else:
                            ema_state[key].copy_(value)
            total+=float(objective.detach())*len(x)
        scheduler.step()
        history.append([epoch,total/len(y),opt.param_groups[0]["lr"]])
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            print(f"{args.tag} fold={fold} epoch={epoch}/{args.epochs} train={total/len(y):.5f} elapsed={time.perf_counter()-start:.0f}s",flush=True)
    if ema_state is not None:
        model.load_state_dict({k:v.to(model.state_dict()[k].dtype) for k,v in ema_state.items()})
    checkpoint=dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                    architecture="RadarV1",dropout=args.dropout,stats_dim=int(data["stats_dim"]),
                    widths=[int(w) for w in args.widths],
                    image_normalisation=args.normalisation,image_fit=fitted,metadata_fit=meta,
                    config=config,fold=fold,train_ids_sha256=train_hash,
                    checkpoint_rule="fixed_final_epoch_no_outer_label_selection")
    torch.save(checkpoint,out/f"fold{fold}.pt")
    vp=predict(model,torch.from_numpy(vx),torch.from_numpy(vs),args.device,tta=True,batch=128)
    plain=predict(model,torch.from_numpy(vx),torch.from_numpy(vs),args.device,tta=False,batch=128)
    ap,coverage=prior(data["angles"][train],data["labels"][train],data["angles"][valid])
    fused=fuse(vp,ap,coverage)
    # Batched test inference limits peak memory on CPU laptops.
    test_prob=[]
    for lo in range(0,0 if args.validation_only else len(data["test_ids"]),512):
        hi=lo+512
        x=transform_images(data["test_images"][lo:hi],args.normalisation,fitted)
        s=meta_transform(data["test_stats"][lo:hi],meta)
        test_prob.append(predict(model,torch.from_numpy(x),torch.from_numpy(s),args.device,tta=True,batch=128))
    test_prob=np.concatenate(test_prob) if test_prob else np.empty(0,dtype=np.float64)
    np.savez(out/f"fold{fold}_predictions.npz",valid_ids=data["ids"][valid],
             probability=vp,plain_probability=plain,fused_probability=fused,
             test_ids=data["test_ids"] if not args.validation_only else np.empty(0,dtype=data["test_ids"].dtype),test_probability=test_prob)
    save_csv(out/f"fold{fold}_history.csv",["epoch","train_loss","lr"],history)
    result=dict(fold=fold,config=config,train_ids_sha256=train_hash,
                val_log_loss=loss(data["labels"][valid],vp),
                val_plain_log_loss=loss(data["labels"][valid],plain),
                val_fused_log_loss=loss(data["labels"][valid],fused),
                valid_rows=int(valid.sum()),pseudo_rows=pseudo_count,
                seconds=time.perf_counter()-start)
    marker.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("DONE "+json.dumps(result),flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tag",required=True)
    p.add_argument("--seed",type=int,default=11)
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--lr",type=float,default=.0015)
    p.add_argument("--dropout",type=float,default=.3)
    p.add_argument("--normalisation",choices=["local","global","local_clip","local_robust"],default="local")
    p.add_argument("--augmentation",choices=["mild","affine"],default="mild")
    p.add_argument("--mixup",type=float,default=0.0)
    p.add_argument("--label-smoothing",dest="label_smoothing",type=float,default=0.0)
    p.add_argument("--weight-decay",dest="weight_decay",type=float,default=1e-4)
    p.add_argument("--widths",nargs=4,type=int,default=[16,32,64,96])
    p.add_argument("--optimizer",choices=["adamw","sgd"],default="adamw")
    p.add_argument("--ema",type=float,default=0.0,
                   help="decay for an exponential moving average of the weights")
    p.add_argument("--no-angle-input",dest="no_angle_input",action="store_true",
                   help="Keep inc_angle and its missing flag out of the metadata branch")
    p.add_argument("--folds",nargs="+",type=int,default=[0,1,2,3,4])
    p.add_argument("--folds-file",type=Path)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--device",default="cpu")
    p.add_argument("--teacher",default="")
    p.add_argument("--pseudo-weight",type=float,default=.25)
    p.add_argument("--pseudo-threshold",dest="pseudo_threshold",type=float,default=.03)
    p.add_argument("--noise",type=float,default=0.02,
                   help="standard deviation of the additive Gaussian augmentation")
    p.add_argument("--subset",choices=["all","no_pure_iceberg"],default="all",
                   help="optionally drop the label-pure iceberg angle block")
    p.add_argument("--initialize-teacher",action="store_true")
    p.add_argument("--validation-only",action="store_true")
    args=p.parse_args()
    if not all(0<=f<=4 for f in args.folds): raise ValueError("Invalid folds")
    torch.set_num_threads(args.threads)
    data=load_data(args.folds_file, include_angle=not args.no_angle_input)
    records=[run_fold(args,data,fold) for fold in args.folds]
    print(json.dumps({"records":records},indent=2),flush=True)
if __name__=="__main__": main()
