"""Align complete five-fold predictions and evaluate fixed fusion rules."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from src.v3.train import ROOT, OUT, save_csv
from src.v2.train import read_folds
from src.v3.fusion import loss, prior, fuse, fold_fusion, pooled_fold_fusion, pool_queries
from src.v2.angle_radius import AngleIndex
from src.v2.angle_fusion import bayes_fusion

def load_metadata(folds_path=None):
    a=np.load(ROOT/"work/cache/train.npz",allow_pickle=False)
    b=np.load(ROOT/"work/cache/test.npz",allow_pickle=False)
    return dict(ids=a["ids"],labels=a["labels"],angles=a["angles"],
                test_ids=b["ids"],test_angles=b["angles"],
                folds=read_folds(a["ids"],a["labels"],folds_path)["fold"])

def load_tag(tag,data,require_test=True):
    probability=np.full(len(data["ids"]),np.nan)
    fused=np.full_like(probability,np.nan)
    tests=[]
    ids=data["ids"]
    for fold in range(5):
        path=OUT/tag/f"fold{fold}_predictions.npz"
        record=json.loads((OUT/tag/f"fold{fold}_metrics.json").read_text())
        if record["fold"]!=fold: raise ValueError("Fold metadata mismatch")
        blob=np.load(path,allow_pickle=False)
        rows=np.flatnonzero(data["folds"]==fold)
        if not np.array_equal(blob["valid_ids"],ids[rows]):
            raise ValueError("OOF ID order mismatch")
        if require_test and not np.array_equal(blob["test_ids"],data["test_ids"]):
            raise ValueError("Test ID order mismatch")
        if not np.isfinite(blob["probability"]).all(): raise ValueError("Invalid OOF")
        probability[rows]=blob["probability"]
        fused[rows]=blob["fused_probability"]
        tests.append(blob["test_probability"])
    if not np.isfinite(probability).all(): raise ValueError("Incomplete OOF")
    return probability,np.mean(tests,axis=0),fused

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tags",nargs="+",required=True)
    p.add_argument("--name",required=True)
    p.add_argument("--folds-file",type=Path)
    p.add_argument("--validation-only",action="store_true")
    args=p.parse_args()
    data=load_metadata(args.folds_file)
    y=data["labels"]; a=data["angles"]; folds=data["folds"]
    report=dict(name=args.name,tags=args.tags,metric="pooled out-of-fold log loss",
                target=.083,official_kaggle_score=None,split=str(args.folds_file or "data/splits/folds.csv"),
                validation_note="Development CV, not an untouched test. V2 design choices were informed by this dataset.",
                per_model={})
    models=[]; test_models=[]
    for tag in args.tags:
        pred,test,old=load_tag(tag,data,not args.validation_only)
        models.append(pred); test_models.append(test)
        report["per_model"][tag]={"image_log_loss":loss(y,pred),"fixed_fold_safe_fusion":loss(y,old)}
    model=np.mean(models,axis=0); test=np.mean(test_models,axis=0)
    fixed,covered=fold_fusion(model,a,y,folds)
    pooled=pooled_fold_fusion(model,a,y,folds)
    # Historical LOO protocol is reproduced only as a clearly separated diagnostic.
    index=AngleIndex(a,y); pos=index.position
    rows=np.flatnonzero(pos>=0)
    loo=np.full(len(y),np.nan)
    exact=index.exact_prior(a[rows],.1,exclude=pos[rows])
    kernel=index.kernel_prior(a[rows],.02,.1,exclude=pos[rows])
    loo[rows]=np.where(np.isfinite(exact),exact,kernel)
    diagnostic=bayes_fusion(model,loo,.75)
    by_fold=[]
    for fold in range(5):
        m=folds==fold
        by_fold.append(dict(fold=fold,rows=int(m.sum()),image=loss(y[m],model[m]),
                            fused=loss(y[m],fixed[m]),pooled=loss(y[m],pooled[m])))
    report.update(image_log_loss=loss(y,model),fold_safe_fused_log_loss=loss(y,fixed),
                  historical_loo_diagnostic_log_loss=loss(y,diagnostic),
                  historical_loo_warning="Other labels from the query's validation fold enter the angle prior; not whole-pipeline five-fold OOF.",
                  coverage=float(covered.mean()),by_fold=by_fold,
                  pooled_fold_safe_fused_log_loss=loss(y,pooled),
                  fixed_clip_001_log_loss=loss(y,np.clip(fixed,.001,.999)),
                  pooled_clip_001_log_loss=loss(y,np.clip(pooled,.001,.999)),
                  pooling_note="Fixed 50% within-query exact-angle averaging. Transductive and batch-dependent; no query labels.",
                  target_met_in_development_cv=bool(min(loss(y,fixed),loss(y,pooled))<.083),
                  claim_first_place=False)
    report["subgroups"]={}
    for name,mask in [("observed_angle",np.isfinite(a)),("missing_angle",~np.isfinite(a)),("angle_prior_covered",covered),("angle_prior_uncovered",~covered)]:
        report["subgroups"][name]=dict(rows=int(mask.sum()),image=loss(y[mask],model[mask]),fused=loss(y[mask],fixed[mask]))
    if args.validation_only:
        report["submission"]=None
        report["validation_only"]=True
        result=ROOT/"results/v3";result.mkdir(parents=True,exist_ok=True)
        (result/f"{args.name}.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report,indent=2))
        return
    # All training labels are valid reference evidence for actual unlabelled test rows.
    ap,mask=prior(a,y,data["test_angles"])
    test_fused=fuse(test,ap,mask)
    test_pooled=fuse(pool_queries(test,data["test_angles"]),ap,mask)
    out=OUT/"submissions"; out.mkdir(parents=True,exist_ok=True)
    submission=out/f"{args.name}.csv"
    save_csv(submission,["id","is_iceberg"],zip(data["test_ids"],test_fused))
    save_csv(out/f"{args.name}_pooled.csv",["id","is_iceberg"],zip(data["test_ids"],test_pooled))
    save_csv(out/f"{args.name}_clip001.csv",["id","is_iceberg"],zip(data["test_ids"],np.clip(test_fused,.001,.999)))
    save_csv(out/f"{args.name}_pooled_clip001.csv",["id","is_iceberg"],zip(data["test_ids"],np.clip(test_pooled,.001,.999)))
    save_csv(out/f"{args.name}_oof.csv",["id","fold","label","image_probability","fused_probability"],
             zip(data["ids"],folds,y,model,fixed))
    # For strict replay, these are also saved in aligned NumPy form.
    np.savez(out/f"{args.name}_predictions.npz",train_ids=data["ids"],test_ids=data["test_ids"],
             image_oof=model,image_test=test,fused_oof=fixed,pooled_oof=pooled,test_probability=test_fused,test_pooled_probability=test_pooled)
    report["submission"]={"path":str(submission.relative_to(ROOT)),"rows":len(test_fused),
                          "finite":bool(np.isfinite(test_fused).all()),
                          "unique_ids":len(set(data["test_ids"].tolist())),
                          "min":float(test_fused.min()),"max":float(test_fused.max()),
                          "sha256":hashlib.sha256(submission.read_bytes()).hexdigest()}
    result=ROOT/"results/v3"; result.mkdir(parents=True,exist_ok=True)
    (result/f"{args.name}.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
