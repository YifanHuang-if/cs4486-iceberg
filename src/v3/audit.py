"""Audit frozen split identity and compare angle protocols on the saved baseline."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from src.v3.fusion import fold_fusion,loss
from src.v2.angle_radius import AngleIndex
from src.v2.angle_fusion import bayes_fusion
ROOT=Path(__file__).resolve().parents[2]
def main():
    cached=np.load(ROOT/"work/cache/train.npz",allow_pickle=False)
    ids,y,a=cached["ids"],cached["labels"],cached["angles"]
    config=json.loads((ROOT/"configs/validation.json").read_text())
    raw=(ROOT/"data/splits/folds.csv").read_bytes()
    canonical=raw.replace(bytes([13,10]),bytes([10]))
    assert hashlib.sha256(canonical).hexdigest()==config["folds_sha256"]
    with (ROOT/"data/splits/folds.csv").open(newline="") as f:
        rows={r["id"]:r for r in csv.DictReader(f)}
    folds=np.array([int(rows[i]["fold"]) for i in ids])
    assert all(int(rows[i]["label"])==int(label) for i,label in zip(ids,y))
    source=ROOT/"results/baseline_v1/oof_predictions.csv"
    with source.open(newline="") as f:
        predictions={r["id"]:float(r["probability"]) for r in csv.DictReader(f)}
    model=np.array([predictions[i] for i in ids])
    safe,coverage=fold_fusion(model,a,y,folds)
    index=AngleIndex(a,y); pos=index.position; observed=np.flatnonzero(pos>=0)
    exact=index.exact_prior(a[observed],.1,exclude=pos[observed])
    kernel=index.kernel_prior(a[observed],.02,.1,exclude=pos[observed])
    loo=np.full(len(y),np.nan);loo[observed]=np.where(np.isfinite(exact),exact,kernel)
    hybrid=bayes_fusion(model,loo,.75)
    report=dict(base_commit="661c5cec0af3dd754fa49396794879fdd084a3db",
                frozen_split_sha256=hashlib.sha256(canonical).hexdigest(),
                split_verified=True,train_rows=len(y),baseline_image_oof=loss(y,model),
                baseline_fold_safe_fusion=loss(y,safe),baseline_historical_loo_hybrid=loss(y,hybrid),
                note="The image predictions are identical in both comparisons; only the angle evidence protocol differs.",
                v2_full_oof_available=False,v2_checkpoint_available=False,
                reference_document_best_claim=.11897,
                reference_document_claim_verified_locally=False,
                official_target=.083,official_score=None)
    out=ROOT/"results/v3";out.mkdir(parents=True,exist_ok=True)
    (out/"validation_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
