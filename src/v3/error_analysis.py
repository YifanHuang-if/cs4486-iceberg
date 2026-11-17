"""Decompose full OOF errors without changing or tuning predictions."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from src.v3.train import ROOT,save_csv
from src.v3.evaluate import load_metadata
from src.v3.fusion import loss,prior
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--name",required=True)
    args=parser.parse_args()
    data=load_metadata()
    npz=np.load(ROOT/"outputs/v3/submissions"/f"{args.name}_predictions.npz",allow_pickle=False)
    if not np.array_equal(npz["train_ids"],data["ids"]): raise ValueError("OOF IDs mismatch")
    y=data["labels"];p=npz["fused_oof"];a=data["angles"];fold=data["folds"]
    row_loss=-(y*np.log(np.clip(p,1e-7,1-1e-7))+(1-y)*np.log1p(-np.clip(p,1e-7,1-1e-7)))
    matched=np.zeros(len(y),bool);count=np.zeros(len(y),int);neardistance=np.full(len(y),np.nan)
    for f in range(5):
        ref=a[(fold!=f)&np.isfinite(a)]
        idx=np.flatnonzero((fold==f)&np.isfinite(a))
        distance=np.abs(a[idx,None]-ref[None,:])
        count[idx]=(distance==0).sum(1)
        neardistance[idx]=distance.min(1)
    groups={"exact_angle_match":count>0,"near_angle_only":(count==0)&(neardistance<.06),
            "far_angle":neardistance>=.06,"missing_angle":~np.isfinite(a)}
    rows=[]
    for name,mask in groups.items():
        rows.append(dict(group=name,rows=int(mask.sum()),log_loss=loss(y[mask],p[mask]) if mask.any() else None,
                         total_loss_share=float(row_loss[mask].sum()/row_loss.sum())))
    order=np.argsort(row_loss)[::-1]
    out=ROOT/"outputs/v3/errors";out.mkdir(parents=True,exist_ok=True)
    save_csv(out/f"{args.name}_largest_errors.csv",
             ["id","fold","label","fused_probability","row_log_loss","angle","same_angle_train_count","nearest_train_angle_distance"],
             [(data["ids"][i],fold[i],y[i],p[i],row_loss[i],a[i],count[i],neardistance[i]) for i in order[:50]])
    report=dict(name=args.name,subgroups=rows,
                top_10_loss_share=float(row_loss[order[:10]].sum()/row_loss.sum()),
                top_25_loss_share=float(row_loss[order[:25]].sum()/row_loss.sum()),
                confident_wrong_rows=int((((p<.1)&(y==1))|((p>.9)&(y==0))).sum()))
    (ROOT/"results/v3"/f"{args.name}_errors.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
