"""Evaluate nested angle selection on an already completed image ensemble."""
import argparse,json
import numpy as np
from src.v3.train import ROOT,save_csv
from src.v3.evaluate import load_metadata
from src.v3.fusion import loss
from src.v3.nested_angle import nested_oof,nested_predict

def main():
    p=argparse.ArgumentParser();p.add_argument("--name",required=True);args=p.parse_args()
    data=load_metadata()
    path=ROOT/"outputs/v3/submissions"
    saved=np.load(path/f"{args.name}_predictions.npz",allow_pickle=False)
    if not np.array_equal(saved["train_ids"],data["ids"]): raise ValueError("ID mismatch")
    image=saved["image_oof"];y=data["labels"];angles=data["angles"];folds=data["folds"]
    pred,records=nested_oof(image,angles,y,folds)
    pooled,unused=nested_oof(image,angles,y,folds,pool=True)
    test,test_record=nested_predict(saved["image_test"],angles,y,data["test_angles"])
    test_pooled,_=nested_predict(saved["image_test"],angles,y,data["test_angles"],pool=True)
    save_csv(path/f"{args.name}_nested.csv",["id","is_iceberg"],zip(data["test_ids"],test))
    save_csv(path/f"{args.name}_nested_pooled.csv",["id","is_iceberg"],zip(data["test_ids"],test_pooled))
    save_csv(path/f"{args.name}_nested_oof.csv",["id","fold","label","probability","pooled_probability"],
             zip(data["ids"],folds,y,pred,pooled))
    report=dict(base=args.name,fold_safe_nested_log_loss=loss(y,pred),
                fold_safe_nested_pooled_log_loss=loss(y,pooled),outer_folds=records,
                full_train_selection=test_record["selected"],official_score=None,
                target=.083,target_met_in_development_cv=bool(min(loss(y,pred),loss(y,pooled))<.083),
                selection_scope="Each outer fold uses only its training labels; 3 inner folds, 12 angle-only candidates.")
    (ROOT/"results/v3"/f"{args.name}_nested.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="outer_folds"},indent=2))
if __name__=="__main__": main()
