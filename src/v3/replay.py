"""Replay stored checkpoints on held-out rows to verify saved probabilities."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.v2.model import RadarV1, predict
from src.v2.train import transform_images
from src.v3.train import ROOT, OUT, load_data, meta_transform, digest_ids
from src.v3.fusion import loss

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tag",required=True)
    p.add_argument("--folds-file",type=Path)
    p.add_argument("--threads",type=int,default=2)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    data=load_data(args.folds_file)
    records=[]
    for fold in range(5):
        train=data["folds"]!=fold
        valid=~train
        checkpoint=torch.load(OUT/args.tag/f"fold{fold}.pt",map_location="cpu",weights_only=True)
        if checkpoint["train_ids_sha256"]!=digest_ids(data["ids"][train]):
            raise ValueError("Training IDs differ from the checkpoint")
        model=RadarV1(dropout=checkpoint["dropout"])
        model.load_state_dict(checkpoint["state_dict"])
        x=transform_images(data["images"][valid],checkpoint["image_normalisation"],checkpoint["image_fit"])
        s=meta_transform(data["stats"][valid],checkpoint["metadata_fit"])
        prediction=predict(model,torch.from_numpy(x),torch.from_numpy(s),"cpu",tta=True,batch=128)
        saved=np.load(OUT/args.tag/f"fold{fold}_predictions.npz",allow_pickle=False)
        if not np.array_equal(saved["valid_ids"],data["ids"][valid]):
            raise ValueError("Validation IDs differ")
        difference=float(np.abs(prediction-saved["probability"]).max())
        if not np.allclose(prediction,saved["probability"],rtol=1e-5,atol=1e-6):
            raise ValueError(f"Prediction replay failed: {difference}")
        records.append(dict(fold=fold,max_absolute_error=difference,log_loss=loss(data["labels"][valid],prediction)))
    report=dict(tag=args.tag,passed=True,records=records)
    path=ROOT/"results/v3"; path.mkdir(parents=True,exist_ok=True)
    (path/f"{args.tag}_replay.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
