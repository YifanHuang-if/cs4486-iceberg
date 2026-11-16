"""Replay held-out predictions from retained second-round CNN weights."""
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from src.experiments.data import ROOT,load_train,read_folds,fold_hash
from src.experiments.metrics import log_loss
from src.refinement.cnn import METADATA,Preparation
from src.refinement.features import training_features
from src.refinement.advanced import cached_prefix_maps,make_model,predict_model


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    data=load_train();raw,names=training_features(data);stats=raw[:,[names.index(n) for n in METADATA]]
    split=ROOT/'data/splits/folds.csv';folds=read_folds(data,split);results=[]
    for name in ['R8_deep','R9_finetuned']:
        directory=ROOT/f'outputs/refinement/primary/{name}';signature=json.loads((directory/'manifest.json').read_text())
        assert signature['data_sha256']==data['sha256'] and signature['folds_sha256']==fold_hash(split)
        for path,digest in signature['source'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
        frame=pd.read_csv(directory/'oof_predictions.csv',dtype={'id':str},float_precision='round_trip').set_index('id').loc[data['ids']]
        assert np.array_equal(frame.label,data['labels']) and np.array_equal(frame.fold,folds.fold)
        family=signature['config']['family'];maps=cached_prefix_maps(data) if family=='transfer' else None;maximum=0.
        for fold in range(5):
            train=folds.fold.to_numpy()!=fold;valid=~train
            checkpoint=torch.load(directory/f'fold_{fold}.pt',map_location='cpu',weights_only=True)
            assert checkpoint['signature']==signature
            assert checkpoint['training_ids']==data['ids'][train].tolist() and checkpoint['validation_ids']==data['ids'][valid].tolist()
            preparation=Preparation.fit(data['images'][train],stats[train],'local');assert preparation.to_dict()==checkpoint['preparation']
            x,meta=preparation.transform(data['images'][valid],stats[valid]);x=maps[valid] if maps is not None else x
            model=make_model(family,pretrained=False);model.load_state_dict(checkpoint['state_dict'])
            p=predict_model(model,x,meta,family);error=float(np.max(np.abs(p-frame.loc[data['ids'][valid],'probability'].to_numpy())))
            assert error<=1e-7;maximum=max(maximum,error)
        result={'name':name,'checkpoints_replayed':5,'maximum_probability_difference':maximum,'oof_log_loss':log_loss(frame.label,frame.probability),'passed':True};results.append(result);print(json.dumps(result),flush=True)
    output=ROOT/'results/refinement/round2';output.mkdir(parents=True,exist_ok=True)
    (output/'verification.json').write_text(json.dumps({'status':'passed','checkpoints_replayed':10,'runs':results},indent=2)+'\n')


if __name__=='__main__':main()
