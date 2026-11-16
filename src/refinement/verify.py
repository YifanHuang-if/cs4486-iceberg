"""Replay saved CNN/tree checkpoints and check that every outer fold stayed held out."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from src.experiments.data import ROOT,load_train,read_folds,fold_hash
from src.experiments.metrics import log_loss
from src.refinement.cnn import METADATA,Preparation,RadarCNN,predict
from src.refinement.features import training_features
from src.refinement.transfer import cached_embeddings,linear_predict


def verify(name,protocol,data,features,names,folds):
    directory=ROOT/f'outputs/refinement/{protocol}/{name}';signature=json.loads((directory/'manifest.json').read_text())
    split=ROOT/('data/splits/folds.csv' if protocol=='primary' else 'data/splits/folds_angle_grouped_steps78.csv')
    assert signature['folds_sha256']==fold_hash(split) and signature['data_sha256']==data['sha256']
    for path,digest in signature['source'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    oof=pd.read_csv(directory/'oof_predictions.csv',dtype={'id':str},float_precision='round_trip')
    assert not oof.id.duplicated().any() and set(oof.id)==set(data['ids'])
    oof=oof.set_index('id').loc[data['ids']].reset_index()
    assert np.array_equal(oof.fold,folds.fold) and np.array_equal(oof.label,data['labels'])
    stats=features[:,[names.index(n) for n in METADATA]];maximum=0.
    if name=='R6_frozen_resnet':transfer_features=np.concatenate([cached_embeddings(data),features],axis=1)
    for fold in range(5):
        train=folds.fold.to_numpy()!=fold;valid=~train
        if name in ['R3_radar_trees','R6_frozen_resnet']:
            provenance=json.loads((directory/f'fold_{fold}_provenance.json').read_text())
            assert set(provenance['training_ids'])==set(data['ids'][train]) and set(provenance['validation_ids'])==set(data['ids'][valid])
            if name=='R3_radar_trees':
                assert not set(provenance['inner_training_ids'])&set(provenance['inner_validation_ids'])
                assert set(provenance['inner_training_ids'])|set(provenance['inner_validation_ids'])==set(provenance['training_ids'])
                model=CatBoostClassifier();model.load_model(str(directory/f'fold_{fold}.cbm'))
                p=model.predict_proba(features[valid])[:,1]
            else:
                with np.load(directory/f'fold_{fold}.npz',allow_pickle=False) as state:p=linear_predict(state,transfer_features[valid])
        else:
            checkpoint=torch.load(directory/f'fold_{fold}.pt',map_location='cpu',weights_only=True)
            assert checkpoint['signature']==signature
            assert checkpoint['training_ids']==data['ids'][train].tolist() and checkpoint['validation_ids']==data['ids'][valid].tolist()
            preparation=Preparation.fit(data['images'][train],stats[train],checkpoint['config']['normalization'])
            assert preparation.to_dict()==checkpoint['preparation']
            images,metadata=preparation.transform(data['images'][valid],stats[valid]);model=RadarCNN();model.load_state_dict(checkpoint['state_dict'])
            p=predict(model,torch.from_numpy(images),torch.from_numpy(metadata),torch.device('cpu'),tta=True)
        error=float(np.max(np.abs(p-oof.loc[valid,'probability'].to_numpy())));assert error<=1e-7
        maximum=max(maximum,error)
    measured=log_loss(oof.label,oof.probability);expected=json.loads((directory/'summary.json').read_text())['oof_log_loss'];assert abs(measured-expected)<1e-9
    result={'name':name,'protocol':protocol,'samples':len(oof),'checkpoints_replayed':5,'maximum_probability_difference':maximum,'oof_log_loss':measured,'passed':True}
    print(json.dumps(result),flush=True);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',default='primary',choices=['primary','angle_grouped']);parser.add_argument('--names',nargs='+',default=['R1_global','R2_local','R3_radar_trees','R6_frozen_resnet']);parser.add_argument('--threads',type=int,default=4);args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    data=load_train();features,names=training_features(data)
    path=ROOT/('data/splits/folds.csv' if args.protocol=='primary' else 'data/splits/folds_angle_grouped_steps78.csv');folds=read_folds(data,path)
    if args.protocol=='angle_grouped':
        known=np.isfinite(data['angles']);assert pd.DataFrame({'angle':data['angles'][known],'fold':folds.fold.to_numpy()[known]}).groupby('angle').fold.nunique().max()==1
    runs=[verify(name,args.protocol,data,features,names,folds) for name in args.names]
    output=ROOT/f'results/refinement/{args.protocol}';output.mkdir(parents=True,exist_ok=True)
    (output/'verification.json').write_text(json.dumps({'status':'passed','checkpoints_replayed':sum(r['checkpoints_replayed'] for r in runs),'runs':runs},indent=2)+'\n')


if __name__=='__main__':main()
