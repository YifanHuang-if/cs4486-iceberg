"""Stream test images, ensemble retained folds, and write a genuine Kaggle candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import ijson
from catboost import CatBoostClassifier
from src.experiments.data import ROOT,load_train,Preprocessor
from src.experiments.model import ExperimentCNN
from src.refinement.cnn import METADATA,Preparation,RadarCNN,predict
from src.refinement.features import radar_features
from src.refinement.context import AngleContext,EmpiricalAngleContext
from src.refinement.transfer import backbone,embeddings,linear_predict,WEIGHTS


def metadata_features(images,angles):
    values={}
    for c in [0,1,2]:
        a=(images[:,c] if c<2 else images[:,0]-images[:,1]).astype(np.float64)
        values[f'c{c}_mean']=a.mean((1,2));values[f'c{c}_std']=a.std((1,2)).clip(1e-6)
        if c<2:
            q=np.quantile(a,[.5,.99],axis=(1,2));values[f'c{c}_q3']=q[0];values[f'c{c}_q7']=q[1]
    values['angle']=np.where(np.isfinite(angles),angles,-1.)
    values['angle_missing']=(~np.isfinite(angles)).astype(float)
    return np.column_stack([values[n] for n in METADATA]).astype(np.float32)


def parse_candidate(candidate):
    context=None
    for suffix in ['empirical_context','fixed_context']:
        if candidate.endswith('_'+suffix):candidate=candidate[:-len(suffix)-1];context=suffix;break
    groups={'R4_cnn_mean':['R1_global','R2_local'],'R5_cnn_tree_mean':['R1_global','R2_local','R3_radar_trees'],
            'R7_cnn_transfer_mean':['R1_global','R2_local','R6_frozen_resnet']}.get(candidate,[candidate])
    if not set(groups)<={'R1_global','R2_local','R3_radar_trees','R6_frozen_resnet','V4_previous'}:raise ValueError('Unknown candidate')
    return groups,context


def load_models(groups,data):
    models=[];hashes={}
    for name in groups:
        for fold in range(5):
            if name=='V4_previous':path=ROOT/f'outputs/steps78/primary/V4_missing/fold_{fold}.pt'
            else:path=ROOT/f'outputs/refinement/primary/{name}/fold_{fold}{".cbm" if name=="R3_radar_trees" else ".npz" if name=="R6_frozen_resnet" else ".pt"}'
            hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            if name=='R3_radar_trees':
                model=CatBoostClassifier();model.load_model(str(path));preparation=None
            elif name=='R6_frozen_resnet':
                with np.load(path,allow_pickle=False) as archive:model={key:archive[key] for key in archive.files}
                preparation=None
            else:
                checkpoint=torch.load(path,map_location='cpu',weights_only=True)
                assert checkpoint['signature']['data_sha256']==data['sha256']
                if name=='V4_previous':
                    config=checkpoint['config'];model=ExperimentCNN(config['dropout'],config['angle_features']);preparation=Preprocessor(**checkpoint['preprocessor'])
                else:model=RadarCNN();preparation=Preparation(**checkpoint['preparation'])
                model.load_state_dict(checkpoint['state_dict']);model.eval()
            models.append((name,model,preparation))
    return models,hashes


def predict_chunk(models,groups,images,angles,transfer_model=None,folds_per_model=5):
    metadata=metadata_features(images,angles);predictions=[];tree_features=None;transfer_features=None
    for name,model,preparation in models:
        if name=='R3_radar_trees':
            if tree_features is None:tree_features,_=radar_features(images,angles)
            p=model.predict_proba(tree_features)[:,1]
        elif name=='R6_frozen_resnet':
            if tree_features is None:tree_features,_=radar_features(images,angles)
            if transfer_features is None:transfer_features=np.concatenate([embeddings(transfer_model,images),tree_features],axis=1)
            p=linear_predict(model,transfer_features)
        elif name=='V4_previous':
            x,m=preparation.transform(images,angles)
            with torch.no_grad():p=model(torch.from_numpy(x),torch.from_numpy(m)).sigmoid().numpy().astype(np.float64)
        else:
            x,m=preparation.transform(images,metadata);p=predict(model,torch.from_numpy(x),torch.from_numpy(m),torch.device('cpu'),tta=True)
        predictions.append(p)
    assert len(models)==folds_per_model*len(groups)
    return np.mean(predictions,axis=0)


def batches(path,size=256):
    batch=[]
    with path.open('rb') as handle:
        for record in ijson.items(handle,'item',use_float=True):
            batch.append(record)
            if len(batch)==size:yield batch;batch=[]
        if batch:yield batch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);parser.add_argument('--threads',type=int,default=4);args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    data=load_train();groups,kind=parse_candidate(args.candidate);models,hashes=load_models(groups,data)
    transfer_model=backbone() if 'R6_frozen_resnet' in groups else None
    context=None
    if kind:context=(EmpiricalAngleContext() if kind=='empirical_context' else AngleContext()).fit(data['ids'],data['angles'],data['labels'])
    source=ROOT/'data/raw/test.json';ids=[];probabilities=[];matching=0
    for records in batches(source):
        chunk_ids=np.asarray([str(r['id']) for r in records]);assert not set(chunk_ids)&set(data['ids'])
        images=np.asarray([[r['band_1'],r['band_2']] for r in records],dtype=np.float32).reshape(-1,2,75,75)
        angles=np.asarray([np.nan if r['inc_angle'] in ('na',None) else float(r['inc_angle']) for r in records])
        p=predict_chunk(models,groups,images,angles,transfer_model)
        if context is not None:p,counts=context.transform(chunk_ids,angles,p);matching+=int((counts>0).sum())
        ids.extend(chunk_ids.tolist());probabilities.extend(p.tolist());print(f'PREDICTED {len(ids)} rows',flush=True)
    frame=pd.DataFrame({'id':ids,'is_iceberg':np.clip(probabilities,1e-7,1-1e-7)})
    assert not frame.id.duplicated().any() and np.isfinite(frame.is_iceberg).all()
    sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv',dtype={'id':str})
    assert set(frame.id)==set(sample.id) and len(frame)==len(sample)
    frame=frame.set_index('id').loc[sample.id].reset_index()
    output=ROOT/'outputs/refinement/submissions';output.mkdir(parents=True,exist_ok=True)
    destination=output/f'{args.candidate}_submission.csv';frame.to_csv(destination,index=False)
    with source.open('rb') as handle:test_hash=hashlib.file_digest(handle,'sha256').hexdigest()
    report={'candidate':args.candidate,'rows':len(frame),'base_models':groups,'fold_checkpoints':len(models),'context':kind,
            'context_strength':context.strength if context else None,'matching_test_samples':matching,
            'data_sha256':data['sha256'],'test_sha256':test_hash,'checkpoint_sha256':hashes,
            'submission_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'official_kaggle_score':None,
            'submitted_to_kaggle':False,'official_threshold_verified':False,'inference_threads':args.threads,
            'public_pretrained_backbone':WEIGHTS.url if transfer_model is not None else None,
            'inference_source':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/refinement/inference.py','src/refinement/context.py','src/refinement/cnn.py','src/refinement/features.py','src/refinement/transfer.py']},
            'note':'This is a prediction file, not a leaderboard result. Upload it to the competition and retain the official scoring response.'}
    destination.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'submission':str(destination),'rows':len(frame),'official_score':None}))


if __name__=='__main__':main()
