"""Frozen public ImageNet features with nested selection of a regularized classifier."""
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18,ResNet18_Weights
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.experiments.data import ROOT,load_train,read_folds,fold_hash
from src.experiments.metrics import metrics
from src.refinement.features import training_features
from src.refinement.train import aggregate,BASE_COMMIT

WEIGHTS=ResNet18_Weights.IMAGENET1K_V1


def backbone():
    model=resnet18(weights=WEIGHTS,progress=False);model.fc=nn.Identity();model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def embeddings(model,images):
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(images),32):
            x=images[start:start+32]
            rgb=np.concatenate([x,(x[:,0]-x[:,1])[:,None]],axis=1)
            q=np.quantile(rgb,[.01,.99],axis=(2,3),keepdims=True)
            rgb=np.clip((rgb-q[0])/np.maximum(q[1]-q[0],1e-6),0,1).astype(np.float32)
            t=F.interpolate(torch.from_numpy(rgb),size=(128,128),mode='bilinear',align_corners=False)
            t=(t-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
            outputs.append(model(t).numpy())
    return np.concatenate(outputs)


def cached_embeddings(data):
    source=hashlib.sha256((ROOT/'src/refinement/transfer.py').read_bytes()).hexdigest()
    path=ROOT/f'outputs/refinement_cache/resnet18_{data["sha256"]}_{source}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as a:
            assert np.array_equal(a['ids'],data['ids']);return a['features']
    features=embeddings(backbone(),data['images']);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,features=features,ids=data['ids']);return features


def linear_predict(state,features):
    return expit(((features-state['mean'])/state['scale'])@state['coefficient'].ravel()+state['intercept'][0])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',default='primary',choices=['primary','angle_grouped']);args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    data=load_train();raw,names=training_features(data);features=np.concatenate([cached_embeddings(data),raw],axis=1).astype(np.float64)
    split=ROOT/('data/splits/folds.csv' if args.protocol=='primary' else 'data/splits/folds_angle_grouped_steps78.csv');folds=read_folds(data,split)
    if args.protocol=='primary':assert fold_hash(split)==json.loads((ROOT/'configs/validation.json').read_text())['folds_sha256']
    config={'name':'R6_frozen_resnet','C_candidates':[.001,.01,.1,1.],'inner_folds':3,'seed':42,'resnet_input_size':128}
    signature={'config':config,'base_commit':BASE_COMMIT,'protocol':args.protocol,'data_sha256':data['sha256'],'folds_sha256':fold_hash(split),
               'pretrained_source':WEIGHTS.url,'pretrained_parameters_official':11689512,'pretrained_dataset':'ImageNet-1K','backbone_frozen':True,
               'source':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/refinement/transfer.py','src/refinement/features.py']}}
    out=ROOT/f'outputs/refinement/{args.protocol}/R6_frozen_resnet';out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==signature
    else:manifest.write_text(json.dumps(signature,indent=2)+'\n')
    for fold in range(5):
        if (out/f'metrics_fold_{fold}.json').exists():continue
        tick=time.perf_counter();train=folds.fold.to_numpy()!=fold;valid=~train
        pipeline=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,solver='lbfgs'))
        selector=GridSearchCV(pipeline,{'logisticregression__C':config['C_candidates']},cv=StratifiedKFold(3,shuffle=True,random_state=42+fold),scoring='neg_log_loss',n_jobs=1)
        with threadpool_limits(limits=1):selector.fit(features[train],data['labels'][train])
        scaler,classifier=selector.best_estimator_.steps[0][1],selector.best_estimator_.steps[1][1]
        state={'mean':scaler.mean_,'scale':scaler.scale_,'coefficient':classifier.coef_,'intercept':classifier.intercept_}
        np.savez(out/f'fold_{fold}.npz',**state)
        probability=linear_predict(state,features[valid]);assert np.allclose(probability,selector.predict_proba(features[valid])[:,1],rtol=0,atol=1e-10)
        frame=folds.loc[valid].copy();frame['probability']=probability;frame.to_csv(out/f'oof_fold_{fold}.csv',index=False)
        result=metrics(data['labels'][valid],probability,folds.loc[valid,'angle_missing']);result.update({'fold':fold,'C':selector.best_params_['logisticregression__C'],'training_seconds':time.perf_counter()-tick})
        (out/f'metrics_fold_{fold}.json').write_text(json.dumps(result,indent=2)+'\n')
        (out/f'fold_{fold}_provenance.json').write_text(json.dumps({'training_ids':data['ids'][train].tolist(),'validation_ids':data['ids'][valid].tolist(),'signature':signature},indent=2)+'\n')
        pd.DataFrame({'fold':fold,'C':config['C_candidates'],'inner_cv_log_loss':-selector.cv_results_['mean_test_score']}).to_csv(out/f'history_fold_{fold}.csv',index=False)
        print(f"DONE R6_frozen_resnet fold={fold} loss={result['oof_log_loss']:.5f}",flush=True)
    aggregate(data,folds,out,config,args.protocol)


if __name__=='__main__':main()
