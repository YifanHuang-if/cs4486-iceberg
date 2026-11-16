"""Train stronger radar models on unchanged folds; save auditable held-out outputs."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from src.experiments.data import ROOT,load_train,read_folds,fold_hash
from src.experiments.metrics import log_loss,metrics
from src.refinement.features import training_features
from src.refinement.cnn import METADATA,Preparation,RadarCNN,augment,predict

BASE_COMMIT='f8c9ff017fe8a818784b5e8e02791543f3a31c2e'


def source_hashes():
    paths=['src/refinement/features.py','src/refinement/cnn.py','src/refinement/train.py']
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def atomic_save(value,path):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def train_fold(data,folds,stats,config,fold,out,signature,device):
    seed=config['seed']+fold;torch.manual_seed(seed);np.random.seed(seed)
    train=folds.fold.to_numpy()!=fold;valid=~train
    preparation=Preparation.fit(data['images'][train],stats[train],config['normalization'])
    x,m=preparation.transform(data['images'],stats)
    x=torch.from_numpy(x);m=torch.from_numpy(m);y=torch.from_numpy(data['labels'])
    loader=DataLoader(TensorDataset(x[train],m[train],y[train]),batch_size=64,shuffle=True,
                      generator=torch.Generator().manual_seed(seed),num_workers=0)
    augmentation_rng=torch.Generator().manual_seed(seed+10000)
    model=RadarCNN().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    def schedule(epoch):
        return (epoch+1)/3 if epoch<3 else .01+.99*(1+math.cos(math.pi*(epoch-3)/(config['epochs']-3)))/2
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    criterion=nn.BCEWithLogitsLoss();history=[];best=float('inf');best_epoch=0;seconds=0.;start=1
    best_state=None;resume=out/f'fold_{fold}_resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location=device,weights_only=True)
        assert saved['signature']==signature
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng'].cpu());loader.generator.set_state(saved['loader_rng'].cpu());augmentation_rng.set_state(saved['augmentation_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch'];best_state=saved['best_state'];seconds=saved['seconds'];start=saved['epoch']+1
    for epoch in range(start,config['epochs']+1):
        if history and history[-1]['epoch']>=config['min_epochs'] and history[-1]['epoch']-best_epoch>=config['patience']:break
        tick=time.perf_counter();model.train();total=0.
        for images,metadata,target in loader:
            images=augment(images.to(device),augmentation_rng);metadata=metadata.to(device);target=target.to(device)
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(images,metadata),target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=loss.item()*len(target)
        probability=predict(model,x[valid],m[valid],device)
        score=log_loss(data['labels'][valid],probability)
        history.append({'fold':fold,'epoch':epoch,'train_loss':float(total/train.sum()),'validation_log_loss':score,'learning_rate':optimizer.param_groups[0]['lr']})
        if score<best:
            best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        scheduler.step();seconds+=time.perf_counter()-tick
        atomic_save({'signature':signature,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                     'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                     'loader_rng':loader.generator.get_state(),'augmentation_rng':augmentation_rng.get_state(),
                     'history':history,'best':best,'best_epoch':best_epoch,'best_state':best_state,'epoch':epoch,'seconds':seconds},resume)
        if epoch%5==0:print(f"{config['name']} fold={fold} epoch={epoch} val={score:.5f} best={best:.5f} seconds={seconds:.1f}",flush=True)
    checkpoint={'signature':signature,'config':config,'state_dict':best_state,'preparation':preparation.to_dict(),'metadata_columns':METADATA,
                'fold':fold,'best_epoch':best_epoch,'training_ids':data['ids'][train].tolist(),'validation_ids':data['ids'][valid].tolist()}
    atomic_save(checkpoint,out/f'fold_{fold}.pt')
    reloaded=torch.load(out/f'fold_{fold}.pt',map_location=device,weights_only=True);model.load_state_dict(reloaded['state_dict'])
    plain=predict(model,x[valid],m[valid],device);tta=predict(model,x[valid],m[valid],device,tta=True)
    assert abs(log_loss(data['labels'][valid],plain)-best)<1e-7
    record=metrics(data['labels'][valid],tta,folds.loc[valid,'angle_missing'].to_numpy())
    record.update({'fold':fold,'best_epoch':best_epoch,'epochs_run':len(history),'plain_log_loss':best,'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':seconds})
    frame=folds.loc[valid].copy();frame['probability']=tta;frame['plain_probability']=plain
    frame.to_csv(out/f'oof_fold_{fold}.csv',index=False);pd.DataFrame(history).to_csv(out/f'history_fold_{fold}.csv',index=False)
    (out/f'metrics_fold_{fold}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f"DONE {config['name']} fold={fold} TTA={record['oof_log_loss']:.5f} plain={best:.5f} epoch={best_epoch}",flush=True)


def aggregate(data,folds,out,config,protocol):
    if not all((out/f'metrics_fold_{k}.json').exists() for k in range(5)):return
    frame=pd.concat([pd.read_csv(out/f'oof_fold_{k}.csv',dtype={'id':str},float_precision='round_trip') for k in range(5)])
    assert not frame.id.duplicated().any() and set(frame.id)==set(data['ids'])
    frame=frame.set_index('id').loc[data['ids']].reset_index();assert np.array_equal(frame.fold,folds.fold)
    result=metrics(data['labels'],frame.probability,folds.angle_missing)
    rows=[json.loads((out/f'metrics_fold_{k}.json').read_text()) for k in range(5)]
    result.update({'name':config['name'],'protocol':protocol,'config':config,'fold_losses':[r['oof_log_loss'] for r in rows],
                   'fold_std':float(np.std([r['oof_log_loss'] for r in rows])),'training_seconds':sum(r['training_seconds'] for r in rows),
                   'strict_0_083_local_target_met':bool(result['oof_log_loss']<.083),'official_kaggle_score':None})
    if 'plain_probability' in frame:result['plain_log_loss']=log_loss(frame.label,frame.plain_probability)
    frame.to_csv(out/'oof_predictions.csv',index=False);pd.DataFrame(rows).to_csv(out/'fold_scores.csv',index=False)
    histories=list(out.glob('history_fold_*.csv'))
    if histories:pd.concat([pd.read_csv(p) for p in sorted(histories)]).to_csv(out/'history.csv',index=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete':config['name'],'protocol':protocol,'oof_log_loss':result['oof_log_loss']}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--protocol',choices=['primary','angle_grouped'],default='primary');parser.add_argument('--folds',nargs='+',type=int)
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda']);args=parser.parse_args()
    config=json.loads(args.config.read_text());torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    data=load_train();raw,names=training_features(data);stats=raw[:,[names.index(n) for n in METADATA]]
    split=ROOT/('data/splits/folds.csv' if args.protocol=='primary' else 'data/splits/folds_angle_grouped_steps78.csv')
    if args.protocol=='primary':assert fold_hash(split)==json.loads((ROOT/'configs/validation.json').read_text())['folds_sha256']
    folds=read_folds(data,split);out=ROOT/f'outputs/refinement/{args.protocol}/{config["name"]}';out.mkdir(parents=True,exist_ok=True)
    signature={'config':config,'protocol':args.protocol,'data_sha256':data['sha256'],'folds_sha256':fold_hash(split),
               'source':source_hashes(),'base_commit':BASE_COMMIT,'device':str(device),'python':platform.python_version(),
               'packages':{p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy']}}
    manifest=out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==signature,'Run changed; use another name'
    else:manifest.write_text(json.dumps(signature,indent=2)+'\n')
    for fold in args.folds or range(5):
        if fold not in range(5):raise ValueError('Invalid fold')
        if not (out/f'metrics_fold_{fold}.json').exists():train_fold(data,folds,stats,config,fold,out,signature,device)
    aggregate(data,folds,out,config,args.protocol)


if __name__=='__main__':main()
