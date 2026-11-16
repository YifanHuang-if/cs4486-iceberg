"""Second-round CNNs: deeper spatial model and task-finetuned ResNet final block."""
import argparse
import copy
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18,ResNet18_Weights
from src.experiments.data import ROOT,load_train,read_folds,fold_hash
from src.experiments.metrics import metrics,log_loss
from src.refinement.cnn import METADATA,Preparation,augment,view
from src.refinement.features import training_features
from src.refinement.train import atomic_save,aggregate,BASE_COMMIT

CONFIGS={
    'R8_deep':{'name':'R8_deep','family':'deep','epochs':60,'min_epochs':30,'patience':15,'lr':.0008,'seed':142,'ema':.98,'batch_size':64},
    'R9_finetuned':{'name':'R9_finetuned','family':'transfer','epochs':24,'min_epochs':12,'patience':8,'lr':.0001,'seed':242,'ema':.95,'batch_size':32},
}


class DeepRadarCNN(nn.Module):
    def __init__(self):
        super().__init__();layers=[];channels=3
        for i,out in enumerate([16,32,64,128]):
            for _ in range(2):
                layers.extend([nn.Conv2d(channels,out,3,padding=1,bias=False),nn.GroupNorm(4,out),nn.SiLU()]);channels=out
            if i<3:layers.extend([nn.MaxPool2d(2),nn.Dropout2d(.05)])
        self.features=nn.Sequential(*layers)
        self.metadata=nn.Sequential(nn.Linear(12,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(272,128),nn.SiLU(),nn.Dropout(.4),nn.Linear(128,1))

    def forward(self,x,metadata):
        x=torch.cat([x,(x[:,0:1]-x[:,1:2])/math.sqrt(2)],dim=1)
        x=self.features(x);x=torch.cat([x.mean((2,3)),x.amax((2,3)),self.metadata(metadata)],dim=1)
        return self.head(x).squeeze(1)


class FinetunedTail(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__();network=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,progress=False)
        self.tail=network.layer4[-1]
        self.metadata=nn.Sequential(nn.Linear(12,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU())
        self.head=nn.Sequential(nn.Dropout(.4),nn.Linear(1040,128),nn.SiLU(),nn.Dropout(.4),nn.Linear(128,1))

    def forward(self,x,metadata):
        x=self.tail(x);x=torch.cat([x.mean((2,3)),x.amax((2,3)),self.metadata(metadata)],dim=1)
        return self.head(x).squeeze(1)


def make_model(family,pretrained=True):
    return DeepRadarCNN() if family=='deep' else FinetunedTail(pretrained)


def pretrained_prefix():
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1,progress=False)
    prefix=nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,model.layer4[0]).eval()
    for parameter in prefix.parameters():parameter.requires_grad_(False)
    return prefix


def prefix_maps(prefix,images):
    rgb=np.concatenate([images,images.mean(1,keepdims=True)],axis=1)
    q=np.quantile(rgb,[.01,.99],axis=(2,3),keepdims=True)
    rgb=np.clip((rgb-q[0])/np.maximum(q[1]-q[0],1e-6),0,1).astype(np.float32)
    rgb=torch.from_numpy(rgb)
    output=np.empty((len(images),8,512,4,4),dtype=np.float16)
    with torch.no_grad():
        for start in range(0,len(images),32):
            for code in range(8):
                batch=F.interpolate(view(rgb[start:start+32],code),size=(128,128),mode='bilinear',align_corners=False)
                batch=(batch-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
                output[start:start+len(batch),code]=prefix(batch).numpy().astype(np.float16)
    return output


def cached_prefix_maps(data):
    source=hashlib.sha256((ROOT/'src/refinement/advanced.py').read_bytes()).hexdigest()
    cache=ROOT/f'outputs/refinement_cache/prefix_{data["sha256"]}_{source}.npy'
    if cache.exists():return np.load(cache,mmap_mode='r',allow_pickle=False)
    prefix=pretrained_prefix();temporary=cache.with_suffix('.tmp.npy')
    result=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float16,shape=(len(data['ids']),8,512,4,4))
    for start in range(0,len(data['ids']),128):
        result[start:start+128]=prefix_maps(prefix,data['images'][start:start+128]);result.flush()
        print('PREFIX',min(start+128,len(data['ids'])),flush=True)
    del result;temporary.replace(cache);return np.load(cache,mmap_mode='r',allow_pickle=False)


def predict_model(model,x,metadata,family,tta=True,batch_size=64):
    model.eval();probabilities=[]
    with torch.no_grad():
        for start in range(0,len(x),batch_size):
            meta=torch.as_tensor(metadata[start:start+batch_size]);predictions=[]
            for code in range(8 if tta else 1):
                if family=='transfer':batch=torch.tensor(np.asarray(x[start:start+batch_size,code]),dtype=torch.float32)
                else:batch=view(torch.as_tensor(x[start:start+batch_size]),code)
                predictions.append(model(batch,meta).sigmoid().numpy().astype(np.float64))
            probabilities.append(np.mean(predictions,axis=0))
    return np.concatenate(probabilities)


def train_fold(data,folds,raw,names,maps,config,fold,out,signature):
    seed=config['seed']+fold;torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+9000)
    train=folds.fold.to_numpy()!=fold;valid=~train;indices=np.flatnonzero(train)
    stats=raw[:,[names.index(n) for n in METADATA]]
    preparation=Preparation.fit(data['images'][train],stats[train],'local')
    images,metadata=preparation.transform(data['images'],stats)
    x=maps if config['family']=='transfer' else images;y=torch.from_numpy(data['labels'])
    model=make_model(config['family']);ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'],eta_min=config['lr']*.02)
    history=[];best=float('inf');best_epoch=0;best_state=None;seconds=0.;start=1
    resume=out/f'fold_{fold}_resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True);assert saved['signature']==signature
        model.load_state_dict(saved['state_dict']);ema.load_state_dict(saved['ema']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng']);rng.set_state(saved['rng']);history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch'];best_state=saved['best_state'];seconds=saved['seconds'];start=saved['epoch']+1
    for epoch in range(start,config['epochs']+1):
        if history and history[-1]['epoch']>=config['min_epochs'] and history[-1]['epoch']-best_epoch>=config['patience']:break
        tick=time.perf_counter();model.train();order=indices[torch.randperm(len(indices),generator=rng).numpy()];total=0.
        for offset in range(0,len(order),config['batch_size']):
            selected=order[offset:offset+config['batch_size']]
            if config['family']=='transfer':
                views=torch.randint(8,(len(selected),),generator=rng).numpy();batch=torch.tensor(np.asarray(x[selected,views]),dtype=torch.float32)
            else:batch=augment(torch.from_numpy(x[selected]),rng)
            meta=torch.from_numpy(metadata[selected]);target=y[selected]
            optimizer.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(batch,meta),target)
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            with torch.no_grad():
                for shadow,current in zip(ema.parameters(),model.parameters()):shadow.mul_(config['ema']).add_(current,alpha=1-config['ema'])
                for shadow,current in zip(ema.buffers(),model.buffers()):shadow.copy_(current)
            total+=loss.item()*len(selected)
        probability=predict_model(ema,x[valid],metadata[valid],config['family'],tta=False)
        score=log_loss(data['labels'][valid],probability)
        history.append({'fold':fold,'epoch':epoch,'train_loss':total/len(indices),'validation_log_loss':score,'learning_rate':optimizer.param_groups[0]['lr']})
        if score<best:best=score;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in ema.state_dict().items()}
        scheduler.step();seconds+=time.perf_counter()-tick
        atomic_save({'signature':signature,'state_dict':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),'rng':rng.get_state(),'history':history,'best':best,'best_epoch':best_epoch,'best_state':best_state,'seconds':seconds,'epoch':epoch},resume)
        if epoch%5==0:print(f'{config["name"]} fold={fold} epoch={epoch} loss={score:.5f} best={best:.5f} seconds={seconds:.1f}',flush=True)
    ema.load_state_dict(best_state);probability=predict_model(ema,x[valid],metadata[valid],config['family'])
    plain=predict_model(ema,x[valid],metadata[valid],config['family'],tta=False)
    checkpoint={'signature':signature,'config':config,'state_dict':best_state,'preparation':preparation.to_dict(),'training_ids':data['ids'][train].tolist(),'validation_ids':data['ids'][valid].tolist()}
    atomic_save(checkpoint,out/f'fold_{fold}.pt')
    frame=folds.loc[valid].copy();frame['plain_probability']=plain;frame['probability']=probability;frame.to_csv(out/f'oof_fold_{fold}.csv',index=False)
    pd.DataFrame(history).to_csv(out/f'history_fold_{fold}.csv',index=False)
    result=metrics(data['labels'][valid],probability,folds.loc[valid,'angle_missing']);result.update({'fold':fold,'best_epoch':best_epoch,'epochs_run':len(history),'plain_log_loss':log_loss(data['labels'][valid],plain),'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':seconds})
    (out/f'metrics_fold_{fold}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DONE',config['name'],fold,result['oof_log_loss'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',choices=list(CONFIGS),required=True);parser.add_argument('--folds',type=int,nargs='+',default=list(range(5)));parser.add_argument('--threads',type=int,default=4);args=parser.parse_args()
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    config=CONFIGS[args.name];data=load_train();raw,names=training_features(data);split=ROOT/'data/splits/folds.csv';folds=read_folds(data,split)
    assert fold_hash(split)==json.loads((ROOT/'configs/validation.json').read_text())['folds_sha256']
    sources=['src/refinement/advanced.py','src/refinement/cnn.py','src/refinement/features.py','src/refinement/train.py']
    signature={'config':config,'base_commit':BASE_COMMIT,'protocol':'primary','data_sha256':data['sha256'],'folds_sha256':fold_hash(split),
               'source':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},'torch':str(torch.__version__),'pretrained_source':ResNet18_Weights.IMAGENET1K_V1.url if config['family']=='transfer' else None}
    out=ROOT/f'outputs/refinement/primary/{args.name}';out.mkdir(parents=True,exist_ok=True);manifest=out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==signature
    else:manifest.write_text(json.dumps(signature,indent=2)+'\n')
    maps=cached_prefix_maps(data) if config['family']=='transfer' else None
    for fold in args.folds:
        if not (out/f'metrics_fold_{fold}.json').exists():train_fold(data,folds,raw,names,maps,config,fold,out,signature)
    aggregate(data,folds,out,config,'primary')


if __name__=='__main__':main()
