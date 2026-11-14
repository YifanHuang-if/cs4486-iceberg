"""Train an image-only CNN on the frozen five-fold split."""
from pathlib import Path
import json
import hashlib
import random
import time
import platform
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((3,3)), nn.Flatten(),
            nn.Linear(576,64), nn.ReLU(), nn.Dropout(0.25), nn.Linear(64,1))

    def forward(self, images):
        return self.network(images).squeeze(1)

def log_loss(labels, probabilities):
    probabilities = np.clip(np.asarray(probabilities,dtype=np.float64),1e-7,1-1e-7)
    return float(-np.mean(labels*np.log(probabilities)+(1-labels)*np.log1p(-probabilities)))

def main():
    config = json.loads((ROOT/'configs/baseline.json').read_text(encoding='utf-8-sig'))
    out = ROOT/'results/baseline_v1'
    weights = ROOT/'outputs/baseline_v1'
    out.mkdir(parents=True,exist_ok=True)
    weights.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    validation = json.loads((ROOT/'configs/validation.json').read_text(encoding='utf-8-sig'))
    fold_hash = hashlib.sha256((ROOT/'data/splits/folds.csv').read_text().encode()).hexdigest()
    if fold_hash != validation['folds_sha256']:
        raise ValueError('Frozen fold hash mismatch')
    with (ROOT/'data/raw/train.json').open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != validation['train_file_sha256']:
            raise ValueError('Training file hash mismatch')
    records = json.loads((ROOT/'data/raw/train.json').read_text())
    folds = pd.read_csv(ROOT/'data/splits/folds.csv').set_index('id')
    ids = [r['id'] for r in records]
    assert set(ids)==set(folds.index) and folds.index.is_unique
    table = folds.loc[ids].reset_index()
    labels = np.asarray([r['is_iceberg'] for r in records],dtype=np.float32)
    assert np.array_equal(labels,table.label)
    images = np.asarray([[r['band_1'],r['band_2']] for r in records],dtype=np.float32).reshape(-1,2,75,75)
    assert np.isfinite(images).all()
    oof = np.full(len(ids),np.nan)
    history, scores = [], []
    started = time.time()
    for fold in range(5):
        seed = config['seed']+fold
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        train_mask = table.fold.to_numpy()!=fold
        valid_mask = ~train_mask
        mean = images[train_mask].mean(axis=(0,2,3),keepdims=True,dtype=np.float64).astype(np.float32)
        std = images[train_mask].std(axis=(0,2,3),keepdims=True,dtype=np.float64).astype(np.float32).clip(1e-6)
        x = torch.from_numpy((images-mean)/std)
        y = torch.from_numpy(labels)
        loader = DataLoader(TensorDataset(x[train_mask],y[train_mask]),batch_size=config['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
        valid_loader = DataLoader(TensorDataset(x[valid_mask],y[valid_mask]),batch_size=128)
        model = BaselineCNN()
        optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        criterion = nn.BCEWithLogitsLoss()
        best, best_epoch, stale = float('inf'),0,0
        resume_path = weights/f'fold_{fold}_resume.pt'
        epoch = 0
        best_state = None
        fold_history = []
        if resume_path.exists():
            saved = torch.load(resume_path, map_location='cpu', weights_only=True)
            if saved['config'] != config or saved['fold_hash'] != fold_hash:
                raise ValueError('Resume configuration or folds differ')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['torch_rng'])
            loader.generator.set_state(saved['loader_rng'])
            epoch, best, best_epoch, stale = saved['epoch'], saved['best'], saved['best_epoch'], saved['stale']
            best_state, fold_history = saved['best_state'], saved['history']
            history.extend(fold_history)
            print(f'Resuming fold {fold} after epoch {epoch}', flush=True)
        start_epoch = epoch + 1
        if stale >= config['patience']:
            start_epoch = config['max_epochs'] + 1
        for epoch in range(start_epoch,config['max_epochs']+1):
            model.train()
            loss_sum = 0
            for batch, target in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch),target)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward(); optimizer.step()
                loss_sum += loss.item()*len(target)
            model.eval()
            with torch.no_grad():
                prediction = torch.cat([model(batch).sigmoid() for batch,_ in valid_loader]).numpy()
            score = log_loss(labels[valid_mask],prediction)
            history.append(dict(fold=fold,epoch=epoch,train_loss=float(loss_sum/train_mask.sum()),validation_log_loss=score))
            fold_history.append(history[-1])
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            print(f'Fold {fold}, epoch {epoch}: train={loss_sum/train_mask.sum():.5f}, validation={score:.5f}',flush=True)
            if score < best:
                best, best_epoch, stale = score,epoch,0
                best_state = {name: value.detach().clone() for name,value in model.state_dict().items()}
                torch.save({'state_dict':model.state_dict(),'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),'fold':fold,'epoch':epoch,'config':config},weights/f'fold_{fold}.pt')
            else:
                stale += 1
            state = {'config':config, 'fold_hash':fold_hash, 'model':model.state_dict(),
                     'optimizer':optimizer.state_dict(), 'torch_rng':torch.get_rng_state(),
                     'loader_rng':loader.generator.get_state(), 'epoch':epoch, 'best':best,
                     'best_epoch':best_epoch, 'stale':stale, 'best_state':best_state,
                     'history':fold_history}
            temporary = resume_path.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(resume_path)
            if stale >= config['patience']:
                break
        torch.save({'state_dict':best_state,'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),'fold':fold,'epoch':best_epoch,'config':config},weights/f'fold_{fold}.pt')
        checkpoint = torch.load(weights/f'fold_{fold}.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        with torch.no_grad():
            prediction = torch.cat([model(batch).sigmoid() for batch,_ in valid_loader]).numpy()
        assert abs(log_loss(labels[valid_mask],prediction)-best)<1e-7
        oof[valid_mask] = prediction
        scores.append(dict(fold=fold,best_epoch=best_epoch,epochs_run=epoch,samples=int(valid_mask.sum()),log_loss=best))
        pd.DataFrame(scores).to_csv(out/'fold_scores.csv',index=False)
    assert np.isfinite(oof).all()
    table['probability'] = oof
    table.to_csv(out/'oof_predictions.csv',index=False)
    pooled = log_loss(labels,oof)
    assert abs(pooled-np.average([s['log_loss'] for s in scores],weights=[s['samples'] for s in scores]))<1e-10
    prior = np.zeros(len(ids))
    for fold in range(5):
        mask = table.fold.to_numpy()==fold
        prior[mask]=labels[~mask].mean()
    summary = dict(oof_log_loss=pooled,fold_mean=float(np.mean([s['log_loss'] for s in scores])),fold_std=float(np.std([s['log_loss'] for s in scores])),accuracy=float(np.mean((oof>=.5)==labels)),prior_baseline_log_loss=log_loss(labels,prior),parameters=sum(p.numel() for p in model.parameters()),elapsed_seconds=time.time()-started,python=platform.python_version(),torch=str(torch.__version__),device='cpu',config=config,folds_sha256=hashlib.sha256((ROOT/'data/splits/folds.csv').read_text().encode()).hexdigest())
    for missing in [0,1]:
        mask = table.angle_missing.to_numpy()==missing
        summary[f'angle_missing_{missing}_log_loss']=log_loss(labels[mask],oof[mask])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    frame=pd.DataFrame(history)
    for fold in range(5):
        part=frame[frame.fold==fold]
        axes.flat[fold].plot(part.epoch,part.train_loss,label='Training BCE')
        axes.flat[fold].plot(part.epoch,part.validation_log_loss,label='Validation Log Loss')
        axes.flat[fold].set(title=f'Fold {fold}',xlabel='Epoch',ylabel='Loss')
        axes.flat[fold].legend()
    axes.flat[5].axis('off')
    fig.tight_layout(); fig.savefig(out/'learning_curves.png',dpi=120);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
