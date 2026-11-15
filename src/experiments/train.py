"""Five-fold CPU training with baseline-compatible RNG order and safe epoch resume."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .data import ROOT, Preprocessor, fold_hash, load_train, read_folds
from .metrics import metrics, log_loss
from .model import ExperimentCNN, augment_flips

BASE_COMMIT = 'f672b0c3218503348cab41656c1d4d7d6fc783f6'


def fingerprints():
    paths = ['src/experiments/data.py', 'src/experiments/model.py', 'src/experiments/metrics.py', 'src/experiments/train.py']
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def predict(model, loader):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x, m).sigmoid() for x, _, m in loader]).numpy().astype(np.float64)


def validate_config(config):
    if config['angle_features'] not in (0, 1, 2) or not 0 <= config['dropout'] < 1:
        raise ValueError('Invalid model configuration')
    for key in ['batch_size', 'max_epochs', 'patience', 'threads']:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError('Invalid '+key)
    if config['learning_rate'] <= 0 or config['weight_decay'] < 0:
        raise ValueError('Invalid optimizer configuration')


def fit_fold(data, mapping, config, fold, out, signature, stop_after=None):
    seed = config['seed'] + fold
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_mask = mapping.fold.to_numpy() != fold
    validation_mask = ~train_mask
    preprocessor = Preprocessor.fit(data['images'][train_mask], data['angles'][train_mask])
    x, metadata = preprocessor.transform(data['images'], data['angles'])
    x, y, metadata = torch.from_numpy(x), torch.from_numpy(data['labels']), torch.from_numpy(metadata)
    train_loader = DataLoader(TensorDataset(x[train_mask], y[train_mask], metadata[train_mask]),
                              batch_size=config['batch_size'], shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    valid_loader = DataLoader(TensorDataset(x[validation_mask], y[validation_mask], metadata[validation_mask]),
                              batch_size=128)
    model = ExperimentCNN(config['dropout'], config['angle_features'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()
    augmentation_rng = torch.Generator().manual_seed(seed + 10000)
    resume_path = out / f'fold_{fold}_resume.pt'
    best_path = out / f'fold_{fold}.pt'
    epoch, best_epoch, stale, best, seconds = 0, 0, 0, float('inf'), 0.0
    best_state, history = None, []
    if resume_path.exists():
        saved = torch.load(resume_path, map_location='cpu', weights_only=True)
        if saved['signature'] != signature:
            raise ValueError('Resume signature differs')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']); train_loader.generator.set_state(saved['loader_rng'])
        augmentation_rng.set_state(saved['augmentation_rng'])
        epoch, best_epoch, stale, best = saved['epoch'], saved['best_epoch'], saved['stale'], saved['best']
        best_state, history, seconds = saved['best_state'], saved['history'], saved['training_seconds']
        print(f"Resume {config['name']} fold={fold} after epoch={epoch}", flush=True)
    start_epoch = epoch + 1 if stale < config['patience'] else config['max_epochs'] + 1
    for epoch in range(start_epoch, config['max_epochs'] + 1):
        started = time.perf_counter(); model.train(); total_loss = 0.0
        for batch, target, meta in train_loader:
            if config['augmentation']:
                batch = augment_flips(batch, augmentation_rng)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch, meta), target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward(); optimizer.step()
            total_loss += loss.item() * len(target)
        probabilities = predict(model, valid_loader)
        score = log_loss(data['labels'][validation_mask], probabilities)
        history.append({'fold': fold, 'epoch': epoch, 'train_loss': float(total_loss / train_mask.sum()), 'validation_log_loss': score})
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        seconds += time.perf_counter() - started
        atomic_save({'signature': signature, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'torch_rng': torch.get_rng_state(), 'loader_rng': train_loader.generator.get_state(),
                     'augmentation_rng': augmentation_rng.get_state(), 'epoch': epoch,
                     'best_epoch': best_epoch, 'stale': stale, 'best': best,
                     'best_state': best_state, 'history': history, 'training_seconds': seconds}, resume_path)
        pd.DataFrame(history).to_csv(out / f'history_fold_{fold}.csv', index=False)
        if epoch % 5 == 0:
            print(f"{config['name']} fold={fold} epoch={epoch} val={score:.5f} best={best:.5f}", flush=True)
        if stop_after is not None and epoch >= stop_after and epoch < config['max_epochs'] and stale < config['patience']:
            return None
        if stale >= config['patience']:
            break
    checkpoint = {'state_dict': best_state, 'preprocessor': preprocessor.to_dict(), 'fold': fold,
                  'best_epoch': best_epoch, 'config': config, 'signature': signature,
                  'training_ids': data['ids'][train_mask].tolist(), 'validation_ids': data['ids'][validation_mask].tolist()}
    atomic_save(checkpoint, best_path)
    reloaded = torch.load(best_path, map_location='cpu', weights_only=True)
    model.load_state_dict(reloaded['state_dict'])
    probabilities = predict(model, valid_loader)
    assert abs(log_loss(data['labels'][validation_mask], probabilities) - best) < 1e-7
    record = metrics(data['labels'][validation_mask], probabilities, ~np.isfinite(data['angles'][validation_mask]))
    record.update({'fold': fold, 'seed': seed, 'best_epoch': best_epoch, 'epochs_run': epoch,
                   'parameters': sum(p.numel() for p in model.parameters()), 'training_seconds': seconds})
    predictions = mapping.loc[validation_mask].copy()
    predictions['probability'] = probabilities
    predictions.to_csv(out / f'oof_fold_{fold}.csv', index=False)
    pd.DataFrame(history).to_csv(out / f'history_fold_{fold}.csv', index=False)
    (out / f'metrics_fold_{fold}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f"DONE {config['name']} fold={fold} loss={record['oof_log_loss']:.5f} epoch={best_epoch} seconds={seconds:.1f}", flush=True)
    return record


def run(config_path, protocol='primary', folds_path=None, selected_folds=None, output_root=None, stop_after=None):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    torch.set_num_threads(config['threads']); torch.use_deterministic_algorithms(True)
    data = load_train()
    path = Path(folds_path or ROOT / 'data/splits/folds.csv')
    if protocol == 'primary':
        frozen = json.loads((ROOT / 'configs/validation.json').read_text())
        if fold_hash(path) != frozen['folds_sha256']:
            raise ValueError('Primary fold mapping changed')
    mapping = read_folds(data, path)
    out = Path(output_root or ROOT / 'outputs/steps78') / protocol / config['name']
    out.mkdir(parents=True, exist_ok=True)
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'packages': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'pandas']}}
    signature = {'config': config, 'data_sha256': data['sha256'], 'folds_sha256_lf': fold_hash(path),
                 'training_source': fingerprints(), 'protocol': protocol, 'environment': environment}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['signature'] != signature:
            raise ValueError('Existing run differs; use a new output directory')
    else:
        manifest_path.write_text(json.dumps({'signature': signature, 'source_base_commit': BASE_COMMIT}, indent=2) + '\n')
    for fold in (range(5) if selected_folds is None else selected_folds):
        if fold not in range(5):
            raise ValueError('Invalid fold')
        if not (out / f'metrics_fold_{fold}.json').exists():
            fit_fold(data, mapping, config, fold, out, signature, stop_after)
    if all((out / f'metrics_fold_{k}.json').exists() for k in range(5)):
        oof = pd.concat([pd.read_csv(out / f'oof_fold_{k}.csv', dtype={'id': str}) for k in range(5)])
        if oof.id.duplicated().any() or set(oof.id) != set(data['ids']):
            raise ValueError('Invalid OOF coverage')
        oof = oof.set_index('id').loc[data['ids']].reset_index()
        assert np.array_equal(oof.fold, mapping.fold) and np.array_equal(oof.label, data['labels'])
        rows = [json.loads((out / f'metrics_fold_{k}.json').read_text()) for k in range(5)]
        result = metrics(data['labels'], oof.probability.to_numpy(), mapping.angle_missing.to_numpy())
        result.update({'name': config['name'], 'protocol': protocol,
                       'fold_losses': [r['oof_log_loss'] for r in rows],
                       'fold_mean': float(np.mean([r['oof_log_loss'] for r in rows])),
                       'fold_std_population': float(np.std([r['oof_log_loss'] for r in rows])),
                       'parameters': rows[0]['parameters'], 'training_seconds': sum(r['training_seconds'] for r in rows),
                       'config': config, 'folds_sha256_lf': fold_hash(path)})
        weighted = np.average([r['oof_log_loss'] for r in rows], weights=[r['samples'] for r in rows])
        assert np.isclose(result['oof_log_loss'], weighted, atol=1e-12)
        oof.to_csv(out / 'oof_predictions.csv', index=False)
        pd.DataFrame(rows).to_csv(out / 'fold_scores.csv', index=False)
        pd.concat([pd.read_csv(out / f'history_fold_{k}.csv') for k in range(5)]).to_csv(out / 'history.csv', index=False)
        (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'complete': config['name'], 'protocol': protocol, 'oof_log_loss': result['oof_log_loss']}), flush=True)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--protocol', choices=['primary', 'angle_grouped'], default='primary')
    p.add_argument('--folds', type=Path)
    p.add_argument('--only-folds', nargs='+', type=int)
    p.add_argument('--output-root', type=Path)
    a = p.parse_args()
    run(a.config, a.protocol, a.folds, a.only_folds, a.output_root)
