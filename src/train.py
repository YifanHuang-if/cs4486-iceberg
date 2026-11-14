"""Reproducible fold training; held-out IDs never enter fitting or augmentation."""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import ROOT, Preprocessor, aligned_folds, fold_hash, load_train
from src.evaluate import metrics
from src.models.cnn import IcebergCNN, augment_flips


def source_fingerprints():
    paths = ['src/data.py', 'src/evaluate.py', 'src/models/cnn.py', 'src/train.py']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def predict_batches(model, x, metadata, batch_size=256):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            values.append(torch.sigmoid(model(x[start:start+batch_size], metadata[start:start+batch_size]))
                          .cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def train_fold(data, folds, config, fold, output_dir, device='cpu', threads=2):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config['seed'] + fold
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    training, validation = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    preprocess = Preprocessor.fit(data['images'][training], data['angles'][training])
    x_train, m_train = preprocess.transform(data['images'][training], data['angles'][training])
    x_val, m_val = preprocess.transform(data['images'][validation], data['angles'][validation])
    x_train, m_train = torch.from_numpy(x_train).to(device), torch.from_numpy(m_train).to(device)
    x_val, m_val = torch.from_numpy(x_val).to(device), torch.from_numpy(m_val).to(device)
    y_train = torch.from_numpy(data['labels'][training].astype(np.float32)).to(device)
    model = IcebergCNN(config['angle_features'], config['dropout']).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()
    order_rng = torch.Generator().manual_seed(seed + 10000)
    augmentation_rng = torch.Generator(device=device).manual_seed(seed + 20000)
    best, best_epoch, stale = float('inf'), 0, 0
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        order = torch.randperm(len(training), generator=order_rng)
        total_loss = 0.0
        for start in range(0, len(order), config['batch_size']):
            index = order[start:start+config['batch_size']].to(device)
            batch = x_train[index]
            if config['augmentation']:
                batch = augment_flips(batch, augmentation_rng)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch, m_train[index]), y_train[index])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training objective')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(index)
        p_val = predict_batches(model, x_val, m_val)
        score = metrics(data['labels'][validation], p_val, data['angles'][validation])['log_loss']
        history.append({'epoch': epoch, 'training_bce': total_loss / len(training),
                        'validation_log_loss': score, 'elapsed_seconds': time.perf_counter() - start_time})
        if score < best - config['min_delta']:
            best, best_epoch, stale = score, epoch, 0
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'config': config, 'preprocessor': preprocess.to_dict(), 'fold': fold,
                        'seed': seed, 'best_epoch': best_epoch, 'validation_log_loss': score,
                        'training_ids': data['ids'][training].tolist(),
                        'validation_ids': data['ids'][validation].tolist(),
                        'train_sha256': data['train_sha256']}, output_dir / f'fold_{fold}.pt')
        else:
            stale += 1
        if epoch % 5 == 0:
            print(f"{config['name']} fold={fold} epoch={epoch} val={score:.5f} best={best:.5f}", flush=True)
        if epoch >= config['min_epochs'] and stale >= config['patience']:
            break
    checkpoint = torch.load(output_dir / f'fold_{fold}.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    probabilities = predict_batches(model, x_val, m_val)
    record = metrics(data['labels'][validation], probabilities, data['angles'][validation])
    record.update({'fold': fold, 'best_epoch': best_epoch, 'epochs_run': epoch,
                   'parameters': parameters, 'seed': seed,
                   'training_seconds': time.perf_counter() - start_time})
    pd.DataFrame(history).to_csv(output_dir / f'history_fold_{fold}.csv', index=False)
    pd.DataFrame({'id': data['ids'][validation], 'fold': fold, 'label': data['labels'][validation],
                  'probability': probabilities, 'angle_missing': ~np.isfinite(data['angles'][validation])})\
        .to_csv(output_dir / f'oof_fold_{fold}.csv', index=False)
    (output_dir / f'metrics_fold_{fold}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f"DONE {config['name']} fold={fold} logloss={record['log_loss']:.5f} "
          f"best_epoch={best_epoch} seconds={record['training_seconds']:.1f}", flush=True)
    return record


def run(config_path, folds_path=None, selected_folds=None, output_root=None, device='cpu', threads=2):
    config = json.loads(Path(config_path).read_text())
    folds_path = Path(folds_path or ROOT / 'data/splits/folds.csv')
    data = load_train()
    folds = aligned_folds(data, folds_path)
    selected_folds = list(range(5)) if selected_folds is None else selected_folds
    out = Path(output_root or ROOT / 'outputs/experiments') / config['name']
    out.mkdir(parents=True, exist_ok=True)
    signature = {'config': config, 'train_sha256': data['train_sha256'],
                 'folds_sha256_lf': fold_hash(folds_path), 'source': source_fingerprints(),
                 'device': device, 'threads': threads}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior['signature'] != signature:
            raise ValueError(f'Existing run differs: use a new output directory: {out}')
    else:
        manifest = {'signature': signature, 'python': platform.python_version(), 'platform': platform.platform(),
                    'packages': {p: importlib.metadata.version(p) for p in ['torch','numpy','pandas','ijson']},
                    'source_base_commit': '18ae3c8f7b5d5ccf8059efbe99a5e3757e05fcea'}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    for fold in selected_folds:
        if fold not in range(5):
            raise ValueError('Invalid fold selection')
        if not (out / f'metrics_fold_{fold}.json').exists():
            train_fold(data, folds, config, fold, out, device, threads)
    if all((out / f'metrics_fold_{k}.json').exists() for k in range(5)):
        oof = pd.concat([pd.read_csv(out / f'oof_fold_{k}.csv', dtype={'id':str}) for k in range(5)])
        if oof.id.duplicated().any() or set(oof.id) != set(data['ids']):
            raise ValueError('Incomplete or duplicate OOF IDs')
        oof = oof.set_index('id').loc[data['ids']].reset_index()
        if not np.array_equal(oof.label, data['labels']) or not np.array_equal(oof.fold, folds):
            raise ValueError('OOF labels/folds disagree')
        records = [json.loads((out / f'metrics_fold_{k}.json').read_text()) for k in range(5)]
        result = metrics(data['labels'], oof.probability.to_numpy(), data['angles'])
        result.update({'name': config['name'], 'fold_log_losses': [r['log_loss'] for r in records],
                       'fold_mean': float(np.mean([r['log_loss'] for r in records])),
                       'fold_std_population': float(np.std([r['log_loss'] for r in records])),
                       'total_training_seconds': sum(r['training_seconds'] for r in records),
                       'config': config, 'folds_sha256_lf': fold_hash(folds_path)})
        weighted = np.average([r['log_loss'] for r in records], weights=[r['samples'] for r in records])
        if not np.isclose(weighted, result['log_loss'], atol=1e-12):
            raise ValueError('Pooled OOF does not match weighted fold scores')
        oof.to_csv(out / 'oof.csv', index=False)
        pd.DataFrame(records).to_csv(out / 'fold_metrics.csv', index=False)
        (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'completed': config['name'], 'oof_log_loss': result['log_loss']}), flush=True)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--folds', type=Path)
    p.add_argument('--only-folds', nargs='+', type=int)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    run(a.config, a.folds, a.only_folds, a.output_root, a.device, a.threads)
