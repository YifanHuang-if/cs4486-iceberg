"""Recompute published metrics; optionally replay saved checkpoints against raw training data."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data import ROOT, Preprocessor, load_train, fold_hash, sha256_file
from src.evaluate import metrics
from src.models.cnn import IcebergCNN
from src.train import predict_batches, source_fingerprints
from src.run_experiments import PRIMARY, GROUPED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', action='store_true', help='Also requires train.json and saved weights')
    parser.add_argument('--protocol', choices=['primary', 'angle_grouped', 'all'], default='all')
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = load_train() if args.checkpoints else None
    count = 0
    for protocol, names, split_path in [
        ('primary', PRIMARY, ROOT / 'data/splits/folds.csv'),
        ('angle_grouped', GROUPED, ROOT / 'data/splits/folds_angle_grouped_v1.csv')]:
        if args.protocol != 'all' and protocol != args.protocol:
            continue
        split = pd.read_csv(split_path, dtype={'id': str}).set_index('id')
        for name in names:
            directory = ROOT / 'results/models' / protocol / name
            summary = json.loads((directory / 'summary.json').read_text())
            manifest = json.loads((directory / 'manifest.json').read_text())
            signature = manifest['signature']
            assert signature['source'] == source_fingerprints(), 'Training source changed'
            assert signature['folds_sha256_lf'] == fold_hash(split_path), 'Fold hash changed'
            assert signature['config'] == json.loads((ROOT / f'configs/experiments/{name}.json').read_text())
            oof = pd.read_csv(directory / 'oof.csv', dtype={'id': str}).set_index('id')
            assert not oof.index.duplicated().any() and set(oof.index) == set(split.index)
            oof = oof.loc[split.index]
            assert np.array_equal(oof.label, split.label) and np.array_equal(oof.fold, split.fold)
            assert np.array_equal(oof.angle_missing.astype(bool), split.angle_missing.astype(bool))
            # Only the missing/observed mask affects subgroup scoring.
            angles = np.where(split.angle_missing.astype(bool), np.nan, 0.)
            measured = metrics(oof.label.to_numpy(), oof.probability.to_numpy(), angles)
            for key, value in measured.items():
                assert np.isclose(value, summary[key], atol=1e-12), (name, key)
            fold_metrics = pd.read_csv(directory / 'fold_metrics.csv').set_index('fold')
            history = pd.read_csv(directory / 'history.csv')
            for k in range(5):
                mask = oof.fold == k
                score = metrics(oof.loc[mask, 'label'].to_numpy(), oof.loc[mask, 'probability'].to_numpy(), angles[mask])
                assert np.isclose(score['log_loss'], fold_metrics.loc[k, 'log_loss'], atol=1e-12)
                h = history[history.fold == k]
                assert int(h.loc[h.validation_log_loss.idxmin(), 'epoch']) == int(fold_metrics.loc[k, 'best_epoch'])
                assert np.isclose(h.validation_log_loss.min(), score['log_loss'], atol=1e-12)
                assert h.epoch.max() == fold_metrics.loc[k, 'epochs_run']
            assert np.isclose(fold_metrics.log_loss.std(ddof=0), summary['fold_std_population'], atol=1e-12)
            assert np.isclose(fold_metrics.log_loss.mean(), summary['fold_mean'], atol=1e-12)
            if args.checkpoints:
                assert signature['train_sha256'] == data['train_sha256']
                for item in json.loads((directory / 'checkpoints.json').read_text()):
                    path = ROOT / item['path']; k = item['fold']
                    assert sha256_file(path) == item['sha256']
                    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                    folds = split.loc[data['ids'], 'fold'].to_numpy()
                    training, validation = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
                    assert checkpoint['training_ids'] == data['ids'][training].tolist()
                    assert checkpoint['validation_ids'] == data['ids'][validation].tolist()
                    fitted = Preprocessor.fit(data['images'][training], data['angles'][training])
                    assert fitted.to_dict() == checkpoint['preprocessor']
                    assert checkpoint['config'] == signature['config']
                    x, meta = fitted.transform(data['images'][validation], data['angles'][validation])
                    config = checkpoint['config']
                    model = IcebergCNN(config['angle_features'], config['dropout'])
                    model.load_state_dict(checkpoint['state_dict'])
                    predictions = predict_batches(model, torch.from_numpy(x), torch.from_numpy(meta))
                    saved = oof.loc[data['ids'][validation], 'probability'].to_numpy()
                    assert np.allclose(predictions, saved, rtol=0, atol=1e-7)
            count += 1
            print(f'PASS {protocol}/{name}: OOF, metrics, histories, source/config/folds' +
                  (' and 5 checkpoint replays' if args.checkpoints else ''), flush=True)
    print(f'Verified {count} completed five-fold experiments.', flush=True)


if __name__ == '__main__':
    main()
