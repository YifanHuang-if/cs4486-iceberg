"""Audit saved CV evidence and optionally replay every held-out checkpoint."""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from .data import ROOT, Preprocessor, file_hash, fold_hash, load_train, read_folds
from .metrics import metrics, log_loss
from .model import ExperimentCNN
from .run_suite import PRIMARY, GROUPED
from .train import fingerprints, predict


def close(actual, expected, context, tolerance=1e-9):
    if not np.isclose(actual, expected, rtol=0, atol=tolerance):
        raise ValueError(f'{context}: {actual} != {expected}')


def read_csv(path):
    return pd.read_csv(path, dtype={'id': str}, float_precision='round_trip')


def verify_run(protocol, name, mapping, data=None):
    directory = ROOT / f'results/steps78/{protocol}/{name}'
    summary = json.loads((directory/'summary.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    signature = manifest['signature']
    config = json.loads((ROOT/f'configs/steps78/{name}.json').read_text())
    folds_path = ROOT / ('data/splits/folds.csv' if protocol == 'primary' else 'data/splits/folds_angle_grouped_steps78.csv')
    assert signature['config'] == config == summary['config']
    assert signature['protocol'] == summary['protocol'] == protocol
    assert signature['training_source'] == fingerprints()
    assert signature['folds_sha256_lf'] == summary['folds_sha256_lf'] == fold_hash(folds_path)
    frozen = json.loads((ROOT/'configs/validation.json').read_text())
    assert signature['data_sha256'] == frozen['train_file_sha256']
    oof = read_csv(directory/'oof_predictions.csv')
    assert not oof.id.duplicated().any() and set(oof.id) == set(mapping.id)
    oof = oof.set_index('id').loc[mapping.id].reset_index()
    for column in ['fold', 'label', 'angle_missing']:
        assert np.array_equal(oof[column], mapping[column]), column
    recomputed = metrics(oof.label, oof.probability, oof.angle_missing.astype(bool))
    for key, value in recomputed.items(): close(value, summary[key], f'{name} {key}')
    folds = read_csv(directory/'fold_scores.csv').sort_values('fold')
    history = read_csv(directory/'history.csv')
    assert folds.fold.tolist() == list(range(5))
    assert not history.duplicated(['fold', 'epoch']).any()
    assert set(history.fold) == set(range(5))
    losses = []
    replay_maximum = 0.0
    checkpoints = json.loads((directory/'checkpoints.json').read_text())
    assert [entry['fold'] for entry in checkpoints] == list(range(5))
    for fold in range(5):
        held_out = oof.fold.to_numpy() == fold
        fold_result = folds.iloc[fold]
        score = log_loss(oof.loc[held_out, 'label'], oof.loc[held_out, 'probability'])
        losses.append(score)
        for key, value in metrics(oof.loc[held_out, 'label'], oof.loc[held_out, 'probability'],
                                  oof.loc[held_out, 'angle_missing'].astype(bool)).items():
            close(value, fold_result[key], f'{name} fold {fold} {key}')
        h = history[history.fold == fold].sort_values('epoch')
        assert h.epoch.tolist() == list(range(1, int(fold_result.epochs_run)+1))
        assert int(fold_result.best_epoch) == int(h.loc[h.validation_log_loss.idxmin(), 'epoch'])
        close(score, h.validation_log_loss.min(), f'{name} retained epoch')
        assert int(fold_result.seed) == config['seed'] + fold
        assert len(h) <= config['max_epochs']
        if len(h) < config['max_epochs']:
            assert len(h) - int(fold_result.best_epoch) == config['patience']
        if data is not None:
            entry = checkpoints[fold]
            path = ROOT/entry['path']
            assert path.stat().st_size == entry['bytes'] and file_hash(path) == entry['sha256']
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            assert checkpoint['signature'] == signature and checkpoint['config'] == config
            assert checkpoint['fold'] == fold and checkpoint['best_epoch'] == int(fold_result.best_epoch)
            assert checkpoint['validation_ids'] == data['ids'][held_out].tolist()
            assert checkpoint['training_ids'] == data['ids'][~held_out].tolist()
            assert not set(checkpoint['validation_ids']) & set(checkpoint['training_ids'])
            fitted = Preprocessor.fit(data['images'][~held_out], data['angles'][~held_out])
            assert fitted.to_dict() == checkpoint['preprocessor']
            x, metadata = fitted.transform(data['images'][held_out], data['angles'][held_out])
            loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(data['labels'][held_out]),
                                              torch.from_numpy(metadata)), batch_size=128)
            model = ExperimentCNN(config['dropout'], config['angle_features'])
            model.load_state_dict(checkpoint['state_dict'])
            assert sum(p.numel() for p in model.parameters()) == int(summary['parameters'])
            replay = predict(model, loader)
            difference = float(np.abs(replay - oof.loc[held_out, 'probability'].to_numpy()).max())
            assert difference <= 1e-7
            close(log_loss(data['labels'][held_out], replay), score, f'{name} checkpoint loss')
            replay_maximum = max(replay_maximum, difference)
    close(np.mean(losses), summary['fold_mean'], f'{name} fold mean')
    close(np.std(losses), summary['fold_std_population'], f'{name} fold std')
    close(np.average(losses, weights=folds.samples), summary['oof_log_loss'], f'{name} weighted loss')
    result = {'protocol': protocol, 'experiment': name, 'samples': len(oof), 'folds': 5,
              'oof_log_loss': recomputed['oof_log_loss'], 'metrics_and_provenance_passed': True,
              'checkpoints_replayed': 5 if data is not None else 0,
              'maximum_checkpoint_probability_difference': replay_maximum if data is not None else None}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoints', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    data = load_train() if args.checkpoints else None
    decision = json.loads((ROOT/'results/steps78/sensitivity_decision.json').read_text())
    frozen = json.loads((ROOT/'configs/validation.json').read_text())
    assert fold_hash(ROOT/'data/splits/folds.csv') == frozen['folds_sha256']
    entries = []
    for protocol, names, path in [('primary', PRIMARY, 'data/splits/folds.csv')] + (
        [('angle_grouped', GROUPED, 'data/splits/folds_angle_grouped_steps78.csv')] if decision['triggered'] else []):
        mapping = read_folds(data, ROOT/path) if data is not None else read_csv(ROOT/path)
        if data is not None and protocol == 'angle_grouped':
            known = np.isfinite(data['angles'])
            grouped = pd.DataFrame({'angle': data['angles'][known], 'fold': mapping.fold.to_numpy()[known]})
            assert grouped.groupby('angle').fold.nunique().max() == 1
        entries += [verify_run(protocol, name, mapping, data) for name in names]
    scores = {r['experiment']: r['oof_log_loss'] for r in entries if r['protocol'] == 'primary'}
    delta = min(scores['V3_angle'], scores['V4_missing']) - scores['V1_flips']
    close(delta, decision['delta_best_angle_minus_reference'], 'Supplement trigger')
    assert decision['triggered'] == (delta <= -decision['threshold_reduction'])
    reproduction_path = ROOT/'results/steps78/baseline_reproduction.json'
    if reproduction_path.exists():
        reproduction = json.loads(reproduction_path.read_text())
        close(scores['V0_baseline'], reproduction['current_environment_V0_loss'], 'Baseline replay')
        if reproduction.get('same_environment_replay_passed'):
            assert reproduction['maximum_probability_difference_from_unchanged_script'] <= 1e-7
    report = {'status': 'passed', 'experiment_count': len(entries),
              'fold_models': sum(r['folds'] for r in entries),
              'checkpoints_replayed': sum(r['checkpoints_replayed'] for r in entries),
              'checks': ['OOF ID/label/fold/missingness coverage', 'probability range and metrics',
                         'best epochs and early stopping', 'config/source/data/fold provenance',
                         'pooled loss versus size-weighted fold loss', 'predeclared sensitivity trigger'],
              'runs': entries}
    if args.checkpoints: report['checks'] += ['checkpoint hashes', 'disjoint train/validation IDs',
                                            'training-fold-only preprocessing', 'held-out checkpoint replay',
                                            'exact-angle group separation']
    (ROOT/'results/steps78/verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f"PASS: {len(entries)} experiments; {report['checkpoints_replayed']} checkpoints replayed")


if __name__ == '__main__': main()
