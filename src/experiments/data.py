"""Validated training data, frozen ID mappings and training-fold-only preprocessing."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def fold_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def load_train():
    source = ROOT / 'data/raw/train.json'
    digest = file_hash(source)
    frozen = json.loads((ROOT / 'configs/validation.json').read_text(encoding='utf-8-sig'))
    if digest != frozen['train_file_sha256']:
        raise ValueError('Training data differs from the frozen baseline source')
    cache = ROOT / 'outputs/steps78_cache' / f'train_{digest}.npz'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
    else:
        records = json.loads(source.read_text())
        data = {
            'ids': np.asarray([r['id'] for r in records]),
            'images': np.asarray([[r['band_1'], r['band_2']] for r in records], dtype=np.float32).reshape(-1, 2, 75, 75),
            'labels': np.asarray([r['is_iceberg'] for r in records], dtype=np.float32),
            'angles': np.asarray([np.nan if r['inc_angle'] in ('na', None) else float(r['inc_angle']) for r in records], dtype=np.float64),
        }
        del records
        with cache.with_suffix('.tmp').open('wb') as handle:
            np.savez(handle, **data)
        cache.with_suffix('.tmp').replace(cache)
    n = len(data['ids'])
    if len(set(data['ids'])) != n or data['images'].shape != (n, 2, 75, 75):
        raise ValueError('Invalid IDs or image shape')
    if not np.isfinite(data['images']).all() or not np.isin(data['labels'], [0, 1]).all():
        raise ValueError('Invalid images or labels')
    if np.isinf(data['angles']).any():
        raise ValueError('Infinite incidence angle')
    data['sha256'] = digest
    return data


def read_folds(data, path):
    frame = pd.read_csv(path, dtype={'id': str})
    if frame.id.duplicated().any() or set(frame.id) != set(data['ids']):
        raise ValueError('Fold IDs must exactly cover the training data')
    frame = frame.set_index('id').loc[data['ids']].reset_index()
    if not np.equal(frame.fold, frame.fold.astype(int)).all() or set(frame.fold) != set(range(5)):
        raise ValueError('Invalid fold assignment')
    if not np.array_equal(frame.label, data['labels']):
        raise ValueError('Fold labels differ from training labels')
    if not np.array_equal(frame.angle_missing.astype(bool), ~np.isfinite(data['angles'])):
        raise ValueError('Fold missingness differs from training data')
    for fold in range(5):
        if set(frame.loc[frame.fold == fold, 'label']) != {0, 1}:
            raise ValueError('Both classes are required in every fold')
    return frame


@dataclass
class Preprocessor:
    channel_mean: list
    channel_std: list
    angle_median: float
    angle_mean: float
    angle_std: float

    @classmethod
    def fit(cls, images, angles):
        # Match baseline_v1's float64 reductions followed by float32 casting.
        mean = images.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
        std = images.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32).clip(1e-6)
        observed = angles[np.isfinite(angles)]
        median = float(np.median(observed)) if len(observed) else 0.0
        filled = np.where(np.isfinite(angles), angles, median)
        return cls(mean.tolist(), std.tolist(), median, float(filled.mean()), max(float(filled.std()), 1e-6))

    def transform(self, images, angles):
        mean = np.asarray(self.channel_mean, dtype=np.float32)[None, :, None, None]
        std = np.asarray(self.channel_std, dtype=np.float32)[None, :, None, None]
        missing = ~np.isfinite(angles)
        filled = np.where(missing, self.angle_median, angles)
        x = np.ascontiguousarray((images - mean) / std, dtype=np.float32)
        metadata = np.column_stack(((filled - self.angle_mean) / self.angle_std, missing)).astype(np.float32)
        return x, metadata

    def to_dict(self):
        return asdict(self)
