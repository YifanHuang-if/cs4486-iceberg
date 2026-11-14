"""Training data and fold-local preprocessing; no global fitting."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import ijson
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fold_hash(path):
    """Canonicalize Windows CRLF to LF before checking the frozen fold hash."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def load_train(data_dir=None, cache_dir=None):
    data_dir = Path(data_dir or ROOT / 'data/raw')
    cache_dir = Path(cache_dir or ROOT / 'outputs/cache')
    path = data_dir / 'train.json'
    fingerprint = sha256_file(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f'train_{fingerprint[:16]}.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            result = {k: z[k] for k in z.files}
    else:
        ids, images, labels, angles = [], [], [], []
        with path.open('rb') as f:
            for row in ijson.items(f, 'item', use_float=True):
                image = np.asarray([row['band_1'], row['band_2']], dtype=np.float32)
                if image.shape != (2, 5625) or not np.isfinite(image).all():
                    raise ValueError(f"Invalid image: {row['id']}")
                if row['is_iceberg'] not in (0, 1):
                    raise ValueError('Invalid label')
                value = row['inc_angle']
                angle = np.nan if value in ('na', None) else float(value)
                if not np.isnan(angle) and not np.isfinite(angle):
                    raise ValueError('Invalid angle')
                ids.append(row['id']); images.append(image.reshape(2, 75, 75))
                labels.append(row['is_iceberg']); angles.append(angle)
        result = {'ids': np.asarray(ids), 'images': np.stack(images),
                  'labels': np.asarray(labels, dtype=np.int64),
                  'angles': np.asarray(angles, dtype=np.float64)}
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate training IDs')
        np.savez(cache, **result)
    result['train_sha256'] = fingerprint
    return result


def aligned_folds(data, path):
    table = pd.read_csv(path, dtype={'id': str})
    if table.id.duplicated().any() or set(table.id) != set(data['ids']):
        raise ValueError('Fold IDs must match training IDs exactly once')
    table = table.set_index('id').loc[data['ids']]
    if table['fold'].isna().any() or not np.equal(table['fold'], table['fold'].astype(int)).all():
        raise ValueError('Invalid fold values')
    folds = table['fold'].to_numpy(dtype=np.int64)
    if set(folds) != set(range(5)):
        raise ValueError('Expected folds 0 through 4')
    if 'label' in table and not np.array_equal(table.label, data['labels']):
        raise ValueError('Fold labels disagree with source data')
    for k in range(5):
        if set(data['labels'][folds == k]) != {0, 1}:
            raise ValueError('Each validation fold needs both classes')
    return folds


@dataclass
class Preprocessor:
    channel_mean: list
    channel_std: list
    angle_median: float
    angle_mean: float
    angle_std: float

    @classmethod
    def fit(cls, images, angles):
        mean = images.mean(axis=(0, 2, 3), dtype=np.float64)
        std = np.maximum(images.std(axis=(0, 2, 3), dtype=np.float64), 1e-6)
        observed = angles[np.isfinite(angles)]
        median = float(np.median(observed)) if len(observed) else 0.0
        filled = np.where(np.isfinite(angles), angles, median)
        return cls(mean.tolist(), std.tolist(), median, float(filled.mean()),
                   max(float(filled.std()), 1e-6))

    def transform(self, images, angles):
        mean = np.asarray(self.channel_mean, dtype=np.float32)[None, :, None, None]
        std = np.asarray(self.channel_std, dtype=np.float32)[None, :, None, None]
        x = np.ascontiguousarray((images - mean) / std, dtype=np.float32)
        missing = ~np.isfinite(angles)
        filled = np.where(missing, self.angle_median, angles)
        metadata = np.column_stack(((filled - self.angle_mean) / self.angle_std,
                                    missing.astype(np.float32))).astype(np.float32)
        return x, metadata

    def to_dict(self):
        return asdict(self)
