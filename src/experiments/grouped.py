"""Generate a separate exact-angle-group sensitivity split without editing primary folds."""
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from .data import ROOT, load_train, read_folds, fold_hash


def create_grouped_split():
    data = load_train(); ids, angles, y = data['ids'], data['angles'], data['labels'].astype(int)
    missing = ~np.isfinite(angles)
    groups = np.array([f'angle:{float(a).hex()}' if np.isfinite(a) else f'unknown:{i}' for i, a in zip(ids, angles)])
    order = np.argsort(ids); strata = y*2 + missing.astype(int)
    folds = np.full(len(ids), -1)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, validation) in enumerate(splitter.split(ids[order], strata[order], groups[order])):
        folds[order[validation]] = fold
    mapping = pd.DataFrame({'id': ids, 'fold': folds, 'label': y, 'angle_missing': missing.astype(int)})
    path = ROOT / 'data/splits/folds_angle_grouped_steps78.csv'
    content = mapping.to_csv(index=False, lineterminator='\n')
    if path.exists() and path.read_text() != content:
        raise ValueError('Refusing to replace a different grouped fold mapping')
    path.write_text(content)
    read_folds(data, path)
    for fold in range(5):
        assert not set(groups[folds == fold]) & set(groups[folds != fold])
    summary = {'folds_sha256_lf': fold_hash(path), 'training_sha256': data['sha256'],
               'method': 'StratifiedGroupKFold; exact observed angles; singleton missing angles; sorted IDs; joint label/missingness strata',
               'seed': 42, 'numpy': np.__version__, 'scikit_learn': sklearn.__version__,
               'known_angle_groups_disjoint': True, 'scene_identity_verified': False,
               'counts': mapping.groupby('fold').agg(samples=('id', 'size'), icebergs=('label', 'sum'), missing_angles=('angle_missing', 'sum')).reset_index().to_dict(orient='records')}
    out = ROOT / 'outputs/steps78'; out.mkdir(parents=True, exist_ok=True)
    (out / 'grouped_split.json').write_text(json.dumps(summary, indent=2) + '\n')
    return path


if __name__ == '__main__': print(create_grouped_split())
