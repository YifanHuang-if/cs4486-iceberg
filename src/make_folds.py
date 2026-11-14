"""Create deterministic five-fold splits and review EDA similarity candidates."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import ijson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_FOLDS = 5
OUT = ROOT / 'results' / 'validation'
OUT.mkdir(parents=True, exist_ok=True)
records = []
with (ROOT / 'data/raw/train.json').open('rb') as handle:
    records = list(ijson.items(handle, 'item', use_float=True))
ids = [record['id'] for record in records]
images = np.asarray([[r['band_1'], r['band_2']] for r in records], dtype=np.float64)
labels = np.asarray([r['is_iceberg'] for r in records], dtype=int)
missing = np.asarray([r['inc_angle'] in ('na', None) for r in records])
assert len(set(ids)) == len(ids)
assert images.shape == (len(ids), 2, 5625)
assert np.isfinite(images).all()

# Center each channel independently to remove inter-channel offset effects.
centered = images - images.mean(axis=2, keepdims=True)
norms = np.linalg.norm(centered, axis=2, keepdims=True)
normalized = centered / np.maximum(norms, 1e-12)
lookup = {sample_id: index for index, sample_id in enumerate(ids)}
candidates = pd.read_csv(ROOT / 'results/eda/similar_image_candidates.csv')
for band in range(2):
    candidates[f'band_{band+1}_correlation'] = [
        float(np.dot(normalized[lookup[a], band], normalized[lookup[b], band]))
        for a, b in zip(candidates.id_a, candidates.id_b)
    ]
candidates['minimum_channel_correlation'] = candidates[['band_1_correlation','band_2_correlation']].min(axis=1)
candidates = candidates.sort_values('minimum_channel_correlation', ascending=False)
candidates.to_csv(OUT / 'candidate_review.csv', index=False)
fig, axes = plt.subplots(6, 4, figsize=(10, 14))
for row, pair in enumerate(candidates.head(6).itertuples()):
    for column, (sample_id, band) in enumerate([(pair.id_a,0),(pair.id_b,0),(pair.id_a,1),(pair.id_b,1)]):
        image = images[lookup[sample_id],band].reshape(75,75)
        lo, hi = np.percentile(image,[2,98])
        axes[row,column].imshow(image,cmap='gray',vmin=lo,vmax=hi)
        axes[row,column].set_title(f'{sample_id}, band {band+1}',fontsize=9)
        axes[row,column].axis('off')
fig.tight_layout()
fig.savefig(OUT / 'candidate_review.png',dpi=120)
plt.close(fig)

# Group identical images, and conservative high-correlation candidates.
parent = list(range(len(ids)))
def root(index):
    while parent[index] != index:
        index = parent[index]
    return index

def union(first, second):
    parent[root(second)] = root(first)

hashes = {}
for index, image in enumerate(images):
    digest = hashlib.sha256(image.astype('<f8').tobytes()).hexdigest()
    if digest in hashes:
        union(index, hashes[digest])
    hashes[digest] = index
for pair in candidates.itertuples():
    if pair.minimum_channel_correlation >= 0.98:
        union(lookup[pair.id_a], lookup[pair.id_b])
groups = np.asarray([root(i) for i in range(len(ids))])
# This dataset must pass the singleton check before using sample stratification.
if len(set(groups)) != len(ids):
    raise ValueError('Related image groups detected; use a group-aware split before proceeding.')

# Stratify jointly by class and missing angle, using sorted IDs for stable input order.
folds = np.full(len(ids), -1, dtype=int)
rng = np.random.default_rng(SEED)
offset = 0
strata = labels * 2 + missing.astype(int)
for stratum in sorted(set(strata)):
    indices = np.asarray(sorted(np.flatnonzero(strata == stratum), key=lambda i: ids[i]))
    rng.shuffle(indices)
    assignments = (np.arange(len(indices)) + offset) % N_FOLDS
    folds[indices] = assignments
    offset = (offset + len(indices)) % N_FOLDS
mapping = pd.DataFrame({'id':ids,'fold':folds,'label':labels,'angle_missing':missing.astype(int)})
assert set(folds) == set(range(N_FOLDS))
assert len(mapping) == len(ids) and mapping.id.is_unique
for fold in range(N_FOLDS):
    assert set(labels[folds == fold]) == {0,1}
    assert not (set(np.asarray(ids)[folds == fold]) & set(np.asarray(ids)[folds != fold]))
counts = mapping.groupby(['fold','label','angle_missing']).size().unstack(['label','angle_missing'],fill_value=0)
assert (counts.max() - counts.min()).max() <= 1
path = ROOT / 'data/splits/folds.csv'
path.parent.mkdir(parents=True,exist_ok=True)
content = mapping.to_csv(index=False,lineterminator='\n')
if path.exists() and path.read_text(encoding='utf-8') != content:
    raise ValueError('Existing frozen folds differ. Review and version changes explicitly.')
path.write_text(content,encoding='utf-8')
summary = mapping.groupby('fold').agg(samples=('id','size'),icebergs=('label','sum'),missing_angles=('angle_missing','sum'))
summary['ships'] = summary.samples-summary.icebergs
summary['training_samples'] = len(ids)-summary.samples
summary.to_csv(OUT/'fold_summary.csv')
config = {'n_folds':N_FOLDS,'seed':SEED,'rng':'numpy.default_rng (PCG64)','stratification':['label','angle_missing'],'algorithm':'Sort IDs within joint strata, shuffle, distribute round-robin with carried offset','fold_numbering':'0 through 4','folds_sha256':hashlib.sha256(content.encode()).hexdigest(),'train_file_sha256':hashlib.file_digest((ROOT/'data/raw/train.json').open('rb'),'sha256').hexdigest(),'candidate_pairs_reviewed':len(candidates),'max_minimum_channel_correlation':float(candidates.minimum_channel_correlation.max()),'grouping_threshold':0.98,'confirmed_groups_larger_than_one':0,'numpy_version':np.__version__}
(ROOT/'configs/validation.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
print(summary.to_string())
print(json.dumps(config,indent=2))
