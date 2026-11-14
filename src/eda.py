from pathlib import Path
import csv
import hashlib
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import ijson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'eda'
OUT.mkdir(parents=True, exist_ok=True)
rows, examples, thumbnails = [], {}, []
issues = []
for split in ['train', 'test']:
    with (ROOT / 'data' / 'raw' / f'{split}.json').open('rb') as handle:
        for record in ijson.items(handle, 'item', use_float=True):
            expected = {'id', 'band_1', 'band_2', 'inc_angle'} | ({'is_iceberg'} if split == 'train' else set())
            if set(record) != expected:
                issues.append(f'{split}/{record.get("id")}: unexpected fields')
            bands = np.asarray([record['band_1'], record['band_2']], dtype=np.float64)
            if bands.shape != (2, 5625) or not np.isfinite(bands).all():
                raise ValueError(f'Invalid image: {split}/{record["id"]}')
            label = record.get('is_iceberg')
            if split == 'train' and label not in [0, 1]:
                raise ValueError('Invalid training label')
            angle = record['inc_angle']
            missing = angle is None or angle == 'na'
            angle = np.nan if missing else float(angle)
            if not missing and not np.isfinite(angle):
                raise ValueError('Nonfinite angle')
            digest = hashlib.sha256(bands.astype('<f8').tobytes()).hexdigest()
            row = dict(split=split, id=record['id'], label=label, inc_angle=angle, angle_missing=missing, image_hash=digest)
            for i in range(2):
                for name, value in [('mean', bands[i].mean()), ('std', bands[i].std()), ('min', bands[i].min()), ('max', bands[i].max())]:
                    row[f'band_{i+1}_{name}'] = float(value)
            rows.append(row)
            if split == 'train':
                small = bands.reshape(2, 15, 5, 15, 5).mean(axis=(2, 4)).ravel()
                small = small - small.mean()
                thumbnails.append(small / max(np.linalg.norm(small), 1e-12))
            if split == 'train' and label not in examples:
                examples[label] = (record['id'], bands.reshape(2, 75, 75))
    print(f'Finished {split}', flush=True)
frame = pd.DataFrame(rows)
frame.to_csv(OUT / 'sample_metadata.csv', index=False)
train = frame[frame.split == 'train']
test = frame[frame.split == 'test']
duplicates = frame[frame.duplicated('image_hash', keep=False)]
duplicates.to_csv(OUT / 'exact_image_duplicates.csv', index=False)
with (ROOT / 'data/raw/sample_submission.csv').open(newline='') as handle:
    reader = csv.DictReader(handle)
    header = reader.fieldnames
    submission_ids = [r['id'] for r in reader]
summary = {
    'train_samples': len(train), 'test_samples': len(test),
    'train_labels': train.label.value_counts().sort_index().to_dict(),
    'missing_angles': frame.groupby('split').angle_missing.sum().to_dict(),
    'duplicate_ids_within_split': int(frame.duplicated(['split', 'id']).sum()),
    'overlapping_train_test_ids': len(set(train.id) & set(test.id)),
    'exact_duplicate_image_rows': len(duplicates),
    'cross_split_image_hash_overlap': len(set(train.image_hash) & set(test.image_hash)),
    'submission_header': header, 'submission_ids_match_test_order': submission_ids == test.id.tolist(),
    'image_shape': [2,75,75], 'all_image_values_finite': True,
    'schema_issues': issues,
    'environment': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib','ijson']}}
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
frame.groupby('split').agg({'inc_angle':['count','min','median','max'], 'band_1_mean':['min','median','max'], 'band_2_mean':['min','median','max']}).to_csv(OUT / 'distribution_summary.csv')
fig, axes = plt.subplots(1,3,figsize=(13,4))
train.label.value_counts().sort_index().plot.bar(ax=axes[0],color=['steelblue','darkorange'])
axes[0].set(xticklabels=['Ship','Iceberg'],title='Training class counts',ylabel='Samples',xlabel='Class')
for split, group in frame.groupby('split'):
    axes[1].hist(group.inc_angle.dropna(), bins=40, alpha=.5, density=True,label=split)
    axes[2].hist(group.band_1_mean,bins=40,alpha=.5,density=True,label=split)
axes[1].set(title='Observed incidence angles',xlabel='Angle (degrees)',ylabel='Density')
axes[2].set(title='Band 1 image means',xlabel='Mean radar value',ylabel='Density')
axes[1].legend(); axes[2].legend()
fig.tight_layout(); fig.savefig(OUT/'distributions.png',dpi=160); plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(8,7))
for label,(sample_id,bands) in examples.items():
    for band in range(2):
        lo,hi=np.percentile(bands[band],[2,98])
        axes[int(label),band].imshow(bands[band],cmap='gray',vmin=lo,vmax=hi)
        axes[int(label),band].set_title(f'{["Ship","Iceberg"][int(label)]}, band {band+1}: {sample_id}')
        axes[int(label),band].axis('off')
fig.tight_layout();fig.savefig(OUT/'representative_images.png',dpi=160);plt.close(fig)
print(json.dumps(summary,indent=2))
train.groupby('label').agg(samples=('id','size'),missing_angles=('angle_missing','sum'),median_angle=('inc_angle','median')).to_csv(OUT/'angle_by_class.csv')
fig,axes=plt.subplots(1,2,figsize=(10,4))
frame.groupby('split').angle_missing.mean().mul(100).plot.bar(ax=axes[0],rot=0)
axes[0].set(title='Missing incidence angles',ylabel='Missing (%)',xlabel='Dataset')
for split,group in frame.groupby('split'):
    axes[1].hist(group.band_2_mean,bins=40,alpha=.5,density=True,label=split)
axes[1].set(title='Band 2 image means',xlabel='Mean radar value',ylabel='Density');axes[1].legend()
fig.tight_layout();fig.savefig(OUT/'missingness_and_band2.png',dpi=160);plt.close(fig)

# Similar thumbnails flag candidates for review, not proven duplicate scenes.
similarity = np.asarray(thumbnails) @ np.asarray(thumbnails).T
np.fill_diagonal(similarity, -np.inf)
nearest = similarity.argmax(axis=1)
candidates = []
for index, neighbor in enumerate(nearest):
    if index < neighbor and similarity[index, neighbor] >= 0.95:
        candidates.append({'id_a': train.iloc[index].id, 'id_b': train.iloc[neighbor].id, 'thumbnail_cosine': float(similarity[index, neighbor]), 'same_label': bool(train.iloc[index].label == train.iloc[neighbor].label)})
pd.DataFrame(candidates, columns=['id_a','id_b','thumbnail_cosine','same_label']).to_csv(OUT/'similar_image_candidates.csv',index=False)
print('Similar-image candidate pairs:', len(candidates))
