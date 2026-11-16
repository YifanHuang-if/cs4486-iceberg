"""Steps 9-10: audit handed-off evidence, analyze OOF errors, and infer test probabilities.

Run from the repository root: python -m src.finalize [--predict]
No training or Kaggle submission is performed. Original experiment files are preserved.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .experiments.data import ROOT, Preprocessor, file_hash, fold_hash, load_train, read_folds
from .experiments.model import ExperimentCNN
from .experiments.train import fingerprints

PRIMARY = ['V0_baseline', 'V1_flips', 'V2_dropout', 'V3_angle', 'V4_missing', 'V5_no_flips']
GROUPED = ['V1_flips', 'V3_angle', 'V4_missing']
RESULTS = ROOT / 'results/final'
PRIVATE = ROOT / 'outputs/final'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def individual_loss(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1-y) * np.log1p(-p))


def metric_row(frame):
    y = frame.label.to_numpy(dtype=int)
    p = frame.probability.to_numpy(dtype=float)
    pred = p >= .5
    return {'n': len(frame), 'ships': int((y == 0).sum()), 'icebergs': int(y.sum()),
            'log_loss': float(individual_loss(y, p).mean()),
            'accuracy': float((pred == y).mean()), 'errors': int((pred != y).sum()),
            'high_confidence_errors': int((((p >= .9) & (y == 0)) | ((p <= .1) & (y == 1))).sum()),
            'false_icebergs': int((pred & (y == 0)).sum()),
            'missed_icebergs': int((~pred & (y == 1)).sum())}


def aligned_oof(path, mapping):
    frame = pd.read_csv(path, dtype={'id': str}, float_precision='round_trip')
    if frame.id.duplicated().any() or set(frame.id) != set(mapping.id):
        raise ValueError(f'Incomplete or duplicate OOF IDs: {path}')
    frame = frame.set_index('id').loc[mapping.id].reset_index()
    for col in ['fold', 'label', 'angle_missing']:
        if not np.array_equal(frame[col], mapping[col]):
            raise ValueError(f'OOF {col} mismatch: {path}')
    p = frame.probability.to_numpy()
    if not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise ValueError('Invalid OOF probabilities')
    return frame


def load_model(path, config):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['config'] != config:
        raise ValueError('Checkpoint configuration differs')
    model = ExperimentCNN(config['dropout'], config['angle_features'])
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    return model, Preprocessor(**saved['preprocessor']), saved


def infer(model, preprocessor, images, angles, batch_size=128):
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x, m = preprocessor.transform(images[start:start+batch_size], angles[start:start+batch_size])
            pieces.append(model(torch.from_numpy(x), torch.from_numpy(m)).sigmoid().numpy().astype(np.float64))
    return np.concatenate(pieces)


def audit(data):
    records, all_oof = [], {}
    expected_sources = fingerprints()
    for protocol, names, split in [('primary', PRIMARY, 'folds.csv'),
                                    ('angle_grouped', GROUPED, 'folds_angle_grouped_steps78.csv')]:
        split_path = ROOT / 'data/splits' / split
        mapping = read_folds(data, split_path)
        if protocol == 'angle_grouped':
            known = np.isfinite(data['angles'])
            groups = pd.DataFrame({'angle': data['angles'][known], 'fold': mapping.fold.to_numpy()[known]})
            assert groups.groupby('angle').fold.nunique().max() == 1
        for name in names:
            directory = ROOT / 'results/steps78' / protocol / name
            frame = aligned_oof(directory/'oof_predictions.csv', mapping)
            summary = json.loads((directory/'summary.json').read_text())
            signature = json.loads((directory/'manifest.json').read_text())['signature']
            config = json.loads((ROOT/f'configs/steps78/{name}.json').read_text())
            assert signature['training_source'] == expected_sources
            assert signature['config'] == config == summary['config']
            assert signature['data_sha256'] == data['sha256']
            assert signature['folds_sha256_lf'] == fold_hash(split_path)
            assert signature['protocol'] == protocol
            loss = float(individual_loss(frame.label, frame.probability).mean())
            assert abs(loss-summary['oof_log_loss']) < 1e-9
            fold_scores = pd.read_csv(directory/'fold_scores.csv').sort_values('fold')
            history = pd.read_csv(directory/'history.csv')
            checkpoints = json.loads((directory/'checkpoints.json').read_text())
            assert [c['fold'] for c in checkpoints] == list(range(5))
            replayed = np.zeros(len(frame), dtype=float)
            for item in checkpoints:
                k = item['fold']; path = ROOT/item['path']
                assert path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256']
                model, prep, saved = load_model(path, config)
                mask = mapping.fold.to_numpy() == k
                assert saved['fold'] == k and saved['signature'] == signature
                assert saved['training_ids'] == data['ids'][~mask].tolist()
                assert saved['validation_ids'] == data['ids'][mask].tolist()
                fitted = Preprocessor.fit(data['images'][~mask], data['angles'][~mask])
                for key, value in fitted.to_dict().items():
                    assert np.allclose(value, saved['preprocessor'][key], rtol=1e-12, atol=1e-12), key
                h = history[history.fold == k]
                assert saved['best_epoch'] == int(h.loc[h.validation_log_loss.idxmin(), 'epoch'])
                fold_loss = float(individual_loss(frame.loc[mask, 'label'], frame.loc[mask, 'probability']).mean())
                assert abs(fold_loss - fold_scores.iloc[k].oof_log_loss) < 1e-9
                replayed[mask] = infer(model, prep, data['images'][mask], data['angles'][mask])
            max_difference = float(np.max(np.abs(replayed-frame.probability.to_numpy())))
            replay_loss = float(individual_loss(frame.label, replayed).mean())
            # Cross-version CPU inference is checked numerically, not claimed bitwise identical.
            assert max_difference < 2e-6, (name, max_difference)
            assert abs(replay_loss-loss) < 2e-6, (name, replay_loss, loss)
            record = {'protocol': protocol, 'experiment': name, 'n': len(frame),
                      'checkpoints_verified': 5, 'oof_log_loss': loss, 'replayed_log_loss': replay_loss,
                      'max_probability_difference': max_difference}
            records.append(record); all_oof[(protocol, name)] = frame
            print('Verified', protocol, name, f'max difference={max_difference:.3g}', flush=True)
    write_json(RESULTS/'handoff_verification.json', {'status': 'passed',
        'source_commit': 'f8c9ff017fe8a818784b5e8e02791543f3a31c2e',
        'bundle_code_commit': 'c8ccc8c528a82c3911a93c186f4fcd87dc1149b7',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'pandas': pd.__version__, 'platform': platform.platform()},
        'probability_tolerance': 2e-6, 'loss_tolerance': 2e-6, 'training_data_sha256': data['sha256'],
        'runs': records})
    return all_oof


def analyze(data, frames):
    rows = []
    angles = data['angles']
    for (protocol, name), frame in frames.items():
        frame = frame.copy(); frame['angle'] = angles
        groups = [('all', 'all', np.ones(len(frame), dtype=bool)),
                  ('angle_available', 'observed', np.isfinite(angles)),
                  ('angle_available', 'missing', ~np.isfinite(angles))]
        groups += [('class', label, frame.label.to_numpy() == y) for y, label in [(0,'ship'), (1,'iceberg')]]
        for lo, hi in [(20,30),(30,35),(35,40),(40,50)]:
            groups.append(('angle_degrees', f'[{lo},{hi})', (angles >= lo) & (angles < hi)))
        for kind, group, mask in groups:
            if mask.any(): rows.append({'protocol': protocol, 'experiment': name, 'group_type': kind,
                                       'group': group, **metric_row(frame.loc[mask])})
    grouped = pd.DataFrame(rows); grouped.to_csv(RESULTS/'subgroup_metrics.csv', index=False)
    observed = grouped[(grouped.protocol == 'primary') & (grouped.group_type == 'angle_available') & (grouped.group == 'observed')]
    chosen = observed.sort_values(['log_loss', 'experiment']).iloc[0].experiment
    # Criterion: observed-angle OOF loss, since the test set has complete angles.
    selection = {'selected_variant': chosen, 'protocol': 'primary',
        'criterion': 'Lowest observed-angle pooled OOF Log Loss among V0-V5; grouped checks are supporting evidence.',
        'inference': 'Equal mean of five primary-fold checkpoint probabilities; no TTA or probability calibration.',
        'limitations': ['Development folds also selected epochs and compared variants; no independent unbiased final score.',
            'Missing angles identify ships in training; overall CV gain overstates gain on observed-angle samples.',
            'Historical baseline uses a different runtime and is not a controlled improvement comparator.',
            'Grouped validation covers only V1, V3 and V4; exact angles are proxy groups.'],
        'checkpoints': json.loads((ROOT/f'results/steps78/primary/{chosen}/checkpoints.json').read_text()),
        'configuration': json.loads((ROOT/f'configs/steps78/{chosen}.json').read_text()),
        'kaggle_status': 'not_submitted', 'top5_status': 'not_established'}
    write_json(RESULTS/'model_selection.json', selection)
    final = frames[('primary', chosen)].copy()
    final['angle'] = angles
    final['loss'] = individual_loss(final.label, final.probability)
    final['prediction'] = (final.probability >= .5).astype(int)
    final['correct'] = final.prediction == final.label
    final['high_confidence_error'] = ((final.label == 0) & (final.probability >= .9)) | ((final.label == 1) & (final.probability <= .1))
    final.sort_values('loss', ascending=False).to_csv(PRIVATE/'oof_error_details.csv', index=False)
    final[~final.correct].sort_values('loss', ascending=False).head(20).to_csv(PRIVATE/'top20_errors.csv', index=False)
    baseline = frames[('primary', 'V0_baseline')]
    base_correct = (baseline.probability.to_numpy() >= .5) == baseline.label.to_numpy()
    corrected = int((~base_correct & final.correct.to_numpy()).sum())
    introduced = int((base_correct & ~final.correct.to_numpy()).sum())
    error_summary = {**metric_row(final), 'corrected_vs_V0': corrected, 'new_errors_vs_V0': introduced,
                     'threshold': .5, 'high_confidence_thresholds': [.1,.9],
                     'test_angle_complete': True, 'analysis_type': 'Descriptive post-selection OOF analysis'}
    write_json(RESULTS/'error_summary.json', error_summary)
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 160})
    selected = final[~final.correct].sort_values('loss', ascending=False).head(6)
    fig, axes = plt.subplots(3,4,figsize=(12,9))
    for j, (index, r) in enumerate(selected.iterrows()):
        for band in range(2):
            ax=axes[j//2, 2*(j%2)+band]; img=data['images'][index,band]
            lo,hi=np.percentile(img,[2,98]); ax.imshow(img,cmap='gray',vmin=lo,vmax=hi); ax.axis('off')
            angle='missing' if not np.isfinite(r.angle) else f'{r.angle:.1f} deg'
            ax.set_title(f'{r.id} / band {band+1}\n'+('iceberg' if r.label else 'ship')+f'; p={r.probability:.3f}\n{angle}; loss={r.loss:.2f}',fontsize=12)
    fig.suptitle('Six highest-loss OOF errors: '+chosen+'\nEach image uses its own 2nd-98th percentile display scale',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(RESULTS/'error_gallery.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    cm=np.array([[int(((final.label==i)&(final.prediction==j)).sum()) for j in [0,1]] for i in [0,1]])
    axes[0].imshow(cm,cmap='Blues');axes[0].set_xticks([0,1],['ship','iceberg']);axes[0].set_yticks([0,1],['ship','iceberg'])
    axes[0].set_xlabel('Predicted');axes[0].set_ylabel('True');axes[0].set_title('OOF confusion matrix (threshold 0.5)')
    for i in range(2):
        for j in range(2): axes[0].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    curve=[]
    for k in range(10):
        mask=(final.probability>=k/10)&((final.probability<(k+1)/10) if k<9 else (final.probability<=1))
        if mask.any(): curve.append({'bin':k,'n':int(mask.sum()),'mean_probability':float(final.loc[mask,'probability'].mean()),'iceberg_fraction':float(final.loc[mask,'label'].mean())})
    cal=pd.DataFrame(curve);cal.to_csv(RESULTS/'calibration_bins.csv',index=False)
    axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].plot(cal.mean_probability,cal.iceberg_fraction,'o-')
    axes[1].set(xlabel='Mean predicted probability',ylabel='Observed iceberg fraction',title='Descriptive OOF reliability plot',xlim=(0,1),ylim=(0,1))
    fig.tight_layout();fig.savefig(RESULTS/'confusion_calibration.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ['V0_baseline','V3_angle',chosen]:
        a=grouped[(grouped.protocol=='primary')&(grouped.experiment==name)&(grouped.group_type=='angle_degrees')]
        axes[0].plot(a.group,a.log_loss,'o-',label=name)
    for _,r in a.iterrows(): axes[0].annotate(f'n={int(r.n)}',(r.group,r.log_loss),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    axes[0].set(xlabel='Observed incidence angle (degrees)',ylabel='Log Loss',title='OOF performance by angle interval');axes[0].legend(fontsize=8)
    comparison=pd.read_csv(ROOT/'results/steps78/primary_comparison.csv')
    x=np.arange(len(comparison));axes[1].bar(x-.18,comparison.oof_log_loss,.36,label='All samples');axes[1].bar(x+.18,comparison.observed_angle_log_loss,.36,label='Observed angles')
    axes[1].set_xticks(x,[n.split('_')[0] for n in comparison.experiment]);axes[1].set(ylabel='Log Loss',title='Missingness changes the overall comparison');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(RESULTS/'angle_robustness.png');plt.close(fig)
    return selection


def stream_records(path, chunk_size=1024*1024):
    """Stream a top-level JSON array without loading the 1.5 GB test file into RAM."""
    decoder=json.JSONDecoder()
    with Path(path).open(encoding='utf-8-sig') as handle:
        buffer=handle.read(chunk_size).lstrip()
        if not buffer.startswith('['): raise ValueError('Expected JSON array')
        buffer=buffer[1:]; first=True
        while True:
            buffer=buffer.lstrip()
            while not buffer:
                extra=handle.read(chunk_size)
                if not extra: raise ValueError('Truncated JSON array')
                buffer+=extra;buffer=buffer.lstrip()
            if buffer.startswith(']'):
                if (buffer[1:]+handle.read()).strip(): raise ValueError('Trailing JSON content')
                return
            if not first:
                if not buffer.startswith(','): raise ValueError('Expected record separator')
                buffer=buffer[1:].lstrip()
            while True:
                try: item,end=decoder.raw_decode(buffer);break
                except json.JSONDecodeError:
                    extra=handle.read(chunk_size)
                    if not extra: raise ValueError('Truncated or invalid JSON record')
                    buffer=(buffer+extra).lstrip()
            if not isinstance(item,dict): raise ValueError('Expected record object')
            yield item;buffer=buffer[end:];first=False


def predict_test(selection):
    name=selection['selected_variant'];config=selection['configuration']
    models=[load_model(ROOT/item['path'],config)[:2] for item in selection['checkpoints']]
    ids=[]; all_folds=[];batch=[];observed=[]
    def flush(records):
        x=np.asarray([[r['band_1'],r['band_2']] for r in records],dtype=np.float32).reshape(-1,2,75,75)
        angles=np.asarray([np.nan if r['inc_angle'] in ('na',None) else float(r['inc_angle']) for r in records])
        if not np.isfinite(x).all() or not np.isfinite(angles).all(): raise ValueError('Unexpected test missingness/nonfinite values')
        ids.extend(r['id'] for r in records);observed.extend(angles.tolist())
        all_folds.append(np.column_stack([infer(m,p,x,angles) for m,p in models]))
    for r in stream_records(ROOT/'data/raw/test.json'):
        if 'is_iceberg' in r: raise ValueError('Unexpected test labels')
        if len(r['band_1'])!=5625 or len(r['band_2'])!=5625: raise ValueError('Bad band length')
        batch.append(r)
        if len(batch)==128:
            flush(batch);batch=[]
            if len(ids)%1024==0: print('Test inference:',len(ids),'records',flush=True)
    if batch: flush(batch)
    probabilities=np.concatenate(all_folds)
    sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv',dtype={'id':str})
    if list(sample.columns)!=['id','is_iceberg'] or sample.id.duplicated().any(): raise ValueError('Invalid sample submission')
    if len(set(ids))!=len(ids) or set(ids)!=set(sample.id): raise ValueError('Test/submission IDs differ')
    folds=pd.DataFrame(probabilities,columns=[f'fold_{i}' for i in range(5)],index=ids).loc[sample.id]
    p=folds.mean(axis=1).to_numpy()
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all(): raise ValueError('Invalid predictions')
    submission=pd.DataFrame({'id':sample.id,'is_iceberg':p});out=PRIVATE/'submission.csv'
    submission.to_csv(out,index=False,float_format='%.17g')
    folds.index.name='id';folds.to_csv(PRIVATE/'test_fold_predictions.csv',float_format='%.17g')
    reread=pd.read_csv(out,dtype={'id':str},float_precision='round_trip')
    assert reread.id.tolist()==sample.id.tolist() and np.array_equal(reread.is_iceberg.to_numpy(),p)
    write_json(RESULTS/'submission_manifest.json',{'variant':name,'rows':len(ids),'file':'outputs/final/submission.csv',
        'sha256':file_hash(out),'test_sha256':file_hash(ROOT/'data/raw/test.json'),
        'min_probability':float(p.min()),'max_probability':float(p.max()),'mean_probability':float(p.mean()),
        'mean_probability_std_across_folds':float(probabilities.std(axis=1).mean()),
        'test_angle_min':min(observed),'test_angle_max':max(observed),'test_missing_angles':0,
        'checkpoints':selection['checkpoints'],'aggregation':'Equal mean of probabilities, no TTA or calibration',
        'status':'File generated and validated; not submitted to Kaggle; leaderboard score unavailable'})
    print('Validated submission:',out,'rows=',len(ids),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predict',action='store_true');args=parser.parse_args()
    RESULTS.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    data=load_train();frames=audit(data);selection=analyze(data,frames)
    print('Selected:',selection['selected_variant'],flush=True)
    if args.predict: predict_test(selection)


if __name__=='__main__':main()
