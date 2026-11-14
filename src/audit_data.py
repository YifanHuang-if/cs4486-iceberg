"""Complete a wider similarity screen, audit frozen folds, and add an angle-group sensitivity split."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
import sklearn
from src.data import ROOT, load_train, aligned_folds, fold_hash


def transform(images, k, flipped):
    result = np.rot90(images, k, axes=(-2,-1))
    return np.flip(result,axis=-1) if flipped else result


def normalized(images):
    flat = images.reshape(len(images),2,-1)
    centered = flat - flat.mean(axis=-1,keepdims=True)
    return centered / np.maximum(np.linalg.norm(centered,axis=-1,keepdims=True),1e-12)


def main():
    out = ROOT/'results/audit';out.mkdir(parents=True,exist_ok=True)
    data = load_train()
    ids, y, angles, images = [data[k] for k in ['ids','labels','angles','images']]
    folds = aligned_folds(data,ROOT/'data/splits/folds.csv')
    config = json.loads((ROOT/'configs/validation.json').read_text())
    if data['train_sha256'] != config['train_file_sha256']:
        raise ValueError('Training file differs from the frozen validation source')
    if fold_hash(ROOT/'data/splits/folds.csv') != config['folds_sha256']:
        raise ValueError('Frozen fold hash mismatch after LF canonicalization')
    expected = np.full(len(ids),-1); rng=np.random.default_rng(42);offset=0
    strata = y*2 + (~np.isfinite(angles)).astype(int)
    for s in sorted(set(strata)):
        index=np.array(sorted(np.flatnonzero(strata==s),key=lambda i:ids[i]));rng.shuffle(index)
        expected[index]=(np.arange(len(index))+offset)%5;offset=(offset+len(index))%5
    if not np.array_equal(expected,folds):raise ValueError('Frozen split could not be reproduced')
    print('Frozen sample-stratified folds reproduced exactly.',flush=True)

    # All unordered training pairs are screened under eight right-angle rotations/reflections.
    small=images.reshape(len(images),2,15,5,15,5).mean(axis=(3,5))
    original=normalized(small);full_original=normalized(images)
    candidates={}
    with threadpool_limits(limits=2):
        for k in range(4):
            for flipped in [False,True]:
                other=normalized(transform(small,k,flipped))
                sims=np.minimum(original[:,0]@other[:,0].T,original[:,1]@other[:,1].T)
                ai,bi=np.where(np.triu(sims>=.90,k=1))
                transformed_full=normalized(transform(images,k,flipped))
                for a,b in zip(ai,bi):
                    correlations=(full_original[a]*transformed_full[b]).sum(axis=-1)
                    score=float(correlations.min())
                    key=(int(a),int(b))
                    if key not in candidates or score>candidates[key]['minimum_full_correlation']:
                        candidates[key]={'id_a':ids[a],'id_b':ids[b],
                            'rotation_quarters':k,'horizontal_reflection':flipped,
                            'minimum_thumbnail_correlation':float(sims[a,b]),
                            'band_1_full_correlation':float(correlations[0]),
                            'band_2_full_correlation':float(correlations[1]),
                            'minimum_full_correlation':score,'same_label':bool(y[a]==y[b]),
                            'same_primary_fold':bool(folds[a]==folds[b])}
                print(f'Screened transform {k}/{flipped}: {len(ai)} candidate orientations',flush=True)
    columns=['id_a','id_b','rotation_quarters','horizontal_reflection','minimum_thumbnail_correlation',
             'band_1_full_correlation','band_2_full_correlation','minimum_full_correlation','same_label','same_primary_fold']
    table=pd.DataFrame(list(candidates.values()),columns=columns).sort_values('minimum_full_correlation',ascending=False)
    table.to_csv(out/'similarity_screen.csv',index=False)
    if len(table):
        n=min(6,len(table));fig,axes=plt.subplots(n,4,figsize=(10,2.3*n),squeeze=False)
        lookup={s:i for i,s in enumerate(ids)}
        for row,pair in enumerate(table.head(n).itertuples()):
            a=images[lookup[pair.id_a]];b=transform(images[lookup[pair.id_b]],pair.rotation_quarters,pair.horizontal_reflection)
            for col,(im,title) in enumerate([(a[0],pair.id_a+' B1'),(b[0],pair.id_b+' B1'),(a[1],pair.id_a+' B2'),(b[1],pair.id_b+' B2')]):
                low,high=np.percentile(im,[2,98]);axes[row,col].imshow(im,cmap='gray',vmin=low,vmax=high)
                axes[row,col].set_title(title,fontsize=8);axes[row,col].axis('off')
        fig.suptitle('Highest-scoring similarity candidates (transformed B; display scaling only)',fontsize=10)
        fig.tight_layout();fig.savefig(out/'similarity_review.png',dpi=100);plt.close(fig)

    # Keep existing primary folds. Same exact angle forms only a proxy group in this sensitivity study.
    groups=np.array([f'angle:{float(a).hex()}' if np.isfinite(a) else f'unknown:{s}' for s,a in zip(ids,angles)])
    sorted_index=np.argsort(ids);grouped=np.full(len(ids),-1)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    for k,(_,val) in enumerate(splitter.split(ids[sorted_index],strata[sorted_index],groups[sorted_index])):
        grouped[sorted_index[val]]=k
    mapping=pd.DataFrame({'id':ids,'fold':grouped,'label':y,'angle_missing':(~np.isfinite(angles)).astype(int)})
    path=ROOT/'data/splits/folds_angle_grouped_v1.csv'
    content=mapping.to_csv(index=False,lineterminator='\n')
    if path.exists() and path.read_text()!=content:raise ValueError('Refusing to replace a different grouped split')
    path.write_text(content)
    aligned_folds(data,path)
    diagnostics=[]
    for k in range(5):
        val=grouped==k;training=~val
        assert not (set(groups[val])&set(groups[training]))
        diagnostics.append({'fold':k,'samples':int(val.sum()),'icebergs':int(y[val].sum()),
            'ships':int((y[val]==0).sum()),'missing_angles':int((~np.isfinite(angles[val])).sum())})
    pd.DataFrame(diagnostics).to_csv(out/'grouped_fold_summary.csv',index=False)
    repeated=pd.DataFrame({'angle':angles,'label':y,'fold':folds}).dropna().groupby('angle').agg(
        samples=('label','size'),classes=('label','nunique'),primary_folds=('fold','nunique'))
    repeated=repeated[repeated.samples>1]
    repeated.to_csv(out/'repeated_angle_groups.csv')
    summary={'primary_fold_hash_verified':True,'primary_folds_reproduced':True,
        'primary_folds_sha256_lf':fold_hash(ROOT/'data/splits/folds.csv'),
        'grouped_folds_sha256_lf':fold_hash(path),'train_sha256':data['train_sha256'],
        'all_unordered_training_pairs':len(ids)*(len(ids)-1)//2,'transforms_screened':8,
        'thumbnail_minimum_channel_threshold':.90,'candidate_pairs':len(table),
        'full_correlation_flag_threshold':.98,'pairs_above_full_threshold':int((table.minimum_full_correlation>=.98).sum()),
        'maximum_full_minimum_correlation':float(table.minimum_full_correlation.max()) if len(table) else None,
        'repeated_angle_groups':len(repeated),'repeated_angle_groups_crossing_primary_folds':int((repeated.primary_folds>1).sum()),
        'single_class_repeated_angle_groups':int((repeated.classes==1).sum()),
        'grouped_splitter':'StratifiedGroupKFold, joint label/missingness strata, exact observed angle groups, singleton unknowns',
        'seed':42,'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,
        'limitations':'Screening does not establish scene identity and does not exhaust translations or scale changes.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
