"""Reuse a scored model's test predictions without repeating all CNN inference.

Invert its known linear context update; replay clipped rows from weights because
clipping is not invertible. Then apply a separately declared odds-based update.
"""
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from src.experiments.data import ROOT,load_train
from src.refinement.context import EmpiricalAngleContext
from src.refinement.odds_context import OddsContext
from src.refinement.inference import batches,load_models,predict_chunk


def invert_context(probability,angle,context):
    if not np.isfinite(angle):return float(probability)
    count,positive=context.groups.get(float(angle).hex(),(0,0.))
    if count==0:return float(probability)
    return float((probability*(context.strength+count)-positive)/context.strength)


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    data=load_train();context=EmpiricalAngleContext().fit(data['ids'],data['angles'],data['labels'])
    directory=ROOT/'outputs/refinement/submissions'
    original=directory/'R4_cnn_mean_empirical_context_submission.csv'
    report=json.loads(original.with_suffix('.json').read_text())
    assert hashlib.sha256(original.read_bytes()).hexdigest()==report['submission_sha256']
    assert report['context_strength']==context.strength
    frame=pd.read_csv(original,dtype={'id':str},float_precision='round_trip').set_index('id')
    groups=['R1_global','R2_local'];models,hashes=load_models(groups,data);assert hashes==report['checkpoint_sha256']
    ids=[];angles=[];base=[];replayed=0
    for records in batches(ROOT/'data/raw/test.json'):
        chunk_ids=np.array([r['id'] for r in records]);a=np.array([np.nan if r['inc_angle'] in ('na',None) else float(r['inc_angle']) for r in records])
        old=frame.loc[chunk_ids,'is_iceberg'].to_numpy()
        clipped=(old<=1e-7)|(old>=1-1e-7)
        values=np.array([invert_context(p,angle,context) for p,angle in zip(old,a)])
        if clipped.any():
            x=np.asarray([[r['band_1'],r['band_2']] for r,c in zip(records,clipped) if c],dtype=np.float32).reshape(-1,2,75,75)
            values[clipped]=predict_chunk(models,groups,x,a[clipped]);replayed+=int(clipped.sum())
        assert np.isfinite(values).all() and values.min()>=-1e-9 and values.max()<=1+1e-9
        values=np.clip(values,0,1)
        restored,_=context.transform(chunk_ids,a,values)
        assert np.max(np.abs(np.clip(restored,1e-7,1-1e-7)-old))<=1e-7
        ids.extend(chunk_ids);angles.extend(a);base.extend(values)
    ids=np.asarray(ids);angles=np.asarray(angles);base=np.asarray(base)
    base_frame=pd.DataFrame({'id':ids,'inc_angle':angles,'probability':base});base_frame.to_csv(directory/'R4_cnn_mean_base_predictions.csv',index=False)
    model=OddsContext(weight=1.,empirical=True,pool=True).fit(data['ids'],data['angles'],data['labels'])
    probability,counts=model.transform(ids,angles,base)
    sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv',dtype={'id':str})
    result=pd.DataFrame({'id':ids,'is_iceberg':np.clip(probability,1e-7,1-1e-7)}).set_index('id').loc[sample.id].reset_index()
    assert len(result)==len(sample) and not result.id.duplicated().any() and np.isfinite(result.is_iceberg).all()
    destination=directory/'R4_cnn_mean_empirical_odds_pool_submission.csv';result.to_csv(destination,index=False)
    metadata={'candidate':'R4_cnn_mean_empirical_odds_pool','base_models':groups,'rows':len(result),'context':'empirical odds, weight 1; 50/50 query group pooling',
              'data_sha256':data['sha256'],'test_sha256':report['test_sha256'],'checkpoint_sha256':hashes,'matching_test_samples':int((counts>0).sum()),
              'base_recovery':'Invert known linear context for unclipped rows; replay clipped rows from CNN weights','clipped_rows_replayed':replayed,
              'previous_submission_sha256':report['submission_sha256'],'submission_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
              'source':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/refinement/recontextualize.py','src/refinement/odds_context.py','src/refinement/context.py']},
              'official_kaggle_score':None,'submitted_to_kaggle':False,'official_threshold_verified':False}
    destination.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
