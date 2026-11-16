"""Check a real candidate file and replay its inference path on held-out records."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from src.experiments.data import ROOT,load_train,read_folds
from src.refinement.inference import parse_candidate,load_models,predict_chunk
from src.refinement.context import EmpiricalAngleContext


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',default='R4_cnn_mean_empirical_context');args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    data=load_train();folds=read_folds(data,ROOT/'data/splits/folds.csv')
    groups,context=parse_candidate(args.candidate)
    if groups!=['R1_global','R2_local'] or context!='empirical_context':raise ValueError('This replay checks the selected CNN candidate')
    models,hashes=load_models(groups,data);replays=[]
    expected=pd.read_csv(ROOT/f'results/refinement/primary/{args.candidate}/oof_predictions.csv',dtype={'id':str},float_precision='round_trip').set_index('id')
    for fold in range(5):
        train=folds.fold.to_numpy()!=fold;valid=~train
        selected=[models[fold],models[5+fold]]
        probability=predict_chunk(selected,groups,data['images'][valid],data['angles'][valid],folds_per_model=1)
        model=EmpiricalAngleContext().fit(data['ids'][train],data['angles'][train],data['labels'][train])
        probability,counts=model.transform(data['ids'][valid],data['angles'][valid],probability)
        difference=float(np.max(np.abs(probability-expected.loc[data['ids'][valid],'probability'].to_numpy())))
        assert difference<=1e-7
        replays.append({'fold':fold,'rows':int(valid.sum()),'maximum_probability_difference':difference,'matching_angles':int((counts>0).sum())})
    path=ROOT/f'outputs/refinement/submissions/{args.candidate}_submission.csv'
    manifest=json.loads(path.with_suffix('.json').read_text())
    frame=pd.read_csv(path,dtype={'id':str},float_precision='round_trip');sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv',dtype={'id':str})
    assert list(frame.columns)==['id','is_iceberg'] and np.array_equal(frame.id,sample.id)
    assert len(frame)==manifest['rows'] and not frame.id.duplicated().any()
    assert np.isfinite(frame.is_iceberg).all() and frame.is_iceberg.between(0,1,inclusive='neither').all()
    assert manifest['data_sha256']==data['sha256'] and manifest['checkpoint_sha256']==hashes
    for source,digest in manifest['inference_source'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['submission_sha256']
    result={'status':'passed','candidate':args.candidate,'rows':len(frame),'sample_ids_exact_order':True,
            'probability_min':float(frame.is_iceberg.min()),'probability_max':float(frame.is_iceberg.max()),
            'full_inference_path_held_out_replays':replays,'submission_sha256':manifest['submission_sha256'],
            'official_kaggle_score':None,'official_threshold_verified':False}
    (ROOT/'results/refinement/submission_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
