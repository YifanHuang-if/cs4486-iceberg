"""Retain every measured result and compare fixed ensembles on aligned OOF IDs."""
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from src.experiments.data import ROOT,load_train,read_folds,fold_hash
from src.experiments.metrics import metrics,log_loss
from src.refinement.context import AngleContext,EmpiricalAngleContext

NAMES=['R1_global','R2_local','R3_radar_trees','R6_frozen_resnet']


def read_prediction(path,data,folds):
    frame=pd.read_csv(path,dtype={'id':str},float_precision='round_trip')
    assert not frame.id.duplicated().any() and set(frame.id)==set(data['ids'])
    frame=frame.set_index('id').loc[data['ids']].reset_index()
    assert np.array_equal(frame.label,data['labels']) and np.array_equal(frame.fold,folds.fold)
    return frame.probability.to_numpy()


def contextualize(data,folds,probability,empirical=False):
    result=probability.copy();counts=np.zeros(len(result),dtype=int);strengths=[]
    for fold in range(5):
        train=folds.fold.to_numpy()!=fold;valid=~train
        model=(EmpiricalAngleContext() if empirical else AngleContext()).fit(data['ids'][train],data['angles'][train],data['labels'][train])
        result[valid],counts[valid]=model.transform(data['ids'][valid],data['angles'][valid],probability[valid]);strengths.append(model.strength)
    return result,{'matching_samples':int((counts>0).sum()),'fold_prior_strengths':strengths,'empirical':empirical}


def save_result(name,probability,data,folds,out,details=None):
    destination=out/name;destination.mkdir(parents=True,exist_ok=True)
    result=metrics(data['labels'],probability,folds.angle_missing)
    losses=[log_loss(data['labels'][folds.fold==k],probability[folds.fold==k]) for k in range(5)]
    result.update({'name':name,'fold_losses':losses,'fold_std':float(np.std(losses)),
                   'data_sha256':data['sha256'],'evaluation_source':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/refinement/evaluate.py','src/refinement/context.py']},
                   'official_kaggle_score':None,'official_threshold_verified':False,
                   'local_loss_below_0_083':bool(result['oof_log_loss']<.083),**(details or {})})
    frame=folds.copy();frame['probability']=probability;frame.to_csv(destination/'oof_predictions.csv',index=False)
    (destination/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',choices=['primary','angle_grouped'],default='primary');parser.add_argument('--names',nargs='+');args=parser.parse_args()
    data=load_train();split=ROOT/('data/splits/folds.csv' if args.protocol=='primary' else 'data/splits/folds_angle_grouped_steps78.csv');folds=read_folds(data,split)
    output=ROOT/f'results/refinement/{args.protocol}';output.mkdir(parents=True,exist_ok=True)
    predictions={};records=[]
    for name in args.names or NAMES:
        source=ROOT/f'outputs/refinement/{args.protocol}/{name}'
        p=read_prediction(source/'oof_predictions.csv',data,folds);predictions[name]=p
        record=save_result(name,p,data,folds,output)
        original=json.loads((source/'summary.json').read_text());assert abs(record['oof_log_loss']-original['oof_log_loss'])<1e-9
        for filename in ['manifest.json','history.csv','fold_scores.csv']:
            if (source/filename).exists():shutil.copyfile(source/filename,output/name/filename)
        checkpoints=[{'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(source.glob('fold_*')) if path.suffix in ['.pt','.cbm','.npz'] and '_resume' not in path.name]
        (output/name/'checkpoints.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
        records.append(record)
    if args.protocol=='primary':
        previous=read_prediction(ROOT/'results/steps78/primary/V4_missing/oof_predictions.csv',data,folds)
        predictions['V4_previous']=previous;records.append(save_result('V4_previous',previous,data,folds,output))
        if {'R1_global','R2_local'}<=set(predictions):
            predictions['R4_cnn_mean']=(predictions['R1_global']+predictions['R2_local'])/2
            records.append(save_result('R4_cnn_mean',predictions['R4_cnn_mean'],data,folds,output,{'weights':{'R1_global':.5,'R2_local':.5}}))
        for ensemble,members in [('R5_cnn_tree_mean',['R1_global','R2_local','R3_radar_trees']),('R7_cnn_transfer_mean',['R1_global','R2_local','R6_frozen_resnet'])]:
            if set(members)<=set(predictions):
                p=sum(predictions[n] for n in members)/len(members);predictions[ensemble]=p
                records.append(save_result(ensemble,p,data,folds,output,{'weights':{n:1/len(members) for n in members}}))
    bases=list(predictions)
    for name in bases:
        if name=='R3_radar_trees' or name=='R5_cnn_tree_mean':continue
        for empirical,suffix in [(False,'fixed_context'),(True,'empirical_context')]:
            p,details=contextualize(data,folds,predictions[name],empirical)
            if args.protocol=='angle_grouped':
                assert details['matching_samples']==0 and np.array_equal(p,predictions[name])
            records.append(save_result(name+'_'+suffix,p,data,folds,output,{'base_model':name,**details}))
    table=pd.DataFrame([{k:r[k] for k in ['name','oof_log_loss','accuracy','observed_angle_log_loss','missing_angle_log_loss','fold_std','local_loss_below_0_083']} for r in records])
    table.to_csv(output/'comparison.csv',index=False)
    (output/'protocol.json').write_text(json.dumps({'protocol':args.protocol,'folds_sha256':fold_hash(split),'n_samples':len(data['ids']),'official_score_available':False,
                                                  'selection_warning':'This is development CV. CNN epochs and model comparisons reuse validation labels; adaptive additions are disclosed in the plan.'},indent=2)+'\n')
    print(table.to_string(index=False))


if __name__=='__main__':main()
