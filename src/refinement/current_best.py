"""Reproduce the complete postprocessing studies and the current best candidate."""
import hashlib
import json
import numpy as np
import pandas as pd
from src.experiments.data import ROOT,load_train,read_folds,fold_hash
from src.refinement.evaluate import read_prediction,save_result
from src.refinement.odds_context import OddsContext


def main():
    data=load_train();split=ROOT/'data/splits/folds.csv';folds=read_folds(data,split)
    first=read_prediction(ROOT/'outputs/refinement/primary/R1_global/oof_predictions.csv',data,folds)
    second=read_prediction(ROOT/'outputs/refinement/primary/R2_local/oof_predictions.csv',data,folds)
    base=(first+second)/2
    assert np.array_equal(base,read_prediction(ROOT/'results/refinement/primary/R4_cnn_mean/oof_predictions.csv',data,folds))
    source={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/refinement/current_best.py','src/refinement/odds_context.py','src/refinement/context.py']}
    output=ROOT/'results/refinement/studies';output.mkdir(parents=True,exist_ok=True);records=[]
    for width in [.0001,.001,.005,.01,.05,.1]:
        probability=base.copy();matching=0
        for fold in range(5):
            train=(folds.fold.to_numpy()!=fold)&np.isfinite(data['angles']);valid=(folds.fold.to_numpy()==fold)&np.isfinite(data['angles'])
            distance=data['angles'][valid,None]-data['angles'][train][None,:]
            weight=np.exp(-.5*(distance/width)**2);count=weight.sum(1)
            probability[valid]=(base[valid]+weight@data['labels'][train])/(1+count);matching+=int((count>.5).sum())
        name=f'R4_gaussian_{width}'
        records.append(save_result(name,probability,data,folds,output,{'gaussian_bandwidth':width,'prior_strength':1,'weighted_matches_over_half':matching,'source':source}))
    for pool in [False,True]:
        for empirical in [False,True]:
            for weight in [.5,1.]:
                probability=base.copy();matching=0;strengths=[]
                for fold in range(5):
                    train=folds.fold.to_numpy()!=fold;valid=~train
                    model=OddsContext(weight,empirical,pool).fit(data['ids'][train],data['angles'][train],data['labels'][train])
                    probability[valid],counts=model.transform(data['ids'][valid],data['angles'][valid],base[valid]);matching+=int((counts>0).sum());strengths.append(model.context.strength)
                name=f'R4_odds_weight{weight}_empirical{empirical}_pool{pool}'
                records.append(save_result(name,probability,data,folds,output,{'weight':weight,'empirical':empirical,'query_group_pooling':pool,'matching_samples':matching,'fold_prior_strengths':strengths,'source':source}))
    columns=['name','oof_log_loss','accuracy','observed_angle_log_loss','missing_angle_log_loss','fold_std','local_loss_below_0_083']
    table=pd.DataFrame([{key:r[key] for key in columns} for r in records]);table.to_csv(output/'comparison.csv',index=False)
    best=min(records,key=lambda r:r['oof_log_loss'])
    target={'candidate':'R4_cnn_mean_empirical_odds_pool','study_name':best['name'],'development_oof_log_loss':best['oof_log_loss'],
            'fold_losses':best['fold_losses'],'samples':1604,'folds_sha256':fold_hash(split),'base_model_weights':{'R1_global':.5,'R2_local':.5},
            'context':{'log_odds_weight':1.,'empirical_train_only':True,'individual_query_weight':.5,'group_mean_query_weight':.5},
            'strict_official_threshold':.083,'official_kaggle_score':None,'official_threshold_verified':False,
            'local_threshold_met':bool(best['oof_log_loss']<.083),'source':source,
            'limitation':'Exploratory selection on development folds; query grouping is transductive and uses no query labels.'}
    assert best['name']=='R4_odds_weight1.0_empiricalTrue_poolTrue'
    (ROOT/'results/refinement/current_best.json').write_text(json.dumps(target,indent=2)+'\n')
    print(json.dumps(target,indent=2))


if __name__=='__main__':main()
