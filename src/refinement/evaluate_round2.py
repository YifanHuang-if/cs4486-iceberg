"""Keep complete second-round OOF comparisons without overwriting the first study."""
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from src.experiments.data import ROOT,load_train,read_folds
from src.refinement.evaluate import read_prediction,save_result,contextualize
from src.refinement.odds_context import OddsContext


def main():
    data=load_train();folds=read_folds(data,ROOT/'data/splits/folds.csv');output=ROOT/'results/refinement/round2';output.mkdir(parents=True,exist_ok=True)
    predictions={};records=[]
    for name in ['R1_global','R2_local','R8_deep','R9_finetuned']:
        source=ROOT/f'outputs/refinement/primary/{name}'
        assert (source/'summary.json').exists(),f'{name} must finish all five folds'
        predictions[name]=read_prediction(source/'oof_predictions.csv',data,folds)
        if name in ['R8_deep','R9_finetuned']:
            records.append(save_result(name,predictions[name],data,folds,output))
            for filename in ['manifest.json','fold_scores.csv','history.csv']:shutil.copyfile(source/filename,output/name/filename)
            checkpoints=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(source.glob('fold_[0-4].pt'))]
            (output/name/'checkpoints.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
    ensembles={'R4_cnn_mean':['R1_global','R2_local'],'R10_three_cnn_mean':['R1_global','R2_local','R8_deep'],
               'R11_cnn_transfer_mean':['R1_global','R2_local','R9_finetuned'],'R12_four_model_mean':['R1_global','R2_local','R8_deep','R9_finetuned']}
    for name,members in ensembles.items():
        probability=np.mean([predictions[m] for m in members],axis=0);predictions[name]=probability
        records.append(save_result(name,probability,data,folds,output,{'weights':{member:1/len(members) for member in members}}))
    for name in ['R8_deep','R9_finetuned']+list(ensembles):
        p,details=contextualize(data,folds,predictions[name],empirical=True)
        records.append(save_result(name+'_empirical_context',p,data,folds,output,details))
        probability=predictions[name].copy();counts=np.zeros(len(probability),dtype=int)
        for fold in range(5):
            train=folds.fold.to_numpy()!=fold;valid=~train
            model=OddsContext(weight=1.,empirical=True,pool=True).fit(data['ids'][train],data['angles'][train],data['labels'][train])
            probability[valid],counts[valid]=model.transform(data['ids'][valid],data['angles'][valid],predictions[name][valid])
        records.append(save_result(name+'_empirical_odds_pool',probability,data,folds,output,{'weight':1.,'query_pool_individual_weight':.5,'query_group_pooling_is_transductive':True,'matching_samples':int((counts>0).sum())}))
    columns=['name','oof_log_loss','accuracy','observed_angle_log_loss','missing_angle_log_loss','fold_std','local_loss_below_0_083']
    comparison=pd.DataFrame([{key:r[key] for key in columns} for r in records]);comparison.to_csv(output/'comparison.csv',index=False)
    best=comparison.sort_values('oof_log_loss').iloc[0]
    (output/'selection.json').write_text(json.dumps({'candidate':best['name'],'development_oof_log_loss':float(best.oof_log_loss),
        'official_kaggle_score':None,'official_threshold_verified':False,'warning':'Exploratory model and postprocessing selection reuses development CV. No official target claim.'},indent=2)+'\n')
    print(comparison.sort_values('oof_log_loss').to_string(index=False))


if __name__=='__main__':main()
