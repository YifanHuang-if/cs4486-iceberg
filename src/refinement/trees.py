"""Radar-statistic boosting with iteration selection inside each outer training fold."""
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from src.experiments.data import ROOT,load_train,read_folds,fold_hash
from src.experiments.metrics import metrics,log_loss
from src.refinement.features import training_features
from src.refinement.train import aggregate,BASE_COMMIT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',choices=['primary','angle_grouped'],default='primary');args=parser.parse_args()
    data=load_train();features,names=training_features(data)
    split=ROOT/('data/splits/folds.csv' if args.protocol=='primary' else 'data/splits/folds_angle_grouped_steps78.csv')
    frozen=json.loads((ROOT/'configs/validation.json').read_text())
    if args.protocol=='primary':assert fold_hash(split)==frozen['folds_sha256']
    folds=read_folds(data,split);config={'name':'R3_radar_trees','seed':42,'max_iterations':1600,'depth':5,'learning_rate':.03,'l2_leaf_reg':5,'inner_fraction':.2}
    out=ROOT/f'outputs/refinement/{args.protocol}/R3_radar_trees';out.mkdir(parents=True,exist_ok=True)
    signature={'config':config,'protocol':args.protocol,'data_sha256':data['sha256'],'folds_sha256':fold_hash(split),'base_commit':BASE_COMMIT,
               'feature_names':names,'catboost_version':catboost.__version__,
               'source':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/refinement/features.py','src/refinement/trees.py']}}
    manifest=out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==signature
    else:manifest.write_text(json.dumps(signature,indent=2)+'\n')
    for fold in range(5):
        if (out/f'metrics_fold_{fold}.json').exists():continue
        tick=time.perf_counter();train=np.flatnonzero(folds.fold.to_numpy()!=fold);valid=np.flatnonzero(folds.fold.to_numpy()==fold)
        strata=data['labels'][train]*2+folds.angle_missing.to_numpy()[train]
        inner_train,inner_valid=train_test_split(train,test_size=.2,random_state=1042+fold,stratify=strata)
        parameters=dict(depth=5,learning_rate=.03,l2_leaf_reg=5,loss_function='Logloss',eval_metric='Logloss',random_seed=42+fold,
                        thread_count=4,allow_writing_files=False,verbose=False)
        selector=CatBoostClassifier(iterations=1600,**parameters)
        selector.fit(features[inner_train],data['labels'][inner_train],eval_set=(features[inner_valid],data['labels'][inner_valid]),early_stopping_rounds=100)
        iterations=selector.get_best_iteration()+1
        model=CatBoostClassifier(iterations=iterations,**parameters);model.fit(features[train],data['labels'][train])
        model.save_model(str(out/f'fold_{fold}.cbm'))
        reloaded=CatBoostClassifier();reloaded.load_model(str(out/f'fold_{fold}.cbm'))
        probability=model.predict_proba(features[valid])[:,1]
        assert np.allclose(probability,reloaded.predict_proba(features[valid])[:,1],rtol=0,atol=1e-12)
        frame=folds.iloc[valid].copy();frame['probability']=probability;frame.to_csv(out/f'oof_fold_{fold}.csv',index=False)
        record=metrics(data['labels'][valid],probability,folds.iloc[valid].angle_missing)
        record.update({'fold':fold,'iterations':iterations,'training_seconds':time.perf_counter()-tick})
        (out/f'metrics_fold_{fold}.json').write_text(json.dumps(record,indent=2)+'\n')
        provenance={'training_ids':data['ids'][train].tolist(),'validation_ids':data['ids'][valid].tolist(),
                    'inner_training_ids':data['ids'][inner_train].tolist(),'inner_validation_ids':data['ids'][inner_valid].tolist(),
                    'iterations':iterations,'signature':signature}
        (out/f'fold_{fold}_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        history=selector.get_evals_result()
        pd.DataFrame({'fold':fold,'iteration':np.arange(len(history['learn']['Logloss']))+1,'inner_training_log_loss':history['learn']['Logloss'],
                      'inner_validation_log_loss':history['validation']['Logloss']}).to_csv(out/f'history_fold_{fold}.csv',index=False)
        print(f"DONE R3_radar_trees fold={fold} loss={record['oof_log_loss']:.5f} iterations={iterations}",flush=True)
    aggregate(data,folds,out,config,args.protocol)


if __name__=='__main__':main()
