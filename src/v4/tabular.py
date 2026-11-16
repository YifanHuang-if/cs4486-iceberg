"""Fixed-budget complementary tree baselines; all fitted state is fold-local."""
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from src.v3.train import OUT, fit_meta, meta_transform, digest_ids
from src.v3.shape_model import get_features
from src.v3.evaluate import load_metadata
from src.v3.fusion import prior, fuse, loss

def make_model(kind, seed):
    if kind == 'hist':
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=.035,
            max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=5.,
            early_stopping=False, random_state=seed)
    return ExtraTreesClassifier(n_estimators=600, min_samples_leaf=3,
        max_features=.8, n_jobs=2, random_state=seed)

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['hist','extra'],required=True)
    p.add_argument('--tag',required=True);p.add_argument('--verify',action='store_true')
    args=p.parse_args();data=load_metadata();a,b=get_features()
    out=OUT/args.tag;out.mkdir(parents=True,exist_ok=True)
    for fold in range(5):
        tr=data['folds']!=fold;va=~tr;start=time.perf_counter()
        fitted=fit_meta(a[tr]);tx=meta_transform(a[tr],fitted)
        vx=meta_transform(a[va],fitted);test=meta_transform(b,fitted)
        if args.verify:
            checkpoint=joblib.load(out/f'fold{fold}.joblib')
            assert checkpoint['train_ids_sha256']==digest_ids(data['ids'][tr])
            assert checkpoint['metadata_fit']==fitted
            model=checkpoint['model'];saved=np.load(out/f'fold{fold}_predictions.npz')
            np.testing.assert_allclose(model.predict_proba(vx)[:,1],saved['probability'],atol=1e-12)
            np.testing.assert_allclose(model.predict_proba(test)[:,1],saved['test_probability'],atol=1e-12)
            print('REPLAY PASS',args.tag,fold,flush=True);continue
        if (out/f'fold{fold}_metrics.json').exists(): continue
        model=make_model(args.kind,601+fold)
        with threadpool_limits(limits=2):
            model.fit(tx,data['labels'][tr]);pv=model.predict_proba(vx)[:,1]
            pt=model.predict_proba(test)[:,1]
        ap,c=prior(data['angles'][tr],data['labels'][tr],data['angles'][va])
        fused=fuse(pv,ap,c)
        np.savez(out/f'fold{fold}_predictions.npz',valid_ids=data['ids'][va],probability=pv,
            plain_probability=pv,fused_probability=fused,test_ids=data['test_ids'],test_probability=pt)
        joblib.dump(dict(model=model,metadata_fit=fitted,
            train_ids_sha256=digest_ids(data['ids'][tr])),out/f'fold{fold}.joblib')
        report=dict(fold=fold,config=dict(tag=args.tag,kind=args.kind,seed=601+fold,
            model_parameters=model.get_params()),train_ids_sha256=digest_ids(data['ids'][tr]),
            val_log_loss=loss(data['labels'][va],pv),val_fused_log_loss=loss(data['labels'][va],fused),
            valid_rows=int(va.sum()),seconds=time.perf_counter()-start)
        (out/f'fold{fold}_metrics.json').write_text(json.dumps(report,indent=2))
        print('DONE',args.tag,fold,report['val_log_loss'],report['val_fused_log_loss'],flush=True)

if __name__=='__main__':main()
