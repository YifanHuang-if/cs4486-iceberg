"""Reuse inner MLP OOF predictions; refit the fusion with an alternate angle prior."""
import argparse,json
import numpy as np
import torch
from src.v3.train import ROOT,OUT,digest_ids
from src.v3.evaluate import load_metadata
from src.v4 import nested_fusion as nested
from src.v4.angle_interpolation import interpolated_prior

def main():
    # Separate artifacts keep the original kernel experiment immutable.
    nested.prior=interpolated_prior
    original_get_fit=nested.get_fit
    def get_fit(fold,data,features):
        source=OUT/'linux_nested_proxy';out=OUT/'linux_interpolation_proxy';out.mkdir(parents=True,exist_ok=True)
        path=out/f'fold{fold}_fit.json';tr=data['folds']!=fold
        ids_hash=digest_ids(data['ids'][tr])
        if path.exists():
            record=json.loads(path.read_text());assert record['train_ids_sha256']==ids_hash;return record
        original_get_fit(fold,data,features)
        saved=np.load(source/f'fold{fold}_inner_oof.npz')
        assert np.array_equal(saved['ids'],data['ids'][tr])
        fit,candidates=nested.select_fit(data['angles'][tr],data['labels'][tr],saved['probability'],saved['inner_folds'])
        record=dict(fold=fold,train_ids_sha256=ids_hash,selected=fit,candidates=candidates,
            method='adjacent angle-group posterior interpolation with distance attenuation',
            inner_proxy_source='linux_nested_proxy',ridge=nested.RIDGE)
        path.write_text(json.dumps(record,indent=2));print('FIT INTERPOLATION',fold,fit,flush=True);return record
    nested.get_fit=get_fit
    nested.main()

if __name__=='__main__':main()
