"""Check feature isolation, aligned augmentation, TTA and neural outputs."""
import unittest
import numpy as np
import torch
from src.refinement.cnn import Preparation,RadarCNN,augment,predict,view
from src.refinement.features import radar_features
from src.refinement.context import AngleContext,EmpiricalAngleContext
from src.refinement.inference import metadata_features,parse_candidate
torch.set_num_threads(2)


class RefinementTests(unittest.TestCase):
    def test_saved_linear_classifier_matches_pipeline(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from src.refinement.transfer import linear_predict
        import io
        rng=np.random.default_rng(12)
        raw=rng.normal(size=(48,8)).astype(np.float32)
        y=(raw[:,0]+raw[:,1]>.1).astype(int)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1)).fit(raw[:36].astype(np.float64),y[:36])
        scaler,classifier=model.steps[0][1],model.steps[1][1]
        buffer=io.BytesIO()
        np.savez(buffer,mean=scaler.mean_,scale=scaler.scale_,coefficient=classifier.coef_,intercept=classifier.intercept_)
        buffer.seek(0)
        with np.load(buffer,allow_pickle=False) as state:
            np.testing.assert_allclose(linear_predict(state,raw[36:]),model.predict_proba(raw[36:].astype(np.float64))[:,1],rtol=0,atol=1e-12)

    def test_streaming_metadata_matches_training_extractor(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(3,2,75,75)).astype(np.float32);a=np.array([30.,np.nan,40.])
        full,names=radar_features(x,a)
        from src.refinement.cnn import METADATA
        np.testing.assert_array_equal(metadata_features(x,a),full[:,[names.index(n) for n in METADATA]])

    def test_empirical_context_and_candidate_validation(self):
        model=EmpiricalAngleContext().fit(['a','b','c','d'],[30.,30.,40.,40.],[0,0,1,1])
        self.assertGreater(model.strength,0)
        p,count=model.transform(['e','f'],[30.,50.],[.5,.6]);self.assertLess(p[0],.5);self.assertEqual(p[1],.6)
        self.assertEqual(parse_candidate('R4_cnn_mean_empirical_context'),(['R1_global','R2_local'],'empirical_context'))
        original=np.random.default_rng(4).random(23)
        unchanged,counts=model.transform([f'unseen{i}' for i in range(23)],np.arange(100.,123.),original)
        np.testing.assert_array_equal(unchanged,original)
        np.testing.assert_array_equal(counts,np.zeros(23))
        with self.assertRaises(ValueError):parse_candidate('../wrong')

    def test_context_excludes_fit_ids_and_unseen_groups(self):
        model=AngleContext().fit(['a','b','c'],[30.,30.,40.],[1,1,0])
        p,count=model.transform(['d','e','f'],[30.,50.,np.nan],[.2,.7,.4])
        np.testing.assert_allclose(p,[2.2/3,.7,.4]);np.testing.assert_array_equal(count,[2,0,0])
        with self.assertRaises(ValueError):model.transform(['a'],[30.],[.2])
    def test_features_independent_of_other_samples(self):
        rng=np.random.default_rng(2);images=rng.normal(size=(3,2,75,75)).astype(np.float32);angles=np.array([30.,np.nan,40.])
        batch,names=radar_features(images,angles)
        single,other_names=radar_features(images[:1],angles[:1])
        self.assertEqual(names,other_names);np.testing.assert_array_equal(batch[0],single[0]);self.assertTrue(np.isfinite(batch).all())

    def test_fold_local_imputation_and_normalization(self):
        x=np.arange(32,dtype=np.float32).reshape(4,2,2,2);meta=np.zeros((4,12),dtype=np.float32)
        meta[:,0]=[20,40,-1,100];meta[:,1]=[0,0,1,0]
        preparation=Preparation.fit(x[:3],meta[:3],'global');self.assertEqual(preparation.angle_median,30.)
        _,transformed=preparation.transform(x[2:3],meta[2:3]);self.assertEqual(transformed[0,0],0.)
        self.assertEqual(preparation.image_mean,x[:3].mean((0,2,3)).tolist())

    def test_augmentation_pairing_and_rng_isolation(self):
        x=torch.randn(4,1,75,75);paired=torch.cat([x,x+1],dim=1);state=torch.get_rng_state().clone()
        actual=augment(paired,torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        torch.testing.assert_close(actual[:,1]-actual[:,0],torch.ones_like(x[:,0]),rtol=1e-5,atol=1e-5)

    def test_tta_group_invariance(self):
        model=RadarCNN().eval();x=torch.randn(3,2,75,75);meta=torch.randn(3,12)
        a=predict(model,x,meta,torch.device('cpu'),tta=True)
        b=predict(model,view(x,5),meta,torch.device('cpu'),tta=True)
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-7)

    def test_trainable_and_batch_independent_evaluation(self):
        model=RadarCNN();x=torch.randn(4,2,75,75);meta=torch.randn(4,12)
        value=model(x,meta);value.sum().backward()
        self.assertEqual(tuple(value.shape),(4,));self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval();torch.testing.assert_close(model(x[:1],meta[:1]),model(x,meta)[:1],rtol=1e-5,atol=1e-6)


if __name__=='__main__':unittest.main()
