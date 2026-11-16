import unittest
import numpy as np
from src.v4.joint_fusion import distances,joint_prior,apply

class JointFusionTests(unittest.TestCase):
    def test_exact_prior_ignores_feature_distances(self):
        p,c=joint_prior(np.array([30.,30.,31.]),np.array([1.,1.,0.]),np.array([30.]),np.array([[100.,200.,0.]]))
        self.assertAlmostEqual(p[0],2.1/2.2);self.assertTrue(c[0])

    def test_similarity_changes_nearby_evidence_without_extra_mass(self):
        a=np.array([30.,30.02]);y=np.array([0.,1.]);q=np.array([30.01])
        neutral,_=joint_prior(a,y,q,np.zeros((1,2)))
        p,_=joint_prior(a,y,q,np.array([[0.,10.]]))
        self.assertAlmostEqual(neutral[0],.5);self.assertLess(p[0],.2)

    def test_query_batch_does_not_fit_scaler(self):
        rng=np.random.default_rng(8);r=rng.normal(size=(7,10));q=rng.normal(size=(3,10))
        np.testing.assert_allclose(distances(r,q)[:1],distances(r,q[:1]))

    def test_prediction_has_no_query_label_input_and_missing_unchanged(self):
        rng=np.random.default_rng(9);r=rng.normal(size=(4,10));q=rng.normal(size=(2,10))
        fit=dict(sigma=.02,alpha=.1,bandwidth=1,weights=[.75,1,.75,1])
        p=apply(np.array([.3,.7]),np.array([30.,30.,31.,31.]),np.array([1.,1.,0.,0.]),np.array([np.nan,80.]),r,q,fit)
        np.testing.assert_allclose(p,[.3,.7])

if __name__=='__main__':unittest.main()
