"""Regression checks for validation isolation and pseudo-label safety."""
import unittest
import numpy as np
from src.v3.fusion import prior, fuse, fold_fusion, soft_targets
from src.v3.train import fit_meta, meta_transform

class IsolationTests(unittest.TestCase):
    def test_same_fold_labels_cannot_change_predictions(self):
        a=np.array([31.,31.,31.01,31.01,40.,40.])
        y=np.array([0.,1.,0.,1.,0.,1.]); f=np.array([0,0,1,1,2,2])
        p=np.array([.2,.8,.3,.7,.4,.6])
        first,_=fold_fusion(p,a,y,f)
        y[f==0]=1-y[f==0]
        second,_=fold_fusion(p,a,y,f)
        np.testing.assert_array_equal(first[f==0],second[f==0])
    def test_uncovered_unchanged(self):
        p=np.array([.1,.9])
        result=fuse(p,np.array([.5,.5]),np.array([False,False]),model_weight=.2)
        np.testing.assert_array_equal(p,result)
    def test_empty_reference(self):
        p,c=prior(np.array([np.nan]),np.array([0]),np.array([30.,np.nan]))
        self.assertFalse(c.any()); self.assertTrue(np.isfinite(p).all())
    def test_exact_prior(self):
        p,c=prior(np.array([30.,30.,40.]),np.array([1,1,0]),np.array([30.]))
        self.assertAlmostEqual(p[0],2.1/2.2); self.assertTrue(c[0])
    def test_soft_bounds(self):
        p=soft_targets(np.array([0.,.001,.03,.97,.999,1.]))
        np.testing.assert_allclose(p,[.02,.02,.03,.97,.98,.98])
    def test_imputation_fitted_without_queries(self):
        train=np.array([[30.,1.],[np.nan,2.],[40.,3.]])
        fit=fit_meta(train)
        self.assertEqual(fit["median"][0],35)
        query=np.array([[np.nan,4.]])
        transformed=meta_transform(query,fit)
        self.assertAlmostEqual(transformed[0,0],0)
    def test_prior_does_not_mutate_inputs(self):
        a=np.array([30.,np.nan]); y=np.array([1.,0.])
        prior(a,y,np.array([30.]))
        self.assertTrue(np.isnan(a[1])); np.testing.assert_equal(y,[1,0])

if __name__=="__main__": unittest.main()
