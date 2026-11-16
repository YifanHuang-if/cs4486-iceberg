import unittest
import numpy as np
from src.v4.nested_fusion import select_fit, predict_with_fit

class NestedFusionTests(unittest.TestCase):
    def test_outer_label_mutation_cannot_change_fit_or_prediction(self):
        a=np.array([30,30,30.01,30.02,31,31.01,31.02,32,32.01,33,33.01,34.],float)
        y=np.array([0,0,0,1,1,1,0,1,1,0,0,1.])
        outer=np.arange(12)%3==0;train=~outer
        proxy=np.array([.2,.8,.4,.7,.1,.9,.6,.3])
        inner=np.arange(train.sum())%3
        fit,_=select_fit(a[train],y[train],proxy,inner)
        original=predict_with_fit(np.full(outer.sum(),.4),a[train],y[train],a[outer],fit)
        y[outer]=1-y[outer]
        changed,_=select_fit(a[train],y[train],proxy,inner)
        np.testing.assert_array_equal(original,predict_with_fit(np.full(outer.sum(),.4),a[train],y[train],a[outer],changed))
        self.assertEqual(fit,changed)

    def test_uncovered_preserved_and_exact_uses_own_coefficients(self):
        fit=dict(sigma=.02,alpha=.1,weights=[1,0,.1,2])
        result=predict_with_fit(np.array([.2,.7,.8]),np.array([30.]),np.array([1.]),np.array([30.,80.,np.nan]),fit)
        np.testing.assert_allclose(result,[.2,.7,.8],rtol=1e-12)

if __name__=='__main__':unittest.main()
