"""Behavioral checks for second-round models and target-free prediction pooling."""
import unittest
import numpy as np
import torch
from src.refinement.advanced import DeepRadarCNN,FinetunedTail
from src.refinement.odds_context import OddsContext,query_group_pool
from src.refinement.context import EmpiricalAngleContext
from src.refinement.recontextualize import invert_context
torch.set_num_threads(2)


class RoundTwoTests(unittest.TestCase):
    def test_context_inverse_recovers_unclipped_probabilities(self):
        model=EmpiricalAngleContext().fit(['a','b','c','d','e'],[30.,30.,40.,40.,50.],[1,1,0,0,1])
        angles=np.array([30.,40.,50.,60.,np.nan]);original=np.array([.023,.876,.456,.321,.789])
        transformed,_=model.transform(['u','v','w','x','y'],angles,original)
        recovered=np.array([invert_context(p,a,model) for p,a in zip(transformed,angles)])
        np.testing.assert_allclose(recovered,original,rtol=0,atol=1e-11)

    def test_new_models_train_and_have_batch_independent_inference(self):
        for model,x in [(DeepRadarCNN(),torch.randn(3,2,75,75)),(FinetunedTail(pretrained=False),torch.randn(3,512,4,4))]:
            meta=torch.randn(3,12);model(x,meta).square().sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            model.eval();torch.testing.assert_close(model(x[:1],meta[:1]),model(x,meta)[:1],atol=1e-6,rtol=1e-5)

    def test_pooling_uses_only_query_predictions_and_preserves_singletons(self):
        p=query_group_pool([30.,30.,40.,np.nan],[.2,.8,.1,.9])
        np.testing.assert_allclose(p,[.35,.65,.1,.9],atol=1e-12)
        permutation=np.array([3,1,0,2]);angles=np.array([30.,30.,40.,np.nan]);values=np.array([.2,.8,.1,.9])
        np.testing.assert_array_equal(query_group_pool(angles[permutation],values[permutation]),p[permutation])

    def test_context_never_reads_held_out_labels_and_requires_disjoint_ids(self):
        model=OddsContext(weight=1,empirical=True,pool=False).fit(['a','b','c','d'],[30.,30.,40.,40.],[1,1,0,0])
        p,n=model.transform(['e','f','g'],[30.,40.,50.],[.1,.9,.723456789])
        self.assertGreater(p[0],.1);self.assertLess(p[1],.9);self.assertEqual(p[2],.723456789)
        np.testing.assert_array_equal(n,[2,2,0])
        with self.assertRaises(ValueError):model.transform(['a'],[30.],[.2])


if __name__=='__main__':unittest.main()
