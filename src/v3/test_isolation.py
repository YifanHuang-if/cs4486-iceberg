"""Regression checks for validation isolation and pseudo-label safety."""
import unittest
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
import torch
import numpy as np
from src.v3.fusion import prior, fuse, fold_fusion, soft_targets
from src.v3.train import fit_meta, meta_transform, run_fold
from src.v3.fusion import pool_queries, pooled_fold_fusion

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


    def test_query_pooling_never_pools_missing_angles(self):
        p=np.array([.1,.9,.2,.8])
        result=pool_queries(p,np.array([30.,30.,np.nan,np.nan]))
        np.testing.assert_allclose(result,[.3,.7,.2,.8])

    def test_training_and_teacher_exclude_heldout_labels(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(4)
        data=dict(images=rng.normal(size=(10,2,75,75)).astype(np.float32),
                  test_images=rng.normal(size=(2,2,75,75)).astype(np.float32),
                  ids=np.array([str(i) for i in range(10)]),
                  test_ids=np.array(["test0","test1"]),
                  stats=rng.normal(size=(10,15)).astype(np.float32),
                  test_stats=rng.normal(size=(2,15)).astype(np.float32),
                  angles=np.linspace(30,40,10),test_angles=np.array([31.,39.]),
                  labels=np.array([0,1]*5,dtype=np.float32),folds=np.arange(10)%5)
        args=Namespace(tag="original",seed=7,epochs=1,batch=4,lr=.001,dropout=.3,
                       normalisation="local",augmentation="mild",folds=[0],
                       folds_file=None,threads=1,device="cpu",teacher="",
                       pseudo_weight=.25,initialize_teacher=False,validation_only=False)
        with tempfile.TemporaryDirectory() as temp, patch("src.v3.train.OUT",Path(temp)):
            run_fold(args,data,0)
            args.folds=[0,1]
            resumed=run_fold(args,data,0)
            self.assertEqual(resumed["config"]["tag"],"original")
            args.folds=[0]
            original=np.load(Path(temp)/"original/fold0_predictions.npz")["test_probability"]
            data["labels"][data["folds"]==0]=1-data["labels"][data["folds"]==0]
            args.tag="changed_outer_labels"
            run_fold(args,data,0)
            changed=np.load(Path(temp)/"changed_outer_labels/fold0_predictions.npz")["test_probability"]
            np.testing.assert_array_equal(original,changed)
            # A controlled teacher-prediction fixture exercises actual pseudo rows.
            teacher_file=Path(temp)/"original/fold0_predictions.npz"
            with np.load(teacher_file) as stored:
                payload={k:stored[k] for k in stored.files}
            payload["test_probability"]=np.array([.99,.01])
            np.savez(teacher_file,**payload)
            data["test_angles"][:]=np.nan
            args.tag="student";args.teacher="original";args.initialize_teacher=True
            student_record=run_fold(args,data,0)
            self.assertEqual(student_record["pseudo_rows"],2)
            student=np.load(Path(temp)/"student/fold0_predictions.npz")["test_probability"]
            data["labels"][data["folds"]==0]=1-data["labels"][data["folds"]==0]
            args.tag="student_changed_outer_labels"
            run_fold(args,data,0)
            changed_student=np.load(Path(temp)/"student_changed_outer_labels/fold0_predictions.npz")["test_probability"]
            np.testing.assert_array_equal(student,changed_student)


    def test_nested_selection_excludes_outer_labels(self):
        from src.v3.nested_angle import nested_oof
        a=np.array([30.,30.,30.01,30.01,40.,40.])
        y=np.array([0.,1.,0.,1.,0.,1.]);f=np.array([0,0,1,1,2,2])
        p=np.array([.2,.8,.3,.7,.4,.6])
        first,parameters=nested_oof(p,a,y,f)
        y[f==0]=1-y[f==0]
        second,changed_parameters=nested_oof(p,a,y,f)
        np.testing.assert_array_equal(first[f==0],second[f==0])
        self.assertEqual(parameters[0],changed_parameters[0])


    def test_shape_features_are_per_image_and_flip_invariant(self):
        from src.v3.shape_model import features
        rng=np.random.default_rng(8)
        images=rng.normal(size=(2,2,75,75)).astype(np.float32)
        angles=np.array([31.,np.nan])
        together=features(images,angles)
        alone=features(images[:1],angles[:1])
        np.testing.assert_allclose(together[:1],alone,rtol=1e-5,atol=1e-5)
        flipped=features(images[:,:,:,::-1].copy(),angles)
        np.testing.assert_allclose(together,flipped,rtol=1e-5,atol=1e-5,equal_nan=True)

if __name__=="__main__": unittest.main()
