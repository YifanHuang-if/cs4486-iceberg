# V4 execution and new comparisons

This continuation restores the V3/V4 source at PR 9 commit
557defaa6a3d2cbd831d84de9ea07effbe825e88. The original Windows V3 checkpoints
and row-level predictions are not available in this workspace or the searched
Library files. Therefore the 0.10114 handoff claim is not verified here.
New Linux runs use separate tags and never overwrite historical evidence.

## Prespecified first batch

1. Rerun the two V3 CNN settings and the three-seed shape MLP on the unchanged
   five folds, with PyTorch 2.5.1 CPU. Epoch counts and model settings are unchanged.
2. Evaluate the two pre-existing V4 rules: full reliability gating and exact-match
   preservation. Publish improvements and regressions against the same probabilities.
3. Train two complementary shape-feature models with fixed budgets: histogram
   gradient boosting and extremely randomized trees. No outer-fold early stopping
   or calibration is allowed. Evaluate individual models and equal-weight additions
   to the recovered CNN/MLP ensemble; retain all comparisons.
4. Test a training-only, inner-cross-validated angle kernel, with a fixed finite
   search space declared in its source before scoring outer-fold predictions.

All scores are development CV, not untouched validation or official Kaggle
scores. Repeated experiments introduce selection bias. Predict all 8,424 test IDs;
decimal precision is not an established scoring-mask identifier. The goal remains
Log Loss below 0.083, without a guarantee or rank claim.

Every completed full-five-fold result is synchronized to PR 9. This work includes
AI implementation and execution; no student contribution hours are inferred.
