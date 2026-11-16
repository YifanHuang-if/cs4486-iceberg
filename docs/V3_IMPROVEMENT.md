# V3 validation repair and improvement plan

Base source: GitHub main commit 661c5cec0af3dd754fa49396794879fdd084a3db (PR #7).
Reference: Iceberg_V2_Improvement_Summary.docx, supplied 2026-11-16.
Target: log loss below 0.083. This is a target, not a guaranteed outcome or an official rank.

## What carries forward
- Compact radar CNNs, local/global normalization diversity, dihedral test-time averaging.
- Exact-angle evidence followed by a Gaussian-angle fallback (sigma 0.02, alpha 0.1).
- Model log-odds weight 0.75 and angle log-odds weight 1, fixed from the supplied document.
- Soft pseudo-label distillation, with strict outer-fold teacher isolation.

## Audit findings
1. V2 angle_radius.evaluate excludes the query row but retains other labels from the
   query's validation fold. Its LOO angle + five-fold image hybrid is a separate
   diagnostic; it is not a five-fold evaluation of the entire prediction pipeline.
2. V2 blend --calibrate fits and scores Platt scaling on the same OOF labels.
   Such a score is a development fit, not a held-out calibration score.
3. V2 pseudo.py applies maximum(p, 0.98) for positive labels and minimum(p, 0.02)
   for negative labels. That pushes probabilities toward the endpoints, opposite
   to clipping into [0.02, 0.98].
4. V2 train accepts the same pseudo-label file for every fold. A teacher trained
   with outer validation labels would transfer those labels indirectly into the
   student. No test labels are needed for this failure.
5. V2 load_cache imputes missing angles using the full-training median.
   V3 fits the median, metadata standardization, and global image moments only on
   the training portion of each outer fold.
6. V2 bayes_fusion says uncovered rows are unchanged, but scales their logits.
   V3 leaves uncovered probabilities exactly unchanged.
7. Decimal precision alone does not establish which test rows are artificial or
   scored. V3 predicts every one of the 8,424 official test IDs; it never assumes
   a subset is exempt from scoring.

## Prespecified runs
- v3_mild_s11: RadarV1, local normalization, seed 11, mild dihedral/noise
  augmentation, 40 fixed epochs, AdamW, cosine learning rate, eight-view TTA.
- v3_global_s29: RadarV1, global normalization, seed 29, affine + dihedral
  augmentation, same fixed budget.
- Fixed equal-probability ensemble, then fold-isolated angle fusion.
- A student, if run, uses only its corresponding fixed-epoch teacher fold and
  the same training-fold angle evidence. High-confidence image/angle agreement
  is required; targets are clipped into [0.02, 0.98], weight 0.25.
- Report image-only and fused pooled OOF losses, each fold, observed/missing-angle
  subgroups, and a separately named historical LOO diagnostic.
- The target is evaluated on the entire frozen course split. Exact-angle grouped
  evaluation is a separate robustness experiment, never mixed into random-fold OOF.

## Validation boundaries
No validation label is accepted by the angle-prior prediction function.
No outer validation loss is used to stop training, select a checkpoint, fit a
calibrator, or generate a teacher's pseudo-labels. The final fixed epoch is saved.

This remains development CV: prior experiments and the supplied document have
already influenced design choices on this dataset. An untouched evaluation is
required for an independent generalization estimate. In particular, local CV,
LOO diagnostics, projected scores and the official Kaggle private leaderboard
must not be described as interchangeable.

## Files and recovery
All new code is in src/v3/. Results go to outputs/v3/ (weights, predictions,
history) and results/v3/ (small shareable reports). The original workspace and
historical outputs are preserved. No raw competition data or weights are placed
in ordinary Git commits.

## AI assistance
On 2026-11-16 ChatGPT audited V2, implemented fold-isolated fusion and pseudo-label
training, added regression checks and executed the recorded experiments.
Students must review and understand these changes. No student hours or
independent authorship are inferred.

## Additional fixed diagnostic
The 50/50 within-query exact-angle pooling used in the earlier refinement
is also evaluated with the new image probabilities. This uses no query labels
but depends on which rows are predicted together. It is reported separately and
must not be described as ordinary independent-sample inference.

## External evidence
The official competition data page states that machine-generated images are
excluded from scoring:
https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data
This does not verify the decimal-precision heuristic for identifying those rows.
The second-place solution used angle-group folds and substantial CNN/XGBoost
ensembling:
https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/writeups/beluga-2nd-place-solution-overview
A small CPU ensemble is not evidence of equivalent official performance.
