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
- Report neural-model and fused pooled OOF losses, each fold, observed/missing-angle
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

## Reproduction

From the repository root with the existing Python environment:

```powershell
.venv\Scripts\python.exe -m src.v3.prepare --raw data/raw
.venv\Scripts\python.exe -m unittest src.v3.test_isolation -q
.venv\Scripts\python.exe -m src.v3.train --tag v3_mild_s11 --epochs 40 --threads 4
.venv\Scripts\python.exe -m src.v3.train --tag v3_global_s29 --seed 29 --normalisation global --augmentation affine --epochs 40 --threads 4
.venv\Scripts\python.exe -m src.v3.evaluate --tags v3_mild_s11 v3_global_s29 --name v3_supervised_ensemble
.venv\Scripts\python.exe -m src.v3.evaluate_nested --name v3_supervised_ensemble
```

The script scripts/run_v3.ps1 wraps these supervised commands and supports an
optional student run. A CPU smoke test is not a GPU compatibility claim.

Student pilot (only fold 0; not a five-fold score):

```powershell
.venv\Scripts\python.exe -m src.v3.train --tag v3_student_s47 --seed 47 --teacher v3_mild_s11 --initialize-teacher --lr 0.0003 --epochs 15 --folds 0 --threads 4
```

To continue an unchanged pilot across all folds, omit --folds 0. Completed
matching folds are skipped. A changed seed, normalization, epoch budget or other
training setting requires a different tag. This resumes completed folds, not an
interrupted epoch.

Grouped robustness run (finite equal-angle groups never cross folds; missing
angles are treated as separate singleton groups, as in the supplied split):

```powershell
.venv\Scripts\python.exe -m src.v3.train --tag v3_grouped_s11 --epochs 40 --folds-file results/v2/folds_angle_grouped.csv --validation-only --threads 4
.venv\Scripts\python.exe -m src.v3.evaluate --tags v3_grouped_s11 --name v3_grouped --folds-file results/v2/folds_angle_grouped.csv --validation-only
```

## Additional angle and confidence checks

- Nested angle selection: 12 combinations of Gaussian width and shrinkage, scored
  using three inner folds of each outer training set. No outer query label enters
  this search. Image fusion weight remains the documented fixed 0.75.
- A fixed [0.001, 0.999] probability clipping variant is reported, matching the
  supplied document's submission convention. It is not silently substituted for
  the unclipped metric.
- These are development comparisons; selecting a final variant using these
  results does not create an untouched test score.

## Measured intermediate results

All values below are pooled predictions over all 1,604 training rows on the
unchanged five-fold course split. The neural models already include metadata;
"model" means before the explicit angle-prior fusion, not image-only inputs.

| Candidate | Model log loss | Fold-isolated angle fusion |
|---|---:|---:|
| Local-normalization CNN | 0.204424 | 0.115420 |
| Global-normalization CNN | 0.260439 | 0.115521 |
| Equal mean of the two CNNs | 0.206748 | 0.108068 |
| Three-seed shape/statistics MLP | 0.228151 | 0.113870 |
| Equal mean of both CNN families and the shape MLP | 0.194428 | 0.103447 |

The shape model uses 90 per-image/statistical/angle features and three independently
seeded MLPs per fold. It has no validation-based epoch selection.
The best currently completed candidate is still above 0.083.

The pseudo-label student and exact-angle-grouped robustness experiments are still
in progress at this checkpoint. The student pilot improved fold 0 from 0.088291
to 0.087745 after training on 4,776 soft pseudo-labels; this is not a five-fold
student result.

Both supervised CNN families passed checkpoint replay over all folds, with
maximum probability differences below 6e-8. All 15 shape MLP states also passed
validation and test prediction replay. The supervised ensemble submission has
8,424 unique IDs in sample-submission order and passed probability checks.

Ten isolation tests passed before the latest shape-feature regression check;
the final verification record will state the final count.
