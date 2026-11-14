# Data preparation contract

## Inputs

Read the provided JSON records without modifying the source files. Reshape each radar band to 75 by 75 in row-major order and stack band_1 then band_2. The model input will have shape (N, 2, 75, 75), with float32 values. Preserve sample IDs and record order. Labels are 0 for ship and 1 for iceberg. Test labels are unavailable.

## Missing angles

Treat the literal na and null as missing. Preserve a binary missingness indicator. The image-only baseline does not consume angles. For subsequent angle fusion, fit the imputation median and scaling statistics using only the training fold, then apply those values to validation and test data. Keep missing-angle samples.

## Image normalization

Retain the supplied radar values. For the baseline, compute one mean and standard deviation per channel using only the training fold. Apply these saved statistics to validation and test samples, with a small denominator floor for constant channels. The EDA script reports raw statistics and does not fit global model preprocessing. Percentile display scaling in example figures is for visualization only.

## Validation handoff

Step 5 should save an ID-to-fold mapping and document the random seed, shuffling and stratification. Review exact duplicates and similar-image candidates before freezing folds. Thumbnail similarity is a screening heuristic, not proof of shared acquisition or leakage. Verify suspicious pairs visually before deciding whether grouping is justified. No reliable acquisition-group field is provided in the records.

## Robustness and follow-up experiments

Compare performance separately for observed and missing angles and across angle ranges. Keep augmentation within the training fold and apply spatial transformations consistently to both image channels. Assess augmentation and angle fusion against the same baseline and folds. Report Log Loss as the primary metric.

## Local data and shared outputs

Keep raw JSON files local. Share the script, environment versions, aggregate tables, figures and ID-level diagnostic metadata in this private team repository. The script reads JSON incrementally to avoid loading the full test file into memory. Running it regenerates results/eda/.
