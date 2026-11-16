# Fixed five-fold validation

## Protocol

Use data/splits/folds.csv for every baseline, improvement and ablation. Fold numbers are 0 through 4. For run k, train on fold != k and validate only on fold == k. Every training example is held out once across the five runs. The final two test-data files are never used as labeled training or validation data.

The custom deterministic stratified algorithm sorts IDs within joint label/missing-angle strata, shuffles with NumPy default_rng seed 42, then assigns samples round-robin with an offset carried between strata. Each stratum differs by at most one sample across folds. This preserves class balance and distributes the 133 missing-angle ships across folds. It is not a scikit-learn splitter. The instructor has not supplied a mandatory fold mapping; replace this protocol only if an official mapping becomes available, with explicit versioning.

## Similarity review

The 182 EDA candidates were rechecked at full resolution after independently centering each channel. The largest minimum of the two channel correlations was 0.849112, below the conservative 0.98 screening threshold. The six highest-ranked pairs were plotted for visual review. No exact duplicate image was found in EDA, and no candidate met the additional grouping threshold. Therefore this version uses sample-level stratification.

This is a heuristic screen of EDA candidates, not an exhaustive search for rotated, shifted or same-scene acquisitions. Acquisition metadata is unavailable. A future verified relationship should trigger a versioned grouped-split review rather than silently changing the frozen file.

## Leakage prevention and preprocessing

For each fold, fit image normalization and angle imputation/scaling only on that fold's training samples. Apply those same saved values to validation and test images. Apply data augmentation only in training. Do not select preprocessing statistics using validation or test distributions. Use validation labels only for model selection/evaluation, and acknowledge that repeated tuning can overfit CV.

## Evaluation contract

Return the probability of iceberg (label 1). Compute binary Log Loss using natural logarithms, clipping probabilities to [1e-7, 1-1e-7]. Save one out-of-fold probability per ID from the model that did not train on that ID. Report pooled OOF Log Loss as the primary CV score, alongside individual fold scores and their unweighted mean and standard deviation. Pooled loss equals the sample-count-weighted mean of fold losses. Accuracy is supplementary.

Also report observed-angle and missing-angle subgroup losses. Missingness occurs only among ships in this training data, while the test angles are complete. No performance score exists until the baseline has been trained. This internal split does not establish the course Top-5 ranking.

## Reproduce and verify

Run `python src/make_folds.py` from the repository root using requirements-eda.txt. No additional packages are required. The script verifies unique IDs, complete fold coverage, both classes in each fold, disjoint train/validation IDs and balanced strata. It refuses to overwrite a different existing folds.csv. configs/validation.json records the algorithm, seed, versions, raw training-file hash and frozen-fold hash. results/validation/fold_summary.csv contains fold counts.
