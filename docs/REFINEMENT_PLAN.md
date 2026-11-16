# Refinement plan: a strict target is not an observed result

The requested target is official Kaggle Log Loss strictly below 0.083. A local cross-validation score cannot certify that target. The Kaggle page exposes a Late Submission entry, but authenticated scoring availability must be checked with an actual submission. The competition excludes machine-generated test images from scoring, which is another reason local CV and leaderboard values differ.

This round retains the original primary folds and supplements them with the existing exact-angle-group split. No training labels are changed or removed, no validation sample is used as a training sample within its fold, and no test ground-truth labels or public prediction files are imported. A lower training loss, a selected single fold, or a rounded number will never be used as the target result.

Before scored runs, the initial batch is:

1. R1_global: a four-block Conv/BatchNorm/SiLU network, average-plus-max spatial pooling, image-statistic/angle metadata, AdamW, a cosine learning-rate schedule, paired dihedral/affine augmentation and an 80-epoch budget. Select the retained epoch with validation loss, with a minimum 40 epochs and patience 20. Report ordinary and eight-view TTA predictions separately.
2. R2_local: the same network/training settings, with per-image channel standardization and preserved global image statistics in the metadata branch. This tests a complementary representation.
3. R3_radar_trees: CatBoost on independently extracted image statistics, thresholded-component geometry, polarization difference and angle features. The number of trees is selected on an inner split of the outer training fold and the model is refitted on that entire training fold before outer evaluation.
4. Evaluate fixed equal-weight averages of the two CNNs, and of the two CNNs plus the feature model. These weights are not optimized on the reported OOF labels. Any further adaptive model selection will be disclosed.

Every initial configuration completes all five primary folds. Supplementary grouped checks are reported separately. Preprocessing for the CNN metadata is fitted on training-fold samples only. Geometric transforms are aligned between channels. Image statistics are calculated within each image and do not use targets. Validation labels select CNN epochs, so development/model-selection optimism remains; this is not an independent final test.

The current execution environment has no available CUDA device. CPU trials provide measured evidence; a GPU-compatible runner will also be retained. No paid computing resources are provisioned.

## Public ideas consulted, not copied implementations

- David Austin's first-person winner interview describes extensive CNN ensembles, additional image features and four-fold validation: https://pyimagesearch.com/2018/03/26/interview-david-austin-1st-place-25000-kaggles-popular-competition/
- Beluga's second-place writeup describes angle-group validation, varied CNNs and tree models: https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/writeups/beluga-2nd-place-solution-overview
- Official data description, including the unscored generated test images: https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/data
- Official evaluation: https://www.kaggle.com/c/statoil-iceberg-classifier-challenge

These sources motivate independent experiments, not a claim that the winner's system was reproduced. AI-generated implementation and execution must be disclosed; students must understand and accurately describe their own contributions under the course's academic-integrity rules.

## Additional context hypothesis, recorded during the initial runs

Training-data inspection found 878 distinct observed-angle groups, including 217 repeated groups; six groups contain both labels. Before either CNN has completed its five folds, add a fixed-strength context experiment: for a held-out sample with a matching observed angle in the training fold, combine the image probability with that group's training-label counts as `(probability + positive_count) / (1 + group_count)`. With no match (or a missing angle), leave the probability unchanged. The prior strength is fixed at one, not searched on OOF scores. The context code rejects IDs that were included in fitting.

This feature is conditional on train/query angle overlap; it must be reported separately from image-only generalization. It can improve random-fold CV when related angle groups cross folds. Under the existing grouped protocol, all known-angle queries have zero matching training groups, so it must reduce exactly to the underlying prediction. No grouped result may be replaced with its primary-fold score. A candidate using this context can only certify the requested official threshold through actual Kaggle scoring.

After observing the fixed-strength update on the previous V4 predictions, also evaluate an empirical-Bayes variant. Its concentration is estimated by beta-binomial marginal likelihood on repeated groups from the training fold alone (log-concentration bounded to [-6,6]); held-out outcomes do not set this parameter. The image probability supplies the query-specific prior mean. This is a separately disclosed adaptive addition to the development study, not a preregistered independent final test. Save all estimated fold concentrations and both context variants.

## Transfer-feature addition

While the two CNNs are still training, add R6_frozen_resnet as a complementary feature experiment. The publicly pretrained ImageNet-1K ResNet-18 backbone is frozen; its official full-model parameter count is 11,689,512, below the course limit. SAR HH, HV and their difference are separately normalized by each image's 1st/99th percentiles, resized to 128 pixels, then normalized with the ImageNet constants. This deliberately differs from the official natural-image crop preprocessing. Concatenate the 512 frozen features with the radar statistics and fit a standardized logistic classifier; choose C from [0.001,0.01,0.1,1] using three-fold CV strictly inside each outer training fold. The public backbone sees no task labels. Its usefulness must be measured, not assumed. Reference: https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.resnet18.html

Also evaluate the fixed equal-weight mean of R1, R2 and R6, including the same two context variants. The initial CNN/tree average retains its original three constituents. Repeated model comparisons introduce selection optimism even though no ensemble weights are fitted to outer-fold labels.

Before the primary CNN runs finish, select R2_local and R6_frozen_resnet for the separate grouped sensitivity check. R2 tests the per-image normalization route when equal angles cannot cross the split; R6 provides an inexpensive independent feature-model comparison. These are supplementary checks of those two individual methods, not a grouped estimate of every primary ensemble. Context updates must have zero effect on these grouped predictions.
