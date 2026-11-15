# Steps 7-8: controlled experiments on baseline_v1

This batch starts from main commit `f672b0c3218503348cab41656c1d4d7d6fc783f6` and its published baseline_v1 implementation. The older stride-convolution E0-E5 experiments belong to a different model implementation and are not used in this batch's ablations.

The following configurations and comparison rules are fixed before the scored experiments. All runs use the existing primary folds, channel normalization fitted on each training fold, Adam (learning rate 0.001, weight decay 0.0001), batch size 32, maximum 30 epochs, patience 7, and seed 42 + fold. There is no minimum-epoch override, scheduler, pretrained weight, class weighting or test-time augmentation. CPU PyTorch 2.5.1 and four threads per process match the published baseline settings. Runtime package versions are recorded; Python/NumPy may differ from the original Windows environment. V0 is reproduced first and its OOF predictions compared with the published baseline.

| ID | Reference | Single intended change |
| --- | --- | --- |
| V0_baseline | Published baseline_v1 | Reproduce the current architecture and training protocol |
| V1_flips | V0 | Add random horizontal/vertical training flips |
| V2_dropout | V1 | Change dropout from 0.25 to 0.40; keep weight decay fixed |
| V3_angle | V1 | Add an imputed/scaled incidence-angle branch |
| V4_missing | V3 | Also provide the missing-angle indicator |
| V5_no_flips | V3 | Remove training flips |

The image backbone retains the baseline's convolution/max-pooling structure, 3x3 adaptive pooling and 64-unit hidden layer. V0-V2 use its existing output layer. For metadata variants, replace only the final output layer with a head that combines the 64 image features and 8 metadata features (metadata MLP: input -> 16 -> 8). All shared image weights have the same initial values within a fold; the added branch/head changes the parameter count, which is disclosed in results. Flip randomness uses a separate generator so enabling flips does not consume the model/dropout random stream or alter the sample-order generator.

Each configuration trains five independent fold models from scratch. Retain the checkpoint with minimum validation Log Loss and combine held-out predictions by sample ID. Report pooled OOF Log Loss, fold mean and population standard deviation, accuracy, observed-angle and missing-angle subgroup losses, individual fold changes, and epoch histories. Negative results are retained. Tests and checkpoint replays verify implementation/integrity, not independent generalization.

## Conditional angle-group check

If either V3 or V4 lowers primary pooled OOF loss by at least 0.005 relative to V1, repeat V1, V3 and V4 on a separate split that keeps exact equal observed angles together. Use StratifiedGroupKFold with joint label/missingness strata, sorted IDs, seed 42 and unique singleton groups for missing angles. Primary folds are unchanged. Grouped results are reported separately, without pooling scores across protocols. If the trigger is not met, explicitly record that the check was not triggered.

Exact angles are proxy groups, not verified acquisition identifiers. All missing-angle training samples are ships while test angles are complete; therefore an overall improvement from the missing indicator must be interpreted together with observed-angle performance. Checkpoint and method selection reuse validation labels, and this bounded single-seed batch is subject to selection optimism. No course Top-5 ranking or final competition score is established. Final inference/submission and detailed error analysis remain subsequent steps.
