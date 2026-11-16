# V4 execution checkpoint

Completed full-five-fold results on the unchanged 1,604-row split:

| Run | Development Log Loss |
|---|---:|
| linux_extra | 0.12328378 |
| linux_hist | 0.11836720 |
| linux_shape | 0.11387032 |
| linux_shape_interpolation | 0.11946831 |
| linux_shape_joint | 0.11267956 |
| linux_shape_nested | 0.11039162 |

The nested shape-only rule improves its own control from 0.11387032 to 0.11039162. It has **not** beaten the historical full CNN/MLP ensemble at 0.103447, nor verified the unsynchronized 0.10114 claim. Interpolation and joint weighting are exploratory controls; unsuccessful results are retained.

The original V3 Windows checkpoints and OOF files were not accessible. The two original CNN settings are being rerun under distinct Linux tags. Historical files remain unchanged.

## In-progress training snapshot

- linux_local_s11: 4/5 complete fold artifacts; incomplete runs are not full-five-fold scores.
- linux_global_s29: 4/5 complete fold artifacts; incomplete runs are not full-five-fold scores.
- linux_student_s47: 1/5 complete fold artifacts; incomplete runs are not full-five-fold scores.

30 isolation/regression checks passed. All five shape-MLP checkpoints and both tree-model families were reloaded and their validation/test probabilities reproduced. Raw data, model weights and row-level predictions are not part of these Git commits.

PR 9 targets the V3 branch (PR 8), not main. No official Kaggle score is available for these candidates; the target <0.083 remains unmet.

The pending full-ensemble V4 reliability comparisons require the remaining CNN folds. This snapshot does not claim those comparisons are complete.
