# V5: verified score, bottleneck analysis and what actually moved the needle

Status: measured results only. Nothing here is an official Kaggle score.
Target 0.083 has **not** been reached.

## 1. Verified starting point

The latest pushed branch was `pr9` (`58a4737`). Its own numbers, read from
`results/v4/*.json` and `docs/V4_EXECUTION_RESULTS.md`:

| Run (full five folds) | Development log loss |
|---|---:|
| linux_extra | 0.12328378 |
| linux_hist | 0.11836720 |
| linux_shape | 0.11387032 |
| linux_shape_interpolation | 0.11946831 |
| linux_shape_joint | 0.11267956 |
| linux_shape_nested | 0.11039162 |

None of the six V4 candidates beat the earlier V3 three-family ensemble:

| V3 candidate | Model log loss | Fold-isolated angle fusion |
|---|---:|---:|
| Local-normalisation CNN | 0.204424 | 0.115420 |
| Global-normalisation CNN | 0.260439 | 0.115521 |
| Shape/statistics MLP | 0.228151 | 0.113870 |
| Equal mean of both CNN families and the shape MLP | 0.194428 | **0.103447** |

A later figure of 0.10114 appears in the handoff prose but has no result file on
the branch; `docs/V3_SYNC_STATUS.md` explicitly says not to cite it. So the
verified best before this work was **0.103447**.

## 2. Where the loss actually is

Group split of the V3 ensemble (`results/v3/v3_diverse_ensemble_errors.json`):

| Group | Rows | Log loss | Share of total loss |
|---|---:|---:|---:|
| Exact angle match | 764 | 0.0248 | 11.4% |
| Nearby angle only | 697 | 0.2069 | 86.9% |
| Far angle | 10 | 0.2726 | 1.6% |
| Missing angle | 133 | 0.0005 | 0.04% |

To reach 0.083 the total must fall from 165.9 to 133.1 loss units. The other
907 rows contribute only 21.7, so the 697 nearby rows must fall from 0.2069 to
**about 0.160**. That is the whole problem.

## 3. The angle layer is exhausted (measured, not assumed)

`work/probe_angle_ceiling.py`, leave-one-out inside each training fold:

| Quantity | Value |
|---|---:|
| Median distance to the nearest training angle (nearby rows) | 0.0019° |
| Nearby rows within 0.002° | 495 / 697 |
| Nearest-neighbour label agreement, exact groups | 98.6% |
| Nearest-neighbour label agreement, nearby rows | **83.9%** |
| Agreement at ≤0.002° / >0.005° | 83.0% / 84.2% |
| Best angle-only log loss on nearby rows | **0.4408** |

Agreement is **not** flat in distance, as a finer binning shows:

| Distance to nearest training angle | Rows | Agreement |
|---|---:|---:|
| exact (0) | 764 | 98.6% |
| 1e-4 to 5e-4 | 121 | 89.3% |
| 5e-4 to 2e-3 | 221 | 79.5% |
| 2e-3 to 6e-2 | 338 | 84.5% |

So nearby angles are different passes, and the reliability wobbles by band
rather than decaying smoothly. That is why a calibrated per-band prior was
tested; it lost to the Gaussian kernel (section 4) because a single nearest
neighbour throws away the kernel's multi-neighbour aggregation. Either way, an
84%-accurate signal cannot beat a 0.44 log loss on its own: the image model has
to carry those rows.

## 4. What was tried

### Positive

| Change | Evidence |
|---|---|
| Group-wise nested stacking of image and angle evidence (separate prior weight for exact and nearby rows) | 0.11690 → 0.11328 on the earlier OOF; 0.11556 → 0.10721 on a fresh single-seed model |
| Shape/statistics MLP for ensemble diversity | Model loss 0.1880 → 0.1768; fused 0.10519 → 0.10307 |
| Extra mild-augmentation seeds | 1 seed 0.10721 → 3 seeds 0.10499 |
| Higher dropout (0.5) | Fold 0 model loss 0.18530 → 0.16821, though pooled only 0.1959 → 0.1978 |
| Pseudo-label distillation from the four-model ensemble | Best single model 0.1959 → 0.1833; ensemble 0.10307 → 0.10246 |

### Negative, with the measured cost

| Attempt | Result |
|---|---|
| Affine augmentation | Model loss 0.3205 pooled, fold 4 diverged to 0.587 |
| 120 epochs instead of 40 | Fold 0 model loss 0.2139 vs 0.1756; train loss collapsed to 0.0011 (memorisation) |
| Global temperature / probability clipping | 0.116900 → 0.116834 nested; no gain |
| Regime-specific fusion weights | 0.116900 → 0.116220; the fixed 0.75 is already optimal on disagreement rows |
| Flip-invariant thumbnail kNN | 0.4732 loss, 77.5% accuracy; similarity saturates at 0.998 and does not separate exact from nearby rows |
| Transductive same-angle query pooling | 0.100679 → 0.101057; on the 46 poolable CV rows 0.0727 → 0.0795 |
| Seven-model ensemble instead of four | 0.10307 → 0.10311 (no gain) |
| Second distillation seed on top of the first | 0.10246 → 0.10277 (weaker student, no gain) |
| Excluding the 5,000 machine-generated test rows from pseudo-labels | 0.10246 → 0.10259; they were 36-46% of the pseudo set but act as useful extra augmentation |
| Centred-thumbnail image witness as a level-2 feature | 0.10246 → 0.10430; the witness is only 75.2% accurate on nearby rows and adds nothing beyond the CNN probability |
| Mixup alpha 0.2 in the distilled student | Worse on three of three folds measured (0.1861 vs 0.1632, 0.2249 vs 0.2057, 0.1942 vs 0.1928); 20 epochs is too short for mixup to converge |
| Affine augmentation with 120 epochs | Fold 0 0.18773: much better than affine at 40 epochs (0.29507) but still worse than any good 40-epoch model (0.158-0.168). The underfitting explanation was partly right, and the result is still not competitive |
| Mild augmentation with 120 epochs and dropout 0.5 | Fold 0 0.22065 vs 0.21391 at dropout 0.3. Higher dropout does not rescue the long schedule; 40 epochs is the sweet spot |
| Distance-binned calibrated prior with a single nearest neighbour | 0.116785 vs 0.099339 for the Gaussian kernel in the same harness. Coarse bins throw away the kernel's multi-neighbour aggregation |
| Removing `inc_angle` from the CNN metadata branch | Model loss 0.21684 and stacked 0.12726, versus 0.18328 / 0.10492 for the identical student that keeps the angle. Added to the six-model ensemble it made things worse, 0.10194 → 0.10537. The external claim that dropping the angle helps does not transfer to this pipeline |
| Second-round distillation with the six-model teacher | Student model loss 0.18571 versus 0.18328; swapped into the ensemble 0.10191 versus 0.10194, i.e. noise |
| GLCM texture features with a histogram gradient-boosted tree | Best tree over 106 features reached OOF 0.24481 alone; adding it to the ensemble gave 0.10280 versus 0.10194. A different function class was not enough because the member is far weaker than the CNNs |
| GLCM appended to the shape MLP descriptor | Model loss 0.22685 versus 0.22749, but the ensemble went 0.10194 → 0.10228 when swapped in, and keeping both shape MLPs gave 0.10187, i.e. noise |
| Label smoothing 0.03 in the distilled student | Model loss 0.18529 versus 0.18328; in the ensemble 0.10228 added, 0.10281 swapped |
| Non-linear level-2 combiner (boosted trees on the fusion features) | 0.128274 versus 0.101938 for the logistic stacker. The linear log-odds form is the right specification; a tree overfits the level-2 mapping |

### Round 4 verdict

Four more interventions were tested and all four lost. The complete list of
measured attempts is now 25, of which exactly five helped. Adding a member of a
different function class (a boosted tree on 106 hand-crafted features) did not
help because the member is far weaker than the CNNs and the ensemble already
contains a non-CNN MLP.

### Round 20: the training budget had to grow with the pseudo-label set

Loosening the pseudo-label threshold from 0.03 to 0.25 raised the student's
training pool from about 4,800 rows per epoch to 7,543, i.e. 58% more steps per
epoch. The 20-epoch budget was therefore no longer calibrated, so it was swept
again at a fixed seed on folds 0-1 (reference 0.16004 at 20 epochs):

| Epochs | Fold 0 | Fold 1 | Mean |
|---:|---:|---:|---:|
| 12 | 0.16356 | 0.17992 | 0.17174 |
| 20 (previous) | 0.15196 | 0.16812 | 0.16004 |
| 28 | 0.15540 | 0.15951 | 0.15746 |
| **40** | 0.15940 | 0.14442 | **0.15191** |
| 56 | 0.16668 | 0.15456 | 0.16062 |

The optimum moved to 40 epochs, worth **0.0081** on the fold-0-1 screen, which is
the largest single-recipe gain since the optimiser tuning. The two levers
interact: more pseudo-data needs a longer schedule, and the longer schedule only
became affordable because the threshold change made better use of it.

### Round 20 addendum: both leaderboard halves are ±0.011 from cross validation

Five submissions gave a consistent pattern:

| Quantity | Value |
|---|---:|
| Five-fold CV of the best submission | 0.098287 |
| Its private score | 0.08752 (CV − 0.0108) |
| Its public score | 0.10899 (CV + 0.0107) |

The two boards bracket the cross-validation estimate almost symmetrically, which
means the 3,424 scored rows split into one systematically harder half and one
easier half rather than one being intrinsically better predicted.

Two consequences for the targets:

| Target | Current | Needed | Required CV |
|---|---:|---:|---:|
| private below 0.083 | 0.08752 | −0.0045 | about 0.094 |
| public below 0.086 | 0.10899 | −0.023 | about 0.075 |

The private target is within reach of the current rate of progress. The public
target would require a cross-validation level of 0.075, which is 0.023 below the
current 0.098 — four times everything this project has gained so far.

The machine-generated rows were proved irrelevant to both boards by a controlled
experiment: `v5_best_allrows.csv` changed the predictions of 3,659 machine rows
(mean shift 0.087, 850 rows by more than 0.20, maximum 0.74) and both scores were
identical to five decimal places.

### Round 23: two mechanisms borrowed from the literature, both negative

A web sweep for small-data radar classification found two mechanisms this project
had never used: stochastic test-time augmentation and BatchNorm adaptation
(AdaBN). Both were measured on a held-out fold with the checkpoint unchanged.

| Mechanism | Fold 0 loss | Verdict |
|---|---:|---|
| Fixed 8-view dihedral TTA (current) | 0.159400 | baseline |
| Stochastic TTA, 8 views x 1 noise pass | 0.159696 | no gain |
| Stochastic TTA, 8 views x 2 | 0.159550 | no gain |
| Stochastic TTA, 8 views x 4 | 0.159403 | no gain |
| BatchNorm adaptation, 1 pass | 0.168460 | worse |
| BatchNorm adaptation, 2 passes | 0.176972 | worse |

Stochastic TTA converges to the fixed-view result because the model was trained
with exactly that noise level, so extra noise views average back to the same
prediction. BN adaptation hurts because the per-image local normalisation already
removes most of the domain shift; re-estimating the statistics on 321 target
images only adds estimation noise.

The 16th-place public repository was also read in full. Its four distinguishing
choices were assessed: full Keras ImageDataGenerator augmentation, input resizing,
per-fold learning-rate selection, and random-subspace stacking. The first was
already measured here as strongly negative (affine augmentation 0.295 versus
0.159), resizing would destroy the small bright returns that the `local_clip`
experiment proved are informative, per-fold selection introduces selection bias,
and the remaining one applies to a stacker with far more parameters than ours.

### Round 22: three more borrowed or untested ideas, all negative

The first-place team's published account says their decisive step was finding an
unsupervised grouping in which one block was entirely one class, then retraining
on the other block to remove the resulting bias. That idea had never been tried
here, and this data has the same structure: 132 exact-angle groups are entirely
iceberg, covering 583 rows or 36.4% of the training set.

| Block (exact-angle groups) | Groups | Rows | Share | Label rate |
|---|---:|---:|---:|---:|
| Pure iceberg | 132 | 583 | 36.4% | 1.000 |
| Pure ship | 79 | 210 | 13.1% | 0.000 |
| Mixed | 6 | 17 | 1.1% | — |
| Singleton | 661 | 661 | 41.2% | — |

So a specialist was trained on the complement (about 446 pure-iceberg rows dropped
per fold, determined from training-fold labels only, then 6,200-6,600 pseudo rows
added). It was weaker than the baseline member and did not help:

| Model | Image OOF | Stacked OOF |
|---|---:|---:|
| e40a (full data) | 0.16135 | 0.10162 |
| spec1 (pure-iceberg block removed) | 0.17730 | 0.10474 |

Added to the current best ensemble: 0.098188 → 0.098435. Per-fold, spec1 won on
folds 0 and 2 and lost on 1, 3 and 4. The borrowed idea does not transfer: our
pure-iceberg block is 36% of the data and carries real signal, whereas the
first-place gain came from retraining an entire 100-model suite, not from a single
specialist grafted onto an existing ensemble.

Two other untested knobs were also measured and rejected:

| Change | Result |
|---|---|
| Percentile-clipped local normalisation (`local_clip`) | Fold 0 0.17016 versus 0.15940. Clipping the 1%/99% tail removes the bright return that distinguishes a ship |
| Threshold-0.10 diversity member (`d10`) | Model 0.16458, but the ensemble went 0.098188 → 0.098349 |

### Round 21: a better single model that the ensemble cannot use

The 40-epoch recipe was completed on all five folds (`e40a`) and it is the best
single model this project has produced:

| Model | Image OOF | Stacked OOF |
|---|---:|---:|
| pt25b (best before) | 0.16484 | 0.10075 |
| **e40a (40 epochs, threshold 0.25)** | **0.16135** | **0.10162** |

But swapping it into the ensemble moved the pooled score from 0.098287 to only
0.098188, i.e. 0.0001. Training for twice as long made the member better and the
ensemble no better.

The per-fold detail explains why:

| Fold | 20 epochs | 40 epochs |
|---:|---:|---:|
| 0 | 0.15064 | 0.15940 |
| 1 | 0.16362 | 0.14442 |
| 2 | 0.19229 | 0.19573 |
| 3 | 0.16563 | 0.15624 |
| 4 | 0.15614 | 0.15094 |

Two folds improve, two degrade, and the net gain is small enough that the
ensemble's correlated errors absorb it.

The angle kernel width was re-swept on the current ensemble and 0.012 is still
optimal, so the fusion layer is not the problem:

| sigma | Pooled OOF |
|---:|---:|
| 0.005 | 0.105004 |
| 0.008 | 0.099847 |
| **0.012** | **0.098188** |
| 0.020 | 0.099547 |
| 0.030 | 0.102593 |

A methodological note: `pth25` and `pt25a` are the same configuration and seed but
differ in thread count (5 versus 6), and they scored differently (0.15196 versus
0.15064 on fold 0). CPU thread count is therefore a real source of run-to-run
variance, and comparisons across different thread counts are not exact.

### Round 19: the pseudo-label confidence threshold was far too strict

Every student so far accepted a test row only if its fused probability was below
0.03 or above 0.97, which yields about 3,500 rows out of 8,424. That threshold had
never been varied. Sweeping it on folds 0-1 at the tuned settings (reference
0.16631) shows the threshold was costing most of the available signal:

| Threshold | Pseudo rows | Fold 0 | Fold 1 | Mean |
|---:|---:|---:|---:|---:|
| 0.03 (previous) | ~3,500 | 0.15201 | 0.18061 | 0.16631 |
| 0.10 | 5,267 | 0.15219 | 0.17026 | 0.16122 |
| **0.25** | 6,260 | 0.15196 | 0.16812 | **0.16004** |
| 0.40 | 6,847 | 0.14401 | 0.18235 | 0.16318 |
| 0.49 | 7,165 | 0.14913 | 0.17170 | 0.16042 |

Anything from 0.10 to 0.49 beats 0.03. Soft targets from a reasonable teacher are
useful even at moderate confidence; discarding two thirds of the test images was
the mistake. 0.25 was taken as the operating point.

Five-fold members at threshold 0.25 are the strongest single models produced in
this project:

| Model | Image OOF | Stacked OOF |
|---|---:|---:|
| previous best single (hp_d) | 0.17227 | 0.09998 |
| **pt25b** | **0.16484** | **0.10075** |
| pt25a | 0.16567 | 0.10125 |

Ensemble effect:

| Ensemble | Pooled five-fold OOF |
|---|---:|
| previous best (8 models, threshold 0.03) | 0.099368 |
| swap two weakest tuned members for pt25a/pt25b | **0.098287** |
| four threshold members + shape + stu5 | 0.099561 |
| mixed nine members | 0.098934 |

### Round 14: the hyperparameters had never been tuned

Every model up to this point inherited V3's optimiser settings (learning rate
1.5e-3, batch 64, weight decay 1e-4), and none of those three numbers had ever
been varied. A one-fold screen showed that this was the real weak spot.

Fold 0, distilled student, dropout 0.5 (baseline 0.16323 / 0.07876):

| Setting | Fold-0 model loss | Fold-0 fused |
|---|---:|---:|
| lr 5e-4 | 0.17578 | 0.08163 |
| lr 3e-3 | 0.16132 | 0.07808 |
| weight decay 1e-2 | 0.16085 | 0.07811 |
| batch 32 | 0.15694 | 0.07652 |
| batch 32 + lr 3e-3 | 0.15346 | 0.07536 |
| **batch 32 + lr 3e-3 + weight decay 1e-2** | **0.14961** | **0.07349** |
| lr 5e-3 (same other settings) | 0.14770 | 0.07470 |
| lr 8e-3 | 0.16812 | 0.08383 |
| lr 1.2e-2 | 0.21178 | diverge |

Halving the batch doubles the number of optimiser steps per epoch, and the
higher learning rate and stronger decay both help on top. The model was
**under-optimised**, not over-regularised.

Five-fold confirmation (`hp_star`, batch 32, lr 3e-3, decay 1e-2):

| Model | Image OOF | Stacked OOF |
|---|---:|---:|
| stu1 (inherited settings) | 0.18328 | 0.10492 |
| **hp_star (tuned)** | **0.17279** | **0.10217** |

And the learning rate that looked best on fold 0 (5e-3) did **not** transfer:
its five-fold image loss was 0.18278, worse than 3e-3. That is another instance
of the single-fold trap documented in section 6.

Ensemble effect, all measured with the same nested stacker:

| Ensemble | Pooled five-fold OOF |
|---|---:|
| s101 s202 s404 v5_shape stu1 stu5 (previous best) | 0.101938 |
| swap stu1 → hp_star | 0.100985 |
| + hp_b (second tuned seed) | 0.100226 |
| + hp_d (dropout 0.4, stronger teacher) | 0.099762 |
| **all six tuned members + shape MLP + stu5** | **0.099368** |

That last set is the current best and it **drops the three original CNNs
entirely**. Side by side:

| Ensemble | Pooled five-fold OOF |
|---|---:|
| old base + 3 tuned (8 models) | 0.099762 |
| **tuned only + shape + stu5 (8 models)** | **0.099368** |

The original s101/s202/s404 models score about 0.196 image loss and their
contribution is no longer worth their weight once several tuned members exist.

Further tuning dimensions were then swept on folds 0-1 at a fixed seed
(reference 0.16631) and none improved on the settled recipe:

| Dimension | Result |
|---|---|
| epochs 30 | 0.16553, within noise of 20 |
| pseudo-label weight 0.5 | 0.17280 |
| teacher rebuilt from the tuned 8-model ensemble | 0.17696 (a stronger teacher made a *worse* student) |
| stronger teacher at weight 0.5 | 0.17846 |
| wider widths 24-48-96-128 | 0.16724 |
| narrower widths 12-24-48-72 | 0.17538 |
| SGD momentum 0.9, lr 0.03 | 0.24230 |
| SGD momentum 0.9, lr 0.06 | 0.24775 |
| weight EMA, decay 0.999 | 0.19222 |
| weight EMA, decay 0.995 | 0.16864 |
| weight EMA, decay 0.990 | 0.16891 |

EMA was expected to help and did not: the cosine schedule already anneals the
learning rate to 1% of its peak, so the final weights are effectively averaged.
The 0.999 setting is worse still because its averaging window is longer than the
whole 800-step run, so it is dominated by early weights.

Three further tuned seeds (image loss 0.1764, 0.1772, 0.1780) were then added,
giving 11 models at 0.100065, worse than the 8-model 0.099368. Adding members has
stopped paying: reasonable compositions now span 0.0994 to 0.1001.

Tuning the three numbers that had never been touched moved the pooled score by
**0.00257**, more than everything else in rounds 2-13 combined. Individual tuned
members ranged from 0.1723 to 0.1870 image loss depending on seed, which is why
the ensemble still needs several of them.

Two further knobs were screened at a fixed seed after that, folds 0-1, reference
`hp_d` at 0.16631:

| Setting | Fold 0 | Fold 1 | Mean |
|---|---:|---:|---:|
| dropout 0.4, 20 epochs (reference) | 0.15201 | 0.18061 | **0.16631** |
| dropout 0.3, 20 epochs | 0.15613 | 0.19633 | 0.17623 |
| dropout 0.4, 30 epochs | 0.15226 | 0.17881 | 0.16553 |

Dropout 0.4 is confirmed better than 0.3; 30 epochs is within noise of 20, so the
cheaper schedule stands. The settled recipe is **batch 32, lr 3e-3, weight decay
1e-2, dropout 0.4, 20 epochs, distilled from the six-model teacher**.

Three more tuned seeds were then trained (image loss 0.1749, 0.1775, 0.1820) and
added to the pool, giving 11 models at **0.100270**, worse than the 8-model
0.099368. Ensemble composition is now at the noise floor: reasonable sets land
between 0.0994 and 0.1003, and taking the minimum is itself a selection
statistic. The submission uses the 8-model set at 0.099762, chosen for a concrete
reason (only three tuned members need test inference) rather than as the CV
minimum.

The other lesson from this round is a repeat of section 6: the epoch/dropout
screens were first run at a *different* seed from the reference, and seed
variance (0.172-0.187) swamped the effect being measured. They had to be redone
at a fixed seed.

### Round 5: nested greedy selection, and a leak caught in the harness

Greedy blending is what the first-place team used, so it was implemented with
the selection loop inside each outer training split. Two harnesses were built;
the first was wrong and the error is worth recording.

| Harness | Single-member score | Verdict |
|---|---:|---|
| First attempt (`work/v5_select.py`) | 0.102106 | wrong, should be 0.104918 |
| Calibrated (`work/v5_greedy.py`) | 0.104918 | matches `v5_fuse` exactly |

The first harness fitted the level-2 stacker on the same rows it scored, which
is an in-sample estimate, and that inflated every number by about 0.0028. After
fixing it, the honest result is:

| Rule over 26 available members | Pooled five-fold OOF |
|---|---:|
| Nested greedy selection with replacement | 0.105607 |
| Uniform weights over all 26 | 0.106404 |
| **Curated six-member set (current best)** | **0.101938** |

Greedy selection *lost* to the hand-curated set, and the weights it picked vary
wildly between folds (for example `g50/stu5/stu9/v5_shape` on fold 2 versus
`s202/s404/stu4/v5_shape` on fold 3). On 321-row folds the selection itself
overfits. This is a negative result for the technique but a useful one: the
automated method did not beat the manual choice, so the current ensemble is not
an artefact of hand tuning being left on the table.

### A calibration warning produced by that last experiment

The binned-prior harness scored the Gaussian kernel at 0.099339, which looks
better than this document's honest 0.101938. It is not better: that harness fits
the level-2 combiner on rows whose prior was computed with the held-out fold's
labels in the reference set. The gap, about 0.0026, is the size of the optimism
that the earlier V2 numbers (0.11897, 0.13136) carry for the same reason. Only
`work/v5_fuse.py` keeps the reference set strictly inside the outer training
folds.

### Why the fusion weight cannot be pushed harder

Rows where the image model and the angle prior disagree in direction are 231
rows but carry **50.7%** of the pooled loss. On those rows the image model is
right only **40.7%** of the time while the prior is right **59.3%**. The image
model is systematically wrong exactly where it contradicts the group evidence;
no reweighting of the two signals fixes that.

Loss concentration confirms the same picture: the worst 25 rows carry 43% of
the pooled loss and the worst 100 carry 80%.

## 5. Current best

Six-model ensemble: three mild-augmentation CNNs at seeds 101/202/404, the
shape/statistics MLP, a distilled student (seed 909, dropout 0.5, 20 epochs),
and a mixup student (seed 909, mixup 0.2). Image probabilities are averaged,
then the nested group-wise stacker is applied.

| Rule | Pooled five-fold OOF |
|---|---:|
| Image ensemble only | 0.173983 |
| Fixed V3 fusion (w = 0.75, sigma 0.02) | 0.104053 |
| **Nested group-wise stacking (sigma 0.012, alpha 0.1)** | **0.101938** |

Group split of the same model:

| Group | Rows | Log loss | Share |
|---|---:|---:|---:|
| Exact | 756 | 0.02230 | 10.3% |
| Nearby only | 700 | 0.20375 | 87.2% |
| Far | 15 | 0.22016 | 2.0% |
| Missing | 133 | 0.00542 | 0.4% |

This is a **0.00151 improvement** on the verified 0.103447. The nearby group
moved from 0.2069 to 0.2038 — about 8% of the 0.044 reduction it needs.

A caveat on selection: the six-model set was chosen after comparing several
compositions on the same folds (0.1019 to 0.1028 across reasonable choices), so
the headline number carries a small selection optimism that the nested stacker
does not remove.

The headline number is not an artefact of one implementation. A second,
deliberately simpler code path (`work/v5_independent_check.py`) rebuilds the
priors, refits the stacker and recomputes the loss by hand:

| Path | Pooled OOF |
|---|---:|
| `work/v5_fuse.py` (produces the submission) | 0.101938 |
| `work/v5_independent_check.py` | 0.101938 |
| Difference | 4.45e-07 |

Per-fold losses from the independent path: 0.077, 0.118, 0.112, 0.111, 0.087.

Individual model losses for reference (fixed fusion column is the plain V3 rule
with sigma 0.02 and weight 0.75):

| Tag | Description | Image OOF | Fixed fusion |
|---|---|---:|---:|
| s101 | mild aug, dropout 0.3, seed 101 | 0.19590 | 0.11195 |
| s202 | mild aug, dropout 0.3, seed 202 | 0.19383 | 0.11163 |
| s404 | mild aug, dropout 0.3, seed 404 | 0.19730 | 0.10840 |
| stu1 | distilled student, dropout 0.5, seed 909 | **0.18328** | **0.10302** |
| stu5 | mixup 0.2 student, dropout 0.5 | 0.19953 | 0.11142 |
| stu2 | distilled student, seed 1111 | 0.18673 | 0.10578 |
| stu3 | student, machine rows excluded | — | — |
| stu6 | mixup 0.4 student | fold 0 0.2035, abandoned | — |
| stu7 | global + mixup student | fold 0-2 0.2118/0.2552/0.2112, abandoned | — |
| d50a | dropout 0.5, seed 505 | 0.19780 | 0.11093 |
| d50b | dropout 0.5, seed 707 | 0.19600 | 0.11144 |
| g50 | global normalisation, dropout 0.5 | 0.20021 | 0.10936 |
| v5_shape | shape/statistics MLP, 3 seeds | 0.22749 | 0.11356 |

The mixup student is individually *worse* than the plain student (0.1995 vs
0.1833) yet still improves the ensemble: its errors decorrelate from the other
members. That is the only lever that still pays, and it pays about 0.0005 per
genuinely different member.

## 6. Honest conclusion

- Best verified five-fold score: **0.101938** (previous best 0.103447).
- Target **0.083 is not reached**; nearby rows sit at 0.2038 and would need 0.160.
- Ensembling saturates slowly: members 4, 5, 6 moved the score by 0.002, 0.001
  and 0.001 respectively, and only genuinely decorrelated members helped.
- Rounds 2 and 3 tested, and recorded as negative: machine-row exclusion, an
  image-witness feature, mixup at three strengths, a global-normalisation
  student, affine and mild schedules at 120 epochs, and a distance-binned
  calibrated prior. Only the mixup student helped, and only through
  decorrelation.
- Fold-0 model losses across 25 configurations span 0.159 to 0.295, and the
  ranking there does not survive to the pooled five-fold score. Single-fold
  comparisons are not trustworthy on 321 rows.
- The remaining gap is image-model quality on the 700 nearby rows. Closing it
  needs a different modelling regime (dozens of decorrelated members on GPU, a
  materially stronger architecture, or more aggressive semi-supervised use of
  the test images). It is not reachable by more fusion tuning, more
  regularization tricks, or longer CPU training.

## 7. Submission provenance (required before every upload)

A submission is only accepted as evidence once
`work/verify_submission_provenance.py` passes. It runs four independent checks
and writes `<name>_provenance.json`:

| Check | What it proves |
|---|---|
| A. Manifest hash | The file on disk is the exact file the pipeline wrote |
| B. Regenerate | Re-running the generator from the same inputs reproduces the file **byte for byte**, so the values are not hand-edited |
| C. Checkpoint replay | Every `(tag, fold)` checkpoint is reloaded and re-run on a fixed block of 512 test rows; the cached predictions must match |
| D. Value sanity | Not constant, thousands of distinct values, inside `[0,1]`, and correlated with each member model |

Measured on `v5_mix6.csv` (the file that scored private 0.08944):

| Check | Result |
|---|---|
| A | `6357b994cd4bea775c5a3837353e49f8d2e2fc2b51dd3e3c41286928087586d6`, matches manifest |
| B | Regenerated file is byte-identical, same hash |
| C | 30 checkpoints replayed, maximum absolute difference **0.00e+00** |
| D | 8,424 rows, 7,171 distinct values, range [0.001, 0.999], 14.9% at clip, correlation with members 0.806-0.952 |

Thresholds: check C passes below 2e-6, check D requires more than 1,000 distinct
values and a minimum member correlation above 0.5.

What these checks do **not** cover, and must not be claimed: they do not prove
the checkpoints were trained honestly (that is the job of the fold-isolated CV in
section 3), and they cannot reproduce the Kaggle-side score, which depends on
the hidden test labels.

## 8. What 0.083 actually requires (leaderboard arithmetic)

The earlier projection used cross-validation per-group losses, which understate
the test side. Decomposing the measured private score instead
(`work/probe_lb_decomposition.py`) gives a much tighter target.

Measured on the 3,424 real test rows:

| Quantity | Value |
|---|---:|
| Exact-angle rows | 1,950 (57.0%) |
| Nearby-only rows | 1,428 (41.7%) |
| Measured private score | 0.08944 |
| CV exact-row loss | 0.0223 |
| Implied test near-row loss | **0.184** |
| Near-row loss required for 0.083 | **0.1686** |
| Relative improvement needed on near rows | **8.4%** |

So the gap is an 8.4% relative improvement on 1,428 rows, not the 21.5% implied
by the CV decomposition. The test near rows are already doing better than CV
near rows (0.184 versus 0.204) because the prior uses all 1,604 training labels
and exact coverage is higher.

The pooling intervention touches 2,389 rows in exact-angle groups (mean size
4.46, maximum 36), of which **598 are nearby-only rows, i.e. 41.9% of the target
population**. A 20% loss reduction on exactly those rows would close the whole
8.4%, so this is a testable route rather than a hope.

Caveats: the arithmetic assumes the CV exact-row loss transfers to the test exact
rows, and the private board is a subset of the 3,424 rows, so its exact/near mix
may differ from the shares above.

That assumption was swept rather than trusted (`work/probe_lb_sensitivity.py`):

| Assumed exact-row loss | Implied test near loss | Improvement needed |
|---:|---:|---:|
| 0.010 | 0.2008 | 7.7% |
| 0.015 | 0.1940 | 8.0% |
| 0.0223 (CV value) | 0.1840 | 8.4% |
| 0.030 | 0.1735 | 8.9% |
| 0.040 | 0.1598 | 9.7% |

Across a four-fold range of the assumption the required improvement stays
between **7.7% and 9.7%**. So "8-10% better on the 1,428 nearby rows" is a
robust statement, not a consequence of one guessed input.

Averaging a same-label group of size k reduces the standard deviation of the
mean logit by about the square root of k; the pooling groups average 4.46
members, so a 15-25% local loss reduction is a plausible order of magnitude. That
is why the pooling test is worth one submission: 41.9% coverage times roughly a
20% local gain lands on the 8.4% requirement.

### Why the test side scores better than cross validation

`work/probe_prior_density.py` measures the angle evidence each row actually
receives on the rows that need the prior:

| Quantity (median) | Cross validation | Test | Ratio |
|---|---:|---:|---:|
| Kernel mass | 5.12 | 6.19 | 1.16 |
| Contributing neighbours | 11.6 | 14 | 1.21 |

Test rows get 16-20% more angle evidence than cross-validation rows. That is a
large part of why the measured private score (0.08944) beats cross validation
(0.101938): cross validation is a deliberately pessimistic proxy because every
query row sees four fifths of the labels instead of all of them.

## 9. Pre-registered decision rule for the pooling A/B

Two pooling files exist. The registered test is **`v5_mix6_poolne.csv`**, which
modifies only the 598 nearby-only rows that sit in an exact-angle group and
leaves the 1,950 exact-matched rows untouched. The rationale is design, not
result: the exact rows already score about 0.02, so pooling them cannot help and
can only drag an impure group's minority member toward the majority. Pooling
them would confound the test. Every group mean is still computed over all real
members; only the modified rows are restricted.

`v5_mix6_pool50.csv` (2,389 rows touched) is retained for completeness but is not
the registered test.

The rule is fixed before seeing either score so the result cannot be
rationalised afterwards:

| Outcome on private | Decision |
|---|---|
| Better than 0.08944 by more than 0.0005 | Adopt pooling, and report it as a verified gain |
| Within ±0.0005 of 0.08944 | Inconclusive. Keep the simpler unpooled submission and record pooling as unproven |
| Worse than 0.08944 by more than 0.0005 | Reject pooling and record it in the negative table |

No further pooling variants will be submitted before this rule resolves. Repeated
submissions of re-tuned variants would turn the private score into a selection
statistic and destroy its value as evidence.

## 10. Reproduce

```powershell
$env:PYTHONPATH='.'
.venv\Scripts\python.exe -m src.v3.prepare --raw data/raw
.venv\Scripts\python.exe -m src.v3.train --tag s101 --seed 101 --epochs 40 --threads 8 --validation-only
.venv\Scripts\python.exe -m src.v3.train --tag s202 --seed 202 --epochs 40 --threads 8 --validation-only
.venv\Scripts\python.exe -m src.v3.train --tag s404 --seed 404 --epochs 40 --threads 8 --validation-only
.venv\Scripts\python.exe -m src.v3.shape_model --tag v5_shape
.venv\Scripts\python.exe work\build_teacher.py
.venv\Scripts\python.exe -m src.v3.train --tag stu1 --seed 909 --epochs 20 --dropout 0.5 --teacher v5_teacher --pseudo-weight 0.25 --threads 6 --validation-only
.venv\Scripts\python.exe -m src.v3.train --tag stu5 --seed 909 --epochs 20 --dropout 0.5 --teacher v5_teacher --pseudo-weight 0.25 --mixup 0.2 --threads 6 --validation-only
.venv\Scripts\python.exe work\v5_fuse.py --tags s101 s202 s404 v5_shape stu1 stu5 --name fuse_6final
.venv\Scripts\python.exe work\v5_predict.py --tags s101 s202 s404 v5_shape stu1 stu5
.venv\Scripts\python.exe work\v5_submit.py --tags s101 s202 s404 v5_shape stu1 stu5 --name v5_mix6
.venv\Scripts\python.exe work\validate_submission.py v5_mix6
.venv\Scripts\python.exe work\verify_submission_provenance.py v5_mix6
```

The stacker coefficients are fitted inside each outer training split, so the
reported 0.103074 never sees the held-out fold's labels. The submission script
refits the same coefficients on all 1,604 rows for deployment.
