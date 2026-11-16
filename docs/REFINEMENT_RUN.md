# Reproduce the current best and resume further experiments

The selected complete candidate is `R4_cnn_mean_empirical_odds_pool`: full primary five-fold development Log Loss **0.131364008783**, not an official Kaggle score. Official private Score < 0.083 is unverified. Read `REFINEMENT_RESULTS.md` and `results/refinement/run_status.json` before reporting results.

## Environment and data

Use a separate Python 3.12 environment. The execution record uses Python 3.12.14, CPU PyTorch 2.5.1 and torchvision 0.20.1:

```powershell
python -m venv .venv-refinement
.venv-refinement\Scripts\Activate.ps1
python -m pip install -r requirements-refinement.txt
```

Place official `train.json`, `test.json`, and `sample_submission.csv` in `data/raw/`, under the competition rules. Preserve the checked-in primary folds. Extract the companion experiment bundle into the repository root for model weights, OOF predictions and resume state. It includes the old V4 OOF predictions used in the comparison and the grouped fold map, but no raw competition images.

If the grouped map is absent, reproduce it with `python -m src.experiments.grouped`. R6 and R9 download the official public ImageNet-1K ResNet-18 weights on first use. Model-size and source information are recorded in manifests. The dependency file installs CPU builds; CUDA training was not exercised in this round.

## Reproduce completed comparisons

```powershell
python -m unittest discover -s tests -v
python -m src.refinement.run_suite --jobs 1 --grouped
python -m src.refinement.verify
python -m src.refinement.verify --protocol angle_grouped --names R2_local R6_frozen_resnet
python -m src.refinement.current_best
```

`--jobs 2` allows two independent training processes at greater memory/CPU cost. `current_best` reproduces all six neighborhood and eight odds/pooling comparisons from aligned R1/R2 OOF predictions. It does not fit ensemble weights to validation labels. All 1,604 labeled samples contribute once to each full primary result. Negative results are retained.

Source/config/data/fold/runtime signatures protect checkpoint reuse. For exact source hashes, use LF line endings (`git config core.autocrlf false` before cloning). When moving to an incompatible runtime, keep delivered evidence untouched and train in a separate checkout/output directory; do not edit manifests to bypass checks.

## Generate the current-best submission

```powershell
python -m src.refinement.inference --candidate R4_cnn_mean_empirical_context --threads 4
python -m src.refinement.verify_submission
python -m src.refinement.recontextualize
```

The first command streams test images and averages ten retained CNN fold checkpoints with eight-view TTA. The second replays the complete inference route on held-out training records and checks the actual CSV against sample submission IDs and source/model hashes. The third changes postprocessing to the selected empirical odds plus 50/50 query-group pooling.

To avoid repeating all CNN work, `recontextualize` mathematically inverts the first file's known linear context update on unclipped rows. Clipped rows are replayed from the saved CNNs because clipping cannot be inverted. It checks that reconstructing the original context reproduces the original file, retains recovered base probabilities, and pools across the complete unlabelled test query set. No test labels are used.

The selected file is `outputs/refinement/submissions/R4_cnn_mean_empirical_odds_pool_submission.csv`.

Its columns are `id,is_iceberg`; the IDs must exactly match `sample_submission.csv` in order. Sidecar JSON records the hashes and leaves the official score null. Upload the CSV through Kaggle's Late Submission interface if accepted, and preserve the official response including whether it is Public or Private. Only the actual private score strictly below 0.083 verifies the requested target.

## Resume stronger architectures

Further architecture training was paused at the current-best publication checkpoint. R8 has a partial five-fold run; R9 has code/component checks but has not run task training. They are not part of the current best:

```powershell
python -m src.refinement.advanced --name R8_deep
python -m src.refinement.advanced --name R9_finetuned
python -m src.refinement.evaluate_round2
python -m src.refinement.verify_advanced
```

R8 uses a deeper GroupNorm CNN and EMA. R9 fine-tunes only the last residual block of ResNet-18 and the task-specific head, using cached label-independent maps from its frozen prefix. The prefix cache is omitted from the bundle because it can be rebuilt from public weights and official images. `evaluate_round2` refuses to rank these methods until all five folds are complete.

`publish_snapshot` generates this publication checkpoint; its prose describes the current pause and must be updated when a later architecture run finishes. Do not run it blindly after changing the experiment scope.

## Interpretation and disclosure

CNN validation labels select epochs, and repeated development comparisons introduce selection optimism. Exact-angle context uses training-fold labels only. Query pooling is a declared transductive prediction step using unlabelled query probabilities; gains depend on query group composition. The grouped result for R2 is not a grouped estimate for the selected two-CNN ensemble. Read the plans, review the code, and disclose AI assistance accurately in assessed work.
