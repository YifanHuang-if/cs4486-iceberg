# CNN baseline

Run `python -m pip install -r requirements-training.txt` in the project environment, then `python src/train_baseline.py` from the repository root. The frozen folds are reused by ID. The script trains five independent CPU models and reloads the best checkpoint from each fold before creating out-of-fold predictions.

The model takes two 75 by 75 radar channels. Three 3x3 convolution blocks use 16, 32 and 64 channels with ReLU and 2x2 max pooling. Adaptive pooling produces 3x3 maps, followed by a 64-unit hidden layer, dropout 0.25 and one binary logit. Training uses BCEWithLogitsLoss and Adam. See configs/baseline.json for the preset configuration. No pretrained parameters are used.

Normalization is fitted separately on the training portion of each fold. Incidence angles and augmentation are reserved for later comparative experiments. Early stopping selects the minimum validation Log Loss within at most 30 epochs, stopping after seven epochs without improvement. This model-selection use of validation makes the reported CV estimate subject to selection optimism.

Local checkpoints are saved under outputs/baseline_v1/. Shared outputs are under results/baseline_v1/, including fold scores, epoch histories, out-of-fold probabilities, subgroup results and learning curves. Checkpoint files are excluded from Git. Repeated runs restore the latest completed epoch including optimizer and RNG states. Completed folds are re-evaluated without retraining. Each resume checkpoint is written through a temporary file and atomically replaced. Configuration or fold-hash mismatches are rejected. Preserve existing outputs and use a separate run directory when changing the experiment. The original interrupted seven-epoch run remains under outputs/baseline_pause/.

The architecture and initial hyperparameters were selected with ChatGPT assistance. Students should review and explain the implementation and disclose this assistance. Cross-device and cross-version bitwise reproducibility is not guaranteed.

References:
- https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- https://docs.pytorch.org/docs/stable/notes/randomness.html

## Measured baseline results

Pooled OOF Log Loss: 0.2463838524. Fold mean: 0.2463796641. Population standard deviation: 0.0228154955. Accuracy at threshold 0.5: 0.8952618454. A training-fold class-prior predictor scores 0.6912813646. The CNN has 60,433 trainable parameters.

Observed-angle subgroup Log Loss: 0.2373989178. Missing-angle subgroup Log Loss: 0.3457585792. Both subgroups were evaluated using the image-only model. Validation curves show fluctuations and a growing training/validation gap, motivating later controlled regularization and augmentation experiments.

Original CPU training took approximately 175 seconds in this execution environment. A completed-checkpoint rerun reproduced the OOF loss exactly and took approximately 3 seconds. elapsed_seconds in summary.json describes the latest invocation, including resume evaluation, rather than cumulative training time.

This is an internal CV baseline. Kaggle leaderboard scoring and the course Top-5 target have not been established. Best-epoch selection uses the same validation folds, so the result is subject to model-selection optimism.
