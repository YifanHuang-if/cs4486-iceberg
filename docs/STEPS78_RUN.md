# Reproduce steps 7-8

These experiments extend `src/train_baseline.py` from the current baseline_v1. The existing baseline script, primary folds and published baseline_v1 outputs remain unchanged. See `STEPS78_PLAN.md` for the preset variants and comparison rules.

## Environment and inputs

Use Python 3.12 in a separate virtual environment and install `requirements-steps78.txt`. The recorded environment uses Python 3.12.14, CPU PyTorch 2.5.1, NumPy 2.3.5 and pandas 2.2.3. The original published baseline records Python 3.12.6 and different dependencies. Numerical results can vary across versions, platforms and CPU kernels; the result report distinguishes the published score from the reproduced baseline. Do not combine the original EDA lock file and this environment's pins.

Place the original `train.json` in `data/raw/`. It must match the frozen SHA-256 in `configs/validation.json`. The steps 7-8 training suite does not read competition test labels or fit preprocessing on test data. It caches parsed training arrays under ignored `outputs/steps78_cache/`.

Windows PowerShell example from the repository root:

```powershell
py -3.12 -m venv .venv-steps78
.\.venv-steps78\Scripts\python.exe -m pip install -r requirements-steps78.txt
.\.venv-steps78\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-steps78\Scripts\python.exe -m src.experiments.run_suite --jobs 1
.\.venv-steps78\Scripts\python.exe -m src.experiments.summarize
.\.venv-steps78\Scripts\python.exe -m src.experiments.verify --checkpoints
```

On Linux, use `python3.12 -m venv .venv-steps78` and `.venv-steps78/bin/python`. `--jobs 2` runs two independent configurations concurrently, each with four PyTorch threads; use it only with sufficient CPU/RAM. Parallelism changes elapsed training time, which is not ranked as a model-quality metric.

To train one configuration, use `python -m src.experiments.train --config configs/steps78/V1_flips.json`. `--only-folds 0` restricts an initial check to one fold. Do not treat a single fold as the full CV score. Each experiment has its own output directory, so one run cannot silently overwrite another's weights.

## Data handling and ablations

The two radar channels share a sample's horizontal/vertical flips; only training batches are augmented. Image normalization uses training-fold channel means/stds and exactly matches the original baseline's reduction/casting convention. Angle medians and scaling also use training-fold samples only. V3 ignores the missing-indicator column; V4 explicitly consumes it. The test suite checks this distinction.

The original backbone consists of three convolution/ReLU/max-pool blocks, 3x3 adaptive average pooling and a 64-unit hidden layer. V0-V2 retain its final single-logit layer. Metadata variants replace that layer with a fusion head for 64 image and 8 metadata features. No pretrained weights are used, and every fold starts from its own seed. Identical shared image initialization, sample-order seeds and an independent augmentation generator reduce avoidable differences between variants.

## Saved results and resume

`outputs/steps78/<protocol>/<variant>/` contains best weights, epoch-resume state, histories, metrics, OOF predictions and a provenance manifest. A resume checkpoint is written atomically after each epoch and restores optimizer, model, dropout RNG, sample-order RNG and augmentation RNG state. The configuration, data/fold hashes, environment and training-source hashes must match. Deliberate changes require a new output root. Temporary prechecks are excluded from scored results.

`src.experiments.summarize` writes compact evidence to `results/steps78/`, updates the experiment register and generates `docs/STEPS78_RESULTS.md`. This code PR contains the implementation and run instructions. Measured results, the experiment register update, per-sample OOF files and model checkpoints are supplied in the separate experiment bundle. Extract that bundle into the matching repository root to restore its output paths, then rerun summarize and verify. Raw data and weights remain excluded from Git.

`src.experiments.verify` recomputes metrics and checks coverage/provenance; `--checkpoints` additionally requires the original training file and best weights, recomputes fold-local preprocessing, and replays held-out predictions. The supplementary angle-group split is generated only if the preset trigger is met; grouped and primary results remain separate.

The companion bundle also contains the unchanged original script's same-environment reference predictions under `outputs/steps78/reference_baseline_v1/`. After restoring the bundle, reproduce the baseline comparison record with `python -m src.experiments.compare_baseline --reference-dir outputs/steps78/reference_baseline_v1`, followed by summarize. The reference script itself is the unchanged `src/train_baseline.py` already in this repository. The bundle contains retained best weights and measured outputs, but omits optimizer resume states; it supports inference replay, not continuation of those completed optimization runs.

The report states negative results and subgroup caveats. These development CV results do not establish a final competition score. Detailed error analysis, final test inference/submission, the assessed report and presentation remain subsequent work. Students should review and understand the code and accurately record AI assistance and their own actual contribution hours.
