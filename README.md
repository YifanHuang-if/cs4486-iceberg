# CS4486 Iceberg Classification

Competition: https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge

## Setup and exploratory analysis

Use Python 3.12 and open this repository in VS Code. Create and activate a local virtual environment, then run:

```powershell
python -m pip install -r requirements-eda.txt
python src/eda.py
```

Download the competition data after accepting its rules. Place train.json, test.json and sample_submission.csv in data/raw/. Results are written to results/eda/. See docs/EDA.md for findings and preprocessing decisions.

## Shared workflow

A owns steps 4-6: data analysis, validation design and the CNN baseline. B and C prepare subsequent improvement experiments. Coordinate shared interfaces and use the same frozen folds for comparisons. Record configurations, results, AI assistance and actual productive hours. Commit and push completed milestones so teammates can pull them.

Raw data, virtual environments and model weights remain local. The course report and presentation each contribute 50 percent. Confirm the Top-5 evaluation interpretation with the instructor.

## CNN training and controlled experiments

The training implementation for steps 6-8 is available in `src/`, with six experiment configurations in `configs/experiments/`. It uses the existing frozen primary folds and training-fold-only preprocessing. A separate exact-angle-group sensitivity split can be generated locally.

See [training instructions](docs/TRAINING_CODE.md). This branch contains code, configurations and tests. Sample-level predictions, newly generated splits, checkpoints and measured experiment artifacts are provided separately; they have not been uploaded in this code-only change.
