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
