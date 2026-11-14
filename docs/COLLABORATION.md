# Collaboration

A owns exploratory analysis, preprocessing, fixed validation splits and the baseline CNN (steps 4-6). B and C have not started implementation; their later experiment assignments should be agreed before editing shared code.

1. Fetch and pull the latest repository state before starting.
2. Coordinate ownership of shared files. Use separate branches for concurrent experiments.
3. Commit and push meaningful completed milestones.
4. Review changes before integrating concurrent work.
5. Use identical saved folds and metric definitions for comparative experiments.

Fit normalization and imputation on each training fold only. Apply augmentation only to training samples. Keep related images together when a verified relationship requires grouped validation. Document random seeds, dependencies, configurations and results.

All project-authored files, comments, figures and reports use English. Each teammate downloads raw competition data separately. Do not commit credentials, environments or model weights.
