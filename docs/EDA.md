# Exploratory Data Analysis

## Reproduction

From the repository root, run `python -m pip install -r requirements-eda.txt`, then `python src/eda.py`. The executed environment was Python 3.12.6 on Windows. Package versions are pinned in requirements-eda.txt. The script processes all records in both JSON files and writes English-language outputs to results/eda/.

## Verified findings

- Training set: 1,604 records, including 851 ships (53.05%) and 753 icebergs (46.95%). The class imbalance is modest.
- Test set: 8,424 records. Sample submission IDs exactly match test IDs in order, and its columns are id and is_iceberg.
- Every image has two bands of 5,625 finite values, corresponding to two 75 by 75 channels. All records match the expected schema.
- Training incidence angle is missing in 133 records (8.29%). All 133 are ships. Test incidence angles are complete. This association may create an easy validation shortcut that does not transfer to the test set. Compare observed-angle validation performance separately and test angle fusion with an ablation.
- Observed training angles range from 24.7546 to 45.9375 degrees. Test angles range from 23.0805 to 50.6618 degrees. Test coverage extends beyond training coverage.
- Median per-image band 1 mean is -21.0498 in training and -22.9298 in test. Band 2 values are -26.1531 and -24.9241 respectively. Distribution differences motivate careful validation and robustness checks. They do not identify the cause of the shift.
- No repeated IDs within either dataset, no shared train/test IDs, and no exact duplicate two-band images were found, including across datasets. Hashes use the numeric float64 image values.
- A heuristic search using normalized 15 by 15 two-channel thumbnails identified 182 mutually deduplicated nearest-neighbor candidate pairs at cosine similarity >= 0.95. The algorithm retains one directed nearest neighbor per image and only pairs whose first row index is smaller; it is a screening subset, not an exhaustive search. Similarity does not establish duplicate acquisition. Manual review remains a step-5 validation prerequisite for suspicious cases.

## Figures and tables

- distributions.png: class counts, observed angles and band 1 image-mean distributions.
- missingness_and_band2.png: missing-angle rates and band 2 image-mean distributions.
- representative_images.png: the first ship and first iceberg in file order, with both radar channels. Each panel uses its own 2nd-98th percentile display limits; brightness across panels is not directly comparable.
- summary.json: machine-readable integrity checks and runtime versions.
- sample_metadata.csv: IDs, labels, missingness, image hashes and per-image statistics.
- distribution_summary.csv and angle_by_class.csv: numerical summaries.
- exact_image_duplicates.csv and similar_image_candidates.csv: duplicate diagnostics.

## Decisions and scope

Retain every valid sample and preserve both channels. Keep missing-angle records and a missingness indicator for later fusion. Start with an image-only baseline. The shared preprocessing specification is in DATA_PREPARATION.md. Training-fold normalization and angle imputation will be fitted after folds are established. No model-specific normalization has been fitted on the combined datasets.

This step produces descriptive analysis and preprocessing decisions. It does not establish leaderboard performance or a Top-5 result. The course interpretation of Top-5 remains to be confirmed. All claims above are computed observations or explicitly labeled hypotheses. AI assistance is disclosed in AI_USAGE.md.
