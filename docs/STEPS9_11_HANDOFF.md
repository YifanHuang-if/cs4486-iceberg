# Steps 9-11: analysis, final prediction and report draft

**Status: work in progress / not ready to merge as a final submission.** This handoff shares the current implementation and evidence so the group can continue work. Online scoring is complete, but the historical Top-5 score level has not been reached and the report still contains pending personal/team fields.

Source snapshot: `f8c9ff017fe8a818784b5e8e02791543f3a31c2e`.
Experiment bundle training code: `c8ccc8c528a82c3911a93c186f4fcd87dc1149b7`.
The bundle contains all six primary five-fold runs and three supplementary angle-group runs.
The original training source, folds and experimental results are preserved.

## Step 9: completed local analysis

- All 308 entries in the supplied manifest passed size/hash validation. All 45 model files passed their individual hash checks.
- All nine OOF prediction sets contain 1,604 unique IDs with matching labels/folds/missingness. Their pooled Log Loss values reproduce the summaries.
- All 45 models were replayed on held-out images in a local Windows CPU environment. Maximum absolute probability differences are below 2e-6, including the difference caused by changing PyTorch versions. Exact measurements and the runtime are in `results/final/handoff_verification.json`.
- `results/final/subgroup_metrics.csv` includes class, angle-availability and angle-range results, with sample counts. It includes primary and grouped protocols separately.
- `results/final/error_summary.json`: selected V4 makes 147 classification errors, including 25 high-confidence errors. There are 81 false icebergs and 66 missed icebergs. Compared with V0, it corrects 84 errors and introduces 51.
- `outputs/final/oof_error_details.csv` and `top20_errors.csv` contain individual evidence. Keep these local or share through the experiment bundle, consistent with the repository's current handling of per-sample results.
- Figures: `error_gallery.png`, `confusion_calibration.png` and `angle_robustness.png` under `results/final/`.

The 133 missing-angle examples are all ships and all classified correctly by V4. Its observed-angle loss is 0.24584, compared with its overall loss of 0.22683. The [40,50) angle group has loss 0.29867 (640 examples). The [20,30) interval has only one example and cannot support a reliable robustness conclusion. These are descriptive post-selection findings, not causal effects or a new independent test.

## Step 10: selected model and generated submission

V4_missing has the lowest observed-angle OOF Log Loss among the six primary configurations and is supported by the supplementary grouped comparison. `results/final/model_selection.json` records the decision, caveats, configuration and exact checkpoint hashes.

`outputs/final/submission.csv` contains 8,424 test IDs and their iceberg probabilities, in sample-submission order. Each row is the equal mean of five primary V4 model probabilities. Each fold uses its saved training-fold preprocessing. There is no TTA, calibrated transform or test-driven tuning. Individual fold predictions are retained in `outputs/final/test_fold_predictions.csv`.

`results/final/submission_manifest.json` records file hashes and checks. The test JSON is processed in batches, rather than expanded into a large Python list. All rows are retained, including any artificial images identified only by the competition organizer.

The OOF score evaluates one held-out model per training example. It is not a measured score for the five-model test ensemble. On 2026-11-16, the user supplied a completed Kaggle late-submission receipt: **Public Log Loss 0.21791; Private Log Loss 0.20895**. Its filename and description identify the prepared V4 five-fold mean-probability submission. The screenshot does not expose an uploaded-file hash or submission ID. Evidence and its transcription are saved as `results/final/kaggle_submission_20261116.png` and `kaggle_score_receipt.json`.

The user also supplied historical public and private leaderboard screenshots. The fifth-place public score is 0.08805 (beluga), and the fifth-place private score is 0.08883 (Kohei and Medrr). The corresponding excess losses are 0.12986 and 0.12012, so the historical Top-5 score level **has not been reached**. No official late-submission rank is claimed. The instructor's treatment of late submissions is still a course-grading question. Successful inference and scoring do not mean the performance target has been achieved.

The score receipt is a separate record of a manual platform submission. Re-running the analysis/inference script does not submit to Kaggle or reproduce the remote score; its generated metadata may reset its original pre-submission status. Keep `kaggle_score_receipt.json` and the screenshots as the scoring evidence and associate them only with the recorded local submission hash.

## Reproduce

Use a separate environment with `requirements-final.txt`, restore the source snapshot and experiment bundle, and put the original competition files under `data/raw/`.

```powershell
python -m unittest discover -s tests -p test_finalize.py -v
python -m src.finalize --predict
```

The continuation ran on the already-installed Python 3.11 interpreter with PyTorch 2.9.1+cpu, NumPy 2.2.1, pandas 2.2.3 and Matplotlib 3.10.0. It did not repair or replace the default `python` command. Four CPU threads are used. Source/hash checks prevent silent use of unrelated training code, data or weights.

On the current computer, the verified interpreter can be invoked explicitly from the repository root:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m src.finalize --predict
```

## Step 11: report draft and required completion

`report/CS4486_Iceberg_Report_Draft.tex` follows the course's IEEEtran journal template and all eleven section headings. It integrates the team's measured EDA, validation, baseline, improvements, error analysis and final inference. It accurately discloses the recorded AI assistance.

`report/CS4486_Iceberg_Report_Draft.pdf` is the compiled draft, with the subsequent Kaggle scores, archived receipt and same-board historical Top-5 comparison incorporated. Rendered pages are visually checked against the supplied course PDF template; the current page count and PDF hash are recorded in `results/final/delivery_checks.json`. Personal-information fields remain pending. The source can be compiled from `report/` with a LaTeX installation supporting IEEEtran; the local build used Tectonic 0.17.0.

`Steps9-11_Deliverables_Draft.zip` in the parent project folder contains the new analysis code, focused tests, dependency pins, handoff notes, final results, report sources/figures/PDF, and prediction CSVs. It is an add-on to the matching repository and original experiment bundle; raw data and model weights are not duplicated in it. Preserve the relative paths when copying it to a teammate's working copy.

Still required from the team:

- Team name. Member identities, roles and total productive hours have been supplied and are recorded below.
- Each member's biography, genuine reflection, specific independent design/implementation elements, and allocation of their reported total hours across the course-template work categories.
- Leader review and signature after the contribution table is completed.
- Instructor's accepted treatment of late submissions and any additional submission instructions. Actual Kaggle scores and historical fifth-place screenshots have been supplied; the historical score-level target is not met.
- Student review of all AI-assisted implementation and prose against course requirements. No personal understanding or independent authorship is inferred by this draft.

Member information supplied on 2026-11-15 and included in the PDF/source:

| Role | Name | Student ID | Reported actual productive hours | Responsible steps |
| --- | --- | --- | ---: | --- |
| Leader | Huang Yifan | 72400438 | 8 | 4, 5, 6: EDA, fixed validation splits, CNN baseline |
| Member A | Zheng Hongyu | 72400229 | 6 | 9, 10, 11: error/robustness analysis, final model and predictions, report |
| Member B | Hachiware | 72400463 | 7 | 7, 8: improvements, comparison and ablations |
| Total | | | 21 | |

The supplied totals are retained exactly; category-level hours have not been invented or increased for later activity. The leader's name is filled in the attestation, but the signature remains blank. Step 10's selection, inference, submission and online-scoring workflow is complete. Meeting the historical Top-5 performance target remains outstanding and would require further successful model-development work; no particular change is guaranteed to achieve it.

The original local folder remains a downloaded working copy rather than a full Git clone. This draft is shared separately on branch `codex/steps9-11-draft-20261116` for team review, with `main` unchanged until the group chooses to integrate it. Four small prediction/error CSV artifacts under `outputs/final/` are explicitly included for this handoff; other generated outputs, raw data, model weights, archives and local environments remain excluded.
