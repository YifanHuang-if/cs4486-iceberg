# Final version19 source and configuration

This update imports the Python source exported in the team's
`CS4486_final_bundle.zip`, supplied on 23 November 2026. It addresses the mismatch
between GitHub main and that handover. It does not retrain models, replace the
submitted CSV, or claim that the full inference/validation issues are resolved.

## What was synchronized

- All 183 Python source files from the bundle's `05_code/src/` and `05_code/work/`
  are accounted for. Of these, 127 already matched main, two existing files
  changed, and 54 were absent from main. 182 are represented at their normal paths;
  one broken historical document helper is retained as text (see below).
- `work/v5_submit.py` now accepts `--angle-scale` and `--exact-scale`.
  Version19 scales both angle coefficients by **0.90** after fitting.
  Default scaling is still 1.0; it must be explicitly set for version19.
- `src/v3/train.py` includes the exported optional learning-rate warmup. Its
  default is zero, preserving the previous cosine schedule.
- `configs/version19/` retains the supplied member list, fold assignment,
  validation metadata, original submission manifest and source-version note.
  `generation.json` explicitly records the otherwise missing angle-scale setting.
- `source_inventory.json` records the ZIP hash and every exported Python file's
  original and LF-normalized SHA-256. Only CRLF-to-LF conversion was applied to
  the imported Python source; existing unrelated repository files were retained.

The supplied version note names `my-update` / `d097591...` and uncommitted
changes. The export inventory, not that historical commit alone, identifies this
source. Many new `work/` scripts are historical report/presentation helpers with
author-specific paths. They are retained for provenance, not auto-run setup steps.
Their historical narrative claims do not supersede the audited report.

The exported `work/build_slide_companion.py` has an unterminated string literal
at line 22. It is preserved unchanged apart from line-ending normalization as
`docs/version19_archive/build_slide_companion.py.txt`, rather than installed as an
executable Python script. `source_inventory.json` maps its original and published
paths. This helper is not imported by training or submission code. The other
182 imported Python files compile successfully.

## Selected configuration

The seven tags are `v5_shape hp_star hp_b hp_d pt25a pt25b stu5`.
Inference averages five fold predictions per tag, then averages the seven tags.
The final settings are sigma 0.012, alpha 0.1, L2 1.0, clipping 0.001,
angle scale 0.90, and pooling weight 0.0. Real-angle gating remains enabled.
No exact-only override is used.

The supplied `submission_manifest.json` does not serialize angle scale. Its
coefficients and checksum are preserved unchanged; see `generation.json`, the
source-version note and the supplied handover description for the 0.90 setting.
Do not infer this setting from the generator's default value.

## Commands and required inputs

Run from the repository root. Module invocation puts the repository root on
Python's import path:

```sh
python -m work.v5_submit --help
python -m src.v3.train --help
python -m unittest discover -s tests -p test_version19_sync.py -v
```

Once the complete input artifacts are restored, the submission-generation
command for the selected configuration is:

```sh
python -m work.v5_submit --tags v5_shape hp_star hp_b hp_d pt25a pt25b stu5 --name version19_replay --sigma 0.012 --alpha 0.1 --l2 1.0 --clip 0.001 --angle-scale 0.90 --pool-weight 0.0
```

`version19_replay` is a separate output name to avoid overwriting the supplied
submission. This command is documented configuration, **not a claim that it can
currently finish from GitHub alone**. It requires:

- Raw Kaggle files under `data/raw/`, including `sample_submission.csv`.
- Train/test caches under `work/cache/`, with IDs and angles aligned.
- The seven members' five-fold validation prediction files under
  `outputs/v3/<tag>/fold<k>_predictions.npz`.
- Their five-fold test probability arrays under
  `work/predictions/<tag>_fold<k>.npy`.
- The fixed fold file under `data/splits/folds.csv`; its normalized content must
  match the supplied `configs/version19/folds.csv`.
- An existing `work/submissions/` directory.

The original CSV's SHA-256 is
`48c197c8534315158402997d7449fc30b717c0468ff1424dfbe6ded59ea8c524`.
The reported Kaggle score pair is Private **0.08708**, Public **0.10724**.
Neither these scores nor a matching source inventory alone prove numerical replay.

## Remaining reproduction and validation limitations

The supplied handover contains trained checkpoints but omits the individual
member validation/test prediction caches listed above. The imported generic
`v5_predict.py` cannot reconstruct the supplied shape-MLP checkpoint through its
CNN fallback. Restore the original caches or implement and validate the missing
architecture-specific inference path before claiming a complete replay. Raw data
and model weights are not added by this source-only update.

The shared-array prior overwrite and nested-validation concerns documented in
the reviewed report are preserved in this historical source. They are not fixed
by publishing it. The 0.098753 recorded development loss is not certified as
leakage-free CV; 0.096300 in the manifest is a level-2 in-sample value. The official
Kaggle result remains a separate observed score.

Environment versions in the handover are team-reported, not a validated lock
file. Do not equate successful CLI/synthetic checks in the local audit runtime
with a reproduction of the original training environment.

## Report

The reviewed report and its evidence/source package are currently on
[`codex/report-reviewed-20261123`](https://github.com/YifanHuang-if/cs4486-iceberg/tree/codex/report-reviewed-20261123/report).
That report's statement about the earlier main snapshot refers to commit
`638f8e4`; this synchronization is a later source publication, not a rerun.
