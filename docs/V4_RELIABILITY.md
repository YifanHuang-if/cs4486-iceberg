# V4 experimental candidate: evidence-aware angle reliability

## Motivation

The published V3 diverse-ensemble error breakdown attributes approximately **86.9% of pooled log loss** to the 697 examples with a nearby training angle but no exact match. Equal full angle-prior weight for exact matches and poorly supported kernel estimates may be excessively confident.

## Proposed modification

This is an unscored experimental candidate, **not an achieved improvement**.

Keep V3 CNN and shape/statistics MLP probabilities and the same frozen five folds. For every held-out fold, derive angle evidence **only from its training labels**. Modify the angle log-odds contribution with a reliability gate:

- Effective sample size of angle weights: `n_eff = sum(weights)^2 / sum(weights^2)`.
- Count-aware gate: `n_eff / (n_eff + tau)`.
- For a query without an exact angle match, additionally multiply by a nearest-angle distance penalty `exp(-0.5*(distance / distance_scale)^2)`.
- Missing angles or negligible evidence leave the original model prediction unchanged.
- Keep the documented model log-odds weight of 0.75.

Fixed diagnostic arguments: `sigma=0.02, alpha=0.1, tau=8, distance_scale=0.02`. These are predefined for comparison, **not** chosen by searching the held-out folds.

## How to reproduce

Run from the project root with the V3 predictions and raw data/cache available locally:

```powershell
.venv\Scripts\python.exe -m unittest src.v4.test_reliability -v
.venv\Scripts\python.exe -m src.v4.evaluate_reliability --tags v3_mild_s11 v3_global_s29 v3_shape_mlp --name v4_reliability_fixed
```

The second command compares against the original V3 rule on identical OOF model probabilities. It saves `results/v4/v4_reliability_fixed.json` with five fold-level losses, subgroup losses and the full pooled OOF Log Loss. No new CNN training, Kaggle submission or model-weight upload occurs.

## Acceptance criteria

1. Confirm 1,604 aligned OOF rows and the frozen split.
2. Run the leakage tests, including mutation of held-out labels.
3. Compare against V3 on the entire five folds and observed-angle subgroup, not only the best fold.
4. If improved, run an exact-angle-grouped robustness experiment and inspect test inference behavior before adopting the rule.
5. Report worsening as well as improvements. Tuning parameters on these OOF results makes the result exploratory development CV.

## Current evidence

The source files and five synthetic regression tests were created and passed in an isolated Python/NumPy test environment. **No competition data or V3 saved model predictions were accessible to this execution environment**, so **there is no measured V4 Log Loss yet**. The best V3 score verifiable in the GitHub branch is 0.103447; the later reported 0.10114 still awaits its final metrics artifact. No verified result is below 0.083 and no Kaggle first-place standing is established.

This candidate intentionally leaves V3 files unchanged.

## AI usage

ChatGPT proposed and implemented this candidate. The team must inspect, understand, run, and document its results; do not claim student hours or independent implementation where they did not occur.
