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
.venv\Scripts\python.exe -m unittest src.v4.test_reliability src.v4.test_groups -v
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

## Continuity correction and focused subgroup audit

The initial V4 implementation scaled only the angle log-odds, while applying the
model log-odds weight of 0.75 regardless of how small the reliability gate was.
That was discontinuous: even nearly zero angle evidence could change a highly
confident image prediction. The corrected method interpolates in log-odds
between the unchanged model probability (gate = 0) and the complete V3
fusion formula (gate = 1). No unverified performance claim is made for this fix.

The V4 evaluator additionally computes four mutually exclusive held-out angle
evidence groups: exact match, nearby only (nearest training angle < 0.06),
farther away, and missing. The output records the count, V3 and V4 subgroup
Log Loss, proportion of total Log Loss, and mean reliability. Evidence groups
are derived solely from training-side angles and held-out angle values,
without using held-out labels to form a group.

Eleven synthetic regression tests passed in the isolated Python/NumPy test
environment, covering fold isolation, zero/near-zero gating, full V3
interpolation, evidence grouping, and invalid parameters. The full evaluator
still requires saved V3 prediction artifacts and raw competition data to
produce an actual Kaggle-related score.

## One-command local evaluation

From a Windows PowerShell terminal at the project root, with the V3 saved
predictions present locally:

```powershell
.\scripts\run_v4.ps1
```

The runner checks `work/cache/train.npz`, `work/cache/test.npz`,
`data/splits/folds.csv`, and all fifteen V3 fold-level prediction files plus
their metrics files, then runs both test suites and the fixed five-fold
comparison. It reports precisely which files are missing rather than
automatically retraining the expensive CNNs. The genuine experiment report is
written to `results/v4/v4_reliability_fixed.json` only after the required
artifacts are available.

## Exact-match-preserving V4 ablation

The observed V3 error breakdown showed much lower loss on exact-angle matches
than on samples supported only by nearby angles. A second V4 variant therefore
uses the **full original V3 fusion at exact-angle matches** (reliability=1)
and the evidence-aware gate only when no exact match is available.

- `v4_reliability_fixed.json`: count-and-distance gating for exact and nearby evidence.
- `v4_exact_preserving.json`: full original fusion for exact matches; gated fusion for neighbors.

The updated `scripts/run_v4.ps1` computes both variants against the **same**
unchanged V3 OOF predictions. This is a hypothesis-driven ablation, not a
verified improvement or an independent holdout evaluation. The extra variant
means choosing the best of the two using the same development folds introduces
additional selection bias. Both scores and subgroup losses must be disclosed.

The local synthetic suite now contains 12 tests, including exact-match
preservation and nearby-angle gating. **No experiment on the actual 1,604
training rows has run in this execution environment.** Both genuine results
remain pending local V3 prediction artifacts.
