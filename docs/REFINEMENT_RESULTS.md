# Current best refinement result

The current best **complete five-fold development Log Loss is 0.131364008783**. The previous V4 was 0.226832367121. This is a 42.1% reduction on the same frozen primary split. **An official Kaggle private Score < 0.083 has not been verified.** Even the development score remains above 0.083; no rank or official score is inferred from CV.

## Measured progression

| Method | Five-fold OOF Log Loss |
|---|---:|
| Previous V4 | 0.22683237 |
| Global CNN | 0.19957376 |
| Local CNN | 0.19427151 |
| Equal CNN mean | 0.18746227 |
| Mean + empirical context | 0.13951653 |
| Mean + empirical odds + query pooling | 0.13136401 |

The underlying models are two 93,425-parameter CNNs trained on the same five folds. They use global versus per-image normalization, radar/angle metadata, paired geometric augmentation, cosine learning-rate decay and eight-view TTA. Their equal-weight mean is the image probability prior. The final context method combines image odds with an exact-angle odds ratio estimated solely from the training fold. Concentration is fitted by training-only empirical Bayes. A fixed 50/50 mixture of individual and same-angle **unlabelled query** probabilities precedes that update.

Query pooling is transductive: predictions depend on the unlabelled samples supplied together. It does not use validation/test labels. Exact training-angle matches exist for 764 of 1,604 primary validation samples. These gains depend partly on related angle groups crossing the primary split; they are not proof of scene-independent generalization.

The eight odds/pooling combinations and six Gaussian-neighborhood experiments are all retained in [the study table](../results/refinement/studies/comparison.csv). Neighborhood smoothing did not beat the best exact-angle method. The first-round table also retains negative CatBoost and frozen-ResNet results (0.276057 and 0.333765). CNN epoch selection and repeated method comparisons reuse development labels; no independent final test was used to choose the best row.

## Separate grouped validation

The local-normalization CNN completed all five exact-angle-group folds at **0.197300901013**, versus previous grouped V4 0.231229971174. The frozen ResNet comparison completed the same grouped protocol at 0.331327423985. Under grouped validation, exact training/query angle overlap is zero and the original count context leaves predictions unchanged. This grouped score belongs to the individual local CNN; the selected two-CNN ensemble has not completed a matching grouped evaluation.

## Further models at this checkpoint

R8_deep has one completed fold and a saved partial second fold. R9_finetuned has implementation and component checks, but no completed task-training fold. These methods are excluded from the current-best full-fold selection. See [exact run status](../results/refinement/run_status.json). The user requested publishing the currently achieved best result, so further architecture training was paused. No incomplete fold is represented as a five-fold score.

## Verification and files

Thirty saved fold models from the completed full configurations were reloaded and their held-out predictions replayed. The implementation test log contains 21 passing checks. All completed comparisons cover the same 1,604 labeled IDs and preserve the original primary mapping. Model/data/source hashes, epoch histories, fold scores, configuration records and verification metadata are included. The current candidate CSV contains 8,424 test IDs and is retained in the companion bundle; its hash and aggregate provenance are versioned. No official submission was made.

Per-sample OOF/test predictions and model/resume weights are distributed privately in the companion bundle, while Git contains code and aggregate evidence. Original baseline code, fold mapping and V-series results are preserved. [Run instructions](REFINEMENT_RUN.md) explain reproduction and resumption; [the second-round plan](REFINEMENT_ROUND2_PLAN.md) records adaptive additions. All AI implementation/execution assistance is disclosed in [AI usage](AI_USAGE.md).
