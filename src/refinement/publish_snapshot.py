"""Create the current-best aggregate snapshot without publishing per-sample data."""
import hashlib
import importlib.metadata
import json
import platform
import shutil
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.experiments.data import ROOT


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    output=ROOT/'results/refinement';best=json.loads((output/'current_best.json').read_text());runs=[];records=[]
    for protocol,names in [('primary',['R1_global','R2_local','R3_radar_trees','R6_frozen_resnet','R8_deep','R9_finetuned']),('angle_grouped',['R2_local','R6_frozen_resnet'])]:
        for name in names:
            source=ROOT/f'outputs/refinement/{protocol}/{name}';destination=output/protocol/name
            complete=[];partial=[]
            for fold in range(5):
                path=source/f'metrics_fold_{fold}.json'
                if path.exists():complete.append(fold);records.append({'protocol':protocol,'name':name,**json.loads(path.read_text())})
                elif (source/f'fold_{fold}_resume.pt').exists():
                    checkpoint=torch.load(source/f'fold_{fold}_resume.pt',map_location='cpu',weights_only=True)
                    partial.append({'fold':fold,'last_completed_epoch':checkpoint['epoch'],'best_epoch_so_far':checkpoint['best_epoch']})
                    destination.mkdir(parents=True,exist_ok=True);pd.DataFrame(checkpoint['history']).to_csv(destination/f'partial_history_fold_{fold}.csv',index=False)
            state={'protocol':protocol,'name':name,'completed_folds':complete,'partial_folds':partial,'five_fold_complete':len(complete)==5,
                   'not_started_folds':[k for k in range(5) if k not in complete and k not in [r['fold'] for r in partial]]}
            if len(complete)<5:
                destination.mkdir(parents=True,exist_ok=True)
                if (source/'manifest.json').exists():shutil.copyfile(source/'manifest.json',destination/'manifest.json')
                for fold in complete:
                    for filename in [f'metrics_fold_{fold}.json',f'history_fold_{fold}.csv']:shutil.copyfile(source/filename,destination/filename)
                save(destination/'run_status.json',state)
            else:
                for path in list(destination.glob('partial_history_fold_*.csv'))+[destination/'run_status.json']:
                    if path.exists():path.unlink()
            runs.append(state)
    save(output/'run_status.json',{'status':'Current best publication checkpoint; further model training paused','runs':runs})
    pd.DataFrame(records).to_csv(output/'training_records.csv',index=False)
    verification=[json.loads((output/protocol/'verification.json').read_text()) for protocol in ['primary','angle_grouped']]
    assert sum(v['checkpoints_replayed'] for v in verification)==30
    save(output/'verification.json',{'status':'passed for completed full configurations','checkpoints_replayed':30,'runs':[r for v in verification for r in v['runs']],
                                   'R8_partial_included':False,'R9_started':False})
    shutil.copyfile(ROOT/'outputs/refinement_round2_tests.log',output/'implementation_tests.txt')
    save(output/'environment.json',{'python':platform.python_version(),'device':'cpu','cuda_available':torch.cuda.is_available(),
         'packages':{p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','pandas','scipy','scikit-learn','catboost','ijson','matplotlib']}})
    sources=list((ROOT/'src/refinement').glob('*.py'))+list((ROOT/'configs/refinement').glob('*.json'))+list((ROOT/'tests').glob('test_refinement*.py'))+[ROOT/'requirements-refinement.txt']
    save(output/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)})
    submission=ROOT/'outputs/refinement/submissions/R4_cnn_mean_empirical_odds_pool_submission.json'
    submission_meta=json.loads(submission.read_text()) if submission.exists() else None
    save(output/'target_status.json',{'requested_metric':'Official Kaggle private Score','strict_upper_bound':.083,
         'official_kaggle_score':None,'official_threshold_verified':False,'local_loss_below_0_083':False,
         'selected_candidate':best['candidate'],'development_oof_log_loss':best['development_oof_log_loss'],
         'candidate_submission_created':submission_meta is not None,'submitted_to_kaggle':False})
    if submission_meta:
        save(output/'submission_metadata.json',submission_meta)
    values=[.22683236712121987,.1995737632844357,.194271512749984,.187462266715854,.13951652716561624,best['development_oof_log_loss']]
    labels=['Previous V4','Global CNN','Local CNN','Equal CNN mean','Mean + empirical context','Mean + empirical odds + query pooling']
    fig,ax=plt.subplots(figsize=(10.5,4.5));bars=ax.barh(labels[::-1],values[::-1],color=['#087e8b']+['#64748b']*5)
    ax.bar_label(bars,fmt='%.5f',padding=5);ax.set_xlim(0,.27);ax.set_xlabel('Full five-fold development Log Loss (lower is better)')
    ax.set_title('Current best: 0.13136 — official Kaggle score pending');ax.spines[['top','right']].set_visible(False)
    (output/'figures').mkdir(exist_ok=True);fig.tight_layout();fig.savefig(output/'figures/current_best.png',dpi=160);plt.close(fig)
    grouped=json.loads((output/'angle_grouped/R2_local/summary.json').read_text())['oof_log_loss']
    rows='\n'.join(f'| {label} | {value:.8f} |' for label,value in zip(labels,values))
    generated=f'The current candidate CSV contains {submission_meta["rows"]:,} test IDs and is retained in the companion bundle; its hash and aggregate provenance are versioned.' if submission_meta else 'Candidate test-file generation is not recorded as complete in this snapshot.'
    (ROOT/'docs/REFINEMENT_RESULTS.md').write_text(f'''# Current best refinement result

The current best **complete five-fold development Log Loss is {best['development_oof_log_loss']:.12f}**. The previous V4 was 0.226832367121. This is a {100*(values[0]-values[-1])/values[0]:.1f}% reduction on the same frozen primary split. **An official Kaggle private Score < 0.083 has not been verified.** Even the development score remains above 0.083; no rank or official score is inferred from CV.

## Measured progression

| Method | Five-fold OOF Log Loss |
|---|---:|
{rows}

The underlying models are two 93,425-parameter CNNs trained on the same five folds. They use global versus per-image normalization, radar/angle metadata, paired geometric augmentation, cosine learning-rate decay and eight-view TTA. Their equal-weight mean is the image probability prior. The final context method combines image odds with an exact-angle odds ratio estimated solely from the training fold. Concentration is fitted by training-only empirical Bayes. A fixed 50/50 mixture of individual and same-angle **unlabelled query** probabilities precedes that update.

Query pooling is transductive: predictions depend on the unlabelled samples supplied together. It does not use validation/test labels. Exact training-angle matches exist for 764 of 1,604 primary validation samples. These gains depend partly on related angle groups crossing the primary split; they are not proof of scene-independent generalization.

The eight odds/pooling combinations and six Gaussian-neighborhood experiments are all retained in [the study table](../results/refinement/studies/comparison.csv). Neighborhood smoothing did not beat the best exact-angle method. The first-round table also retains negative CatBoost and frozen-ResNet results (0.276057 and 0.333765). CNN epoch selection and repeated method comparisons reuse development labels; no independent final test was used to choose the best row.

## Separate grouped validation

The local-normalization CNN completed all five exact-angle-group folds at **{grouped:.12f}**, versus previous grouped V4 0.231229971174. The frozen ResNet comparison completed the same grouped protocol at 0.331327423985. Under grouped validation, exact training/query angle overlap is zero and the original count context leaves predictions unchanged. This grouped score belongs to the individual local CNN; the selected two-CNN ensemble has not completed a matching grouped evaluation.

## Further models at this checkpoint

R8_deep has one completed fold and a saved partial second fold. R9_finetuned has implementation and component checks, but no completed task-training fold. These methods are excluded from the current-best full-fold selection. See [exact run status](../results/refinement/run_status.json). The user requested publishing the currently achieved best result, so further architecture training was paused. No incomplete fold is represented as a five-fold score.

## Verification and files

Thirty saved fold models from the completed full configurations were reloaded and their held-out predictions replayed. The implementation test log contains 21 passing checks. All completed comparisons cover the same 1,604 labeled IDs and preserve the original primary mapping. Model/data/source hashes, epoch histories, fold scores, configuration records and verification metadata are included. {generated} No official submission was made.

Per-sample OOF/test predictions and model/resume weights are distributed privately in the companion bundle, while Git contains code and aggregate evidence. Original baseline code, fold mapping and V-series results are preserved. [Run instructions](REFINEMENT_RUN.md) explain reproduction and resumption; [the second-round plan](REFINEMENT_ROUND2_PLAN.md) records adaptive additions. All AI implementation/execution assistance is disclosed in [AI usage](AI_USAGE.md).
''')
    print(json.dumps({'best':best['development_oof_log_loss'],'replayed_checkpoints':30,'submission_created':submission_meta is not None}))


if __name__=='__main__':main()
