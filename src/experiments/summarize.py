"""Build measured comparisons and compact evidence without changing baseline_v1."""
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import ROOT, file_hash
from .metrics import log_loss
from .run_suite import PRIMARY, GROUPED


def collect(names, protocol):
    rows = []
    for name in names:
        source = ROOT / f'outputs/steps78/{protocol}/{name}'
        target = ROOT / f'results/steps78/{protocol}/{name}'
        target.mkdir(parents=True, exist_ok=True)
        summary = json.loads((source/'summary.json').read_text())
        oof = pd.read_csv(source/'oof_predictions.csv', dtype={'id': str}, float_precision='round_trip')
        assert np.isclose(log_loss(oof.label, oof.probability), summary['oof_log_loss'], rtol=0, atol=1e-9)
        for filename in ['summary.json', 'manifest.json', 'oof_predictions.csv', 'fold_scores.csv', 'history.csv']:
            shutil.copyfile(source/filename, target/filename)
        checkpoints = [{'fold': fold, 'path': f'outputs/steps78/{protocol}/{name}/fold_{fold}.pt',
                        'bytes': (source/f'fold_{fold}.pt').stat().st_size,
                        'sha256': file_hash(source/f'fold_{fold}.pt')} for fold in range(5)]
        (target/'checkpoints.json').write_text(json.dumps(checkpoints, indent=2)+'\n')
        rows.append({'experiment': name, 'protocol': protocol,
                     **{k: summary[k] for k in ['oof_log_loss', 'fold_mean', 'fold_std_population', 'accuracy',
                                               'observed_angle_log_loss', 'missing_angle_log_loss', 'parameters', 'training_seconds']}})
    frame = pd.DataFrame(rows)
    frame['delta_vs_protocol_reference'] = frame.oof_log_loss - frame.iloc[0].oof_log_loss
    return frame


def table(frame, columns):
    result = '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    for _, row in frame.iterrows():
        result += '| '+' | '.join(f'{row[c]:.5f}' if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns)+' |\n'
    return result


def main():
    output = ROOT/'results/steps78'; output.mkdir(parents=True, exist_ok=True)
    decision = json.loads((ROOT/'outputs/steps78/sensitivity_decision.json').read_text())
    primary = collect(PRIMARY, 'primary')
    primary.to_csv(output/'primary_comparison.csv', index=False)
    grouped = collect(GROUPED, 'angle_grouped') if decision['triggered'] else None
    if grouped is not None: grouped.to_csv(output/'grouped_comparison.csv', index=False)
    for filename in ['baseline_reproduction.json', 'sensitivity_decision.json', 'grouped_split.json']:
        source = ROOT/'outputs/steps78'/filename
        if source.exists(): shutil.copyfile(source, output/filename)
    contrasts = [('Flips', 'V0_baseline', 'V1_flips'), ('Dropout 0.25 to 0.40', 'V1_flips', 'V2_dropout'),
                 ('Angle branch', 'V1_flips', 'V3_angle'), ('Missing indicator', 'V3_angle', 'V4_missing'),
                 ('Flips in angle model', 'V5_no_flips', 'V3_angle')]
    ablations = []
    indexed = primary.set_index('experiment')
    for question, reference, modified in contrasts:
        a = pd.read_csv(output/f'primary/{reference}/oof_predictions.csv', dtype={'id':str}, float_precision='round_trip').set_index('id')
        b = pd.read_csv(output/f'primary/{modified}/oof_predictions.csv', dtype={'id':str}, float_precision='round_trip').set_index('id').loc[a.index]
        assert np.array_equal(a.label, b.label) and np.array_equal(a.fold, b.fold)
        deltas = [log_loss(b.loc[b.fold == k, 'label'], b.loc[b.fold == k, 'probability'])-
                  log_loss(a.loc[a.fold == k, 'label'], a.loc[a.fold == k, 'probability']) for k in range(5)]
        ablations.append({'comparison': question, 'reference': reference, 'modified': modified,
                          'loss_change': indexed.loc[modified, 'oof_log_loss']-indexed.loc[reference, 'oof_log_loss'],
                          'observed_angle_loss_change': indexed.loc[modified, 'observed_angle_log_loss']-indexed.loc[reference, 'observed_angle_log_loss'],
                          'missing_angle_loss_change': indexed.loc[modified, 'missing_angle_log_loss']-indexed.loc[reference, 'missing_angle_log_loss'],
                          'folds_improved': sum(d < 0 for d in deltas), **{f'delta_fold_{k}': d for k, d in enumerate(deltas)}})
    ablations = pd.DataFrame(ablations); ablations.to_csv(output/'ablations.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, column, title in zip(axes, ['oof_log_loss','observed_angle_log_loss'], ['All OOF samples', 'Observed-angle OOF samples']):
        positions = np.arange(len(primary))
        ax.bar(positions, primary[column], color=['#657786']+['#24769b']*5)
        for i, value in enumerate(primary[column]): ax.text(i, value+.008, f'{value:.3f}', ha='center', fontsize=9)
        ax.set_xticks(positions, [n.split('_')[0] for n in PRIMARY]);ax.set_title(title)
        ax.set_ylim(0, max(primary.oof_log_loss.max(), primary.observed_angle_log_loss.max())+.09)
        ax.set_ylabel('Binary Log Loss (lower is better)');ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Current-environment comparison: V0-V5, fixed primary folds')
    fig.tight_layout();fig.savefig(output/'comparison.png', dpi=150);plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), PRIMARY):
        history = pd.read_csv(output/f'primary/{name}/history.csv')
        for fold, h in history.groupby('fold'):
            color = f'C{fold}'
            ax.plot(h.epoch, h.validation_log_loss, color=color, linewidth=1, label=f'Fold {fold}')
            best = h.loc[h.validation_log_loss.idxmin()]
            ax.scatter(best.epoch, best.validation_log_loss, color=color, s=14)
        ax.set_title(name);ax.set_xlabel('Epoch');ax.set_ylabel('Validation Log Loss');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Validation histories; dots identify retained checkpoints')
    fig.tight_layout();fig.savefig(output/'learning_curves.png', dpi=150);plt.close(fig)

    reproduction = json.loads((output/'baseline_reproduction.json').read_text()) if (output/'baseline_reproduction.json').exists() else None
    best = primary.loc[primary.oof_log_loss.idxmin()]
    report = '# Steps 7-8 measured results\n\n'
    report += 'Executed on 2026-11-15 using the baseline_v1 architecture. These V-series experiments are separate from the older E-series implementation. The preset plan was fixed before scored runs. All listed configurations completed five folds.\n\n'
    report += '## Baseline reproduction and environment\n\n'
    if reproduction:
        report += (f"The historical GitHub baseline scores {reproduction['published_baseline_loss']:.8f}; V0 in the current environment scores {reproduction['current_environment_V0_loss']:.8f}. ")
        if reproduction.get('same_environment_replay_passed'):
            report += (f"An isolated run of the unchanged original train_baseline.py in this same environment scores {reproduction['unchanged_script_same_environment_loss']:.8f}, with maximum per-sample probability difference {reproduction['maximum_probability_difference_from_unchanged_script']:.3g} from V0. ")
        report += 'The historical score was not exactly reproduced here. Runtime/platform numerical differences remain a limitation; the specific component responsible has not been isolated. All claims about improvements below compare methods trained in this same environment. The original baseline_v1 files remain unchanged.\n\n'
    report += '## Primary results\n\n'
    report += table(primary, ['experiment','oof_log_loss','fold_std_population','accuracy','observed_angle_log_loss','missing_angle_log_loss','parameters'])
    report += f"\nLowest pooled primary loss in this batch: **{best.experiment}, {best.oof_log_loss:.5f}**. This is a provisional development result, not a final competition score.\n\n"
    report += '## Controlled comparisons and ablations\n\nNegative loss change favors the modified configuration.\n\n'
    report += table(ablations, ['comparison','reference','modified','loss_change','observed_angle_loss_change','folds_improved'])
    for row in ablations.itertuples():
        direction = 'lower' if row.loss_change < 0 else 'higher'
        report += f'\n- {row.comparison}: pooled loss is {abs(row.loss_change):.5f} {direction}; {row.folds_improved}/5 folds improve.'
    report += '\n\nDropout is the only regularization setting changed in V2; weight decay remains fixed. Angle fusion changes parameter count as well as providing metadata. Comparisons are conditional on the preset architecture/budget and one seed per fold, not universal conclusions or significance tests.\n\n'
    mask = ablations[ablations.modified == 'V4_missing'].iloc[0]
    report += (f"Adding the missing indicator changes observed-angle loss by {mask.observed_angle_loss_change:+.5f} and missing-angle loss by {mask.missing_angle_loss_change:+.5f}. ")
    if mask.loss_change < 0 and mask.observed_angle_loss_change > 0:
        report += 'Its pooled improvement comes from the missing-angle subgroup while observed-angle performance worsens. '
    report += 'All missing-angle training samples are ships and test angles are complete. Therefore pooled loss alone is insufficient for final model selection.\n\n'
    report += '## Conditional angle-group validation\n\n'
    report += f"The predefined trigger was {'met' if decision['triggered'] else 'not met'}: the best angle variant changed primary loss by {decision['delta_best_angle_minus_reference']:+.5f} relative to V1; the threshold was -0.005.\n\n"
    if grouped is not None:
        report += table(grouped, ['experiment','oof_log_loss','fold_std_population','observed_angle_log_loss','missing_angle_log_loss'])
        grouped_index = grouped.set_index('experiment')
        for name in ['V3_angle', 'V4_missing']:
            delta = grouped_index.loc[name, 'oof_log_loss'] - grouped_index.loc['V1_flips', 'oof_log_loss']
            observed_delta = grouped_index.loc[name, 'observed_angle_log_loss'] - grouped_index.loc['V1_flips', 'observed_angle_log_loss']
            report += f'\n- {name} versus V1 on grouped folds: pooled loss change {delta:+.5f}; observed-angle loss change {observed_delta:+.5f}.\n'
        report += '\nThese results use disjoint exact-angle groups (missing angles are singleton groups). Exact angles are only a proxy for acquisition relationships. Compare configurations within this table; do not pool scores with primary CV or interpret a change between splits as a causal estimate of leakage.\n\n'
    report += '## Evidence, scope and handoff\n\n'
    report += '- Each completed run has configurations/source/data/fold/environment hashes, fold scores, complete epoch histories, one OOF probability per training ID and best-checkpoint hashes. Raw data and weights are excluded from Git. Per-sample evidence and weights can be restored from the separate experiment bundle.\n'
    report += '- Audit evidence: [metric and checkpoint verification](../results/steps78/verification.json) and [implementation tests](../results/steps78/tests.json).\n'
    report += '- Normalization, angle imputation and scaling use only training-fold samples. Flips are training-only and aligned across the two channels.\n'
    report += '- Validation labels choose best epochs and compare methods. The resulting selection optimism is not removed by OOF naming. No independent final test or nested CV was run. Fold standard deviation uses ddof=0 and is not a confidence interval.\n'
    report += '- All variants share the 30-epoch cap and patience 7. Some may remain undertrained under this bounded budget. Concurrent execution times are not fair speed benchmarks.\n'
    report += '- Detailed error analysis, final model selection/test prediction, assessed report and slides are subsequent tasks. No final submission or Top-5 claim is made, and no student contribution hours are invented.\n\n'
    report += '![Primary comparison](../results/steps78/comparison.png)\n\n![Validation histories](../results/steps78/learning_curves.png)\n'
    (ROOT/'docs/STEPS78_RESULTS.md').write_text(report)

    register = pd.read_csv(ROOT/'results/experiments.csv')
    new_rows = []
    for frame in [primary] + ([grouped] if grouped is not None else []):
        for row in frame.itertuples():
            new_rows.append({'experiment_id': f'steps78/{row.protocol}/{row.experiment}', 'owner': 'Unassigned (AI-assisted)',
                             'git_commit': '', 'config': f'configs/steps78/{row.experiment}.json',
                             'split': 'data/splits/folds.csv' if row.protocol == 'primary' else 'data/splits/folds_angle_grouped_steps78.csv',
                             'seed': '42+fold', 'cv_log_loss': row.oof_log_loss,
                             'notes': f"5 folds completed; source hashes in per-run manifest; within-protocol {'V0' if row.protocol == 'primary' else 'V1'} reference; no leaderboard score"})
    keep = register[~register.experiment_id.astype(str).str.startswith('steps78/')]
    pd.concat([keep, pd.DataFrame(new_rows)], ignore_index=True).to_csv(ROOT/'results/experiments.csv', index=False)
    print(primary.to_string(index=False))
    if grouped is not None: print(grouped.to_string(index=False))


if __name__ == '__main__': main()
