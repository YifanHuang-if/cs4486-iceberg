"""Publish compact, recomputed evidence from completed experiment outputs."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import ROOT, sha256_file
from src.evaluate import log_loss
from src.run_experiments import PRIMARY, GROUPED


def collect(names, output_root, protocol):
    records = []
    for name in names:
        source = output_root / name
        target = ROOT / 'results/models' / protocol / name
        target.mkdir(parents=True, exist_ok=True)
        summary = json.loads((source / 'summary.json').read_text())
        oof = pd.read_csv(source / 'oof.csv', dtype={'id': str})
        assert np.isclose(log_loss(oof.label, oof.probability), summary['log_loss'], atol=1e-12)
        for file in ['summary.json', 'manifest.json', 'oof.csv', 'fold_metrics.csv']:
            shutil.copyfile(source / file, target / file)
        histories = []
        checkpoints = []
        for fold in range(5):
            history = pd.read_csv(source / f'history_fold_{fold}.csv')
            history.insert(0, 'fold', fold)
            histories.append(history)
            checkpoint = source / f'fold_{fold}.pt'
            checkpoints.append({'fold': fold, 'path': str(checkpoint.relative_to(ROOT)).replace('\\', '/'),
                                'sha256': sha256_file(checkpoint), 'bytes': checkpoint.stat().st_size})
        pd.concat(histories, ignore_index=True).to_csv(target / 'history.csv', index=False)
        (target / 'checkpoints.json').write_text(json.dumps(checkpoints, indent=2) + '\n')
        records.append({'experiment_id': name, 'protocol': protocol,
                        'oof_log_loss': summary['log_loss'], 'fold_mean': summary['fold_mean'],
                        'fold_std_population': summary['fold_std_population'],
                        'accuracy': summary['accuracy'],
                        'observed_angle_log_loss': summary['observed_angle_log_loss'],
                        'missing_angle_log_loss': summary['missing_angle_log_loss'],
                        'training_seconds': summary['total_training_seconds']})
    table = pd.DataFrame(records)
    table['delta_vs_E0'] = table.oof_log_loss - table.iloc[0].oof_log_loss
    return table


def markdown_table(frame, columns):
    header = '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---'] * len(columns)) + ' |\n'
    lines = []
    for _, row in frame.iterrows():
        cells = [f'{row[c]:.5f}' if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return header + '\n'.join(lines) + '\n'


def main():
    out = ROOT / 'results/comparison'; out.mkdir(parents=True, exist_ok=True)
    primary = collect(PRIMARY, ROOT / 'outputs/experiments', 'primary')
    grouped = collect(GROUPED, ROOT / 'outputs/grouped', 'angle_grouped')
    primary.to_csv(out / 'primary.csv', index=False)
    grouped.to_csv(out / 'angle_grouped.csv', index=False)
    comparisons = [
        ('Training flips, image model', 'E0_baseline', 'E1_augmentation'),
        ('Dropout and weight decay package', 'E1_augmentation', 'E2_regularized'),
        ('Angle metadata branch', 'E2_regularized', 'E3_angle'),
        ('Missing-angle indicator', 'E3_angle', 'E4_angle_missing'),
        ('Training flips, angle model', 'E5_angle_no_aug', 'E3_angle')]
    ablations = []
    for question, reference, modified in comparisons:
        a = pd.read_csv(ROOT / 'results/models/primary' / reference / 'oof.csv', dtype={'id': str}).set_index('id')
        b = pd.read_csv(ROOT / 'results/models/primary' / modified / 'oof.csv', dtype={'id': str}).set_index('id').loc[a.index]
        assert np.array_equal(a.label, b.label) and np.array_equal(a.fold, b.fold)
        loss_a, loss_b = log_loss(a.label, a.probability), log_loss(b.label, b.probability)
        deltas = [log_loss(b.loc[b.fold == k, 'label'], b.loc[b.fold == k, 'probability']) -
                  log_loss(a.loc[a.fold == k, 'label'], a.loc[a.fold == k, 'probability']) for k in range(5)]
        ablations.append({'comparison': question, 'reference': reference, 'modified': modified,
                          'reference_log_loss': loss_a, 'modified_log_loss': loss_b,
                          'delta_modified_minus_reference': loss_b - loss_a,
                          'folds_with_lower_loss': sum(d < 0 for d in deltas),
                          **{f'delta_fold_{k}': d for k, d in enumerate(deltas)}})
    ablations = pd.DataFrame(ablations)
    ablations.to_csv(out / 'ablations.csv', index=False)

    # An intercept-only training-fold prevalence is a sanity reference, not a fitted image model.
    baseline_oof = pd.read_csv(ROOT / 'results/models/primary/E0_baseline/oof.csv', dtype={'id': str})
    prior = np.empty(len(baseline_oof))
    for k in range(5):
        prior[baseline_oof.fold == k] = baseline_oof.loc[baseline_oof.fold != k, 'label'].mean()
    prior_loss = log_loss(baseline_oof.label, prior)
    (out / 'sanity_reference.json').write_text(json.dumps({'fold_train_prevalence_oof_log_loss': prior_loss}, indent=2) + '\n')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [2, 1]})
    for ax, table, title in zip(axes, [primary, grouped], ['Primary: fixed sample-stratified folds', 'Sensitivity: exact-angle groups']):
        positions = np.arange(len(table))
        ax.bar(positions, table.oof_log_loss, color='#276c9e', alpha=.85)
        for i, row in table.reset_index(drop=True).iterrows():
            s = json.loads((ROOT / 'results/models' / row.protocol / row.experiment_id / 'summary.json').read_text())
            ax.scatter(np.full(5, i), s['fold_log_losses'], color='#d87526', s=13, zorder=3)
            ax.text(i, max(s['fold_log_losses']) + .018, f'{row.oof_log_loss:.3f}', ha='center', fontsize=9)
        ax.set_xticks(positions, [n.split('_')[0] for n in table.experiment_id])
        ax.set_ylim(0, max(table.oof_log_loss.max() + .13, .65))
        ax.set_ylabel('Binary Log Loss (lower is better)'); ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Bars: pooled OOF loss; dots: individual folds. Compare methods within each panel.', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), PRIMARY):
        history = pd.read_csv(ROOT / 'results/models/primary' / name / 'history.csv')
        for fold, group in history.groupby('fold'):
            ax.plot(group.epoch, group.validation_log_loss, lw=1, label=f'Fold {fold}')
            best = group.loc[group.validation_log_loss.idxmin()]
            ax.scatter(best.epoch, best.validation_log_loss, s=12)
        ax.set_title(name, fontsize=10); ax.grid(alpha=.15)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Log Loss')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Primary validation histories (dots mark retained checkpoints)')
    fig.tight_layout(); fig.savefig(out / 'learning_curves.png', dpi=150); plt.close(fig)

    registry = []
    for frame in [primary, grouped]:
        for row in frame.itertuples():
            registry.append({'experiment_id': f'{row.protocol}/{row.experiment_id}',
                             'owner': 'Unassigned (AI-assisted)', 'git_commit': '',
                             'config': f'configs/experiments/{row.experiment_id}.json',
                             'split': 'data/splits/folds.csv' if row.protocol == 'primary' else 'data/splits/folds_angle_grouped_v1.csv',
                             'seed': '42+fold', 'cv_log_loss': row.oof_log_loss,
                             'notes': 'Completed 5 folds; exact source/config hashes in results/models manifest; CPU; no leaderboard score'})
    pd.DataFrame(registry).to_csv(ROOT / 'results/experiments.csv', index=False)

    best = primary.loc[primary.oof_log_loss.idxmin()]
    text = '# Completed experiments: steps 6-8\n\n'
    text += 'Executed on 2026-11-14: six primary configurations and three supplementary grouped configurations, each on all five folds (45 trained models). All scores below are actual saved held-out predictions. No competition test labels were used.\n\n'
    text += '## Primary results\n\n'
    text += markdown_table(primary, ['experiment_id', 'oof_log_loss', 'fold_mean', 'fold_std_population', 'accuracy', 'observed_angle_log_loss', 'missing_angle_log_loss'])
    text += f'\nThe lowest pooled loss in this batch is **{best.experiment_id}: {best.oof_log_loss:.5f}**. This is a provisional development result, not a final test estimate. The training-fold class-prior sanity reference scores {prior_loss:.5f}.\n\n'
    indexed = primary.set_index('experiment_id')
    angle, mask = indexed.loc['E3_angle'], indexed.loc['E4_angle_missing']
    observed_delta = mask.observed_angle_log_loss - angle.observed_angle_log_loss
    missing_delta = mask.missing_angle_log_loss - angle.missing_angle_log_loss
    text += '### Missing-indicator interpretation\n\n'
    text += (f'Adding the missing indicator changes observed-angle loss by {observed_delta:+.5f} '
             f'and missing-angle loss by {missing_delta:+.5f} relative to E3. ')
    if mask.oof_log_loss < angle.oof_log_loss and observed_delta > 0:
        text += ('The overall improvement is driven by the missing-angle subgroup; observed-angle performance worsens. '
                 'Because competition test angles are complete, the lower overall E4 score alone is insufficient to select it for final submission. ')
    best_observed = primary.loc[primary.observed_angle_log_loss.idxmin()]
    text += (f'The lowest primary observed-angle loss in this batch is {best_observed.experiment_id} '
             f'({best_observed.observed_angle_log_loss:.5f}). This subgroup comparison is also development-set selection, not an independent test score.\n\n')
    text += '## Controlled comparisons\n\nNegative delta means the modified model has lower pooled loss.\n\n'
    text += markdown_table(ablations, ['comparison', 'reference', 'modified', 'delta_modified_minus_reference', 'folds_with_lower_loss'])
    text += '\nThese are conditional comparisons under the fixed budget. Dropout and weight decay are one package; angle fusion changes the architecture. The full result table includes modifications that do not improve the score.\n\n'
    text += '## Angle-group sensitivity\n\n'
    text += markdown_table(grouped, ['experiment_id', 'oof_log_loss', 'fold_std_population', 'observed_angle_log_loss', 'missing_angle_log_loss'])
    text += '\nDo not combine these scores with the primary table or treat the raw difference between protocols as a causal leakage estimate. The exact-angle grouping is a proxy, and fold memberships differ. See the audit and experiment plan for its rationale.\n\n'
    text += '## Evidence and limits\n\n'
    text += '- Each `results/models/<protocol>/<experiment>/` contains one OOF probability per training ID, fold metrics, all epoch histories, source/config/environment metadata and checkpoint hashes. Model weights are local artifacts outside Git.\n'
    text += '- Normalization, angle median and scaling are fitted separately on each training fold. Validation/test images are never augmented. Checkpoints include the fitted preprocessing and training/validation IDs.\n'
    text += '- All experiments share a 30-epoch cap and fixed early stopping. Different retained epochs are expected. Fold standard deviation uses ddof=0 and is not uncertainty from repeated training seeds.\n'
    text += '- Several runs retain checkpoints close to the epoch cap. This small CPU budget does not establish convergence or the best achievable performance of each method.\n'
    text += '- Checkpoint selection and method comparison reuse validation labels. Report the resulting model-selection optimism; no independent held-out final test or nested CV has been run.\n'
    text += '- All 133 missing-angle training images are ships, while competition test angles are complete. Observed-angle subgroup loss is relevant, but subgroup results do not establish robustness on unseen data.\n'
    text += '- Detailed error analysis and final test inference/submission remain steps 9-10. No external ranking, course Top-5 result or student contribution hours are invented.\n\n'
    text += '![Primary and supplementary scores](../results/comparison/comparison.png)\n\n![Primary learning curves](../results/comparison/learning_curves.png)\n'
    (ROOT / 'docs/EXPERIMENT_RESULTS.md').write_text(text)
    print(primary.to_string(index=False)); print(grouped.to_string(index=False))


if __name__ == '__main__':
    main()
