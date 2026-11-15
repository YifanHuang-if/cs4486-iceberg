"""Separate historical environment drift from a training-entry-point regression."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import ROOT, file_hash
from .metrics import log_loss


def aligned(path, ids=None):
    frame = pd.read_csv(path, dtype={'id': str}, float_precision='round_trip').set_index('id')
    if frame.index.duplicated().any(): raise ValueError('Duplicate OOF IDs')
    if ids is not None:
        if set(frame.index) != set(ids): raise ValueError('Different OOF IDs')
        frame = frame.loc[ids]
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', type=Path, help='Isolated unchanged train_baseline.py replay: results/baseline_v1 directory')
    args = parser.parse_args()
    published = aligned(ROOT / 'results/baseline_v1/oof_predictions.csv')
    new = aligned(ROOT / 'outputs/steps78/primary/V0_baseline/oof_predictions.csv', published.index)
    assert np.array_equal(published.label, new.label) and np.array_equal(published.fold, new.fold)
    old_loss, new_loss = log_loss(published.label, published.probability), log_loss(new.label, new.probability)
    result = {'published_baseline_loss': old_loss, 'current_environment_V0_loss': new_loss,
              'difference_from_published': new_loss-old_loss,
              'maximum_probability_difference_from_published': float(np.max(np.abs(new.probability-published.probability))),
              'original_script_sha256': file_hash(ROOT / 'src/train_baseline.py'),
              'comparison_rule': 'All V0-V5 comparisons use the current-environment V0; the historical baseline is separate.'}
    if args.reference_dir:
        reference = aligned(args.reference_dir / 'oof_predictions.csv', published.index)
        assert np.array_equal(reference.label, new.label) and np.array_equal(reference.fold, new.fold)
        error = float(np.max(np.abs(new.probability-reference.probability)))
        exact = bool(np.array_equal(new.probability, reference.probability))
        result.update({'unchanged_script_same_environment_loss': log_loss(reference.label, reference.probability),
                       'maximum_probability_difference_from_unchanged_script': error,
                       'bitwise_probability_equality_to_unchanged_script': exact,
                       'same_environment_replay_passed': bool(error <= 1e-7),
                       'reference_environment': {k: json.loads((args.reference_dir/'summary.json').read_text())[k] for k in ['python','torch','device']}})
        if error > 1e-7: raise ValueError('New entry point does not reproduce the unchanged script in the same environment')
    out = ROOT / 'outputs/steps78/baseline_reproduction.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
