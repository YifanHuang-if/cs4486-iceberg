"""Run the declared primary comparisons and optional angle-group sensitivity checks."""
import argparse
from src.data import ROOT
from src.train import run

PRIMARY = ['E0_baseline', 'E1_augmentation', 'E2_regularized', 'E3_angle',
           'E4_angle_missing', 'E5_angle_no_aug']
GROUPED = ['E0_baseline', 'E3_angle', 'E4_angle_missing']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', choices=['primary', 'grouped', 'all'], default='all')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.protocol in ('primary', 'all'):
        for name in PRIMARY:
            run(ROOT / f'configs/experiments/{name}.json', threads=args.threads)
    if args.protocol in ('grouped', 'all'):
        for name in GROUPED:
            run(ROOT / f'configs/experiments/{name}.json',
                folds_path=ROOT / 'data/splits/folds_angle_grouped_v1.csv',
                output_root=ROOT / 'outputs/grouped', threads=args.threads)


if __name__ == '__main__':
    main()
