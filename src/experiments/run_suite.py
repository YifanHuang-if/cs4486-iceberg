"""Run independent configurations, then the predeclared conditional sensitivity check."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
from .data import ROOT, load_train
from .grouped import create_grouped_split

PRIMARY = ['V0_baseline', 'V1_flips', 'V2_dropout', 'V3_angle', 'V4_missing', 'V5_no_flips']
GROUPED = ['V1_flips', 'V3_angle', 'V4_missing']


def launch(name, protocol='primary', folds=None):
    log = ROOT / 'outputs/steps78/logs' / f'{protocol}_{name}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'src.experiments.train', '--config', str(ROOT / f'configs/steps78/{name}.json'), '--protocol', protocol]
    if folds is not None: command += ['--folds', str(folds)]
    print('START', protocol, name, flush=True)
    with log.open('a') as handle:
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    summary = json.loads((ROOT / f'outputs/steps78/{protocol}/{name}/summary.json').read_text())
    print('COMPLETE', protocol, name, f"loss={summary['oof_log_loss']:.6f}", flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--jobs', type=int, default=1)
    args = p.parse_args()
    if args.jobs not in (1, 2): raise ValueError('Use one or two independent worker processes')
    load_train()  # Prepare the shared read-only cache before concurrent workers.
    launch(PRIMARY[0])
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(launch, PRIMARY[1:]))
    summaries = {name: json.loads((ROOT / f'outputs/steps78/primary/{name}/summary.json').read_text()) for name in PRIMARY}
    reference = summaries['V1_flips']['oof_log_loss']
    delta = min(summaries[n]['oof_log_loss'] for n in ['V3_angle', 'V4_missing']) - reference
    triggered = delta <= -0.005
    decision = {'threshold_reduction': 0.005, 'reference': 'V1_flips', 'delta_best_angle_minus_reference': delta,
                'triggered': triggered, 'selected_grouped_configurations': GROUPED if triggered else [],
                'rule': 'Repeat V1,V3,V4 on angle-grouped folds if either V3 or V4 improves primary loss over V1 by at least 0.005.'}
    (ROOT / 'outputs/steps78/sensitivity_decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps(decision), flush=True)
    if triggered:
        folds = create_grouped_split()
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(launch, name, 'angle_grouped', folds) for name in GROUPED]
            for future in futures: future.result()


if __name__ == '__main__': main()
