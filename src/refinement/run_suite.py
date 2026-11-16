"""Reproduce the primary batch and optional grouped checks without altering old runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from src.experiments.data import ROOT,load_train
from src.refinement.features import training_features


def run(name,arguments,protocol):
    logs=ROOT/'outputs/refinement/logs';logs.mkdir(parents=True,exist_ok=True)
    print('START',protocol,name,flush=True)
    with (logs/f'{protocol}_{name}.log').open('a') as handle:
        subprocess.run([sys.executable]+arguments+['--protocol',protocol],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE',protocol,name,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,choices=[1,2],default=1);parser.add_argument('--grouped',action='store_true');args=parser.parse_args()
    training_features(load_train())
    tasks=[('R3_radar_trees',['-m','src.refinement.trees']),('R6_frozen_resnet',['-m','src.refinement.transfer']),
           ('R1_global',['-m','src.refinement.train','--config','configs/refinement/R1_global.json']),
           ('R2_local',['-m','src.refinement.train','--config','configs/refinement/R2_local.json'])]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(run,name,arguments,'primary') for name,arguments in tasks]
        for future in futures:future.result()
    subprocess.run([sys.executable,'-m','src.refinement.evaluate'],cwd=ROOT,check=True)
    if args.grouped:
        from src.experiments.grouped import create_grouped_split
        create_grouped_split()
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures=[pool.submit(run,name,arguments,'angle_grouped') for name,arguments in tasks if name in ['R2_local','R6_frozen_resnet']]
            for future in futures:future.result()
        subprocess.run([sys.executable,'-m','src.refinement.evaluate','--protocol','angle_grouped','--names','R2_local','R6_frozen_resnet'],cwd=ROOT,check=True)


if __name__=='__main__':main()
