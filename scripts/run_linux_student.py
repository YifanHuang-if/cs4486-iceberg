"""Continue the documented V3 student, waiting for its own supervised fold only."""
import time
from argparse import Namespace
from src.v3.train import OUT, load_data, run_fold
import torch

torch.set_num_threads(2)
args=Namespace(tag='linux_student_s47',seed=47,epochs=15,batch=64,lr=.0003,
    dropout=.3,normalisation='local',augmentation='mild',folds=[0,1,2,3,4],
    folds_file=None,threads=2,device='cpu',teacher='linux_local_s11',
    pseudo_weight=.25,initialize_teacher=True,validation_only=False)
data=load_data()
for fold in args.folds:
    while not (OUT/args.teacher/f'fold{fold}_metrics.json').exists():
        print('Waiting for teacher fold',fold,flush=True);time.sleep(15)
    run_fold(args,data,fold)
