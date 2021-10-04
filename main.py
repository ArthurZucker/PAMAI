"""
__author__ = "Arthur Zucker & Younes Belkada"
Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent

# @TODO Use dcopt


"""

from __future__ import absolute_import
from __future__ import division

from apex import amp
import argparse
import os
import sys
import time
import torch
from runx.logx import logx
from config import assert_and_infer_cfg, update_epoch, cfg

# Import autoresume module
AutoResume = None
try:
    from userlib.auto_resume import AutoResume
except ImportError:
    print(AutoResume)
    
from agents import *

# @TODO Use dcopt
# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--arch', type=str, default='deepv3.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='hashizume',
                    help='hashizume, Official')
parser.add_argument('--num_workers', type=int, default=8,
                    help='cpu worker threads per dataloader instance')

parser.add_argument('--cv', type=int, default=0,
                    help=('Cross-validation split id to use. Default # of splits set'
                    ' to 3 in config'))
parser.add_argument('--num_classes', type=int, default=0,
                    help='number of classes to use')
parser.add_argument('--sr', type=int, default=0,
                    help='sampling rate to use')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout')

parser.add_argument('--adam', action='store_true', default=False,
                    help='use adam optimizer. If false, SGD will be used')

args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

torch.backends.cudnn.benchmark = True
if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

def check_termination(epoch):
    if AutoResume:
        shouldterminate = AutoResume.termination_requested()
        if shouldterminate:
            if args.global_rank == 0:
                progress = "Progress %d%% (epoch %d of %d)" % (
                    (epoch * 100 / args.max_epoch),
                    epoch,
                    args.max_epoch
                )
                AutoResume.request_resume(
                    user_dict={"RESUME_FILE": logx.save_ckpt_fn,
                        "TENSORBOARD_DIR": args.result_dir,
                        "EPOCH": str(epoch)
                        }, message=progress)
                return 1
            else:
                return 1
    return 0


def main():
    if AutoResume:
        AutoResume.init()

    assert args.result_dir is not None, 'need to define result_dir arg'
    logx.initialize(logdir=args.result_dir,
                    tensorboard=True, hparams=vars(args),
                    global_rank=args.global_rank)

    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    prep_experiment(args) # what does that do? 
    
    
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[args.agent]
	
    agent = agent_class(args)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()