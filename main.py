"""
Auteur: Arthur Zucker

- Capture configuration
- Update with argvs 
- launches training through agent

"""

from __future__ import absolute_import
from __future__ import division
from docopt import docopt

# trying to replace docopt
from dataclasses import dataclass
from simple_parsing import ArgumentParser

# create a parser, as usual
parser = ArgumentParser()


@dataclass
class MyModelHyperParameters:
    """Hyperparameters of MyModel"""

    # Learning rate of the Adam optimizer.
    lr: float = 0.05
    # Momentum of the optimizer.
    momentum: float = 0.01
    # Architecture to choose
    arch: str = "denet"
    # Dataset used for training
    dataset: str = "egyptian"
    # Number of workers used for the dataloader
    num_workers: int = 8
    # Sampling rate of the raw audio
    sr: int = 24000  # in HZ
    # Probability of dropout
    dropout: float = 0.3
    # Use of deterministic training, setting constant random seed
    deterministic: bool = True
    # Toggle testing mode, which only runs a few epochs and val
    test_mode: bool = True
    # Rank in the cluster
    global_rank: int = 0


@dataclass
class TrainingConfig:
    """Training configuration settings"""

    data_dir: str = "./data"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"


# automatically add arguments for all the fields of the classes above:
parser.add_arguments(MyModelHyperParameters, dest="hparams")
parser.add_arguments(TrainingConfig, dest="config")

args = parser.parse_args()

# Create an instance of each class and populate its values from the command line arguments:
hyperparameters: MyModelHyperParameters = args.hparams
config: TrainingConfig = args.config


class MyModel:
    def __init__(self, hyperparameters: MyModelHyperParameters, config: TrainingConfig):
        # hyperparameters:
        self.hyperparameters = hyperparameters
        # config:
        self.config = config


m = MyModel(hyperparameters, config)

# from apex import amp

import os
import sys
import time
import torch
from runx.logx import logx

# from config import assert_and_infer_cfg, update_epoch, cfg

# Import autoresume module
AutoResume = None
try:
    from userlib.auto_resume import AutoResume
except ImportError:
    print(AutoResume)

from agents import *


args.hparams.best_record = {
    "epoch": -1,
    "iter": 0,
    "val_loss": 1e10,
    "acc": 0,
    "acc_cls": 0,
    "mean_iu": 0,
    "fwavacc": 0,
}

print(args)

torch.backends.cudnn.benchmark = True
if args.hparams.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args.hparams.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.hparams.test_mode:
    args.hparams.max_epoch = 2


def check_termination(epoch):
    if AutoResume:
        shouldterminate = AutoResume.termination_requested()
        if shouldterminate:
            if args.hparams.global_rank == 0:
                progress = "Progress %d%% (epoch %d of %d)" % (
                    (epoch * 100 / args.hparams.max_epoch),
                    epoch,
                    args.hparams.max_epoch,
                )
                AutoResume.request_resume(
                    user_dict={
                        "RESUME_FILE": logx.save_ckpt_fn,
                        "TENSORBOARD_DIR": args.config.log_dir,
                        "EPOCH": str(epoch),
                    },
                    message=progress,
                )
                return 1
            else:
                return 1
    return 0


def main():

    if AutoResume:
        AutoResume.init()

    assert args.config.log_dir is not None, "need to define log_dir arg"
    logx.initialize(
        logdir=args.config.log_dir,
        tensorboard=True,
        hparams=vars(args.hparams),
        global_rank=args.hparams.global_rank,
    )
    print(args)
    exit(0)
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    prep_experiment(args)  # what does that do?

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[args.agent]

    agent = agent_class(args)
    agent.run()
    agent.finalize()


if __name__ == "__main__":
    main()
