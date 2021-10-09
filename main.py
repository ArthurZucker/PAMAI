"""
Auteur: Arthur Zucker

- Capture configuration
- Update with argvs 
- launches training through agent

"""
from __future__ import absolute_import
from __future__ import division
from numpy.core.fromnumeric import std
# use wandb instead of runx and logx
import wandb

from agents import *

# trying to replace docopt
from dataclasses import dataclass
from simple_parsing import ArgumentParser

# import hyperparameters
from config.hparams import hparams
from config.hparams import sincnet
# from apex import amp
# create a parser, as usual
parser = ArgumentParser()
# automatically add arguments for all the fields of the classes in hparams:
parser.add_arguments(hparams, dest="hparams")
parser.add_arguments(sincnet, dest="sincnet_param")
# parse arguments
args = parser.parse_args()
# Test Mode run two epochs with a few iterations of training and val
if args.hparams.test_mode:
    args.hparams.max_epoch = 2

def main():
    # initialize wandb instance 
    run = wandb.init(config=vars(args.hparams), project="DENET-sweep run (testing code)")
    config = wandb.config
    print("INITIALIZED WANDB")
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config,run)
    # run the model
    agent.run()
    agent.finalize()


if __name__ == "__main__":
    main()
