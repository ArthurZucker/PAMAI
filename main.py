"""
Auteur: Arthur Zucker

- Capture configuration
- Update with argvs 
- launches training through agent

"""
from __future__ import absolute_import
from __future__ import division
# use wandb instead of runx and logx
import wandb

from agents import *

# trying to replace docopt
from dataclasses import dataclass
from simple_parsing import ArgumentParser

# from apex import amp
# create a parser, as usual
parser = ArgumentParser()


@dataclass
class MyModelHyperParameters:
    """Hyperparameters of MyModel"""

    # Learning rate of the Adam optimizer.
    lr: float = 0.05
    # Momentum of the optimizer.
    momentum: float = 0.01
    # Architecture to choose, available are "denet (to come)", "sincnet (to come)", "leaf (to come)", "yolor (to come)"
    arch: str = "denet"
    #Agent to use, the agent has his own trining loop
    agent: str = "DenetAgent"
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
    # Number of epochs
    epochs: int = 10
    # use adam 
    adam: bool = True

# automatically add arguments for all the fields of the classes above:
parser.add_arguments(MyModelHyperParameters, dest="hparams")


args = parser.parse_args()


# Test Mode run two epochs with a few iterations of training and val
if args.hparams.test_mode:
    args.hparams.max_epoch = 2

def main():

    global args

    print(args)
    run = wandb.init(config=vars(args.hparams), project="pytorch-cnn-fashion")
    print("done")

    config = wandb.config
    # print('Iteration: {0} Loss: {1:.2f} Accuracy: {2:.2f}'.format(iter, loss, accuracy))
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config,run)
    
    agent.run()
    agent.finalize()


if __name__ == "__main__":
    main()
