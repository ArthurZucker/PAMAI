from dataclasses import dataclass
from simple_parsing import ArgumentParser

@dataclass
class hparams:
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