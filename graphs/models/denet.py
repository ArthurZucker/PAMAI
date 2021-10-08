"""
An example for the model class
"""
import torch.nn as nn

from graphs.weights_initializer import weights_init

class Denet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


        # initialize weights
        # self.apply(weights_init)

    def forward(self, x):
        return x