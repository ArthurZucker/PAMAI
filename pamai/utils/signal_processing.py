"""
    Most functions are from the github repo of sincNet, simply put in other files for clarity
"""
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.autograd import Variable


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :, getattr(torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda])().long(), :
    ]
    return x.view(xsize)


def sinc(band, t_right):
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = flip(y_right, 0)

    y = torch.cat([y_left, Variable(torch.ones(1)).cuda(), y_right])

    return y


def get_rnd_audio(filename: str, window_size):
    audio, sr = torchaudio.load(filename)

    begin_sample = random.randint(0, (audio.shape[1] - window_size))
    end_sample = begin_sample + window_size
    # print(f'begin and start samples randomly selected : {begin_sample},{end_sample}')
    audio = audio[:, begin_sample:end_sample]
    return audio, begin_sample, end_sample


def extract_label_bat(labels, begining, end):
    # for each 50ms frames, give a single label. Only if it is actually inside a call

    pairs = [[i, j] for i, j in zip(labels[6::2], labels[7::2])]
    val = 0
    for b, e in pairs:
        if begining > b and end < (e + (end - begining) / 2):
            return 1  # 1 is "bat call", 0 is other

    return val
