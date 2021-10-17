"""
    Most functions are from the github repo of sincNet, simply put in other files for clarity
"""
import torchaudio
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math
import random

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    y=torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])

    return y
    
def get_rnd_audio(filename: str, window_size):
    audio,sr = torchaudio.load(filename)
    print(audio.size())
    begin_sample = random.randint(0,audio.size()[1]-window_size)
    end_sample = begin_sample + window_size
    audio = audio[begin_sample:end_sample]
    return audio,begin_sample,end_sample

def extract_label(labels,begining,end):
    for k in labels[2:]:
        if not np.isnan(k):
            print(f'value of k : {k}')
            if k < end and k>begining:
                return 1 

    return 0