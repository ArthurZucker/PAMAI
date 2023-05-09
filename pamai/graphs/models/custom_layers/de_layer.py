from torch._C import memory_format
from torch.nn.functional import layer_norm
import torch
from pamai.graphs.models.custom_layers.MLP_sincnet import MLP
import torch.nn as nn
class DELayer(nn.Module):
    def __init__(self,input_dim,input_channels, data_format="channels_last", sum_channels=True, dropout=0.0, **kwargs):
            super(DELayer, self).__init__(**kwargs)
            self.input_dim = input_dim
            self.input_channels = input_channels
            self.data_format    = data_format
            self.sum_channels   = sum_channels
            self.dropout_rate   = dropout
            self._mlp           = self._get_mlp()
            if self.sum_channels:
                self._lambda_sum = lambda x: torch.sum(x,axis=1,keepdim=True) # keep dim and other params are supposed to be here
    
    def forward(self,x):
        self.out = []
        for i in range(self.input_channels):
            # For each feature map
            feature_map = torch.unsqueeze(x[:, i, :], axis=1)
            # Apply Attention Branch
            self.out.append(self._mlp(feature_map))
            del feature_map
        merge = torch.cat(self.out, axis=-1)
        # Matrix of shape (batch_size, #channels)
        weights = nn.Softmax(dim=1)(merge).unsqueeze(dim=2)
        # Dropout regularization
        if self.train() == 1 and self.dropout_rate != 0.0 : #if training
          weights = nn.dropout(weights, rate=self.dropout_rate)
        # Channel weighting
        refined_features = x*weights
        if self.sum_channels:
          refined_features = self._lambda_sum(refined_features)
        return refined_features

    

    def _get_mlp(self):
        # computes channel wise attention weight
        dim0 = ((self.input_dim-7)/2)+1
        dim1 = int(((dim0-7)/2)+1)
        dim2 = int(((dim1-7)/2)+1)
        hidd_layer = nn.Sequential(
            nn.Conv1d(1,30,kernel_size=7,stride = 2),nn.ReLU(),
            nn.Conv1d(30,30,kernel_size=7,stride = 2),nn.ReLU(),
            nn.Conv1d(30,10,kernel_size=7,stride = 2),nn.ReLU(),
            nn.Flatten(),
            nn.Linear(dim2*10,128),nn.ReLU(),
            nn.Linear(128,64),nn.ReLU(),
            nn.Linear(64,1),nn.ReLU()

        )
        # channel last is used for the first conv layers in the original code
        # but I can't find the equivalent
        # hidd_layer = hidd_layer.to(memory_format = torch.channels_last)
        return hidd_layer


class Print(nn.Module):
        def __init__(self):
            super(Print, self).__init__()

        def forward(self, x):
            print(x.shape)
            return x