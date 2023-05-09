"""
An example for the model class
"""
import torch
import torch.nn as nn
from pamai.graphs.models.custom_layers.de_layer import DELayer
from pamai.graphs.models.custom_layers.sinc_conv import SincConv_fast


class Denet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sinc_conv1 = SincConv_fast(80, 251, self.config.sr)
        self.sinc_dim_out = self.config.input_dim - 251
        # dirty trick to compute dimension
        self.att_dim_out = int(
            self.sinc_dim_out * ([(1 / 3) - (5 / (3 * self.sinc_dim_out)), 1][not self.config.DenetBeforePooling])
        )
        self.attention_layer = DELayer(
            self.sinc_dim_out, input_channels=80, sum_channels=self.config.SumChannel, dropout=config.DenetDropout
        )
        # Time distributed is used in tf, but useless in torch
        self.conv1 = self.get_conv(self.att_dim_out, [80, 1][self.config.SumChannel])  # sinc conv outputs 80 channels
        self.conv1_dim_out = (self.att_dim_out - 5) // 3
        self.conv2 = self.get_conv(self.conv1_dim_out, 60)  # first conv has 60 filters
        self.conv2_dim_out = 1 + (self.conv1_dim_out - 5) // 3
        self.dropout = nn.Dropout(self.config.DenetDropout)
        # input dimension -sinc_kernel_filter size - 2xkernel dimension of convolution which is 5
        self.gru_dim_in = self.conv2_dim_out
        self.gru = nn.GRU(self.gru_dim_in, 1024, bidirectional=True, dropout=self.config.DenetDropout, batch_first=True)
        self.mlp1 = nn.Sequential(nn.Linear(2048 * 60, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU())
        self.mlp2 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.LeakyReLU())
        self.fcn = nn.Linear(512, self.config.nb_classes)
        self.softmax = nn.Softmax(dim=1)

    
    def get_conv(self, input_dim, input_channels, out_channels=60):
        return nn.Sequential(
            nn.Conv1d(input_channels, out_channels, kernel_size=5),
            nn.MaxPool1d(3),
            nn.LayerNorm([out_channels, ((input_dim - 5) // 3) + 1]),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.sinc_conv1(x)

        if self.config.DenetBeforePooling:
            x = self.attention_layer(x)

        x = nn.MaxPool1d(kernel_size=3)(x)

        x = nn.LeakyReLU()(x)
        if not self.config.DenetBeforePooling:
            x = self.attention_layer(x)

        x = self.conv1(x)

        # equivalent of SpatialDropout in tf : (not used for now)
        # x = x.permute(0, 2, 1)   # convert to [batch, channels, time]
        # x = F.dropout2d(x, p, training=self.training)
        # x = x.permute(0, 2, 1)   # back to [batch, time, channels]
        if self.config.DenetDropout != 0:
            x = self.dropout(x)
        x = self.conv2(x)

        if self.config.DenetDropout != 0:
            x = self.dropout(x)

        x, _ = self.gru(x)
        x = nn.Flatten()(x)
        x = self.mlp1(x)
        if self.config.DenetDropout != 0:
            x = self.dropout(x)
        x = self.mlp2(x)
        if self.config.DenetDropout != 0:
            x = self.dropout(x)
        x = self.softmax(self.fcn(x))

        return x
