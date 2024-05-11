import torch
from torch import nn


class ConvolutionSubsampling(nn.Module):
    def __init__(self, in_channels: int = 128, stride=2, kernel_size=2):
        super(ConvolutionSubsampling, self).__init__()

        # Convolution layer with subsampling via stride
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, x):

        x = self.conv(x.permute((0, -1, 1)))
        x = self.activation(x.permute((0, -1, 1)))
        x = self.layer_norm(x)

        return x
