import torch
from torch import nn


class ConvolutionModule(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        dropout: int = 0.1,
    ):
        super(ConvolutionModule, self).__init__()

        self.prenorm = nn.LayerNorm(in_channels)
        self.pointwise = nn.Conv1d(
            in_channels=in_channels, out_channels=2 * in_channels, kernel_size=1
        )
        self.glu = nn.GLU(dim=-2)
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=1,
            groups=in_channels,  # groups equal to the number of input channels corresponds to deptwhise convolution
        )
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.swish = nn.SiLU()
        self.pointwise2 = nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x_forward = self.prenorm(x)
        x_forward = self.pointwise(x_forward.permute((0, -1, 1)))
        x_forward = self.glu(x_forward)  # here channels are reduced by 2
        x_forward = self.depthwise(x_forward)
        x_forward = self.batch_norm(x_forward)
        x_forward = self.swish(x_forward)
        x_forward = self.pointwise2(x_forward)
        x_forward = self.dropout(x_forward).permute((0, -1, 1))
        return x + x_forward
