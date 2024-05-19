import torch
from torch import nn


class ConvolutionModule(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 128,
        dropout: int = 0.1,
    ):
        super(ConvolutionModule, self).__init__()

        self.prenorm = nn.LayerNorm(encoder_dim)
        self.pointwise = nn.Conv1d(
            in_channels=encoder_dim, out_channels=2 * encoder_dim, kernel_size=1
        )
        self.glu = nn.GLU(dim=-2)
        self.depthwise = nn.Conv1d(
            encoder_dim,
            encoder_dim,
            kernel_size=1,
            groups=encoder_dim,  # groups equal to the number of input channels corresponds to deptwhise convolution
        )
        self.batch_norm = nn.BatchNorm1d(encoder_dim)
        self.swish = nn.SiLU()
        self.pointwise2 = nn.Conv1d(
            in_channels=encoder_dim, out_channels=encoder_dim, kernel_size=1
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
