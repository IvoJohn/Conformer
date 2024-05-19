from torch import nn


class ConvolutionSubsampling(nn.Module):
    def __init__(
        self, in_channels: int = 80, encoder_dim: int = 128, stride=2, kernel_size=2
    ):
        super(ConvolutionSubsampling, self).__init__()

        # Convolution layer with subsampling via stride
        self.conv = nn.Conv1d(
            in_channels,
            encoder_dim,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(encoder_dim)

    def forward(self, x):

        x = self.conv(x.permute((0, -1, 1)))
        x = self.activation(x.permute((0, -1, 1)))
        x = self.layer_norm(x)

        return x
