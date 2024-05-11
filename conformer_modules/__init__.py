from pathlib import Path
import sys

# Set up the import path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# Import the necessary modules for the conformer encoder
from conformer_block import ConformerBlock
from convolution import ConvolutionModule
from convolution_subsampling import ConvolutionSubsampling
from feedforward import FeedForward
from multihead import MultiHeadSelfAttention
from conformer_encoder import ConformerEncoder
