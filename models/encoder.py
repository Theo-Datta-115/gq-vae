import torch
import torch.nn as nn
from x_transformers.x_transformers import AttentionLayers, FeedForward
from utils import ConvStack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    This takes a sequence of embedded chars and outputs a sequence of latent vectors + gates

    Inputs:
    - dim : the input dimension

    """

    def __init__(self, dim, depth=4):
        super(Encoder, self).__init__()
        self.dim = dim
        
        self.attn_layers = AttentionLayers(
                dim = dim,
                causal = False,
                depth = depth,
                heads = int(dim / 64),
                attn_dim_head = 64,
                alibi_pos_bias = True,
            )

        self.ff = FeedForward(
            dim = dim,
            dim_out = dim + 1, # For gates
        )
        
    def forward(self, x):

        x = self.attn_layers(x)
        x = self.ff(x)

        # # Split into tokens and gates
        z, l = x.split([self.dim, 1], dim=-1)
        # g = nn.functional.sigmoid(g.squeeze(-1))
        return z, l
