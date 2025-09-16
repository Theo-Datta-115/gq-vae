import torch
import torch.nn as nn
from torch.nn import LayerNorm
from x_transformers import FeedForward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    """
    Given a latent sample z p_phi maps back to characters + gate/mask predictions

    Inputs:
    - dim : the input dimension
    - num_chars : the number of characters in the vocab
    - kernel_size : the maximum length of the token
    - depth : the number of layers in the feedforward network
    """

    def __init__(self, dim, num_chars, kernel_size, depth=4):
        super(Decoder, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.num_chars = num_chars

        self.conv = nn.ConvTranspose1d(dim, num_chars + 1, kernel_size=kernel_size, stride=kernel_size)
        self.lns = nn.ModuleList([LayerNorm(dim, device=device) for _ in range(depth)])
        self.ffs = nn.ModuleList([FeedForward(dim = dim, dim_out = dim).to(device) for _ in range(depth)]) 
        
    def forward(self, x):
        B, T, D = x.shape
        for ln, ff in zip(self.lns, self.ffs):
            x = x + ff(ln(x))
        x = x.permute(0,2,1).contiguous() # B, D, T
        x = self.conv(x) # B, num_chars + 1, T*kernel_size
        chars, lens = x.split([self.num_chars, 1], dim=1)
        chars = chars.reshape(B, self.num_chars, T, self.kernel_size)
        lens = lens.reshape(B, T, self.kernel_size).permute(0,2,1).contiguous() 
        return chars, lens

