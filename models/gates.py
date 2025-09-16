import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from x_transformers import Encoder as Attn

class Gater(nn.Module):
    """
    This takes a sequence of quantized characters and outputs gates

    Inputs:
    - dim : the input dimension
    - kernel_size : the maximim length of the token
    - alpha : the compression cost used in loss term
    """

    def __init__(self, dim, kernel_size, alpha): 
        super(Gater, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha

        self.attn_layers = Attn(
                dim = dim,
                depth = 2,
                heads = int(dim / 64),
                alibi_pos_bias = True,
                use_simple_rmsnorm = True
            )
        
        self.gateconv1 = nn.Conv1d(dim, int(dim/2), kernel_size=(kernel_size*2), padding='same')
        self.gateconv2 = nn.Conv1d(int(dim/2), int(dim/4), kernel_size=(int(kernel_size + kernel_size / 2)), padding='same')
        self.gateconv3 = nn.Conv1d(int(dim/4), 1, kernel_size=(kernel_size), padding='same')

    def forward(self, z_q):
        # ATTN 
        g = self.attn_layers(z_q).permute(0,2,1).contiguous()
        g = self.gateconv3(self.gateconv2(self.gateconv1(g))).reshape(z_q.shape[0], -1)
        g = self.sigmoid(g)
        
        compression = self.alpha * torch.mean(g)
        g_under_half = torch.mean((g < 0.5).float())
        return g, compression, g_under_half

