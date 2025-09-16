import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.quantize_helpers import ReservoirSampler, VectorResample
import time
from models.quantizer import VectorQuantizer


class DecompQuantizer(nn.Module):
    """
    VQ-VAE, decomposed into smaller codebooks, and recomposed
    """

    def __init__(self, decomp, vocab_size, D, alpha, beta):
        super(DecompQuantizer, self).__init__()
        
        self.decomp = decomp
        self.vqs = nn.ModuleList([VectorQuantizer(4, int(D / decomp), alpha, beta) for _ in range(decomp)])

    def forward(self, z, iter):
        
        z_perm = z.view(z.shape[0], z.shape[1], self.decomp, int(z.shape[2] / self.decomp))
        z_decomp = z_perm.permute(2, 0, 1, 3)
        codebook, commitment, codebook_used = 0, 0, 0
        zqs = []

        for z, vq in zip(z_decomp, self.vqs):
            (cd, cmp), z_q, cd_used, _ = vq(z, iter)
            zqs.append(z_q)
            if cd:
                codebook += cd
                commitment += cmp
                codebook_used += cd_used

        z_q = torch.cat(zqs, dim=2)
        codebook, commitment, codebook_used = (codebook / self.decomp), (codebook / self.decomp), (codebook / self.decomp)

        return (codebook, commitment), z_q, codebook_used, 0
        