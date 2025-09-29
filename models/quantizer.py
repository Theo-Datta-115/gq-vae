import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.quantize_helpers import ReservoirSampler, VectorResample
import time
import os
os.environ['CUDA_PATH'] = '/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda/'

from pykeops.torch import LazyTensor
import torchpq
from torchpq.index import IVFPQIndex
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - vocab_size :  vocab size
    - D : dimensionality of embedding space
    - alpha: compression cost used in loss term, enforces importance of low probability 
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, vocab_size, D, alpha, beta):
        super(VectorQuantizer, self).__init__()
        self.vocab_size = vocab_size
        self.D = D
        self.alpha = alpha
        self.beta = beta

        # CODEBOOK 
        self.embedding = nn.Embedding(self.vocab_size, self.D)
        self.embedding.weight.data.uniform_(-1.0 / self.vocab_size, 1.0 / self.vocab_size)
        self.resample = VectorResample(num_samples=131072, vocab_size=vocab_size)

        self.reservoir_re_init = 250 #How many iters before every re-initialization
        self.reservoir_buffer = 500 #How many iters when the model starts training for it to warm up

        self.sigmoid = nn.Sigmoid()

    def build_index(self):
        print("building")

        index = IVFPQIndex(
            d_vector=self.D,
            n_subvectors=64,   # Number of subquantizers
            n_cells=256,       # Number of coarse quantizer clusters
            initial_size=200,  # Initial capacity per cell
            distance="euclidean"  # Or "cosine", depending on your use case
        )

        codebook_data = self.embedding.weight.data.permute(1,0).contiguous()

        index.train(codebook_data)
        ids = torch.arange(self.vocab_size, device=device)
        index.add(codebook_data, ids=ids)

        self.index = index
        print("done building")

    def get_very_efficient_rotation(self, u, q, e):
        # Rotation Trick (https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py)
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
        e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
            torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        return e

    def quantize(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, D, T)
        p.shape = (batch, 1, T)

        B = batch size
        D = dimension of embedding space
        T = length of context inputted (length of sequence)
        """
        (B,T,D) = z.shape
        use_keops = True
        
        # # flatten to size (B*T, D)
        z_flattened = z.view(-1, self.D).contiguous()

        # # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # # d has shape (B*T, V)
        # d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        #     torch.sum(self.embedding.weight**2, dim=1) - 2 * \
        #     torch.matmul(z_flattened, self.embedding.weight.t())

        # # find closest encodings: min_encodings (BT, V), one hot for the codebook vectors
        # min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # min_encodings = torch.zeros(
        #     min_encoding_indices.shape[0], self.vocab_size).to(device)
        # min_encodings.scatter_(1, min_encoding_indices, 1)

        # # get quantized latent vectors (BT, D)
        # z_q = torch.matmul(min_encodings, self.embedding.weight)

        # #reshape to original dimensions -> (B,D,T)
        # z_q = z_q.view(B,T,D)
        # d_min, _ = torch.min(d, dim=1)

        # This uses a graph-based reconstruction to allow for faster indexing of vectors from the codebook when it is fixed
        if not self.training:
            # use IVFPQ
            z_perm = z_flattened.permute(1,0).contiguous()
            distances, topk_ids = self.index.search(z_perm, k=1)
            d_min = distances.squeeze()
            min_encoding_indices = topk_ids.squeeze()
        else:
            # Convert to LazyTensors for efficient computation
            x_i = LazyTensor(z_flattened[:, None, :])  # Shape: (B*T, 1, D)
            x_j = LazyTensor(self.embedding.weight[None, :, :])  # Shape: (1, V, D)

            d = ((x_i - x_j) ** 2).sum(-1)  # Shape: (B*T, V)
            d_min = d.min(1).squeeze()  # Shape: (B*T,)
            min_encoding_indices = d.argKmin(K=1, dim=1).view(-1)  # Shape: (B*T,)

        # Create one-hot encoding for min_encodings
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.vocab_size, device=device
        )
        min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)  # One-hot encoded

        # Get quantized latent vectors directly
        z_q = torch.matmul(min_encodings, self.embedding.weight)  # Shape: (B*T, D)

        # Reshape to original dimensions
        z_q = z_q.view(B, T, D)

        # Rotation Trick
        rotate = False

        if rotate:
            # Rearrange inputs and quantized tensors to flatten sequence dimension
            z_q = rearrange(z_q, 'b t d -> (b t) d')

            # Compute normalized inputs and call the efficient rotation function
            pre_norm_q = self.get_very_efficient_rotation(
                z_flattened / (torch.norm(z_flattened, dim=1, keepdim=True) + 1e-6),  # Normalize x
                z_q / (torch.norm(z_q, dim=1, keepdim=True) + 1e-6),  # Normalize quantized
                z_flattened.unsqueeze(1)
                ).squeeze()

            # Scale pre-normalized quantized values
            quantized = pre_norm_q * (
                torch.norm(z_q, dim=1, keepdim=True) / (torch.norm(z_flattened, dim=1, keepdim=True) + 1e-6)
                ).detach()

            # Rearrange quantized back to original 1D shape
            z_q = rearrange(quantized, '(b t) d -> b t d', b=B, t=T)

        codebook = torch.mean((z_q.detach()-z)**2)
        commitment = self.beta * torch.mean((z_q - z.detach()) ** 2)
            
        # preserve gradients / stop grad
        z_q = z + (z_q - z).detach() 

        #codebook proportion used
        codebook_used = len(torch.unique(min_encoding_indices)) / self.vocab_size

        return (codebook, commitment), z_q, codebook_used, min_encodings, d_min

    def forward(self, z, iter):
        # Do Resampling
        encodings = z.reshape(z.shape[0] * z.shape[1], self.D)
        if (iter < self.reservoir_buffer) and self.training:
            self.resample.res.add(encodings)
            z_q = z
            (codebook, commitment), codebook_used = (None, None), None
            min_encodings = 0
        else:
            # re-initialize the codebook
            if (iter == self.reservoir_buffer or ((iter - self.reservoir_buffer) % self.reservoir_re_init == 0)) and self.training:
                print("re-initializing the codebook")
                idxs, codebook_v = self.resample.resample()
                self.embedding.weight.data[idxs] = codebook_v
                (codebook, commitment), z_q, codebook_used, min_encodings, d = self.quantize(z)
                self.resample.reset()
            else:
                (codebook, commitment), z_q, codebook_used, min_encodings, d = self.quantize(z)
                self.resample.add(encodings, min_encodings, d)

        return (codebook, commitment), z_q, codebook_used, min_encodings
