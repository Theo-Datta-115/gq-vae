import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from models.quantizer import VectorQuantizer
from models.gates import Gater
from models.encoder import Encoder
from models.decoder import Decoder
from models.pqquantizer import VectorQuantization
from models.quantize_decomp import DecompQuantizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

class TokenAE(nn.Module):
    def __init__(self, T, D, embed_dim, n_embeddings, kernel_size, vocab_size, alpha, beta, gamma):
        super(TokenAE, self).__init__()

        """
        T: the length of the inputed, tokenized context. Essentially, the length of the context window.
        D: dimension of embeddings in the dictionary (also used as the dimension of the channels in the Conv layers, but I imagine this can be changed)
        embed_dim: dimension of the embedding of your data
        n_embeddings: number of unique characters in your dataset to embed
        kernel_size: the kernel used in the convolution layers
        vocab_size: length of the Vocab size, which is the number of vectors in the VQ-VAE dictionary
        alpha: compression cost term
        beta: commitment cost term
        """
        self.n_embeddings = n_embeddings
        self.kernel_size = kernel_size
        self.T = T
        self.D = D
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.embedding_layer = nn.Embedding(num_embeddings=self.n_embeddings, embedding_dim=self.embed_dim)
    
        self.gater = Gater(D, kernel_size, alpha)
        self.encoder = Encoder(D)
        self.decoder = Decoder(D, self.n_embeddings, kernel_size)
        self.vector_quantization = VectorQuantizer(vocab_size, D, alpha, beta)
        # self.vector_quantization = DecompQuantizer(8, vocab_size, D, alpha, beta)
        # self.pq_quantization = VectorQuantization(dim=D, codebook_size=10000, commitment_weight=beta, kmeans_iters=500)
        # self.pq_quantization = VectorQuantization(dim=int(D / 4), codebook_size=[16, 16, 16, 16], commitment_weight=beta, kmeans_iters=500)
    
    def load_state_dict(self, state_dict, strict=True):
        # Load the state dict
        super(TokenAE, self).load_state_dict(state_dict, strict)

        # Rebuild the HNSW index with the updated weights
        self.vector_quantization.build_index()

    def forward(self, x, iter=0, verbose=False, tokenizing=False, hardset=None):
        # FIRST ZEROS
        first_zeros = torch.argmax((x == 0).float(), dim=1) - 1

        # EMBED
        xl = x.long().to(self.embedding_layer.weight.device)
        x_embed = self.embedding_layer(xl).float()
        
        # ENCODER
        z_e, len_pred = self.encoder(x_embed) 

        # QUANTIZE
        # (codebook, commitment), z_q, codebook_used = self.pq_quantization(z_e, 0, iter=iter)
        (codebook, commitment), z_q, codebook_used, min_encodings = self.vector_quantization(z_e, iter=iter)

        # GATES
        gates, compression, g_under_half = self.gater(z_q) #OG Encoder, self.alpha. gates

        # HARDSET LAST GATE
        # hardset = torch.zeros_like(x).to(device) 
        # hardset[torch.arange(x.size(0)), first_zeros] = 1
        # gates = torch.clamp(gates + hardset, max=1.0)

        # HARDSET EVERY XTH GATE (UNTIL PADS) – Effectively hard sets fixed-length chunks
        if hardset:
            gates = self.hardset_func(first_zeros, hardset)

        # DECODER
        x_hat, pred_mask_lens = self.decoder(z_q)
        # pred_mask_lens = len_pred.permute(0,2,1).contiguous() # used to try and pred lengths from decoder

        prediction_modes = ['gates', 'lens', 'masks', 'divergence']
        predicting = 'lens'
        
        #Create targets that are expanded
        targets, masks, masks_g = self.diagonalOutputs(x.shape[0], x.shape[1], self.kernel_size, gates, x)
        targets = targets.long().to(device)
        
        # LENGTH PREDICTIONS
        # mask_lens = (masks > 0.5).float().sum(axis=2) - 1 #(B, T), old maksing method
        # mask_loss = nn.functional.cross_entropy(pred_mask_lens.float(), mask_lens.detach().long(), reduction='none') #reduction='none'
        # mask_loss = torch.mean(mask_loss * gates.detach()) * self.gamma #USED WHEN WEIGHTING BY GATES  * gates.detach()
        
        mask_approx = torch.clamp(self.mask_approx_fun(pred_mask_lens), 1e-7, 1 - 1e-7) # add clamping to prevent nan values, which kill some runs
        mask_loss = nn.functional.binary_cross_entropy(mask_approx, masks, reduction='none') #reduction='none'
        mask_loss = torch.mean(torch.mean(mask_loss, dim=2) * gates.detach()) * self.gamma #USED WHEN WEIGHTING BY GATES  * gates.detach()

        ## RECON LOSS
        recon_loss = nn.functional.cross_entropy(x_hat, targets, reduction='none')
        recon_loss = torch.mean(recon_loss * masks.to(device)) #OG
        # recon_loss = torch.mean(recon_loss * masks.to(device) * (targets != 0).to(device)) #MASKS PADS

        #### FOUR CORRECTNESS METRICS
        # corr_char_masks, corr_token_masks, corr_char_pred, corr_token_pred
        correct = (torch.argmax(x_hat, dim=1) == targets)

        #derive predicted masks from the predicted lengths
        # pred_masks = (torch.arange(self.kernel_size).repeat(x.shape[0] * x.shape[1], 1).to(device) <= 
        #                 pred_mask_lens.argmax(dim=1).reshape(-1, 1)).flip([1]).reshape(x.shape[0], x.shape[1], self.kernel_size) # OLD METHOD
        pred_masks = mask_approx

        corr_char_masks = torch.sum(torch.logical_and(masks > 0.5, correct)[gates > 0.5].float()) / torch.sum((masks > 0.5)[gates > 0.5].float())
        corr_token_masks = torch.mean(torch.all(torch.logical_or((masks < 0.5),correct), dim=2)[gates > 0.5].float())
        corr_char_pred = torch.sum((torch.logical_and(pred_masks > 0.5, correct)[gates > 0.5].float()) / torch.sum((pred_masks > 0.5)[gates > 0.5]).float())
        corr_token_pred = torch.mean((torch.all(torch.logical_or((pred_masks < 0.5),correct), dim=2)[gates > 0.5]).float())
        
        compression_val = torch.sum((x != 0).float()) / torch.sum((gates > 0.5).float())
        correct_used_tokens = torch.logical_or(correct[(gates > 0.5)], (masks[(gates > 0.5)] < 0.5))

        corr_token = (corr_char_masks, corr_token_masks, corr_char_pred, corr_token_pred, compression_val, correct_used_tokens)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            print('original data:', x)
            print('targets', targets)
            print('xhat', x_hat)
            print('correct', correct.shape, correct)
            print('tokens', (torch.logical_or(correct, torch.logical_not(masks)))[gates>0.5] )
            print('masks', masks)
            print('gates', gates)
            print('correct token', corr_token_pred)
            print('pred masks', pred_masks)
            print('lens', )
        
        if tokenizing:
            return (recon_loss, codebook, commitment, compression, mask_loss), x_hat, codebook_used, corr_token, gates, g_under_half, pred_mask_lens, masks, (min_encodings, correct_used_tokens, targets)
        else:
            return (recon_loss, codebook, commitment, compression, mask_loss), x_hat, codebook_used, corr_token, gates, g_under_half, pred_masks, masks

    # Function to create masks
    def diagonalOutputs(self, B, T, kernel_size, g, x):
        # flatten x and p, they will later be reshaped.
        x = x.flatten()
        
        # function to create a new targets
        targets = torch.zeros((kernel_size, T * B))
        for i in range(kernel_size):
            targets[kernel_size - i - 1] = x
            x = x.roll(1)
            x[(torch.arange(T * B) % T) == 0] = 0
        targets = targets.reshape(kernel_size, B, T).permute(1,2,0)
        
        # function to create the masks
        # pads the gates, expands them to kernels of kernel_size, flips, hard sets to one, and does a cumulative product, flips back
        # try to predict the masks, weighted by the gates -> 

        g_inv = F.pad((1 - g), (kernel_size - 1, 0), "constant", 0)
        # g_inv[:, -1] = 1 #forces true for EOS
        g_expanded = g_inv.unfold(1, size=kernel_size, step=1).flip([2])
        g_expanded[:,:,0] = 1
        masks = torch.cumprod(g_expanded, dim=2).flip([2]) # this is m bar

        m_expanded = g_expanded.clone()
        m_expanded[:,:,0] = g # true gates on the last column 
        masks_g = torch.cumprod(g_expanded, dim=2).flip([2])

        return targets, masks, masks_g
    
    # Function to take inputs of size (B,K,T) and transfer them to mask-like outputs of size (B,T,K)
    def mask_approx_fun(self, pred_masks):
        pred_masks = pred_masks.permute(0,2,1).contiguous()                    # (B,K,T) -> (B,T,K)
        pred_masks = pred_masks - pred_masks.max(dim=2, keepdim=True).values   # Normalize for numerical stability
        pred_masks = torch.clamp(pred_masks, min=-50, max=50)                  # Clamp to prevent overflows
        pred_masks = torch.exp(pred_masks)                                     # Expanentiate to ensure positive values
        pred_masks = pred_masks + 1e-6                                         # Add small epsilon to prevent nans in the divide 
        pred_masks = torch.cumsum(pred_masks, dim=2)                           # Make sure probabilities are strictly increasing
        pred_masks = pred_masks / (pred_masks[:, :, -1:])                      # Convert to [0,1] range
        return pred_masks

    def hardset_func(self, first_zeros, hardset):
        row_pattern = torch.zeros(self.T, dtype=torch.int)                     # Create new array to represent masking pattern
        row_pattern[hardset-1::hardset] = 1                                    # Set every xth gate in this to 1
        g = row_pattern.repeat(len(first_zeros), 1).float().to(device)         # Expand this to all gates
        
        next_multiple = ((first_zeros // hardset) + 1) * hardset               # Mask all encounters of positive gates after end of token 
        next_multiple = torch.clamp(next_multiple, max=self.T).to(device)      # Make sure that these only stretch until end of content
        indices = torch.arange(self.T).unsqueeze(0).repeat(len(first_zeros), 1).to(device)
        g_mask = (indices < next_multiple.unsqueeze(1)).float()                # Create a mask for these indices   
        gates = g * g_mask
        return gates
 