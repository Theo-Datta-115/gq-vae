import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def charToInts(string):
    d = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8, "j":9, "k": 10,
         "l":11, "m":12, "n":13, "o":14, "p":15, "q":16, "r":17, "s":18, "t": 19, "u": 20,
         "v": 21, "w": 22, "x": 23, "y":24, "z":25, " ":26}
    chars = []
    for c in string:
        chars.append(d[c])
    return chars

def intToChar(arr):
    d = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
    10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't',
    20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: ' ', 27: "_"}
    string = ""
    for c in arr:
        if c in d:
            string += d[c]
        else:
            string += "%"
    return string

def collate_fn(batch):
    BOS_TOKEN = 2 #These need to change as you change the dataset - they should end up stable at like 1, 2
    EOS_TOKEN = 3
    
    batch = torch.stack(batch, dim=0)
    new_batch = torch.zeros((batch.shape[0], batch.shape[1] + 2), dtype=torch.long)
    
    new_batch[:, 0] = BOS_TOKEN
    new_batch[:, 1:batch.shape[1] + 1] = batch
    new_batch[:, -1] = EOS_TOKEN

    return new_batch

class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class ConvStack(nn.Module):
    def __init__(self, dim, depth):
        super(ConvStack, self).__init__()
        self.dim = dim
        KERNEL_MULT = 2
        self.convs = nn.ModuleList([nn.Conv1d(dim, dim, kernel_size=(10 + i * KERNEL_MULT), padding='same', device=device) for i in reversed(range(depth))])
        self.lns = nn.ModuleList([nn.LayerNorm(dim, device=device) for _ in range(depth)]) #mult=10
    
    def forward(self, x):
        x = x.permute(0,2,1).contiguous()
        for ln, conv in zip(self.lns, self.convs):
            x = ln(x.permute(0,2,1)).permute(0,2,1)
            x = x + F.gelu(conv(x))
        x = x.permute(0,2,1).contiguous()
        return x

class CosWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, alpha_f=0.1, t_max=None, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.alpha_f = alpha_f
        self.t_max = t_max if t_max is not None else max_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1  # Scheduler counts epochs
        initial_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]
        
        new_lrs = []
        for initial_lr in initial_lrs:
            new_lrs.append(self._get_lr_for_step(initial_lr, current_step, self.max_steps))
        
        return new_lrs

    def _get_lr_for_step(self, initial_lr, step, max_steps):
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return initial_lr * step / self.warmup_steps  # Linear warmup
        elif step >= max_steps:
            return eta_min
        else:
            step -= self.warmup_steps
            max_steps -= self.warmup_steps
            return eta_min + (initial_lr - eta_min) * (1 + math.cos(math.pi * step / max_steps)) / 2
"""

from torch.nn.attention.flex_attention import create_block_mask, create_mask, flex_attention

class FlexAttn(nn.Module):
    def __init__(self, B,H,C,D, depth, alibi=True, sliding=True, sliding_window=16):
        super(FlexAttn, self).__init__()
        self.sliding_window = sliding_window
        self.alibi = alibi
        self.sliding = sliding
        self.depth = depth

        #Batch size, Heads, Context Len, Head Dim
        query = torch.randn(B, H, C, D, device="cuda", dtype=torch.float32)
        key = torch.randn(B, H, C, D, device="cuda", dtype=torch.float32)
        value = torch.randn(B, H, C, D, device="cuda", dtype=torch.float32)

        flex_attention = torch.compile(flex_attention, dynamic=False)
        self.attns = nn.ModuleList([flex_attention(query, key, value, score_mod=self.score_mod, block_mask=self.mask_mod) for _ in range(depth)])
    
    def mask_mod(b, h, q_idx, kv_idx):
        if self.sliding:
            return sliding_window_mask(b, h, q_idx, kv_idx)
        else:
            return True

    def score_mod(score, b, h, q_idx, kv_idx):
        if self.alibi:
            return self.alibi_and_causal_functional(score, b, h, q_idx, kv_idx)
        else:
            return score
    
    def sliding_window_mask(b, h, q_idx, kv_idx): #block mask
        window = (torch.abs(q_idx - kv_idx) <= self.sliding_window)
        return window

    def alibi_and_causal_functional(score, b, h, q_idx, kv_idx): #score mask
        scale = torch.exp2(-((h + 1) * 8.0 / H))
        bias = (q_idx - kv_idx) * scale
        return score + bias

    def forward(self, x):
        for attn in self.attns:
            x = attn(x)
        return x


        self.attn = FlexAttn(B, int(dim / 64), T, 64, depth, alibi=True, sliding=True, sliding_window=16)


"""