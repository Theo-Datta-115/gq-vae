import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    BOS_TOKEN = 2 #ASCII for BOS
    EOS_TOKEN = 3 #ASCII for EOS
    
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
Deprecated architecture layers that were ablated but later discarded:

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

"""