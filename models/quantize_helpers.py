#Drawn from this file: https://github.com/distsup/DistSup/blob/master/distsup/modules/bottlenecks.py / the encoders file
import torch
from torch import nn
import numpy as np
import time
import random
from fast_pytorch_kmeans import KMeans

"""
Reservoir Sampling achieves the following tasks:
    1. Stores a running history of the values seen from the encoder
    2. Can output KMeans++ on this sampling
    3. Can add samples
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Reservoir Sampling! The num_samples should be kept at some multiple larger than V
class ReservoirSampler(nn.Module):
    def __init__(self, num_samples):
        super(ReservoirSampler, self).__init__()
        self.n = num_samples
        self.reset()

    def reset(self):
        self.i = 0
        self.buffer = None

    def add(self, samples):
        samples = samples.detach()
        if self.buffer is None:
            self.buffer = torch.empty(
                self.n, samples.size(-1), device=samples.device)
        buffer = self.buffer
        if self.i < self.n:
            slots = self.n - self.i
            add_samples = samples[:slots]
            samples = samples[slots:]
            buffer[self.i: self.i + len(add_samples)] = add_samples
            self.i += len(add_samples)
            if not len(samples):
                # print(f"Res size {self.i}")
                return

        #fast version - makes array of indexes, samples if in the correct range
        range = torch.arange(self.i, self.i + samples.shape[0])
        range_rand = (range * torch.rand(samples.shape[0])).int()
        to_change = range_rand < len(buffer)
        if to_change.float().sum() != 0:
            idxs = range_rand[to_change]
            idxs_s = range[to_change] - self.i
            buffer[idxs] = samples[idxs_s]
        self.i += samples.shape[0]

    def contents(self):
        return self.buffer[:self.i]

    
class VectorResample(nn.Module):
    def __init__(self, num_samples, vocab_size):
        super(VectorResample, self).__init__()
        self.frequencies = 1
        self.res = ReservoirSampler(num_samples=num_samples)
        self.register_buffer("embed_prob", torch.zeros(vocab_size))
        self.vocab_size = vocab_size

    def reset(self):
        self.res.reset()
        self.embed_prob = torch.zeros(self.vocab_size).to(device)
    
    def add(self, samples, min_encodings, d, dist_metric=True):
        decay_anc = 0.99
        self.embed_prob.mul_(decay_anc).add_(torch.mean(min_encodings, dim=0), alpha= 1 - decay_anc)
        if dist_metric:
            THRESH = 2500
            if len(samples) > THRESH:
                _, d_large_idx = torch.topk(d, THRESH)
                self.res.add(samples[d_large_idx])
            else:
                self.res.add(samples)
        else:
            self.res.add(samples)
    
    def resample(self):
        # THRESH = 10
        # usage_threshold = 1 / self.vocab_size / THRESH # 10 times less utilized than normal is the threshold
        # idxs = self.embed_prob < usage_threshold

        idxs = (self.embed_prob == 0)
        print(idxs.float().sum(), "resampled vectors")

        encodings = self.res.contents()
        sample_idxs = torch.randperm(len(encodings))[:idxs.int().sum().item()]

        return idxs, encodings[sample_idxs] #usually returns encodings[sample_idxs], or (torch.randn(len(sample_idxs), 512) / 10 + mean)

    def cluster(self):
        encodings = self.res.contents() #.cpu().numpy()
        kmeans = KMeans(n_clusters=self.vocab_size, init_method="kmeans++", verbose=0)
        kmeans.fit(encodings)
        return kmeans.centroids

"""Torch distributed utilities."""

import typing as tp
import torch


def rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def is_distributed():
    return world_size() > 1


def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)


def all_gather(tensor: torch.Tensor):
    if is_distributed():
        n_devices = torch.distributed.get_world_size()
        local_size = torch.tensor(tensor.size(), device=tensor.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(n_devices)]
        torch.distributed.all_gather(all_sizes, local_size)

        max_length = max(size[0] for size in all_sizes)
        length_diff = max_length.item() - local_size[0].item()
        if length_diff:
            pad_size = (length_diff, *tensor.size()[1:])
            padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat((tensor, padding))

        all_tensors_padded = [torch.zeros_like(tensor) for _ in range(n_devices)]
        torch.distributed.all_gather(all_tensors_padded, tensor)
        all_tensors = []
        for tensor_, size in zip(all_tensors_padded, all_sizes):
            all_tensors.append(tensor_[: size[0]])
        return all_tensors
    else:
        return [tensor]


def _is_complex_or_float(tensor):
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)


def _check_number_of_params(params: tp.List[torch.Tensor]):
    # utility function to check that the number of params in all workers is the same,
    # and thus avoid a deadlock with distributed all reduce.
    if not is_distributed() or not params:
        return
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    all_reduce(tensor)
    if tensor.item() != len(params) * world_size():
        # If not all the workers have the same number, for at least one of them,
        # this inequality will be verified.
        raise RuntimeError(f"Mismatch in number of params: ours is {len(params)}, "
                           "at least one worker has a different one.")


def broadcast_tensors(tensors: tp.Iterable[torch.Tensor], src: int = 0):
    """Broadcast the tensors from the given parameters to all workers.
    This can be used to ensure that all workers have the same model to start with.
    """
    if not is_distributed():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = torch.distributed.broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()


def sync_buffer(buffers, average=True):
    """
    Sync grad for buffers. If average is False, broadcast instead of averaging.
    """
    if not is_distributed():
        return
    handles = []
    for buffer in buffers:
        if torch.is_floating_point(buffer.data):
            if average:
                handle = torch.distributed.all_reduce(
                    buffer.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
            else:
                handle = torch.distributed.broadcast(
                    buffer.data, src=0, async_op=True)
            handles.append((buffer, handle))
    for buffer, handle in handles:
        handle.wait()
        if average:
            buffer.data /= world_size


def sync_grad(params):
    """
    Simpler alternative to DistributedDataParallel, that doesn't rely
    on any black magic. For simple models it can also be as fast.
    Just call this on your model parameters after the call to backward!
    """
    if not is_distributed():
        return
    handles = []
    for p in params:
        if p.grad is not None:
            handle = torch.distributed.all_reduce(
                p.grad.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
            handles.append((p, handle))
    for p, handle in handles:
        handle.wait()
        p.grad.data /= world_size()


def average_metrics(metrics: tp.Dict[str, float], count=1.):
    """Average a dictionary of metrics across all workers, using the optional
    `count` as unnormalized weight.
    """
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))