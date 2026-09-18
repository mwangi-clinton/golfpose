# Distributed training utilities for GolfPose DDP training.
#
# Provides:
#   - Process-group setup / teardown (torchrun compatible)
#   - Rank / world-size helpers
#   - All-reduce for scalar metrics
#   - DistributedEvalSampler (no sample duplication)

import os
import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


# ---------------------------------------------------------------------------
# Process group helpers
# ---------------------------------------------------------------------------

def setup_distributed(backend: str = "nccl"):
    """Initialise the distributed process group from torchrun env vars.

    After this call ``torch.distributed.is_initialized()`` returns True and
    the current process' GPU is set to ``LOCAL_RANK``.
    """
    if dist.is_initialized():
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        init_method="env://",
    )


def cleanup_distributed():
    """Destroy the process group (call at the end of training)."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Rank / world-size queries
# ---------------------------------------------------------------------------

def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def reduce_tensor(tensor: torch.Tensor, world_size: int = None, op=dist.ReduceOp.SUM):
    """All-reduce *tensor* across all ranks and return the mean.

    If the process group is not initialised the tensor is returned as-is.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor

    if world_size is None:
        world_size = get_world_size()

    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    if op == dist.ReduceOp.SUM:
        rt /= world_size
    return rt


def barrier():
    """Synchronisation barrier (no-op when not distributed)."""
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# DistributedEvalSampler — evaluation without duplicating tail samples
# ---------------------------------------------------------------------------

class DistributedEvalSampler(Sampler):
    """Sampler that distributes data across ranks **without** duplicating the
    last few samples.  ``DistributedSampler`` pads to make the dataset
    evenly divisible; that padding biases evaluation metrics.  This sampler
    instead assigns ``ceil(N/world)`` to the first ranks and ``floor(N/world)``
    to the rest.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = len(dataset)

    def __iter__(self):
        indices = list(range(self.total_size))
        # split without padding
        per_rank = math.ceil(self.total_size / self.num_replicas)
        start = self.rank * per_rank
        end = min(start + per_rank, self.total_size)
        return iter(indices[start:end])

    def __len__(self):
        per_rank = math.ceil(self.total_size / self.num_replicas)
        start = self.rank * per_rank
        end = min(start + per_rank, self.total_size)
        return end - start
