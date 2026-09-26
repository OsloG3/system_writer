"""PyTorch/XLA (TPU) helpers.

All torch_xla imports are lazy, so the package works unchanged on CPU/CUDA
machines without the `tpu` extra. On a TPU VM every helper below becomes a
thin wrapper over torch_xla.core.xla_model / xla_multiprocessing; off a TPU
they degrade to no-ops or single-process defaults, which keeps training code
device-agnostic and testable on CPU.

Conventions used by the training entrypoints:

- one process per TPU core (xmp.spawn), data-parallel: each rank rolls out /
  trains on a shard of the deals and gradients are averaged with all_reduce;
- static shapes everywhere on XLA (TPUs recompile the graph for every new
  tensor shape, so dynamic batches would thrash the compile cache);
- xm.optimizer_step + xm.mark_step to bound graph growth between steps;
- xm.save for checkpoints (master ordinal only, lazy tensors materialized).
"""

import os

import numpy as np
import torch

_xm = None
_xmp = None
_xr = None
_import_failed = False


def _xm_mod():
    """Lazily import torch_xla; returns the xla_model module or None."""
    global _xm, _xmp, _xr, _import_failed
    if _xm is None and not _import_failed:
        try:
            os.environ.setdefault("PJRT_DEVICE", "TPU")
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.xla_multiprocessing as xmp
            import torch_xla.runtime as xr
            _xm, _xmp, _xr = xm, xmp, xr
        except ImportError:
            _import_failed = True
    return _xm


def available() -> bool:
    return _xm_mod() is not None


def is_xla(device) -> bool:
    return str(getattr(device, "type", device)) == "xla"


def device():
    """The XLA device for this process (requires torch_xla)."""
    xm = _xm_mod()
    if xm is None:
        raise RuntimeError(
            "torch_xla is not installed -- install it with: uv sync --extra tpu")
    return xm.xla_device()


def world_size() -> int:
    """Number of addressable XLA devices (TPU cores); 1 without torch_xla."""
    if _xm_mod() is None:
        return 1
    return _xr.world_size()


def rank() -> int:
    if _xm_mod() is None:
        return 0
    return _xr.global_ordinal()


def is_master() -> bool:
    xm = _xm_mod()
    return xm.is_master_ordinal() if xm is not None else True


def mark_step():
    """Flush the pending lazy graph (call once per training/rollout step)."""
    xm = _xm_mod()
    if xm is not None:
        xm.mark_step()


def barrier():
    xm = _xm_mod()
    if xm is not None:
        xm.barrier()


def print_master(msg: str):
    xm = _xm_mod()
    if xm is not None:
        xm.master_print(msg)
    else:
        print(msg)


def spawn(fn, args=(), nprocs: int | None = None):
    """Run fn(rank, *args) on every TPU core (xmp.spawn; args must be
    picklable -- the library-default start method is used)."""
    xm = _xm_mod()
    if xm is None:
        raise RuntimeError("spawn requires torch_xla")
    _xmp.spawn(fn, args=tuple(args), nprocs=nprocs or world_size())


def optimizer_step(opt):
    """opt.step() the XLA way (gradients must already be all-reduced)."""
    xm = _xm_mod()
    if xm is None:
        opt.step()
    else:
        xm.optimizer_step(opt, barrier=False)
        xm.mark_step()


def average_grads(params, world: int):
    """All-reduce (mean) gradients across ranks -- the data-parallel step.

    Called after backward()/grad-clipping and before optimizer_step(); the
    all_reduce stays inside the lazy graph, so the collective overlaps with
    compilation of the next step.
    """
    xm = _xm_mod()
    if xm is None or world <= 1:
        return
    grads = [p.grad for p in params if p.grad is not None]
    if grads:
        xm.all_reduce(xm.REDUCE_SUM, grads, scale=1.0 / world)


def reduce_sum_np(arr: np.ndarray) -> float:
    """Sum of `arr` across every rank (identical result on all ranks)."""
    xm = _xm_mod()
    total = float(np.sum(arr))
    if xm is None or world_size() <= 1:
        return total
    t = torch.tensor([total], dtype=torch.float64, device=xm.xla_device())
    xm.all_reduce(xm.REDUCE_SUM, [t])
    mark_step()
    return float(t.cpu().item())


def gather_np(arr: np.ndarray) -> np.ndarray:
    """Concatenate equal-length per-rank arrays along axis 0 (same order and
    result on every rank)."""
    xm = _xm_mod()
    arr = np.asarray(arr)
    if xm is None or world_size() <= 1:
        return arr
    t = torch.tensor(arr, dtype=torch.float64, device=xm.xla_device())
    out = xm.all_gather(t, dim=0)
    mark_step()
    return out.cpu().numpy().astype(arr.dtype)


def save(obj, path):
    """torch.save with XLA support: lazy tensors are materialized and only
    the master ordinal writes (files remain plain torch checkpoints)."""
    xm = _xm_mod()
    if xm is not None:
        xm.save(obj, str(path), master_only=True)
    else:
        torch.save(obj, str(path))
