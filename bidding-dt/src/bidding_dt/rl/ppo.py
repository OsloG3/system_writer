"""PPO update over a finished RolloutBuffer.

Clipped surrogate + value MSE + entropy bonus + optional KL(pi || pi_BC)
anchor (annealed externally). Everything runs on decision states in the
supervised representation; masks keep distributions over legal calls only.

XLA/TPU (static=True): tensor shapes must not change between steps or the
TPU recompiles, so the row count is padded up to whole minibatches. Pad rows
are cyclic repeats of real rows and are excluded from every loss via a valid
mask, so the update is numerically identical to the dynamic path. Per-step
stats are accumulated on-device and synced to the host once at the end
(.item()/float() per minibatch would stall the TPU pipeline each time).
Multi-core runs pass world>1: gradients are all-reduced (averaged) across
ranks before the optimizer step, making the update equivalent to one big
minibatch over all ranks' rollouts.
"""

import numpy as np
import torch
from torch.distributions import Categorical

from .. import xla
from .config import RLConfig

_STAT_KEYS = ("policy_loss", "value_loss", "entropy", "kl", "clipfrac",
              "approx_kl")


def _masked_dist(logits, mask):
    return Categorical(logits=logits.float().masked_fill(~mask, float("-inf")))


def ppo_update(model, ref_model, buf_tensors: dict, opt, device,
               cfg: RLConfig, kl_beta: float, ent_coef: float,
               rng: np.random.Generator, use_amp: bool = False,
               temp: float = 1.0, static: bool = False,
               world: int = 1, n_rows: int | None = None) -> dict:
    model.train()
    on_xla = xla.is_xla(device)
    n_alloc = buf_tensors["action"].shape[0]
    n = n_rows if n_rows is not None else n_alloc
    assert 0 < n <= n_alloc
    mb = cfg.minibatch_size
    adv = buf_tensors["adv"][:n]
    if n > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buf_tensors = {**buf_tensors,
                       "adv": torch.cat([adv, buf_tensors["adv"][n:]])}
    if static:
        stats = {k: torch.zeros((), device=device) for k in _STAT_KEYS}
    else:
        stats = {k: 0.0 for k in _STAT_KEYS}
    n_updates = 0

    # padded row count so every minibatch has shape (mb, ...) -- static graph
    n_pad = max(mb, -(-n // mb) * mb) if static else n

    for _ in range(cfg.ppo_epochs):
        perm = rng.permutation(n)
        if static:
            perm = np.resize(perm, n_pad)  # cyclic tile: pads with real rows
        for start in range(0, n_pad, mb):
            rows = perm[start:start + mb]
            sl = torch.as_tensor(rows, device=device)
            b = {k: v[sl] for k, v in buf_tensors.items()}
            if static:
                # valid mask excludes pad rows from every reduction; shape
                # stays (mb,) regardless of n
                vm = (start + torch.arange(len(rows), device=device)
                      ).float().lt(n).float()
                denom = vm.sum().clamp(min=1.0)

                def red(x, vm=vm, denom=denom):
                    return (x * vm).sum() / denom
            else:
                def red(x):
                    return x.mean()
            with torch.autocast(device.type if isinstance(device, torch.device)
                                else "cpu", dtype=torch.bfloat16, enabled=use_amp):
                logits, values = model.forward_last(
                    b["tokens"], b["hand"], b["vuln"], b["row_len"])
            # same temperature as the rollout sampler -> ratio starts at 1
            logits = logits.float()
            if temp != 1.0:
                logits = logits / temp
            dist = _masked_dist(logits, b["mask"])
            logp = dist.log_prob(b["action"])
            ratio = torch.exp(logp - b["logprob"])
            surr1 = ratio * adv[sl]
            surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[sl]
            policy_loss = -red(torch.min(surr1, surr2))
            value_loss = red((values.float() - b["ret"]) ** 2)
            entropy = red(dist.entropy())
            loss = (policy_loss + cfg.vf_coef * value_loss
                    - ent_coef * entropy)

            kl_mean = None
            if ref_model is not None:
                with torch.no_grad():
                    ref_logits, _ = ref_model.forward_last(
                        b["tokens"], b["hand"], b["vuln"], b["row_len"])
                if temp != 1.0:
                    ref_logits = ref_logits.float() / temp
                # masked KL(pi || ref) over legal calls only; both sides are
                # renormalized over the mask (unmasked softmax would leak
                # probability onto illegal calls and understate the KL)
                ref_logp = ref_logits.float().masked_fill(
                    ~b["mask"], float("-inf")).log_softmax(-1)
                p = dist.probs
                diff = torch.where(b["mask"], dist.logits - ref_logp,
                                   torch.zeros_like(ref_logp))
                kl = (p * diff).sum(-1)
                kl_mean = red(kl)
                if kl_beta > 0:
                    loss = loss + kl_beta * kl_mean

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if world > 1:
                xla.average_grads(model.parameters(), world)
            if on_xla:
                xla.optimizer_step(opt)
            else:
                opt.step()

            with torch.no_grad():
                approx_kl = red(b["logprob"] - logp)
                clipfrac = red(((ratio - 1).abs() > cfg.clip_eps).float())
            vals = {"policy_loss": policy_loss.detach(),
                    "value_loss": value_loss.detach(),
                    "entropy": entropy.detach(),
                    "kl": kl_mean.detach() if kl_mean is not None
                          else torch.zeros((), device=device),
                    "clipfrac": clipfrac,
                    "approx_kl": approx_kl}
            if static:
                for k, v in vals.items():
                    stats[k] += v
            else:
                for k, v in vals.items():
                    stats[k] += float(v)
            n_updates += 1

    model.eval()
    if static:
        # single host sync for all metrics instead of one per minibatch
        packed = torch.stack([stats[k] for k in _STAT_KEYS]).cpu()
        stats = {k: float(v) for k, v in zip(_STAT_KEYS, packed)}
    return {k: v / max(n_updates, 1) for k, v in stats.items()}
