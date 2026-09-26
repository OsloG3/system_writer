"""PPO update over a finished RolloutBuffer.

Clipped surrogate + value MSE + entropy bonus + optional KL(pi || pi_BC)
anchor (annealed externally). Everything runs on decision states in the
supervised representation; masks keep distributions over legal calls only.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .config import RLConfig


def _masked_dist(logits, mask):
    return Categorical(logits=logits.float().masked_fill(~mask, float("-inf")))


def ppo_update(model, ref_model, buf_tensors: dict, opt, device,
               cfg: RLConfig, kl_beta: float, ent_coef: float,
               rng: np.random.Generator, use_amp: bool = False,
               temp: float = 1.0) -> dict:
    model.train()
    n = buf_tensors["action"].shape[0]
    adv = buf_tensors["adv"]
    if n > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {k: 0.0 for k in ("policy_loss", "value_loss", "entropy", "kl",
                              "clipfrac", "approx_kl")}
    n_updates = 0

    for _ in range(cfg.ppo_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, cfg.minibatch_size):
            rows = perm[start:start + cfg.minibatch_size]
            sl = torch.as_tensor(rows, device=device)
            b = {k: v[sl] for k, v in buf_tensors.items()}
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
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.float(), b["ret"])
            entropy = dist.entropy().mean()
            loss = (policy_loss + cfg.vf_coef * value_loss
                    - ent_coef * entropy)

            kl_val = 0.0
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
                kl_val = float(kl.mean().detach())
                if kl_beta > 0:
                    loss = loss + kl_beta * kl.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            with torch.no_grad():
                approx_kl = float((b["logprob"] - logp).mean())
                clipfrac = float(((ratio - 1).abs() > cfg.clip_eps).float().mean())
            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy.detach())
            stats["kl"] += kl_val
            stats["clipfrac"] += clipfrac
            stats["approx_kl"] += approx_kl
            n_updates += 1

    model.eval()
    return {k: v / max(n_updates, 1) for k, v in stats.items()}
