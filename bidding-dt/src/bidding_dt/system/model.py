"""The system policy: a fixed-length-state FFN, plus the evolvable z adapters.

`SystemPolicy` exposes the same `forward_last(tokens, hand, vuln, row_len)`
interface as rl/model.py's `RLModel`, so the existing two-table rollout
(rl/rollout.py), league opponents and clipped PPO (rl/ppo.py) drive it
unchanged -- the state, including the four z blocks, is rebuilt from exactly
those arguments (system/state.py). That also makes the PPO logprob recompute
consistent with the rollout sampler: same inputs, same state, same distribution.

A *system* is (frozen hand-inference encoder, adapter). The policy owns the
adapter population (`adapters[i]`) and a name -> encoder registry; the outer
loop switches the active pair (system/population.py). Encoders are deliberately
*not* registered submodules: they are frozen inputs loaded from their own
checkpoints, so they stay out of `parameters()`, the optimizer, grad clipping
and every snapshot's state dict.
"""

import torch
import torch.nn as nn

from ..model.transformer import SwiGLU
from .state import N_SEATS, SUMMARY_DIM, frame_vuln, seat_frames, summarize, summary_features


class MLPBlock(nn.Module):
    """Pre-norm residual SwiGLU block -- the FFN a fixed-length state allows."""

    def __init__(self, cfg):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return x + self.drop(self.ffn(self.ln(x)))


class ZAdapter(nn.Module):
    """One system's projection of the frozen encoder's z into state features.

    This is the mutable part of a system variant: the outer loop breeds
    `child = parent + sigma * N(0,1)` (population.py) and, with
    `evo.inner_target` including "adapter", the inner PPO loop trains it too.

    The output is layer-normed so the z blocks enter the state at the same
    scale as the 0/1 hand and one-hot summary blocks -- without it a
    small-init projection is ~100x smaller than its neighbours and the FFN
    simply learns to ignore what the encoders say.
    """

    def __init__(self, z_in: int, d_z: int, hidden: int = 0):
        super().__init__()
        proj = (nn.Sequential(nn.Linear(z_in, hidden), nn.SiLU(),
                              nn.Linear(hidden, d_z))
                if hidden > 0 else nn.Linear(z_in, d_z))
        self.net = nn.Sequential(proj, nn.LayerNorm(d_z))

    def forward(self, z):
        return self.net(z)


class ZCache:
    """Bounded memo (FIFO eviction) of encoder code indices, keyed by
    (encoder, vuln, prefix bytes).

    z depends only on the public auction and the seat frame, and PPO recomputes
    logprobs for the same rows `ppo_epochs` times per iteration, so what is
    cached is the quantizer's *code indices* (k int64 per seat) -- re-embedding
    them through the frozen codebook reproduces the straight-through output
    exactly, at a fraction of a transformer pass, and keeps the rollout and the
    recompute bit-identical even under autocast.
    """

    def __init__(self, max_entries: int = 50_000):
        self.max = int(max_entries)
        self._store: dict = {}
        self._order: list = []

    def get(self, key):
        return self._store.get(key) if self.max > 0 else None

    def put(self, key, value):
        if self.max <= 0:
            return
        if key in self._store:                    # keep _order 1:1 with _store
            self._store[key] = value
            return
        self._store[key] = value
        self._order.append(key)
        while len(self._order) > self.max:
            self._store.pop(self._order.pop(0), None)

    def clear(self):
        self._store.clear()
        self._order.clear()

    def __len__(self):
        return len(self._store)


class SystemPolicy(nn.Module):
    """[own hand | auction summary | legal mask | z_self z_RHO z_pd z_LHO] ->
    SwiGLU MLP -> 39 call logits + a state value."""

    stochastic = False            # greedy eval may argmax (see rl/rollout._act)

    def __init__(self, cfg, z_in: int, n_systems: int = 1):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.z_in = z_in
        self.state_dim = cfg.hand_dim + SUMMARY_DIM + N_SEATS * cfg.d_z
        self.adapters = nn.ModuleList(ZAdapter(z_in, cfg.d_z, cfg.z_hidden)
                                     for _ in range(n_systems))
        self.active = 0
        self.in_proj = nn.Linear(self.state_dim, d)
        self.ln_in = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(MLPBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.policy_head = nn.Linear(d, cfg.n_actions)
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(),
                                        nn.Linear(d, 1))
        # name -> frozen HandVQ, held in a plain dict and referenced by name:
        # assigning a Module to self would register it, dragging 2.5M frozen
        # parameters into parameters(), the optimizer and every checkpoint
        self._encoders: dict = {}
        self.enc_name: str | None = None
        self.cache = ZCache(cfg.z_cache)

        self.apply(self._init_weights)
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # -- system registry -----------------------------------------------------

    def add_encoder(self, name: str, encoder):
        """Register a frozen hand-inference encoder (hand/model.py HandVQ)."""
        shape = (encoder.cfg.n_codes, encoder.cfg.d_model)
        for other in self._encoders.values():
            assert (other.cfg.n_codes, other.cfg.d_model) == shape, \
                "all encoders in a population must share n_codes and d_model"
        for p in encoder.parameters():
            p.requires_grad_(False)
        self._encoders[name] = encoder.eval()
        if self.enc_name is None:
            self.set_encoder(name)
        return self

    def set_encoder(self, name: str):
        if name not in self._encoders:
            raise KeyError(f"unknown encoder {name!r}; have {list(self._encoders)}")
        self.enc_name = name

    def set_system(self, encoder_name: str, adapter: int):
        self.set_encoder(encoder_name)
        self.active = int(adapter)

    @property
    def encoder(self):
        """The active frozen encoder (None before the first add_encoder)."""
        return None if self.enc_name is None else self._encoders[self.enc_name]

    def encoder_names(self) -> list:
        return list(self._encoders)

    def trainable_params(self, adapter: bool = False):
        """Policy trunk/heads (+ adapters when the inner loop trains them)."""
        skip = self.adapters if not adapter else ()
        skip_ids = {id(p) for m in skip for p in m.parameters()}
        return [p for p in self.parameters()
                if p.requires_grad and id(p) not in skip_ids]

    # -- state ---------------------------------------------------------------

    def _cache_keys(self, tokens, row_len, vuln):
        if self.cache.max <= 0:
            return None
        tk = tokens.cpu().numpy()
        rl = row_len.cpu().numpy()
        vv = vuln.cpu().numpy()
        return [(self.enc_name, int(vv[r]), tk[r, :int(rl[r])].tobytes())
                for r in range(tk.shape[0])]

    @torch.no_grad()
    def _codes(self, tokens, row_len, vuln) -> torch.Tensor:
        """(B, N_SEATS, k) code indices of the four seat frames (memoized)."""
        enc = self.encoder
        if enc is None:
            raise RuntimeError("no encoder registered (policy.add_encoder)")
        k = enc.cfg.n_codes
        B = tokens.shape[0]
        out = torch.zeros(B, N_SEATS, k, dtype=torch.long, device=tokens.device)
        keys = self._cache_keys(tokens, row_len, vuln)
        miss = []
        if keys is None:
            miss = list(range(B))
        else:
            for r, key in enumerate(keys):
                got = self.cache.get(key)
                if got is None:
                    miss.append(r)
                else:
                    out[r] = got
        if miss:
            rows = torch.as_tensor(miss, dtype=torch.long, device=tokens.device)
            # fp32 regardless of the surrounding autocast: the codes must be
            # the same in the rollout and in every PPO recompute
            with torch.autocast(getattr(tokens.device, "type", "cpu"),
                                enabled=False):
                sub = summarize(tokens[rows], row_len[rows])
                frames, fmask, _ = seat_frames(tokens[rows], sub)
                mem, mm = enc.memory(frames, frame_vuln(vuln[rows]), None, fmask)
                _z, aux = enc.quantize(mem, mm)
            codes = aux["idx"].view(N_SEATS, len(miss), k).permute(1, 0, 2)
            out[rows] = codes
            if keys is not None:
                cpu = codes.cpu()
                for i, r in enumerate(miss):
                    self.cache.put(keys[r], cpu[i])
        return out

    def _z_features(self, codes) -> torch.Tensor:
        """Codes -> frozen-encoder z -> the active adapter -> (B, 4*d_z)."""
        enc = self.encoder
        k = enc.cfg.n_codes
        with torch.no_grad():
            z = enc.from_code(enc.vq.codebook(codes.reshape(-1, k)))
            z = z.reshape(codes.shape[0], N_SEATS, -1).float()   # (B,4,k*d)
        z = self.adapters[self.active](z)                        # (B,4,d_z)
        return z.reshape(z.shape[0], -1)

    def state(self, tokens, hand, vuln, row_len) -> torch.Tensor:
        """The fixed-length decision state (B, state_dim)."""
        s = summarize(tokens, row_len)
        z = self._z_features(self._codes(tokens, row_len, vuln))
        feats = summary_features(s, vuln).to(z.dtype)
        return torch.cat([hand.to(z.dtype), feats, z], dim=-1)

    def forward_last(self, tokens, hand, vuln, row_len):
        """Logits + value for the decision position of each row."""
        h = self.drop(self.ln_in(self.in_proj(
            self.state(tokens, hand, vuln, row_len))))
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def forward(self, tokens, hand, vuln, row_len):
        return self.forward_last(tokens, hand, vuln, row_len)

    # -- misc ----------------------------------------------------------------

    def freeze(self) -> "SystemPolicy":
        """Frozen copy for use as a league opponent (cache off)."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.cache = ZCache(0)
        return self.eval()

    def num_params(self, with_adapters: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not with_adapters:
            n -= sum(p.numel() for a in self.adapters for p in a.parameters())
        return n


def load_system_policy(ck: dict, encoders: dict, device="cpu",
                       encoder: str | None = None) -> SystemPolicy:
    """Rebuild a frozen-eval SystemPolicy from a system/train.py checkpoint.

    `encoders` maps names to loaded HandVQ models (the checkpoint stores only
    their names/configs, never their weights); `encoder` selects the active
    one, defaulting to the checkpoint's.
    """
    from .config import SystemModelConfig
    cfg = SystemModelConfig(**ck["model_cfg"])
    policy = SystemPolicy(cfg, z_in=ck["z_in"],
                          n_systems=ck.get("n_systems", 1)).to(device)
    policy.load_state_dict(ck["model"])
    for name, enc in encoders.items():
        policy.add_encoder(name, enc.to(device))
    name = encoder or ck.get("enc_name")
    if name is not None and name in policy.encoder_names():
        policy.set_encoder(name)
    policy.active = int(ck.get("active", 0)) % len(policy.adapters)
    return policy.eval()
