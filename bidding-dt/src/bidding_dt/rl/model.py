"""Actor-critic wrapper: BiddingDT trunk + policy head (reused) + value head.

The value head reads the hidden state at each decision position (the last
real token of a hero-rotated row; trailing pads are irrelevant under causal
attention), predicting the expected final par-diff reward for the acting
seat's side.
"""

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..model.transformer import BiddingDT


class RLModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.dt = BiddingDT(cfg)
        d = cfg.d_model
        self.value_head = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward_last(self, tokens, hand, vuln, row_len):
        """Logits + value at the decision position (row_len - 1) of each row."""
        h = self.dt.hidden(tokens, hand, vuln)                    # (B,L,d)
        idx = (row_len - 1).view(-1, 1, 1).expand(-1, 1, h.shape[-1])
        h_last = h.gather(1, idx).squeeze(1)                      # (B,d)
        return self.dt.head(h_last), self.value_head(h_last).squeeze(-1)

    def load_bc(self, bc_state: dict, strict: bool = True):
        """Load a supervised BiddingDT checkpoint (runs/*/best.pt 'model')."""
        self.dt.load_state_dict(bc_state, strict=strict)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def load_rl_ckpt(path, device="cpu"):
    """Load an RL checkpoint -> (RLModel, ckpt dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = RLModel(ModelConfig(**ck["model_cfg"]))
    model.load_state_dict(ck["model"])
    return model.to(device), ck


def build_from_ckpt(ck: dict, device="cpu") -> tuple[RLModel, str]:
    """(RLModel, kind) from a loaded BC or RL checkpoint dict.

    kind is "rl" when the state dict is a full RLModel (dt.* + value head),
    "bc" when it is a bare BiddingDT (supervised train.py checkpoint). The
    architecture is taken from the checkpoint's own model_cfg, so BC and RL
    models of any size interoperate.
    """
    model = RLModel(ModelConfig(**ck["model_cfg"]))
    state = ck["model"]
    kind = "rl" if any(k.startswith("dt.") for k in state) else "bc"
    if kind == "rl":
        model.load_state_dict(state)
    else:
        model.load_bc(state)
    return model.to(device).eval(), kind


def load_any_ckpt(path, device="cpu") -> RLModel:
    """Load a BC (BiddingDT) or RL (RLModel) checkpoint as a frozen-eval RLModel."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model, _ = build_from_ckpt(ck, device=device)
    return model
