"""Bidding decision transformer: pre-norm MHA + SwiGLU FFN.

Sequence: [COND, HAND, padded hero-rotated calls...]. Trailing PADs never
affect real positions under causal attention, so no key-padding mask is
needed and SDPA can use the fast is_causal path.

The COND token is a learned "unconditional" embedding for now; it is the
designated slot for return-to-go conditioning in the offline-RL phase.
"""

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..data.dataset import IGNORE, ROLE_COND, ROLE_HAND


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_in = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        g, u = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(g) * u)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = cfg.dropout
        self.proj_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class BiddingDT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, d)
        self.role_emb = nn.Embedding(cfg.n_roles, d)
        self.vuln_emb = nn.Embedding(cfg.n_vuln, d)
        self.cond_emb = nn.Parameter(torch.zeros(1, 1, d))
        self.hand_mlp = nn.Sequential(nn.Linear(cfg.hand_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)

        roles = torch.arange(cfg.max_seq_len)
        roles = torch.where(roles == 0, ROLE_COND,
                            torch.where(roles == 1, ROLE_HAND, (roles - 2) % 4))
        self.register_buffer("roles", roles, persistent=False)

        self.apply(self._init_weights)
        nn.init.normal_(self.cond_emb, std=0.02)
        scale = 0.02 / math.sqrt(2 * cfg.n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, std=scale)
            nn.init.normal_(blk.ffn.w_out.weight, std=scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def hidden(self, tokens, hand, vuln):
        """Post-block, post-ln_f hidden states (B, L, d_model)."""
        B, L = tokens.shape
        cond = self.cond_emb + self.vuln_emb(vuln).unsqueeze(1)
        hand_tok = self.hand_mlp(hand).unsqueeze(1)
        call_emb = self.tok_emb(tokens[:, 2:])
        pos = torch.arange(L, device=tokens.device)
        x = torch.cat([cond, hand_tok, call_emb], dim=1)
        x = x + self.pos_emb(pos).unsqueeze(0) + self.role_emb(self.roles[:L]).unsqueeze(0)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward(self, tokens, hand, vuln, targets=None):
        logits = self.head(self.hidden(tokens, hand, vuln))
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.cfg.vocab_size),
            targets[:, :-1].reshape(-1),
            ignore_index=IGNORE,
        )
        return logits, loss

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel() + self.role_emb.weight.numel()
            n -= self.vuln_emb.weight.numel() + self.cond_emb.numel()
        return n


def build_model(cfg: ModelConfig) -> BiddingDT:
    return BiddingDT(cfg)


def model_config_from_ckpt(ckpt: dict) -> ModelConfig:
    return ModelConfig(**ckpt["model_cfg"])
