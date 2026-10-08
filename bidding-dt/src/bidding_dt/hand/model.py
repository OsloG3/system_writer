"""VQ encoder-decoder that samples one player's hand from the public auction.

Encoder (public information only): `[COND(vuln), KNOWN(mask), padded calls...]`
in the same hero-rotated frame as the supervised dataset -- the target seat is
role 0 and the leading PADs say where it sits versus the dealer. Trailing batch
padding is masked out (leading frame PADs are *not* padding: their role
embedding is the positional information).

Bottleneck: `n_codes` learned query slots cross-attend the encoder memory and
each is vector-quantized against a codebook of `codebook_size` entries, so z is
k=16 discrete codes. The decoder sees **only** z -- no auction, no memory --
and draws the 13 cards autoregressively in ascending card-id order, each step a
softmax over the 52 cards with already-drawn and caller-excluded cards removed.

Why a discrete bottleneck + sampling instead of a regression head: an auction
like a multi-2D is *multi-meaning* (long hearts or long spades), and the mean of
those hands is a hand nobody holds. The VQ codes are a discrete latent the
encoder can split between the two readings, and the decoder samples a hand from
whichever reading it got, so the output is always a plausible whole hand rather
than an average. `sample(codes=True)` additionally draws the codes from the
near-tied alternatives (see VectorQuantizer.forward), exploring the competing
readings directly.

Masking: `excl` is a (B, 52) 0/1 vector in the data/hands.py indicator layout
(e.g. `encode_hand(my_own_hand)`) of cards the target cannot hold. It is fed to
the encoder as the KNOWN token (knowing 13 dead cards is genuinely informative)
and hard-masks the decoder's card softmax. Training uses random exclusions
(`train.mask_aug`) so inference-time masking is in distribution.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..data.dataset import ROLE_COND, ROLE_HAND
from ..data.hands import CARD_TO_COUNT
from ..model.transformer import SwiGLU
from .config import HandVQConfig

_NEG = torch.finfo(torch.float32).min   # "-inf" that stays finite under softmax


class MHA(nn.Module):
    """Self-attention (ctx=None) or cross-attention, SDPA-backed.

    mask / ctx_mask are boolean (B,1,1,S) key masks, True = attend.
    """

    def __init__(self, cfg: HandVQConfig, causal: bool = False):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.causal = causal
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = cfg.dropout
        self.proj_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, ctx=None, mask=None, ctx_mask=None):
        self_attn = ctx is None
        if self_attn:
            ctx, m = x, mask
        else:
            m = ctx_mask
        B, T, C = x.shape
        S = ctx.shape[1]
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(ctx).view(B, S, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=m,
            is_causal=self.causal and self_attn and m is None,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        return self.proj_drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class Block(nn.Module):
    """Pre-norm block: self-attn [+ cross-attn] + SwiGLU (as model/transformer)."""

    def __init__(self, cfg: HandVQConfig, causal: bool = False, cross: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MHA(cfg, causal=causal)
        self.ln_c = nn.LayerNorm(cfg.d_model) if cross else None
        self.cross = MHA(cfg) if cross else None
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, ctx=None, mask=None, ctx_mask=None):
        x = x + self.drop(self.attn(self.ln1(x), mask=mask))
        if self.cross is not None:
            x = x + self.drop(self.cross(self.ln_c(x), ctx=ctx, ctx_mask=ctx_mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


def code_perplexity(usage: torch.Tensor) -> float:
    """exp(entropy) of the code-usage distribution: how many codes are live."""
    p = usage / usage.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum())) if p.numel() else 0.0


class VectorQuantizer(nn.Module):
    """Product quantizer: each of the k slots picks 1 of V codebook vectors.

    Straight-through estimator, with the usual codebook + commitment terms.
    Usage is tracked as an EMA over winning codes so (a) `perplexity` reports
    how much of the codebook is alive and (b) dead codes can be reseeded from
    live encoder outputs -- a k=16 x V=1024 product codebook otherwise collapses
    onto a handful of codes and the bottleneck stops carrying hand shape.
    """

    def __init__(self, cfg: HandVQConfig):
        super().__init__()
        self.cfg = cfg
        self.codebook = nn.Embedding(cfg.codebook_size, cfg.code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / math.sqrt(cfg.code_dim),
                         1.0 / math.sqrt(cfg.code_dim))
        # persisted: a resumed run must not read an all-zero usage EMA and
        # "revive" (i.e. wipe) the whole trained codebook on its next check
        self.register_buffer("usage", torch.zeros(cfg.codebook_size))

    def forward(self, z: torch.Tensor, sample: bool = False, temp: float = 1.0):
        """(B,k,code_dim) -> (quantized, aux{vq, commit, idx, ppl, z_pre}).

        `sample` draws each slot's code instead of taking the nearest one, which
        is how one auction yields the competing readings of a multi-meaning bid.
        The temperature is scale-free: distances are measured from the nearest
        code in units of the nearest-to-runner-up gap, so the runner-up is drawn
        exp(-1/temp) as often as the winner (temp=1 -> 37%, 0.5 -> 14%, 2 -> 61%)
        while codes far away keep essentially no mass.
        """
        B, K, D = z.shape
        flat = z.reshape(-1, D)
        f32, cb = flat.float(), self.codebook.weight.float()
        dist = (f32.pow(2).sum(1, keepdim=True) - 2.0 * f32 @ cb.t() + cb.pow(2).sum(1))
        if sample and temp > 0:
            best = dist.topk(2, dim=-1, largest=False).values
            gap = (best[:, 1:] - best[:, :1]).clamp_min(1e-6)
            idx = Categorical(logits=(best[:, :1] - dist) / (temp * gap)).sample()
        else:
            idx = dist.argmin(-1)
        e = self.codebook(idx)
        zq = flat + (e - flat).detach()          # straight-through
        if self.training:
            with torch.no_grad():
                p = torch.bincount(idx, minlength=self.cfg.codebook_size).float()
                p = p / p.sum().clamp_min(1.0)
                d = self.cfg.usage_decay
                self.usage.mul_(d).add_(p, alpha=1.0 - d)
        return zq.view(B, K, D), {
            "vq": F.mse_loss(e.float(), f32.detach()) * self.cfg.codebook_weight,
            "commit": F.mse_loss(f32, e.detach().float()) * self.cfg.commit_beta,
            "idx": idx.view(B, K),
            "ppl": code_perplexity(self.usage),
            "z_pre": flat.detach(),
        }

    @torch.no_grad()
    def revive(self, z_pre: torch.Tensor, thresh: float | None = None) -> int:
        """Reseed (near-)unused codes from encoder outputs; returns the count.

        thresh=None uses cfg.revive_frac/V; pass a large value to reseed the
        whole codebook (data-dependent init on the first step).
        """
        t = self.cfg.revive_frac / self.cfg.codebook_size if thresh is None else thresh
        dead = torch.nonzero(self.usage <= t).squeeze(-1)
        if dead.numel() == 0:
            return 0
        flat = z_pre.reshape(-1, z_pre.shape[-1]).float()
        pick = torch.randint(0, flat.shape[0], (int(dead.numel()),), device=flat.device)
        self.codebook.weight.data[dead] = flat[pick]
        # half of the uniform share: revived codes stay alive for a few hundred
        # steps and are reseeded again if they keep losing
        self.usage[dead] = 0.5 / self.cfg.codebook_size
        return int(dead.numel())


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filter: -inf outside the smallest set of cards holding mass p."""
    ordered, idx = logits.sort(-1, descending=True)
    cum = ordered.softmax(-1).cumsum(-1)
    drop = cum > p
    drop[..., 1:] = drop[..., :-1].clone()
    drop[..., 0] = False
    return logits.scatter(-1, idx, ordered.masked_fill(drop, _NEG))


class HandVQ(nn.Module):
    """Auction (+ known-cards mask) -> k VQ codes -> a sampled 13-card hand."""

    def __init__(self, cfg: HandVQConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # -- encoder (public information)
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, d)
        self.role_emb = nn.Embedding(cfg.n_roles, d)
        self.vuln_emb = nn.Embedding(cfg.n_vuln, d)
        self.cond_emb = nn.Parameter(torch.zeros(1, 1, d))
        self.known_mlp = nn.Sequential(nn.Linear(cfg.n_cards, d), nn.SiLU(),
                                       nn.Linear(d, d))
        self.drop = nn.Dropout(cfg.dropout)
        self.enc_blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_enc_layers))
        self.ln_enc = nn.LayerNorm(d)
        # -- VQ bottleneck (k query slots -> k codes)
        self.slots = nn.Parameter(torch.zeros(1, cfg.n_codes, d))
        self.slot_blocks = nn.ModuleList(Block(cfg, cross=True)
                                         for _ in range(cfg.n_slot_layers))
        self.ln_slots = nn.LayerNorm(d)
        self.to_code = nn.Linear(d, cfg.code_dim, bias=False)
        self.vq = VectorQuantizer(cfg)
        self.from_code = nn.Linear(cfg.code_dim, d, bias=False)
        # -- decoder (z only)
        self.card_emb = nn.Embedding(cfg.n_cards + 1, d)   # +1 = start-of-hand
        self.step_emb = nn.Embedding(cfg.hand_len, d)
        self.dec_blocks = nn.ModuleList(Block(cfg, causal=True, cross=True)
                                        for _ in range(cfg.n_dec_layers))
        self.ln_dec = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.n_cards, bias=False)

        roles = torch.cat([torch.tensor([ROLE_COND, ROLE_HAND], dtype=torch.long),
                           torch.arange(cfg.max_seq_len - 2, dtype=torch.long) % 4])
        self.register_buffer("roles", roles, persistent=False)
        self.register_buffer("card_ids", torch.arange(cfg.n_cards), persistent=False)
        self.register_buffer("card_to_count",
                             torch.from_numpy(CARD_TO_COUNT.astype("int64")),
                             persistent=False)

        self.apply(self._init_weights)
        nn.init.normal_(self.cond_emb, std=0.02)
        nn.init.normal_(self.slots, std=0.02)
        scale = 0.02 / math.sqrt(2 * max(cfg.n_enc_layers, cfg.n_dec_layers))
        for blk in (*self.enc_blocks, *self.slot_blocks, *self.dec_blocks):
            nn.init.normal_(blk.attn.proj.weight, std=scale)
            nn.init.normal_(blk.ffn.w_out.weight, std=scale)
            if blk.cross is not None:
                nn.init.normal_(blk.cross.proj.weight, std=scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding) and m is not self.vq.codebook:
            nn.init.normal_(m.weight, std=0.02)

    # -- encoder -------------------------------------------------------------

    def memory(self, tokens, vuln, excl=None, mask=None):
        """Encoder pass -> (memory (B,2+S,d), key mask (B,1,1,2+S) or None).

        `tokens` is the supervised-dataset layout `[COND, HAND, pads, calls]`;
        the two leading placeholders are dropped and rebuilt here as COND (vuln)
        and KNOWN (excluded cards). `mask` is (B, L) bool, True = real position
        (None = no trailing padding).
        """
        calls = tokens[:, 2:]
        B, S = calls.shape
        if S + 2 > self.cfg.max_seq_len:
            raise ValueError(f"auction of {S} call slots exceeds max_seq_len "
                             f"{self.cfg.max_seq_len} (2 + {S})")
        dev = tokens.device
        cond = self.cond_emb + self.vuln_emb(vuln).unsqueeze(1)
        known_in = (torch.zeros(B, self.cfg.n_cards, device=dev) if excl is None
                    else excl.reshape(B, self.cfg.n_cards).float())
        x = torch.cat([cond, self.known_mlp(known_in).unsqueeze(1),
                       self.tok_emb(calls)], dim=1)
        pos = torch.arange(S + 2, device=dev)
        x = x + self.pos_emb(pos).unsqueeze(0) + self.role_emb(self.roles[:S + 2]).unsqueeze(0)
        x = self.drop(x)
        attn = None if mask is None else mask[:, None, None, :].bool()
        for blk in self.enc_blocks:
            x = blk(x, mask=attn)
        return self.ln_enc(x), attn

    def quantize(self, mem, mem_mask=None, sample: bool = False, temp: float = 1.0):
        """Memory -> z (B,k,d) through the k-slot product quantizer."""
        s = self.slots.expand(mem.shape[0], -1, -1)
        for blk in self.slot_blocks:
            s = blk(s, ctx=mem, ctx_mask=mem_mask)
        z, aux = self.vq(self.to_code(self.ln_slots(s)), sample=sample, temp=temp)
        return self.from_code(z), aux

    # -- decoder -------------------------------------------------------------

    def _decode(self, z, inp):
        """Logits for each position of the drawn-card prefix `inp` (B,T)."""
        T = inp.shape[1]
        x = self.card_emb(inp) + self.step_emb(
            torch.arange(T, device=inp.device)).unsqueeze(0)
        x = self.drop(x)
        for blk in self.dec_blocks:
            x = blk(x, ctx=z)
        return self.head(self.ln_dec(x))

    def _banned(self, inp, excl, hand_len=None):
        """(B,T,n_cards) bool: cards that cannot be drawn at each step.

        Hands are drawn in ascending card id, so a card is impossible when it is
        <= the previous draw, excluded by the caller's known-cards mask, or so
        high that the rest of the hand could not fit above it (at step t there
        must be at least L-1-t unexcluded cards greater than it, L being the
        number of cards drawn -- 13 for a whole hand, fewer when the play bot
        samples only a seat's still-held cards). Together these make every sample
        a valid L-card hand whatever the decoder believes -- an early high draw
        cannot paint the remaining draws into a corner -- and they never touch a
        real target, whose t-th smallest card always has enough cards above it.
        """
        L = self.cfg.hand_len if hand_len is None else int(hand_len)
        B, T = inp.shape
        dev = inp.device
        prev = torch.where(inp >= self.cfg.n_cards,
                           torch.full_like(inp, -1), inp)
        ids = self.card_ids[None, None, :]
        banned = ids <= prev[:, :, None]
        allowed = torch.ones(B, self.cfg.n_cards, dtype=torch.bool, device=dev)
        if excl is not None:
            gone = self._excl_cards(excl)
            allowed = allowed & ~gone
            banned = banned | gone[:, None, :]
        above = allowed.flip(-1).cumsum(-1).flip(-1) - allowed.long()  # cards > c
        need = L - 1 - torch.arange(T, device=dev)
        return banned | (above[:, None, :] < need[None, :, None])

    def _excl_cards(self, excl):
        """(B,52) indicator-layout mask -> (B,52) bool in card-id order."""
        return excl.reshape(-1, self.cfg.n_cards)[:, self.card_to_count] > 0

    def forward(self, tokens, vuln, cards, excl=None, mask=None,
                code_temp: float | None = None):
        """Teacher-forced loss on `cards` (B,13) ascending card ids."""
        sample = code_temp is not None or self.cfg.train_code_temp > 0
        temp = self.cfg.train_code_temp if code_temp is None else code_temp
        mem, mem_mask = self.memory(tokens, vuln, excl, mask)
        z, aux = self.quantize(mem, mem_mask, sample=sample, temp=temp)
        bos = torch.full((cards.shape[0], 1), self.cfg.n_cards,
                         dtype=torch.long, device=cards.device)
        inp = torch.cat([bos, cards[:, :-1]], dim=1)
        # the targets are ascending and disjoint from `excl`, so masking can
        # never remove a target and the cross-entropy stays finite
        logits = self._decode(z, inp).float().masked_fill(
            self._banned(inp, excl), _NEG)
        ce = F.cross_entropy(logits.reshape(-1, self.cfg.n_cards), cards.reshape(-1))
        return {"loss": ce + aux["vq"] + aux["commit"], "ce": ce,
                "vq": aux["vq"], "commit": aux["commit"], "ppl": aux["ppl"],
                "codes": aux["idx"], "z_pre": aux["z_pre"], "logits": logits}

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def generate(self, z, excl=None, temp: float = 1.0, top_p: float | None = None,
                 greedy: bool = False, n_cards: int | None = None):
        """z (B,k,d) -> (B,L) card ids, drawn one at a time.

        L defaults to a whole 13-card hand; the play bot passes a smaller
        `n_cards` to draw only the cards a seat still holds mid-hand (its
        already-played cards are part of `excl`).
        """
        L = self.cfg.hand_len if n_cards is None else int(n_cards)
        if L < 0 or L > self.cfg.hand_len:
            raise ValueError(f"n_cards must be in 0..{self.cfg.hand_len}, got {L}")
        if excl is not None and excl.numel() and \
                int(self._excl_cards(excl).sum(1).max()) > self.cfg.n_cards - L:
            raise ValueError("exclusion mask leaves fewer than "
                             f"{L} cards")
        B = z.shape[0]
        dev = z.device
        out = torch.zeros(B, L, dtype=torch.long, device=dev)
        inp = torch.full((B, 1), self.cfg.n_cards, dtype=torch.long, device=dev)
        for t in range(L):
            logits = self._decode(z, inp)[:, -1].float()
            logits = logits.masked_fill(self._banned(inp, excl, hand_len=L)[:, -1], _NEG)
            if temp != 1.0:
                logits = logits / temp
            if top_p is not None and top_p < 1.0:
                logits = top_p_filter(logits, top_p)
            nxt = logits.argmax(-1) if greedy else Categorical(logits=logits).sample()
            out[:, t] = nxt
            inp = torch.cat([inp, nxt[:, None]], dim=1)
        return out

    @torch.no_grad()
    def sample(self, tokens, vuln, excl=None, mask=None, n: int = 1,
               temp: float = 1.0, top_p: float | None = None, greedy: bool = False,
               codes: bool = False, code_temp: float = 1.0, n_cards: int | None = None):
        """Draw `n` hands per row -> (cards (B,n,L), codes (B,n,k)).

        codes=True samples the VQ codes from the softmax over negative code
        distances instead of taking the nearest one, which is how one auction
        yields genuinely different hand *types* (the multi-meaning-bid case).
        `n_cards` < 13 draws only that many cards per hand (see generate).
        """
        mem, mem_mask = self.memory(tokens, vuln, excl, mask)
        if n > 1:
            mem = mem.repeat_interleave(n, dim=0)
            mem_mask = None if mem_mask is None else mem_mask.repeat_interleave(n, dim=0)
            excl = None if excl is None else excl.repeat_interleave(n, dim=0)
        z, aux = self.quantize(mem, mem_mask, sample=codes, temp=code_temp)
        cards = self.generate(z, excl, temp=temp, top_p=top_p, greedy=greedy,
                              n_cards=n_cards)
        B = tokens.shape[0]
        return (cards.view(B, n, -1),
                aux["idx"].view(B, n, -1) if n > 1 else aux["idx"].unsqueeze(1))

    @torch.no_grad()
    def init_codebook(self, tokens, vuln, excl=None, mask=None) -> int:
        """Seed every code from one batch of encoder outputs (step 0)."""
        mem, mem_mask = self.memory(tokens, vuln, excl, mask)
        z, aux = self.quantize(mem, mem_mask)
        return self.vq.revive(aux["z_pre"], thresh=float("inf"))

    @torch.no_grad()
    def revive_dead(self, z_pre: torch.Tensor) -> int:
        return self.vq.revive(z_pre)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= (self.pos_emb.weight.numel() + self.role_emb.weight.numel()
                  + self.vuln_emb.weight.numel() + self.cond_emb.numel()
                  + self.vq.codebook.weight.numel())
        return n


def build_hand_model(cfg: HandVQConfig) -> HandVQ:
    return HandVQ(cfg)


def hand_config_from_ckpt(ckpt: dict) -> HandVQConfig:
    return HandVQConfig(**ckpt["model_cfg"])


def load_hand_model(path, device="cpu") -> HandVQ:
    """Load a hand/train.py checkpoint -> frozen HandVQ on `device`."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = HandVQ(hand_config_from_ckpt(ck))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()
