"""Dataset of hero-rotated bidding views.

Sample i corresponds to deal `deals[i // 4]` with hero seat `i % 4`
(N=0,E=1,S=2,W=3). The auction is rotated so the hero acts at call
slots s with s % 4 == 0; when the hero is not the dealer, the frame is
front-padded with (4 - o) % 4 PAD tokens, o = (hero - dealer) % 4.

Sequence layout (length MAX_SEQ = 41):
  pos 0        COND token (placeholder; return-conditioning slot for RL)
  pos 1        HAND token (hero's encoded hand)
  pos 2..      calls region: PAD*p then the auction in hero-rotated order
Targets follow the standard LM shift: targets[t] = token at position t+1
at hero positions, IGNORE elsewhere.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .hands import HAND_DIM, HAND_SCALE
from .parse import MAX_CALLS
from .vocab import CALL_PAD, VOCAB_SIZE

MAX_SEQ = 2 + 3 + MAX_CALLS  # COND + HAND + <=3 pads + calls
IGNORE = -100

# Role ids for the role embedding: call slots use slot%4 (0=hero,1=RHO,
# 2=partner,3=LHO); COND and HAND positions get dedicated ids.
ROLE_COND = 4
ROLE_HAND = 5
N_ROLES = 6


class BiddingDataset(Dataset):
    def __init__(self, cache_dir, split="train", max_deals=None):
        cache = Path(cache_dir)
        self.hands = np.load(cache / "hands.npy", mmap_mode="r")
        self.calls = np.load(cache / "calls.npy", mmap_mode="r")
        self.lengths = np.load(cache / "lengths.npy", mmap_mode="r")
        self.dealer = np.load(cache / "dealer.npy", mmap_mode="r")
        self.vuln = np.load(cache / "vuln.npy", mmap_mode="r")
        self.deals = np.load(cache / f"{split}.npy")
        if max_deals is not None:
            self.deals = self.deals[:max_deals]

    def __len__(self):
        return len(self.deals) * 4

    @staticmethod
    def vuln_class(vuln: int, hero: int) -> int:
        """0=neither, 1=we vuln only, 2=they vuln only, 3=both."""
        hero_ns = hero in (0, 2)
        we = vuln == 3 or (vuln == 1 and hero_ns) or (vuln == 2 and not hero_ns)
        they = vuln == 3 or (vuln == 2 and hero_ns) or (vuln == 1 and not hero_ns)
        return int(we) + 2 * int(they)

    def __getitem__(self, i):
        deal = int(self.deals[i >> 2])
        hero = i & 3
        n = int(self.lengths[deal])
        dealer = int(self.dealer[deal])
        o = (hero - dealer) % 4
        p = (4 - o) % 4

        tokens = np.zeros(MAX_SEQ, dtype=np.int64)
        tokens[2 + p: 2 + p + n] = self.calls[deal, :n]

        targets = np.full(MAX_SEQ, IGNORE, dtype=np.int64)
        dnos = np.full(MAX_SEQ, -1, dtype=np.int8)  # hero decision number
        seq_end = 2 + p + n
        pos = 2 + p + o  # first hero call position (2 if dealer else 6)
        j = 0
        while pos < seq_end:
            targets[pos - 1] = tokens[pos]
            dnos[pos - 1] = j
            pos += 4
            j += 1

        hand = np.asarray(self.hands[deal, hero], dtype=np.float32) * HAND_SCALE
        vuln_cls = self.vuln_class(int(self.vuln[deal]), hero)
        return tokens, hand, np.int64(vuln_cls), targets, dnos, np.int64(seq_end)


def collate(batch):
    """Pad to batch max length; returns dict of tensors."""
    seq_len = max(int(b[5]) for b in batch)
    bsz = len(batch)
    tokens = torch.zeros(bsz, seq_len, dtype=torch.long)
    hand = torch.zeros(bsz, HAND_DIM, dtype=torch.float32)
    vuln = torch.zeros(bsz, dtype=torch.long)
    targets = torch.full((bsz, seq_len), IGNORE, dtype=torch.long)
    dnos = torch.full((bsz, seq_len), -1, dtype=torch.short)
    for k, (t, h, v, tg, dn, _sl) in enumerate(batch):
        L = int(_sl)
        tokens[k, :L] = torch.from_numpy(t[:L])
        hand[k] = torch.from_numpy(h)
        vuln[k] = int(v)
        targets[k, : L - 1] = torch.from_numpy(tg[: L - 1])
        dnos[k, : L - 1] = torch.from_numpy(dn[: L - 1])
    return {"tokens": tokens, "hand": hand, "vuln": vuln, "targets": targets, "dnos": dnos}
