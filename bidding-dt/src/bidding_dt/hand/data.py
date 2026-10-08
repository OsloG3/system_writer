"""Auction stores and hand-inference views.

A *store* is a directory of .npy arrays in the data/parse.py cache layout
(hands/calls/lengths/dealer/vuln + train/val/test deal splits): either the
human cache built from `training.txt`, or one written by hand/gen.py from the
bidding models' own rollouts. Both are read the same way, so a hand model can
be trained on human auctions, model auctions, or a mix.

A row is one *view*: a deal, a target seat, and an auction prefix. The prefix
is what makes the model useful mid-auction -- at any point the calls so far are
the whole public story about the target's hand, and inferring from a prefix is
the same task as inferring from the finished auction.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.dataset import BiddingDataset
from ..data.hands import HAND_DIM, HAND_LEN, hand_cards

ARRAYS = ("hands", "calls", "lengths", "dealer", "vuln")


def rotated_tokens(dealer: int, calls, seat: int) -> np.ndarray:
    """Auction in seat order (from `dealer`) -> `seat`-rotated token row.

    Layout is the supervised one: `[COND, HAND, pad*((4-o)%4), calls...]` with
    o = (seat - dealer) % 4, so the target seat is role 0 and the two leading
    slots are placeholders the model replaces with its own COND/KNOWN tokens.
    """
    calls = [int(c) for c in calls]
    p = (4 - (seat - dealer) % 4) % 4
    tokens = np.zeros(2 + p + len(calls), dtype=np.int64)
    tokens[2 + p:] = calls
    return tokens


@dataclass
class AuctionStore:
    """Auction + deal arrays, with the selected deal subset (`deals`)."""
    hands: np.ndarray     # (N,4,52) uint8 indicator, seat order N,E,S,W
    calls: np.ndarray     # (N,MAX_CALLS) uint8 call ids
    lengths: np.ndarray   # (N,) uint8 calls per auction
    dealer: np.ndarray    # (N,) uint8 dealer seat
    vuln: np.ndarray      # (N,) uint8 0=None 1=N-S 2=E-W 3=Both
    deals: np.ndarray     # (M,) deal indices this store contributes
    name: str = ""
    path: str | None = None

    @classmethod
    def load(cls, path, split: str | None = None, max_deals: int | None = None,
             name: str | None = None) -> "AuctionStore":
        """Load a cache dir; `split` selects train/val/test.npy deal indices."""
        path = Path(path)
        arrays = {}
        for key in ARRAYS:
            f = path / f"{key}.npy"
            if not f.exists():
                raise FileNotFoundError(f"{f} missing -- not an auction store")
            arrays[key] = np.load(f, mmap_mode="r")
        n_deals = len(arrays["lengths"])
        if split is not None:
            deals = np.load(path / f"{split}.npy")
        else:
            deals = np.arange(n_deals, dtype=np.int64)
        if max_deals is not None:
            deals = deals[:max_deals]
        return cls(**arrays, deals=deals.astype(np.int64),
                   name=name or path.name, path=str(path))

    def __len__(self) -> int:
        return len(self.deals)

    def hands_of(self, deal: int) -> np.ndarray:
        return np.asarray(self.hands[deal])


class HandDataset(Dataset):
    """Views of an auction store: public prefix -> the target seat's 13 cards.

    prefix="random" draws one prefix length per row per epoch (row count =
    deals x 4); prefix="all" enumerates every prefix as its own row
    (deterministic, ~10x the rows -- used for validation).

    mask_aug > 0 attaches a random exclusion mask to a fraction of the rows:
    `mask_cards` cards drawn from the 39 the target does *not* hold, exactly the
    shape of a caller saying "these are mine (or dummy's), so they are not his".
    """

    def __init__(self, store: AuctionStore, prefix: str = "random",
                 mask_aug: float = 0.0, mask_cards: int = HAND_LEN):
        if prefix not in ("random", "all"):
            raise ValueError(f"unknown prefix mode: {prefix!r}")
        self.store = store
        self.prefix = prefix
        self.mask_aug = float(mask_aug)
        self.mask_cards = int(mask_cards)
        deals = store.deals
        lengths = np.asarray(store.lengths)[deals].astype(np.int64)
        if prefix == "random":
            self.deal = np.repeat(deals, 4)
            self.seat = np.tile(np.arange(4), len(deals))
            self.k = None
        else:
            counts = np.repeat(lengths + 1, 4)          # prefixes 0..n per seat
            if counts.size == 0:                        # empty split
                self.k = np.zeros(0, dtype=np.int64)
                self.deal = self.seat = self.k
                return
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            self.k = np.arange(int(counts.sum())) - np.repeat(starts, counts)
            self.deal = np.repeat(np.repeat(deals, 4), counts)
            self.seat = np.repeat(np.tile(np.arange(4), len(deals)), counts)

    def __len__(self) -> int:
        return len(self.deal)

    def __getitem__(self, i):
        deal = int(self.deal[i])
        seat = int(self.seat[i])
        n = int(self.store.lengths[deal])
        if self.k is None:
            # torch RNG (seeded per DataLoader worker): prefixes vary per epoch
            k = int(torch.randint(0, n + 1, (1,)))
        else:
            k = int(self.k[i])
        tokens = rotated_tokens(int(self.store.dealer[deal]),
                                self.store.calls[deal, :k], seat)
        hand = np.asarray(self.store.hands[deal, seat])
        cards = hand_cards(hand)

        excl = np.zeros(HAND_DIM, dtype=np.float32)
        if self.mask_aug > 0 and self.mask_cards > 0 and \
                float(torch.rand(())) < self.mask_aug:
            free = np.flatnonzero(hand == 0)
            m = min(self.mask_cards, len(free))
            excl[free[torch.randperm(len(free))[:m].numpy()]] = 1.0

        vuln = BiddingDataset.vuln_class(int(self.store.vuln[deal]), seat)
        return tokens, cards.astype(np.int64), np.int64(vuln), excl, len(tokens)


def collate(batch):
    """Pad rows to the batch max; `mask` is True at real (non-padded) slots."""
    bsz = len(batch)
    seq_len = max(int(b[4]) for b in batch)
    tokens = torch.zeros(bsz, seq_len, dtype=torch.long)
    mask = torch.zeros(bsz, seq_len, dtype=torch.bool)
    cards = torch.zeros(bsz, HAND_LEN, dtype=torch.long)
    vuln = torch.zeros(bsz, dtype=torch.long)
    excl = torch.zeros(bsz, HAND_DIM, dtype=torch.float32)
    for r, (t, c, v, e, n) in enumerate(batch):
        tokens[r, :n] = torch.from_numpy(t)
        mask[r, :n] = True
        cards[r] = torch.from_numpy(c)
        vuln[r] = int(v)
        excl[r] = torch.from_numpy(e)
    return {"tokens": tokens, "mask": mask, "vuln": vuln, "excl": excl,
            "cards": cards}
