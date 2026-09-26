"""Vectorized self-play auction environment.

A batch of independent auctions, stepped one call at a time. Observations for
the seat to act are built in exactly the hero-rotated form used by the
supervised dataset (`[COND, HAND, pads, calls]`), so a BC checkpoint drops in
unchanged. Legality masks are maintained incrementally and mirror data/legal.py
(cross-checked in tests).

Rewards are terminal: DD par-diff in IMPs from the N-S perspective
(dd/reward.py); E-W gets the negation. The env does not sample actions --
the rollout worker feeds model logits and calls step().
"""

import numpy as np

from ..data.dataset import BiddingDataset, MAX_SEQ
from ..data.hands import HAND_SCALE
from ..data.parse import MAX_CALLS
from ..data.vocab import (CALL_PASS, CALL_X, CALL_XX, FIRST_BID, VOCAB_SIZE)
from ..dd.reward import DealReward, RewardRequest, deal_rewards
from ..dd.solver import TableCache
from .deals import Deal


class AuctionBatch:
    """B concurrent auctions over a fixed list of deals."""

    def __init__(self, deals: list[Deal], dealer: np.ndarray, vuln: np.ndarray):
        b = len(deals)
        assert dealer.shape == (b,) and vuln.shape == (b,)
        self.deals = deals
        self.b = b
        self.dealer = dealer.astype(np.int64)
        self.vuln = vuln.astype(np.int64)
        self.calls = np.zeros((b, MAX_CALLS), dtype=np.int64)
        self.n_calls = np.zeros(b, dtype=np.int64)
        self.done = np.zeros(b, dtype=bool)
        self._pass_run = np.zeros(b, dtype=np.int64)
        self._highest_bid = np.zeros(b, dtype=np.int64)
        self._last_bid_seat = np.full(b, -1, dtype=np.int64)
        self._doubled = np.zeros(b, dtype=bool)
        self._redoubled = np.zeros(b, dtype=bool)

    # -- queries -----------------------------------------------------------

    def any_active(self) -> bool:
        return bool((~self.done).any())

    def active_idx(self) -> np.ndarray:
        return np.flatnonzero(~self.done)

    def seat_to_act(self) -> np.ndarray:
        return (self.dealer + self.n_calls) % 4

    def legal_masks(self, idx: np.ndarray) -> np.ndarray:
        """(n, 39) bool masks for the deals in idx (hero = seat to act)."""
        hero = self.seat_to_act()[idx]
        high = self._highest_bid[idx]
        n = len(idx)
        cols = np.arange(VOCAB_SIZE)
        mask = np.zeros((n, VOCAB_SIZE), dtype=bool)
        mask[:, CALL_PASS] = True
        mask[:, FIRST_BID:] = cols[None, FIRST_BID:] > high[:, None]
        rel = (self._last_bid_seat[idx] - hero) % 4
        opp = (rel == 1) | (rel == 3)
        own = (rel == 0) | (rel == 2)
        has_bid = high > 0
        mask[:, CALL_X] = has_bid & opp & ~self._doubled[idx]
        mask[:, CALL_XX] = has_bid & own & self._doubled[idx] & ~self._redoubled[idx]
        return mask

    def build_inputs(self, idx: np.ndarray):
        """Model inputs for the deals in idx, matching bid.py:predict().

        Returns (tokens (n,L) int64, hand (n,32) float32, vuln (n,) int64,
                 hero (n,) int64, row_len (n,) int64).
        """
        hero = self.seat_to_act()[idx]
        n = len(idx)
        o = (hero - self.dealer[idx]) % 4
        p = (4 - o) % 4
        nc = self.n_calls[idx]
        row_len = 2 + p + nc
        L = min(int(row_len.max()), MAX_SEQ)
        tokens = np.zeros((n, L), dtype=np.int64)
        for k, i in enumerate(idx):
            start = 2 + int(p[k])
            end = min(start + int(nc[k]), L)
            tokens[k, start:end] = self.calls[i, : end - start]
        hand = np.stack([self.deals[i].encoded[h] for i, h in zip(idx, hero)]
                        ).astype(np.float32) * HAND_SCALE
        vuln_cls = np.array([BiddingDataset.vuln_class(int(self.vuln[i]), int(h))
                             for i, h in zip(idx, hero)], dtype=np.int64)
        return tokens, hand, vuln_cls, hero, row_len

    # -- stepping ------------------------------------------------------------

    def step(self, idx: np.ndarray, actions: np.ndarray):
        """Append one call per deal in idx (parallel to legal_masks order)."""
        actions = np.asarray(actions, dtype=np.int64)
        assert len(actions) == len(idx)
        seat = self.seat_to_act()[idx]
        rows = np.arange(len(idx))
        alive = ~self.done[idx]
        assert alive.all(), "cannot step finished deals"
        nc = self.n_calls[idx]
        assert (nc < MAX_CALLS).all(), "auction exceeded MAX_CALLS"
        self.calls[idx, nc] = actions
        self.n_calls[idx] = nc + 1

        self._pass_run[idx] = np.where(actions == CALL_PASS, self._pass_run[idx] + 1, 0)
        is_bid = actions >= FIRST_BID
        if is_bid.any():
            bi = idx[is_bid]
            self._highest_bid[bi] = actions[is_bid]
            self._last_bid_seat[bi] = seat[is_bid]
            self._doubled[bi] = False
            self._redoubled[bi] = False
        xd = actions == CALL_X
        if xd.any():
            self._doubled[idx[xd]] = True
        xx = actions == CALL_XX
        if xx.any():
            self._redoubled[idx[xx]] = True

        finished = (self._pass_run[idx] >= 3) & (self.n_calls[idx] >= 4)
        finished |= self.n_calls[idx] >= MAX_CALLS
        self.done[idx[finished]] = True

    # -- terminal rewards ----------------------------------------------------

    def rewards(self, cache: TableCache | None = None) -> list[DealReward]:
        """Par-diff IMP rewards (N-S perspective) for all deals; solves DD."""
        reqs = [RewardRequest(d.hands, int(self.dealer[i]), int(self.vuln[i]),
                              self.calls[i, : self.n_calls[i]].tolist())
                for i, d in enumerate(self.deals)]
        return deal_rewards(reqs, cache=cache)


def sample_from_masks(masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform random legal actions (baseline policies and tests)."""
    probs = masks / masks.sum(axis=1, keepdims=True)
    cum = probs.cumsum(axis=1)
    u = rng.random((len(masks), 1))
    return (cum < u).sum(axis=1).clip(max=VOCAB_SIZE - 1)
