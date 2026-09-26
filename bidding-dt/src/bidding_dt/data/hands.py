"""Hand encoding.

Every rank gets its own slot per suit (no rank folding), so a hand is the
52-dim uint8 0/1 indicator vector of the cards it holds:
  [s*13 : s*13+13]  A K Q J T 9 8 7 6 5 4 3 2 in suit s (suit order S,H,D,C)
Suit lengths, HCP etc. are all derivable from these counts; the encoding is
lossless (it uniquely identifies the 13 cards).
"""

import numpy as np

HAND_DIM = 52
NUM_RANKS = 13
SUITS = "SHDC"  # file order
RANKS = "AKQJT98765432"

_RANK_IDX = {ch: i for i, ch in enumerate(RANKS)}


def encode_hand(hand: str) -> np.ndarray:
    """Encode one 'S.H.D.C' hand string into a (52,) uint8 vector."""
    vec = np.zeros(HAND_DIM, dtype=np.uint8)
    suits = hand.split(".")
    if len(suits) != 4:
        raise ValueError(f"bad hand: {hand!r}")
    for s, suit_str in enumerate(suits):
        for ch in suit_str:
            try:
                vec[s * NUM_RANKS + _RANK_IDX[ch]] += 1
            except KeyError:
                raise ValueError(f"bad card char {ch!r} in hand {hand!r}")
    if vec.sum() != 13:
        raise ValueError(f"hand is not 13 cards: {hand!r}")
    return vec


def encode_deal(hands_line: str) -> np.ndarray:
    """Encode a full deal line (4 hands, N E S W order) -> (4, 52) uint8."""
    parts = hands_line.split()
    if len(parts) != 4:
        raise ValueError(f"expected 4 hands, got {len(parts)}")
    return np.stack([encode_hand(p) for p in parts])


# Normalization constants for the model input: slots are already 0/1
# card indicators, so no rescaling is needed.
HAND_SCALE = np.ones(HAND_DIM, dtype=np.float32)
