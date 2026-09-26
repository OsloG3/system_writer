"""Random full-deal generation + vectorized card-indicator encoding.

Card ids follow bid.py's deck order: card = rank*4 + suit, rank A=0..2=12
(RANK_CHARS), suit S,H,D,C (SUIT_CHARS). A deal is a (52,) uint8 array
mapping card id -> seat (N=0 E=1 S=2 W=3).

The policy consumes the (4,52) card-indicator encoding (data/hands.py);
`encode_deals` produces it vectorized from card arrays and is verified
against encode_hand in the tests.
"""

from dataclasses import dataclass

import numpy as np

RANK_CHARS = "AKQJT98765432"
SUIT_CHARS = "SHDC"

_CARD_IDS = np.arange(52, dtype=np.int64)
_SUIT_OF = _CARD_IDS % 4                       # S=0 H=1 D=2 C=3
_COUNT_IDX = _SUIT_OF * 13 + _CARD_IDS // 4    # (52,) in 0..51, hands.py layout


@dataclass(frozen=True, slots=True, eq=False)
class Deal:
    """One deal: hand strings (N,E,S,W 'S.H.D.C'), card->seat map, encoding."""
    hands: tuple[str, str, str, str]
    cards: np.ndarray    # (52,) uint8 card id -> seat
    encoded: np.ndarray  # (4, 52) uint8 card indicators, data/hands.py layout

    def key(self) -> bytes:
        return " ".join(self.hands).encode("ascii")


def hands_from_cards(cards: np.ndarray) -> tuple[str, str, str, str]:
    """(52,) card->seat map -> four 'S.H.D.C' hand strings (N,E,S,W)."""
    out = []
    for seat in range(4):
        mine = np.flatnonzero(cards == seat)
        suits = mine % 4
        ranks = mine // 4
        per_suit = []
        for s in range(4):
            r = np.sort(ranks[suits == s])
            per_suit.append("".join(RANK_CHARS[int(x)] for x in r))
        out.append(".".join(per_suit))
    return tuple(out)


def encode_deals(cards: np.ndarray) -> np.ndarray:
    """(B,52) card->seat maps -> (B,4,52) uint8 card-indicator encodings."""
    cards = np.atleast_2d(cards)
    b = cards.shape[0]
    enc = np.zeros((b, 4, 52), dtype=np.uint8)
    enc[np.arange(b)[:, None], cards, _COUNT_IDX[None, :]] = 1
    return enc


def random_deals(rng: np.random.Generator, n: int) -> list[Deal]:
    """n fresh random deals (uniform over all 52!/(13!^4) partitions)."""
    perm = np.argsort(rng.random((n, 52)), axis=1)  # perm[i, slot] = card id
    slot_seat = (np.arange(52) // 13).astype(np.uint8)
    cards = np.empty((n, 52), dtype=np.uint8)
    cards[np.arange(n)[:, None], perm] = slot_seat
    encoded = encode_deals(cards)
    return [Deal(hands_from_cards(cards[i]), cards[i], encoded[i])
            for i in range(n)]
