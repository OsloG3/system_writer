"""Hand encoding.

Every rank gets its own slot per suit (no rank folding), so a hand is the
52-dim uint8 0/1 indicator vector of the cards it holds:
  [s*13 : s*13+13]  A K Q J T 9 8 7 6 5 4 3 2 in suit s (suit order S,H,D,C)
Suit lengths, HCP etc. are all derivable from these counts; the encoding is
lossless (it uniquely identifies the 13 cards).

Card ids (env/deals.py order: card = rank*4 + suit) are the generation order
of the hand decoder; CARD_TO_COUNT / COUNT_TO_CARD convert between the two
orderings and hand_cards / indicator_from_cards / cards_to_str round-trip a
hand between them.
"""

import numpy as np

HAND_DIM = 52
NUM_RANKS = 13
HAND_LEN = 13                   # cards per hand
SUITS = "SHDC"  # file order
RANKS = "AKQJT98765432"

_RANK_IDX = {ch: i for i, ch in enumerate(RANKS)}

# Card ids follow env/deals.py: card = rank*4 + suit (rank A=0 .. 2=12, suit
# S,H,D,C). The indicator layout above is suit-major (count = suit*13 + rank),
# so the two orderings are permutations of each other:
_IDX = np.arange(HAND_DIM)
CARD_TO_COUNT = (_IDX % 4) * NUM_RANKS + _IDX // 4      # card id -> indicator slot
COUNT_TO_CARD = (_IDX % NUM_RANKS) * 4 + _IDX // NUM_RANKS  # indicator slot -> card id

HCP_WEIGHTS = np.zeros(HAND_DIM, dtype=np.int8)
HCP_WEIGHTS.reshape(4, NUM_RANKS)[:, :4] = (4, 3, 2, 1)  # A K Q J


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
    if vec.sum() != HAND_LEN:
        raise ValueError(f"hand is not {HAND_LEN} cards: {hand!r}")
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


def hand_cards(vec: np.ndarray) -> np.ndarray:
    """(52,) card indicator -> (13,) ascending card ids (env/deals.py order).

    Ascending card id is the canonical order the hand decoder generates in:
    a hand is a set, so one fixed total order makes the sequence well defined
    and lets inference forbid re-drawing (or drawing out of order).
    """
    slots = np.flatnonzero(np.asarray(vec).reshape(-1) > 0)
    cards = np.sort(COUNT_TO_CARD[slots])
    if len(cards) != HAND_LEN:
        raise ValueError(f"expected {HAND_LEN} cards, got {len(cards)}")
    return cards.astype(np.int64)


def indicator_from_cards(cards) -> np.ndarray:
    """Card ids (any order) -> (52,) uint8 indicator vector."""
    vec = np.zeros(HAND_DIM, dtype=np.uint8)
    vec[CARD_TO_COUNT[np.asarray(cards, dtype=np.int64)]] = 1
    return vec


def card_str(card: int) -> str:
    """Card id -> 'AS', 'TD', ... (rank + suit)."""
    card = int(card)
    return RANKS[card // 4] + SUITS[card % 4]


def cards_to_str(cards) -> str:
    """Card ids -> 'S.H.D.C' hand string (inverse of str_to_cards)."""
    vec = indicator_from_cards(cards)
    return ".".join("".join(RANKS[r] for r in range(NUM_RANKS)
                            if vec[s * NUM_RANKS + r]) for s in range(4))


def str_to_cards(hand: str) -> np.ndarray:
    """'S.H.D.C' hand string -> (13,) ascending card ids."""
    return hand_cards(encode_hand(hand))


def suit_lengths(vec: np.ndarray) -> np.ndarray:
    """(52,) indicator -> (4,) suit lengths in S,H,D,C order."""
    return np.asarray(vec).reshape(4, NUM_RANKS).sum(axis=1)


def hcp(vec: np.ndarray) -> int:
    """High-card points (A=4 K=3 Q=2 J=1) of a (52,) indicator vector."""
    return int((np.asarray(vec) * HCP_WEIGHTS).sum())
