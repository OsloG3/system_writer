"""Action vocabulary: PAD + Pass + X + XX + 35 bids (1C..7N).

Bid ids are ordered by (level, denomination) with denominations C<D<H<S<N,
so id order matches bridge ranking: a bid is legal iff its id exceeds the
highest bid id so far.
"""

CALL_PAD = 0
CALL_PASS = 1
CALL_X = 2
CALL_XX = 3
FIRST_BID = 4
DENOMS = "CDHSN"  # ascending rank order
NUM_BIDS = 35
VOCAB_SIZE = FIRST_BID + NUM_BIDS  # 39

_CALL_TO_ID = {"PAD": CALL_PAD, "P": CALL_PASS, "X": CALL_X, "XX": CALL_XX}
for _lvl in range(1, 8):
    for _d, _den in enumerate(DENOMS):
        _CALL_TO_ID[f"{_lvl}{_den}"] = FIRST_BID + (_lvl - 1) * 5 + _d
_ID_TO_CALL = {v: k for k, v in _CALL_TO_ID.items()}

SEATS = "NESW"
SEAT_TO_IDX = {s: i for i, s in enumerate(SEATS)}

VULN_TO_IDX = {"None": 0, "N-S": 1, "E-W": 2, "Both": 3}


def call_to_id(call: str) -> int:
    try:
        return _CALL_TO_ID[call]
    except KeyError:
        raise ValueError(f"unknown call: {call!r}")


def id_to_call(idx: int) -> str:
    return _ID_TO_CALL[idx]


def is_bid(idx: int) -> bool:
    return idx >= FIRST_BID


def bid_level(idx: int) -> int:
    return (idx - FIRST_BID) // 5 + 1


def bid_denom(idx: int) -> str:
    return DENOMS[(idx - FIRST_BID) % 5]


def bid_id(level: int, denom: str) -> int:
    return FIRST_BID + (level - 1) * 5 + DENOMS.index(denom)
