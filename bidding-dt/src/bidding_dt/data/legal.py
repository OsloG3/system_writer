"""Legality masks over the 39-token action space.

Works in the hero-rotated frame: role of a call = (index + role_offset) % 4
with 0=hero, 1=RHO, 2=partner, 3=LHO. Pass a padded hero-frame prefix with
role_offset=0, or unpadded auction-order calls with role_offset=number of
leading pads.

Rules:
  P    always legal
  bid  iff strictly higher than the highest bid so far (ids are ordered)
  X    iff the last bid is an opponent's and has not been doubled
       (intervening passes do not cancel the double)
  XX   iff our side's bid is currently doubled and not redoubled
"""

import numpy as np

from .vocab import CALL_PAD, CALL_PASS, CALL_X, CALL_XX, FIRST_BID, VOCAB_SIZE


def legal_mask(calls, role_offset: int = 0) -> np.ndarray:
    """calls: sequence of call token ids (may contain leading PADs)."""
    mask = np.zeros(VOCAB_SIZE, dtype=bool)
    mask[CALL_PASS] = True

    highest_bid = 0
    bid_role = -1
    doubled = False
    redoubled = False
    for i, c in enumerate(calls):
        c = int(c)
        if c == CALL_PAD:
            continue
        if c >= FIRST_BID:
            highest_bid = c
            bid_role = (i + role_offset) % 4
            doubled = False
            redoubled = False
        elif c == CALL_X:
            doubled = True
        elif c == CALL_XX:
            redoubled = True

    mask[max(FIRST_BID, highest_bid + 1):VOCAB_SIZE] = True
    if highest_bid and bid_role in (1, 3) and not doubled:
        mask[CALL_X] = True
    if highest_bid and bid_role in (0, 2) and doubled and not redoubled:
        mask[CALL_XX] = True
    return mask
