"""Kaplan-Rubens hand evaluator (the 26-step algorithm).

Reference: Richard Pavlicek, "Kaplan-Rubens Evaluation Stats",
https://www.rpbridge.net/8j19.htm. Steps 1-11 score raw honors plus
ten/nine/long-suit adjustments, step 12 scales the subtotal by suit length
(x length / 10), steps 13-26 add guarded-honor, shortness and jack/ten
combinations; the four suit totals are summed, minus 1, plus 0.5 for 4-3-3-3.

Verified against Pavlicek's own evaluator (rpbridge.net/cgi-bin/xhe1.pl):
the 37-HCP 7NT claimer scores 35.6 (the published maximum), a 4-3-3-3
Yarborough -0.5 (the published minimum), and the ten/nine steps, the
"exactly two higher honors" jack/ten rules (an AKQJ suit scores 10.0, not
10.5) and the additive long-suit steps 9-11 all reproduce his per-suit
numbers (see tests/test_conventions.py).

K&R points replace HCP+distribution adjustments throughout the convention
rules: a hand's opening/response/overcall strength is always `kr_points`.
"""

import numpy as np

from ..data.hands import HAND_DIM, HCP_WEIGHTS, NUM_RANKS, encode_hand

# Rank indices in the data/hands.py layout ("AKQJT98765432").
RA, RK, RQ, RJ, RT, R9, R8 = range(7)

_KR_CACHE: dict[str, float] = {}
_KR_CACHE_MAX = 1 << 17          # ~130k hands; cleared wholesale when full


def kr_suit(h: np.ndarray) -> float:
    """K&R value of one suit: h is the (13,) 0/1 holds vector (A..2)."""
    A = bool(h[RA]); K = bool(h[RK]); Q = bool(h[RQ]); J = bool(h[RJ])
    T = bool(h[RT]); n9 = bool(h[R9]); n8 = bool(h[R8])
    n = int(h.sum())

    # steps 1-5: raw honors (ten counts 0.5)
    t = 4.0 * A + 3.0 * K + 2.0 * Q + 1.0 * J + 0.5 * T
    if 2 <= n <= 6:
        # step 6: ten accompanied by the jack or two+ higher honors (J..A)
        if T and (J or (J + Q + K + A) >= 2):
            t += 0.5
        # step 7: nine accompanied by eight, ten, or exactly two higher
        # honors (T..A)
        if n9 and (n8 or T or (T + J + Q + K + A) == 2):
            t += 0.5
        # step 8: 4-6 cards, nine (no eight/ten) with exactly three higher
        # honors (J..A)
        if n >= 4 and n9 and not n8 and not T and (J + Q + K + A) == 3:
            t += 0.5
    # steps 9-11: long suits missing the queen/jack are overvalued by raw
    # counts -- add 1 for each condition met
    if n >= 7 and not (Q and J):
        t += 1.0
    if n >= 8 and not Q:
        t += 1.0
    if n >= 9 and not (Q and J):
        t += 1.0
    # step 12: scale by length
    t = t * n / 10.0

    # steps 13-23: honor combinations (added after the scaling)
    p = 0.0
    if A:
        p += 3.0                                    # 13
    if K:
        p += 2.0 if n >= 2 else 0.5                 # 14 guarded / 15 stiff
    if Q:
        if n >= 3:
            p += 1.0 if (A or K) else 0.75          # 16 / 17
        elif n == 2:
            p += 0.5 if (A or K) else 0.25          # 18 / 19
    if J:
        higher = int(Q) + int(K) + int(A)
        if higher == 2:
            p += 0.5                                # 20
        elif higher == 1:
            p += 0.25                               # 21
    if T:
        higher = int(J) + int(Q) + int(K) + int(A)
        if higher == 2:
            p += 0.25                               # 22
        elif higher == 1 and n9:
            p += 0.25                               # 23
    # steps 24-26: shortness
    if n == 0:
        p += 3.0
    elif n == 1:
        p += 2.0
    elif n == 2:
        p += 1.0
    return t + p


def kr_from_matrix(m: np.ndarray) -> float:
    """K&R total from a (4, 13) 0/1 holds matrix (data/hands.py layout)."""
    total = sum(kr_suit(m[s]) for s in range(4)) - 1.0
    lens = sorted(int(x) for x in m.sum(axis=1))
    if lens == [3, 3, 3, 4]:
        total += 0.5
    return total


def kr_points(hand) -> float:
    """Kaplan-Rubens count of a hand ('S.H.D.C' string or (52,) indicator).

    String inputs are memoized (bounded cache): the training loop evaluates
    the same four hands of a deal many times per iteration.
    """
    if isinstance(hand, str):
        cached = _KR_CACHE.get(hand)
        if cached is not None:
            return cached
        vec = encode_hand(hand)
    else:
        vec = np.asarray(hand)
        if vec.shape != (HAND_DIM,):
            raise ValueError(f"expected a ({HAND_DIM},) vector, got {vec.shape}")
    m = (vec.reshape(4, NUM_RANKS) > 0).astype(np.int64)
    total = kr_from_matrix(m)
    if isinstance(hand, str):
        if len(_KR_CACHE) >= _KR_CACHE_MAX:
            _KR_CACHE.clear()
        _KR_CACHE[hand] = total
    return total


def hcp(hand) -> int:
    """Milton-Work HCP (A=4 K=3 Q=2 J=1); accepts string or (52,) vector."""
    vec = encode_hand(hand) if isinstance(hand, str) else np.asarray(hand)
    return int((vec * HCP_WEIGHTS).sum())
