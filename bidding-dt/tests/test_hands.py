import numpy as np
import pytest

from bidding_dt.data.hands import HAND_DIM, encode_deal, encode_hand


def test_known_hand():
    v = encode_hand("AT7.KT943.A42.K8")
    # spades AT7: A, T and 7 each get their own slot
    assert v[0] == 1 and v[4] == 1 and v[7] == 1
    # hearts KT943: K, T, 9, 4, 3
    h = v[13:26]
    assert h[1] == 1 and h[4] == 1 and h[5] == 1 and h[10] == 1 and h[11] == 1
    assert h.sum() == 5
    assert v.dtype == np.uint8 and v.shape == (HAND_DIM,)
    assert v.sum() == 13 and v.max() == 1


def test_every_rank_own_slot():
    v = encode_hand("8765432.AKQJT9..")
    assert v[:13].tolist() == [0] * 6 + [1] * 7      # 8..2 set, A..9 clear
    assert v[13:19].tolist() == [1] * 6              # AKQJT9
    w = encode_hand("876543.AKQJT92..")             # different low cards
    assert not np.array_equal(v, w)


def test_suit_lengths_recoverable():
    v = encode_hand("98765432.AKQJT..")  # 8 spades, 5 hearts
    assert v[5] == 1                     # the 9 has its own slot
    assert v[:13].sum() == 8 and v[13:26].sum() == 5


def test_bad_hand_raises():
    with pytest.raises(ValueError):
        encode_hand("AKQ.AKQ.AKQ.")          # 9 cards
    with pytest.raises(ValueError):
        encode_hand("AKQJ1.AKQJ.AKQ.AK")     # bad char
    with pytest.raises(ValueError):
        encode_hand("AKQJ.AKQJ.AKQ")         # 3 suits


def test_encode_deal():
    d = encode_deal("AT7.KT943.A42.K8 J6543.75.K8.AQJT K982.AQ6.T73.952 Q.J82.QJ965.7643")
    assert d.shape == (4, HAND_DIM)
    assert d.sum(axis=1).tolist() == [13, 13, 13, 13]
    assert d.max() == 1
