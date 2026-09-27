import numpy as np

from bidding_dt.data.legal import legal_mask
from bidding_dt.data.vocab import (CALL_PASS, CALL_X, CALL_XX, call_to_id,
                                   id_to_call)

# Role of call i is (i + role_offset) % 4 with 0=hero, 1=RHO, 2=partner,
# 3=LHO; the call cycle is RHO -> partner -> LHO -> hero. All fixtures below
# satisfy (n + role_offset) % 4 == 0, i.e. the hero is the seat to act.


def ids(*calls):
    return np.array([call_to_id(c) for c in calls], dtype=np.int64)


def legal_set(mask):
    return {id_to_call(i) for i in np.flatnonzero(mask)}


def test_hero_dealer_empty_auction():
    m = legal_mask(ids(), role_offset=0)
    assert m[CALL_PASS]
    assert m[call_to_id("1C")] and m[call_to_id("7N")]
    assert not m[CALL_X] and not m[CALL_XX]
    assert not m[0]  # PAD is never a legal action


def test_over_opponent_bid():
    # LHO opens 1C, hero acts next
    m = legal_mask(ids("1C"), role_offset=3)
    s = legal_set(m)
    assert "X" in s and "XX" not in s
    assert "P" in s and "1D" in s and "1C" not in s


def test_over_partner_bid():
    # partner opens 1H, LHO passes, hero acts
    m = legal_mask(ids("1H", "P"), role_offset=2)
    s = legal_set(m)
    assert "X" not in s and "XX" not in s
    assert "1S" in s and "1H" not in s


def test_immediate_double_of_lho_opening():
    m = legal_mask(ids("1S"), role_offset=3)
    assert "X" in legal_set(m)


def test_balancing_double_after_everyone_passes():
    # RHO opens 1S; partner, LHO, hero pass; RHO, partner, LHO pass again
    m = legal_mask(ids("1S", "P", "P", "P", "P", "P", "P"), role_offset=1)
    assert "X" in legal_set(m)


def test_redouble_our_doubled_bid():
    # hero opens 1S, RHO doubles, partner and LHO pass
    m = legal_mask(ids("1S", "X", "P", "P"), role_offset=0)
    s = legal_set(m)
    assert "XX" in s and "X" not in s


def test_no_double_of_own_double():
    # RHO 1C, partner P, LHO P, hero X, RHO P, partner P, LHO P -> hero again
    m = legal_mask(ids("1C", "P", "P", "X", "P", "P", "P"), role_offset=1)
    s = legal_set(m)
    assert "X" not in s and "XX" not in s
    assert "1D" in s and "1C" not in s


def test_double_last_bid_only():
    # RHO 1C, partner 1D, LHO 1H -> hero may X the 1H (last bid, opponent)
    m = legal_mask(ids("1C", "1D", "1H"), role_offset=1)
    s = legal_set(m)
    assert "X" in s
    assert "1S" in s and "1H" not in s and "1D" not in s


def test_no_xx_when_double_was_ours():
    # hero 1H, RHO P, partner P, LHO X -> hero may XX (our bid, doubled)
    m = legal_mask(ids("1H", "P", "P", "X"), role_offset=0)
    s = legal_set(m)
    assert "XX" in s
    assert "X" not in s


def test_padded_prefix_equivalent_to_role_offset():
    calls = ids("1S", "X", "P", "P")
    m1 = legal_mask(calls, role_offset=0)
    padded = np.concatenate([ids("PAD", "PAD", "PAD", "PAD"), calls])
    m2 = legal_mask(padded, role_offset=0)
    assert np.array_equal(m1, m2)
