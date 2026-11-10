"""Tests for the conventions package (K&R evaluator, auction reader, rules)
and the RL convention-penalty wiring.

The K&R vectors were checked against Richard Pavlicek's own evaluator
(rpbridge.net/cgi-bin/xhe1.pl), the reference implementation of the
Kaplan-Rubens 26-step algorithm (rpbridge.net/8j19.htm).

Seat arithmetic: with dealer N, the caller after `k` fed calls is seat
k % 4, so tests spell out the full auction (including the opponents' and
partner's passes) up to the hero's decision.
"""

import numpy as np
import pytest

from bidding_dt.conventions import (AuctionReader, ConventionConfig,
                                    HandInfo, call_penalty, call_violations,
                                    check_call, kr_points)
from bidding_dt.conventions.auction import bid_id
from bidding_dt.data.hands import encode_hand
from bidding_dt.data.vocab import call_to_id

B = call_to_id
P, X, XX = "P", "X", "XX"

HAS_ENDPLAY = True
try:
    import endplay  # noqa: F401
except ImportError:  # pragma: no cover
    HAS_ENDPLAY = False


# ---------------------------------------------------------------------------
# Kaplan-Rubens evaluator (vectors verified against Pavlicek's CGI)
# ---------------------------------------------------------------------------

KR_VECTORS = [
    ("AKQJ.AKQ.AKQ.AKQ", 35.6),        # published maximum (7NT claimer)
    ("AKQJ.AKQJ.AKQJ.A", 34.4),        # stiff ace = 5.4
    ("9876.975.974.973", -0.3),        # nine-with-eight bonus, 4333
    ("8765.876.875.874", -0.5),        # published minimum (4333 Yarborough)
    ("Q7654.KJ5.76.QJ4", 7.1),         # queen bonuses, stiff/doubleton
    ("AKT9876.5.43.654", 13.2),        # step 9 + ten-with-2-higher
    ("AKQJT98.AKQJ.A.5", 29.75),       # 7-card solid-ish + stiff ace
    ("AQJT9.KQJT9.A.KQ", 24.9),        # steps 6+7, KQ doubleton
    ("AKQJT9.65.Q4.765", 14.55),       # step 7 (9 with ten), Qx doubleton
    ("T9876.54.32.KQJ5", 7.4),         # T98 with no higher honor
    ("AKQ976.5.7654.32", 13.7),        # step 8 (nine + exactly 3 higher)
    ("JT98.765.743.652", 0.75),        # step 23 (T9 with one higher)
    ("AKT987654.5.43.2", 18.7),        # steps 9+10+11 stack
    ("AKQJT98765432...", 27.65),       # full 13-card suit, three voids
    ("AJ74.KQ6.AJ7.654", 14.0),        # 15 HCP quacky balanced -> 14.0
    ("AK74.QJ6.AQ7.654", 15.0),        # 16 HCP prime balanced -> 15.0
]


@pytest.mark.parametrize("hand,expect", KR_VECTORS)
def test_kr_points_reference_vectors(hand, expect):
    assert kr_points(hand) == pytest.approx(expect, abs=1e-9)


def test_kr_points_vector_input_and_cache():
    hand = "AKQJT98.AKQJ.A.5"
    expect = 29.75
    assert kr_points(hand) == pytest.approx(expect)
    assert kr_points(encode_hand(hand)) == pytest.approx(expect)
    assert kr_points(hand) == pytest.approx(expect)   # cached path


def test_kr_points_range_and_hcp_sanity():
    hi = HandInfo("QJT98.7.AQJ98.75")
    assert -0.5 <= hi.kr <= 35.6
    assert hi.hcp == 10
    assert hi.longest_denom() in (3, 1)               # 5-5 in the pointed suits
    assert HandInfo("AK74.QJ6.AQ7.654").balanced()
    assert not HandInfo("KQJ4.A753.AJT5.4").balanced()


# ---------------------------------------------------------------------------
# AuctionReader: structural classification
# ---------------------------------------------------------------------------

def feed_all(calls, dealer=0):
    r = AuctionReader(dealer, 0)
    for c in calls:
        r.feed(B(c))
    return r


def test_reader_2over1_canonical_and_competition():
    # 1C-2D is a 2/1 even though 1D was available (single "jump")
    r = feed_all(["1C", P])
    assert r.classify(B("2D"), seat=2).is_2over1
    # 1D-2H is a 2/1 too (hero = S, the responder)
    r = feed_all(["1D", P])
    assert r.classify(B("2H"), seat=2).is_2over1
    # with an intervening BID the level-2 single jump is the invitational JS
    r = feed_all(["1C", "1D"])
    info = r.classify(B("2S"), seat=2)
    assert not info.is_2over1 and info.is_jump_shift
    # a double does not consume a level: 1C-(X)-2D stays a 2/1
    r = feed_all(["1C", X])
    assert r.classify(B("2D"), seat=2).is_2over1
    # free 2-level response over an overcall with no jump is still a 2/1
    r = feed_all(["1C", "1H"])
    assert r.classify(B("2D"), seat=2).is_2over1
    # but 1C-(1D)-2H skips 1H -> invitational jump shift (competition)
    r = feed_all(["1C", "1D"])
    assert r.classify(B("2H"), seat=2).is_jump_shift


def test_reader_jump_shift_and_splinter_flags():
    r = feed_all(["1S", P])
    assert r.classify(B("3D"), seat=2).is_jump_shift          # single jump
    info = r.classify(B("4C"), seat=2)                        # double jump
    assert info.splinter and not info.is_jump_shift
    # splinter only below 4H: 1H-3S (double jump) qualifies, 1S-4H does not
    r = feed_all(["1H", P])
    assert r.classify(B("3S"), seat=2).splinter
    r = feed_all(["1S", P])
    assert not r.classify(B("4H"), seat=2).splinter
    # opener's jump-shift rebid is the strong one, not invitational
    r = feed_all(["1C", P, "1H", P])
    info = r.classify(B("2S"), seat=0)
    assert info.is_strong_js and not info.is_jump_shift


def test_reader_xyz_state_machine_with_passes():
    r = feed_all(["1C", P, "1H", P, "1S", P])
    assert r.xyz_stage == "s3"
    assert r.classify(B("2C"), seat=2).xyz_2c
    assert r.classify(B("2D"), seat=2).xyz_2d
    # other calls in a live XYZ auction keep the context flag but are not
    # artificial; a direct level-2 jump is natural there (no inv JS reading)
    info = r.classify(B("2S"), seat=2)
    assert info.xyz and not info.xyz_2c and not info.xyz_2d
    assert not info.is_jump_shift
    r.feed(B("2C"))
    r.feed(B(P))
    assert r.xyz_stage == "relay"
    assert r.classify(B("2D"), seat=0).xyz_relay
    r.feed(B("2D"))
    assert r.xyz_stage == "after_relay"
    r.feed(B(P))                                        # E passes
    assert r.classify(B(P), seat=2).xyz_after_relay
    # an overcall kills XYZ
    r2 = feed_all(["1C", "1D", "1H"])
    assert r2.xyz_stage == "none"
    # opener's 1NT rebid is not a 1-level suit bid -> no XYZ
    r3 = feed_all(["1C", P, "1H", P, "1N", P])
    assert r3.xyz_stage == "none"


def test_reader_gf_tracking():
    # 2/1 sets GF for N-S; reaching game clears it
    r = feed_all(["1S", P, "2C", P, "2D", P])
    assert r.gf[0] and not r.gf[1]
    assert r.classify(B(P), seat=2).gf_active
    r.feed(B("4S"))                    # S (index 6) bids game -> GF cleared
    assert not r.gf[0]
    # strong 2C is GF; opener's 2NT rebid after 2D waiting is invitational
    r = feed_all(["2C"])
    assert r.gf[r.opener % 2]
    r = feed_all(["2C", P, "2D", P, "2N", P])
    assert not r.gf[0]
    assert not r.classify(B(P), seat=2).gf_active
    r = feed_all(["2C", P, "2S", P, "2N", P])            # positive, GF stays
    assert r.gf[0]
    # fourth suit sets GF
    r = feed_all(["1C", P, "1H", P, "2D", P])
    assert r.classify(B("2S"), seat=2).fourth_suit
    r.feed(B("2S"))
    assert r.gf[0]
    # XYZ 2D sets GF
    r = feed_all(["1C", P, "1H", P, "1S", P])
    r.feed(B("2D"))
    assert r.gf[0]


def test_reader_double_contexts():
    assert feed_all(["1D"]).classify(B(X), seat=1).dbl_ctx == "takeout"
    assert feed_all(["1D", P, P]).classify(B(X), seat=3).dbl_ctx \
        == "balancing_takeout"
    assert feed_all(["1C", "1D"]).classify(B(X), seat=2).dbl_ctx == "negative"
    ctx = feed_all(["1D", "1S", "2H", P, "3D"]).classify(B(X), seat=1).dbl_ctx
    assert ctx == "penalty"                # hero (E) bid 1S earlier
    assert feed_all(["1N"]).classify(B(X), seat=1).dbl_ctx == "nt_penalty"


def test_reader_classify_does_not_mutate():
    r = feed_all(["1C", P])
    before = (r.n, r.highest, r.gf.copy(), r.xyz_stage)
    r.classify(B("2D"), seat=2)
    r.classify(B(P), seat=2)
    r.classify(B("3S"), seat=2)
    assert (r.n, r.highest, r.gf, r.xyz_stage) == before


# ---------------------------------------------------------------------------
# Rule checks: codes per call
# ---------------------------------------------------------------------------

CFG = ConventionConfig()


def codes(hand, calls, action, cfg=CFG, dealer=0):
    acts = [B(x) for x in calls]
    vs = call_violations(hand, dealer, 0, acts, B(action), cfg)
    return sorted(v.code for v in vs)


def test_opening_checks():
    # hero = N (empty auction, dealer N)
    assert codes("AK74.QJ6.AQ7.654", [], "1N") == []            # K&R 15.0
    assert codes("AJ74.KQ6.AJ7.654", [], "1N") == []            # K&R 14.0 tol
    assert codes("AJ74.QJ6.QJ7.654", [], "1N") == ["open_1nt_range"]
    assert codes("AKQJT9.65.Q4.765", [], "1N") == ["open_1nt_range"]  # unbal
    assert codes("AQJ4.K73.T75.QJ4", [], "1S") == ["open_major_5card"]
    assert codes("AQJT4.K73.75.KJ4", [], "1S") == []            # 5-card major
    assert codes("432.QJT985.5.KQ4", [], "2H") == []            # weak two
    assert codes("432.KQJT98.5.KQ4", [], "2H") == ["open_weak2"]  # too good
    assert codes("432.KQJT98.5.KQ4", [], "2C") == ["open_2c_strong"]
    assert codes("AKQ976.A5.AKQ4.3", [], "2C") == []            # strong 2C
    assert codes("AKJ.QJT.AQJ.QJT2", [], "2N") == []            # K&R 19.3
    assert codes("AK74.QJ6.AQ7.654", [], "2N") == ["open_2nt_range"]
    assert codes("T73.952.T74.QJT3", [], "1S") \
        == ["open_major_5card", "open_values"]
    # rule of 20: 10 HCP + 6-5 shape may open despite a middling count
    assert "open_values" not in codes("KQJT98.QJT98.5.4", [], "1S")


def test_two_over_one_and_jump_shifts():
    # hero = S (responder) after 1S-P
    assert codes("Q73.K95.A74.KJT5", ["1S", P], "2C") == []
    assert codes("T73.952.Q74.KJT3", ["1S", P], "2C") == ["two_one_gf"]
    assert codes("AK73.KQ95.A74.J5", ["1S", P], "2C") == ["two_one_gf"]  # 2-card
    assert codes("A5.973.KQJT98.54", ["1S", P], "3D") == []     # invite JS
    assert codes("A5.973.KQJT9.543", ["1S", P], "3D") == ["inv_jump_shift"]
    assert codes("AKQ.AKQ.AKQ.AKQJ", ["1S", P], "3D") == ["inv_jump_shift"]
    # competition: single jump at the 2 level is the invitational JS ...
    assert codes("KQJ4.A753.T985.4", ["1C", "1D"], "2S") == ["inv_jump_shift"]
    # ... and jump overcalls follow the same treatment (hero = E over 1D)
    assert codes("KJ73.952.74.T985", ["1D"], "2S") == ["inv_jump_shift"]


def test_splinter_checks():
    # hero = S after 1S-P: 4C = double jump below 4H
    assert codes("KQJ4.A753.AJT5.4", ["1S", P], "4C") == []
    assert codes("43.AKJ5.AJT52.43", ["1S", P], "4C") == ["splinter"]  # no fit
    assert codes("A753.KQJ4.T54.AT", ["1S", P], "4D") == ["splinter"]  # no stiff
    # 4H is not below 4H -> no splinter reading, unchecked natural call
    assert codes("43.AKJ5.AJT52.43", ["1S", P], "4H") == []


def test_strong_2c_responses_and_gf():
    # hero = S after 2C-P
    assert codes("QJT98.7.AQJ98.75", ["2C", P], "2S") == []     # positive
    assert codes("94.752.T9854.T98", ["2C", P], "2S") == ["resp_2c_positive"]
    assert codes("94.752.T9854.T98", ["2C", P], "2D") == []     # waiting
    assert codes("94.752.T9854.T98", ["2C", P], "P") == ["gf_pass"]
    assert codes("94.752.T9854.T98", ["2C", P, "2D", P, "2N", P], "P") == []


def test_overcall_checks():
    # hero = E over N's 1D
    assert codes("KQJ73.A5.74.T985", ["1D"], "1S") == []
    assert codes("KJ73.952.74.T985", ["1D"], "1S") == ["overcall"]
    assert codes("AKQJ.AKQ.AQ7.KQJ", ["1D"], "1S") == []        # 4-card, huge
    assert codes("AK74.QJ6.AQ7.654", ["1D"], "1N") == []        # 1NT overcall
    assert codes("KJ73.952.74.T985", ["1D"], "1N") == ["overcall_1nt"]
    assert codes("AKQ7.QJ64.76.KQ4", ["1D"], "1N") == ["overcall_1nt"]  # no stop
    # hero = W balancing after 1D-P-P
    assert codes("KJ73.952.A74.T98", ["1D", P, P], "1S") == []
    assert codes("9852.74.KJ73.T95", ["1D", P, P], "1S") == ["overcall"]
    # hero = W advancing partner's 1S overcall: invitational jump shift
    assert codes("T5.7.QJ94.KQJT53", ["1D", "1S", P], "3C") == []
    assert codes("T54.7.QJ94.KQJT5", ["1D", "1S", P], "3C") == ["inv_jump_shift"]


def test_double_checks():
    # hero = E: takeout double of 1D
    assert codes("AK73.KQ74.5.AJT4", ["1D"], "X") == []
    assert codes("AKT9876.5.Q4.T98", ["1D"], "X") == ["takeout_double"]
    # hero = W: balancing takeout (lighter values fine)
    assert codes("KJ73.952.A7.T985", ["1D", P, P], "X") == []
    assert codes("T73.952.Q74.T983", ["1D", P, P], "X") == ["takeout_double"]
    # hero = S: negative double after 1C-(1D)
    assert codes("KT73.QJ74.5.T984", ["1C", "1D"], "X") == []
    assert codes("T98.75.KJ5.T9843", ["1C", "1D"], "X") == ["negative_double"]
    # hero = E (bid 1S earlier): penalty double of 3D
    assert codes("KQJ73.A5.74.T985", ["1D", "1S", "2H", P, "3D"], "X") == []
    assert codes("KJ73.952.74.T985", ["1D", "1S", "2H", P, "3D"], "X") \
        == ["penalty_double"]
    # hero = E: penalty double of a 1NT opening
    assert codes("AKJT.AKQ.AKQ.AKQ", ["1N"], "X") == []
    assert codes("752.752.752.T985", ["1N"], "X") == ["nt_penalty_double"]


def test_gf_pass_and_fourth_suit():
    # hero = S: 2/1 established GF, passing below game violates it
    gf = ["1S", P, "2C", P, "2D", P]
    assert codes("Q73.K95.A74.KJT5", gf, "P") == ["gf_pass"]
    assert codes("Q73.K95.A74.KJT5", gf, "3S") == []
    # game reached -> GF cleared
    assert codes("Q73.K95.A74.KJT5", ["1S", P, "2C", P, "4S", P], "P") == []
    # hero = S: fourth suit after 1C-1H-2D (opener reverse, no XYZ)
    fs = ["1C", P, "1H", P, "2D", P]
    assert codes("9854.74.KJ73.T95", fs, "2S") == ["fourth_suit"]
    assert codes("KQJ4.AKJ3.T98.54", fs, "2S") == []


def test_rkcb_4nt():
    # hero = S after 1S-2C-2S: 4NT is RKCB in spades
    auc = ["1S", P, "2C", P, "2S", P]
    assert codes("KQJ.A753.AJT5.43", auc, "4N") == []
    assert codes("43.AKJ5.AJT52.43", auc, "4N") == ["rkcb_4nt"]   # 2-card fit
    assert codes("KQJ.752.T9854.T9", auc, "4N") == ["rkcb_4nt"]   # no values
    # no suit bid by us -> quantitative, unchecked (hero = S over 1NT)
    assert codes("AK74.QJ6.AQ7.654", ["1N", P], "4N") == []


def test_xyz_checks():
    base = ["1C", P, "1H", P, "1S", P]           # hero = S, XYZ live
    assert codes("KQJ4.A753.T985.4", base, "2C") == []        # invite
    assert codes("KQJ4.AKQ3.T985.A", base, "2C") == ["xyz_2c"]  # GF values
    assert codes("T98.752.T985.T98", base, "2D") == ["xyz_2d"]  # junk GF bid
    assert codes("KQJ4.AKJ3.T98.54", base, "2D") == []
    # opener must complete the relay without extras (hero = N)
    relay = base + ["2C", P]
    assert codes("KQJ4.A753.T985.4", relay, "2N") == ["xyz_relay"]
    assert codes("KQJ4.A753.T985.4", relay, "2D") == []
    assert codes("AKQ.AKQ.AKQ.AKQJ", relay, "2N") == []
    # passing the relay: diamond signoff ok, invitational values not
    after = base + ["2C", P, "2D", P]
    assert codes("84.752.T98543.94", after, "P") == []
    assert codes("KQJ4.A753.T985.4", after, "P") == ["xyz_pass"]
    # opener passing responder's GF 2D
    assert codes("AK74.QJ6.AQ7.654", ["1C", P, "1H", P, "1S", P, "2D", P],
                 "P") == ["gf_pass"]


def test_raise_and_nt_response_checks():
    # hero = S over 1S: simple raise
    assert codes("KQJ4.9753.T98.54", ["1S", P], "2S") == []
    assert codes("KQ.A753.T9854.54", ["1S", P], "2S") == ["resp_raise"]  # 2-card
    # limit raise
    assert codes("KQJ4.A753.T985.4", ["1S", P], "3S") == []
    assert codes("94.752.T9854.T98", ["1S", P], "3S") == ["resp_jump_raise"]
    # 1NT response
    assert codes("KJ7.QT94.743.T98", ["1S", P], "1N") == []
    assert codes("AK74.QJ6.AQ7.654", ["1S", P], "1N") == ["resp_1nt"]  # too good
    assert codes("KT74.Q53.T98.754", ["1H", P], "1N") == ["resp_1nt"]  # 4-card S
    # 2NT / 3NT responses
    assert codes("Q73.K95.A74.KJT5", ["1S", P], "2N") == []
    assert codes("KQJ4.A753.T985.4", ["1S", P], "2N") == ["resp_2nt"]  # unbal
    assert codes("AK74.QJ6.AQ7.654", ["1S", P], "3N") == []
    assert codes("KJ7.QT94.743.T98", ["1S", P], "3N") == ["resp_3nt"]
    # opener's 1NT rebid (hero = N)
    assert codes("AK74.QJ6.AQ7.654", ["1C", P, "1H", P], "1N") == []
    assert codes("9854.74.KJ73.T95", ["1C", P, "1H", P], "1N") == ["rebid_1nt"]
    # opener's strong jump-shift rebid (hero = N): 2S over 1H
    assert codes("AKQ4.AKJ.AKQ4.KQ", ["1C", P, "1H", P], "2S") == []
    assert codes("KQ74.QJ6.AQ7.654", ["1C", P, "1H", P], "2S") \
        == ["strong_jump_rebid"]


def test_pass_and_redouble_neutral():
    assert codes("752.752.752.T985", [], "P") == []
    assert codes("752.752.752.T985", ["1D", X], "XX") == []


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_convention_config_overrides():
    cfg = ConventionConfig.from_overrides({"gf_pass": 2.0, "open_values": 0.0},
                                          {"kr_tolerance": 0.75})
    assert cfg.weight("gf_pass") == 2.0
    assert cfg.weight("open_values") == 0.0
    assert cfg.kr_tolerance == 0.75
    with pytest.raises(KeyError):
        ConventionConfig.from_overrides({"no_such_rule": 1.0})
    with pytest.raises(KeyError):
        ConventionConfig.from_overrides(None, {"no_such_param": 1})
    # disabled rule emits nothing
    cfg2 = ConventionConfig.from_overrides({"open_major_5card": 0.0})
    assert codes("AQJ4.K73.T75.QJ4", [], "1S", cfg=cfg2) == []
    # weights sum into the penalty
    pen = call_penalty("AQJ4.K73.T75.QJ4", 0, 0, [], B("1S"))
    assert pen == pytest.approx(CFG.weight("open_major_5card"))


def test_rl_config_conv_fields(tmp_path):
    from bidding_dt.rl.config import RLTrainConfig
    path = tmp_path / "conv.yaml"
    path.write_text(
        "rl:\n"
        "  conv_weight: 1.5\n"
        "  conv_rules:\n"
        "    gf_pass: 2.0\n"
        "    open_values: 0.0\n"
        "  conv_params:\n"
        "    kr_tolerance: 0.75\n")
    cfg = RLTrainConfig.load(path)
    assert cfg.rl.conv_weight == 1.5
    conv = ConventionConfig.from_overrides(cfg.rl.conv_rules, cfg.rl.conv_params)
    assert conv.weight("gf_pass") == 2.0
    assert conv.weight("open_values") == 0.0
    assert conv.kr_tolerance == 0.75


# ---------------------------------------------------------------------------
# Rollout wiring
# ---------------------------------------------------------------------------

def _toy_buf(deal_idx, seat, call_idx):
    from bidding_dt.rl.rollout import RolloutBuffer
    buf = RolloutBuffer()
    buf.deal_idx = list(deal_idx)
    buf.seat = list(seat)
    buf.call_idx = list(call_idx)
    buf.value = [0.0] * len(seat)
    return buf


def test_team_row_rewards_convention_penalty():
    from bidding_dt.rl.rollout import team_row_rewards
    buf = _toy_buf([0, 0], [0, 1], [0, 0])
    kw = dict(n_deals=1, team_imps=np.array([-8.0]), diverge=np.array([0]),
              par_imps_ns=np.zeros(2), reward_scale=1.0,
              team_weight=1.0, par_weight=0.25)
    conv = np.array([2.0, 1.0])
    r0 = team_row_rewards(buf, **kw)
    assert np.allclose(r0, [-8.0, 8.0])
    r = team_row_rewards(buf, conv_pen=conv, conv_weight=0.5, **kw)
    assert np.allclose(r, [-9.0, 7.5])
    # penalty applies after the partner-dev discount (row owns its own break)
    pd = np.full(2, np.log(4.0))
    r = team_row_rewards(buf, partner_dev=pd, conv_pen=conv, conv_weight=0.5,
                         **kw)
    assert np.allclose(r, [-8.0 / 4 - 1.0, 8.0 - 0.5])


def test_convention_penalties_matches_manual():
    """convention_penalties() over a randomly-played batch env equals the
    per-row call_violations recomputation (no DD solver needed)."""
    from bidding_dt.env.auction_env import AuctionBatch, sample_from_masks
    from bidding_dt.env.deals import random_deals
    from bidding_dt.rl.rollout import RolloutBuffer, convention_penalties
    rng = np.random.default_rng(0)
    deals = random_deals(rng, 6)
    dealer = rng.integers(0, 4, 6)
    vuln = rng.integers(0, 4, 6)
    env = AuctionBatch(deals, dealer, vuln)
    buf = RolloutBuffer()
    while env.any_active():
        idx = env.active_idx()
        masks = env.legal_masks(idx)
        actions = sample_from_masks(masks, rng)
        tokens, hand, vuln_cls, hero, row_len = env.build_inputs(idx)
        buf.add_rows(idx, hero, tokens, hand, vuln_cls, row_len, masks,
                     actions, np.zeros(len(idx)), np.zeros(len(idx)),
                     env.n_calls[idx], np.zeros(len(idx)))
        env.step(idx, actions)
    pens, pen_codes = convention_penalties(buf, env, CFG)
    assert pens.shape == (buf.n,)
    assert (pens >= 0).all() and np.isfinite(pens).all()
    for i in range(buf.n):
        di, ci = buf.deal_idx[i], buf.call_idx[i]
        prefix = env.calls[di, :ci].tolist()
        vs = call_violations(deals[di].hands[buf.seat[i]], int(dealer[di]),
                             int(vuln[di]), prefix, buf.action[i], CFG)
        assert pens[i] == pytest.approx(sum(v.weight for v in vs))
        assert pen_codes[i] == [v.code for v in vs]
    # random legal play breaks the system somewhere in 6 full auctions
    assert (pens > 0).any()


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_play_team_deals_conv_pen_flows_into_rewards(tmp_path):
    import torch
    from bidding_dt.config import ModelConfig
    from bidding_dt.dd.solver import TableCache
    from bidding_dt.rl.model import RLModel
    from bidding_dt.rl.rollout import play_team_deals
    from bidding_dt.env.deals import random_deals
    rng = np.random.default_rng(5)
    deals = random_deals(rng, 6)
    dealer = rng.integers(0, 4, 6)
    vuln = rng.integers(0, 4, 6)
    torch.manual_seed(7)
    model = RLModel(ModelConfig.from_preset("tiny")).eval()
    cache = TableCache(tmp_path / "dd.sqlite")
    roll = play_team_deals(model, deals, dealer, vuln, torch.device("cpu"),
                           cache=cache, greedy=False, temp=1.4,
                           conv_cfg=ConventionConfig())
    assert roll.conv_pen is not None
    assert roll.conv_pen.shape == (roll.buf.n,)
    assert (roll.conv_pen >= 0).all()
    assert len(roll.conv_codes) == roll.buf.n
    r0 = roll.row_rewards(1.0, conv_weight=0.0)
    r1 = roll.row_rewards(1.0, conv_weight=0.5)
    assert np.allclose(r0 - r1, 0.5 * roll.conv_pen)
    # without conv_cfg nothing is computed and rewards are unchanged
    ref = play_team_deals(model, deals, dealer, vuln, torch.device("cpu"),
                          cache=cache, greedy=True)
    assert ref.conv_pen is None
    cache.close()
