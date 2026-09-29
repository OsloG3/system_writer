import numpy as np
import pytest

from bidding_dt.bid import auction_over, seat_to_act
from bidding_dt.data.hands import HAND_SCALE
from bidding_dt.data.legal import legal_mask
from bidding_dt.data.parse import MAX_CALLS
from bidding_dt.data.vocab import CALL_PASS, VOCAB_SIZE, call_to_id
from bidding_dt.env.auction_env import AuctionBatch, sample_from_masks
from bidding_dt.env.deals import random_deals

HAS_ENDPLAY = True
try:
    import endplay  # noqa: F401
except ImportError:  # pragma: no cover
    HAS_ENDPLAY = False


def make_env(n=8, seed=0):
    rng = np.random.default_rng(seed)
    deals = random_deals(rng, n)
    dealer = rng.integers(0, 4, n)
    vuln = rng.integers(0, 4, n)
    return AuctionBatch(deals, dealer, vuln), rng


def test_fresh_state():
    env, _ = make_env()
    assert env.any_active()
    assert len(env.active_idx()) == env.b
    assert (env.n_calls == 0).all()
    assert (env.seat_to_act() == env.dealer).all()


def test_passout_terminates_at_four():
    env, _ = make_env(4)
    for _ in range(3):
        idx = env.active_idx()
        env.step(idx, np.full(len(idx), CALL_PASS))
        assert not env.done.any()  # PPP is not over (dealer must pass too)
    idx = env.active_idx()
    env.step(idx, np.full(len(idx), CALL_PASS))
    assert env.done.all()
    assert (env.n_calls == 4).all()


def test_masks_match_legal_module():
    """Vectorized masks must equal data/legal.py on random partial auctions."""
    env, rng = make_env(64, seed=1)
    for _ in range(30):
        idx = env.active_idx()
        if len(idx) == 0:
            break
        got = env.legal_masks(idx)
        for k, i in enumerate(idx):
            hero = int(env.seat_to_act()[i])
            o = (hero - int(env.dealer[i])) % 4
            p = (4 - o) % 4
            calls = env.calls[i, : env.n_calls[i]]
            want = legal_mask(calls, role_offset=p)
            assert np.array_equal(got[k], want), (i, calls, hero)
        env.step(idx, sample_from_masks(got, rng))
    assert env.done.all()


def test_random_auctions_consistent_with_bid_helpers():
    env, rng = make_env(32, seed=2)
    while env.any_active():
        idx = env.active_idx()
        # cross-check seat-to-act against the reference helper
        for i in idx:
            calls = env.calls[i, : env.n_calls[i]].tolist()
            assert int(env.seat_to_act()[i]) == seat_to_act(int(env.dealer[i]), calls)
            assert not auction_over(calls)
        masks = env.legal_masks(idx)
        assert (masks.sum(axis=1) >= 1).all()
        env.step(idx, sample_from_masks(masks, rng))
    for i in range(env.b):
        calls = env.calls[i, : env.n_calls[i]].tolist()
        assert auction_over(calls) or len(calls) == MAX_CALLS


def test_build_inputs_matches_predict_layout():
    """One deal, scripted auction: inputs equal bid.py:predict()'s layout."""
    from bidding_dt.data.dataset import BiddingDataset
    rng = np.random.default_rng(3)
    deals = random_deals(rng, 1)
    env = AuctionBatch(deals, np.array([0]), np.array([1]))
    # script: N 1H, E P, S 2H, then W to act (n=3)
    for a in ("1H", "P", "2H"):
        env.step(env.active_idx(), np.array([call_to_id(a)]))
    idx = env.active_idx()
    tokens, hand, vuln, hero, row_len = env.build_inputs(idx)
    assert hero[0] == 3  # W
    o = (3 - 0) % 4
    p = (4 - o) % 4
    assert p == 1
    assert row_len[0] == 2 + p + 3
    assert tokens[0, :2].tolist() == [0, 0]
    assert tokens[0, 2] == 0  # one pad
    assert tokens[0, 3:6].tolist() == [call_to_id(c) for c in ("1H", "P", "2H")]
    assert np.array_equal(hand[0], deals[0].encoded[3].astype(np.float32) * HAND_SCALE)
    assert vuln[0] == BiddingDataset.vuln_class(1, 3)
    assert tokens.shape[1] <= 29


def test_double_redouble_tracking():
    env, _ = make_env(1, seed=4)
    seq = "1H X P P XX P P P"
    for a in seq.split()[:5]:
        env.step(env.active_idx(), np.array([call_to_id(a)]))
    m = env.legal_masks(env.active_idx())[0]
    # after 1H X P P XX: W to act; no XX available (already redoubled),
    # no X (last bid is ours-side... W is opponent of N: X illegal since doubled)
    assert not m[call_to_id("X")]
    assert not m[call_to_id("XX")]
    for a in seq.split()[5:]:
        env.step(env.active_idx(), np.array([call_to_id(a)]))
    assert env.done.all()


def test_sample_from_masks_respects_legality():
    rng = np.random.default_rng(5)
    masks = np.zeros((10, VOCAB_SIZE), dtype=bool)
    masks[:, CALL_PASS] = True
    masks[:5, call_to_id("7N")] = True
    acts = sample_from_masks(masks, rng)
    assert set(acts[:5].tolist()) <= {CALL_PASS, call_to_id("7N")}
    assert set(acts[5:].tolist()) == {CALL_PASS}


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_rewards_end_to_end():
    env, rng = make_env(6, seed=6)
    while env.any_active():
        idx = env.active_idx()
        env.step(idx, sample_from_masks(env.legal_masks(idx), rng))
    rs = env.rewards()
    assert len(rs) == env.b
    for r in rs:
        assert -24.0 <= r.reward_imps <= 24.0
    # random auctions should mostly lose versus par
    assert np.mean([r.reward_imps for r in rs]) < 5.0
