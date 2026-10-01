import copy

import numpy as np
import pytest
import torch

from bidding_dt.config import ModelConfig
from bidding_dt.data.hands import HAND_DIM
from bidding_dt.data.vocab import CALL_PASS, VOCAB_SIZE
from bidding_dt.rl.config import RLConfig
from bidding_dt.rl.model import RLModel
from bidding_dt.rl.ppo import ppo_update
from bidding_dt.rl.rollout import (RolloutBuffer, UniformPolicy,
                                   par_row_rewards, partner_greedy_dev,
                                   play_deals, play_team_deals,
                                   team_row_rewards)
from bidding_dt.env.deals import random_deals

HAS_ENDPLAY = True
try:
    import endplay  # noqa: F401
except ImportError:  # pragma: no cover
    HAS_ENDPLAY = False


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return RLModel(ModelConfig.from_preset("tiny"))


def test_forward_last_matches_truncated_row():
    model = tiny_model().eval()
    rng = np.random.default_rng(0)
    n, L = 4, 12
    tokens = torch.randint(0, 39, (n, L))
    tokens[:, :2] = 0
    hand = torch.rand(n, 52)
    vuln = torch.randint(0, 4, (n,))
    row_len = torch.tensor([5, 8, 12, 3])
    with torch.no_grad():
        logits, values = model.forward_last(tokens, hand, vuln, row_len)
        for i in range(n):
            Li = int(row_len[i])
            li, vi = model.forward_last(tokens[i:i + 1, :Li], hand[i:i + 1],
                                        vuln[i:i + 1], torch.tensor([Li]))
            assert torch.allclose(logits[i], li[0], atol=1e-5)
            assert torch.allclose(values[i], vi[0], atol=1e-5)
    assert logits.shape == (n, 39) and values.shape == (n,)


def test_finish_per_decision_credit_two_step():
    buf = RolloutBuffer()
    # one seat's two decisions; values 1 and 3; own gated rewards 0 and 2
    buf.deal_idx = [0, 0]
    buf.seat = [0, 0]
    buf.value = [1.0, 3.0]
    buf.finish(np.array([0.0, 2.0]), gamma=1.0, lam=0.5)
    # bandit credit: adv = R - value; ret = R (no chaining / double-count)
    assert np.allclose(buf.adv, [-1.0, -1.0])
    assert np.allclose(buf.ret, [0.0, 2.0])


def test_finish_per_decision_side_signs():
    buf = RolloutBuffer()
    # deal 0: N (row 0, +1) and E (row 1, -1) single decisions
    buf.deal_idx = [0, 0]
    buf.seat = [0, 1]
    buf.value = [0.0, 0.0]
    buf.finish(np.array([1.0, -1.0]), gamma=1.0, lam=1.0)
    assert np.allclose(buf.adv, [1.0, -1.0])  # zero-sum
    assert np.allclose(buf.ret, [1.0, -1.0])


def test_finish_no_leak_to_predivergence_bid():
    """A seat that bids once pre-divergence (reward 0) and once post
    (reward 5) must NOT have the swing leak back onto the first bid."""
    buf = RolloutBuffer()
    buf.deal_idx = [0, 0]
    buf.seat = [2, 2]           # same seat, two calls
    buf.value = [0.0, 0.0]
    buf.finish(np.array([0.0, 5.0]))   # first gated off, second credited
    assert np.allclose(buf.adv, [0.0, 5.0])
    assert buf.ret[0] == 0.0           # pre-divergence bid gets nothing


def _toy_buf(deal_idx, seat, call_idx):
    buf = RolloutBuffer()
    buf.deal_idx = list(deal_idx)
    buf.seat = list(seat)
    buf.call_idx = list(call_idx)
    buf.value = [0.0] * len(seat)
    return buf


def test_par_row_rewards_signs_and_scale():
    buf = _toy_buf([0, 0, 1, 1], [0, 1, 2, 3], [0, 0, 0, 0])
    r = par_row_rewards(buf, np.array([10.0, -4.0]), reward_scale=0.2)
    assert np.allclose(r, [2.0, -2.0, -0.8, 0.8])


def test_team_row_rewards_gating():
    # 1 deal, 2 tables: team swing +10 IMPs (table-0 N-S did better),
    # auctions identical through call 1, diverge at call 2;
    # par-diff: table 0 +4, table 1 -6.
    buf = _toy_buf(
        deal_idx=[0, 0, 1, 1],          # env rows: <b table 0, >=b table 1
        seat=[0, 1, 2, 3],              # N, E, S, W
        call_idx=[1, 2, 3, 0])
    r = team_row_rewards(buf, n_deals=1, team_imps=np.array([10.0]),
                         diverge=np.array([2]),
                         par_imps_ns=np.array([4.0, -6.0]),
                         reward_scale=1.0, team_weight=1.0, par_weight=0.25)
    # row 0: table 0 N, call 1 < diverge -> gated; par +4*0.25
    # row 1: table 0 E, call 2 >= 2 -> team 10*(-1) = -10; par -1
    # row 2: table 1 S, call 3 -> team 10*(+1)*(-1) = -10; par -6*0.25 = -1.5
    # row 3: table 1 W, call 0 -> gated; par -6*(-1)*0.25 = +1.5
    assert np.allclose(r, [1.0, -11.0, -11.5, 1.5])


def test_team_row_rewards_league_uniform_sign():
    # league: learner is N-S at table 0 and E-W at table 1 -> every learner
    # row carries +team_imps (once past divergence), regardless of table.
    buf = _toy_buf(deal_idx=[0, 1], seat=[0, 3], call_idx=[0, 0])
    r = team_row_rewards(buf, n_deals=1, team_imps=np.array([7.0]),
                         diverge=np.array([0]), par_imps_ns=np.zeros(2),
                         reward_scale=0.5, team_weight=1.0, par_weight=0.25)
    assert np.allclose(r, [3.5, 3.5])


def test_to_torch_pads():
    buf = RolloutBuffer()
    tokens = [np.array([0, 0, 1, 5]), np.array([0, 0, 0, 3, 1, 7, 1, 1, 1])]
    buf.tokens = tokens
    buf.hand = [np.zeros(52, np.float32), np.ones(52, np.float32)]
    buf.vuln = [0, 3]
    buf.row_len = [4, 9]
    buf.mask = [np.ones(39, bool), np.ones(39, bool)]
    buf.action = [5, 1]
    buf.logprob = [-1.0, -0.5]
    buf.value = [0.1, 0.2]
    buf.deal_idx = [0, 1]
    buf.seat = [0, 2]
    buf.call_idx = [0, 0]
    buf.finish(np.array([0.0, 0.0]), 1.0, 1.0)
    t = buf.to_torch(torch.device("cpu"))
    assert t["tokens"].shape == (2, 9)
    assert torch.equal(t["tokens"][0], torch.tensor([0, 0, 1, 5, 0, 0, 0, 0, 0]))
    assert t["row_len"].tolist() == [4, 9]
    assert t["mask"].dtype == torch.bool


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_play_deals_and_ppo_update(tmp_path):
    from bidding_dt.dd.solver import TableCache
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    deals = random_deals(rng, 6)
    dealer = rng.integers(0, 4, 6)
    vuln = rng.integers(0, 4, 6)
    model = tiny_model(seed=1)
    cache = TableCache(tmp_path / "dd.sqlite")

    # pure self-play: all four seats recorded
    buf, env, r_ns = play_deals(model, deals, dealer, vuln, torch.device("cpu"),
                                rng, cache=cache)
    assert env.done.all()
    assert buf.n >= 6  # at least the four opening calls per deal
    assert len(r_ns) == 6
    # every deal contributes decisions for all four seats
    seats = {(d, s) for d, s in zip(buf.deal_idx, buf.seat)}
    deals_seen = {d for d, _ in seats}
    assert deals_seen == set(range(6))
    # call_idx is the decision position within its auction and matches
    # the call the env actually played from that seat
    for d, s, c, a in zip(buf.deal_idx, buf.seat, buf.call_idx, buf.action):
        assert 0 <= c < env.n_calls[d]
        assert env.calls[d, c] == a
        assert (dealer[d] + c) % 4 == s

    buf.finish(par_row_rewards(buf, r_ns, 0.2), 1.0, 0.95)
    tensors = buf.to_torch(torch.device("cpu"))
    cfg = RLConfig(ppo_epochs=2, minibatch_size=64)
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    stats = ppo_update(model, ref, tensors, opt, torch.device("cpu"), cfg,
                       kl_beta=0.1, ent_coef=0.01, rng=rng)
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])
    assert np.isfinite(stats["kl"]) and stats["entropy"] > 0
    assert stats["kl"] >= -1e-6  # proper masked KL is nonnegative
    changed = any(not torch.equal(before[n], p.detach())
                  for n, p in model.named_parameters())
    assert changed

    # league mode: only learner-side seats are recorded
    model2 = tiny_model(seed=2)
    learner_ns = np.array([True, False, True, False, True, False])
    buf2, _, r2 = play_deals(model2, deals, dealer, vuln, torch.device("cpu"),
                             rng, cache=cache, opponent=UniformPolicy(),
                             learner_ns=learner_ns)
    for d, s in zip(buf2.deal_idx, buf2.seat):
        assert (s % 2 == 0) == bool(learner_ns[d])
    cache.close()


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_play_team_deals_greedy_selfplay_identical(tmp_path):
    """Greedy self-play: one policy at both tables -> identical auctions,
    zero team swing, divergence gate closed, reward is par-only."""
    from bidding_dt.dd.solver import TableCache
    rng = np.random.default_rng(3)
    deals = random_deals(rng, 4)
    dealer = rng.integers(0, 4, 4)
    vuln = rng.integers(0, 4, 4)
    model = tiny_model(seed=5).eval()
    cache = TableCache(tmp_path / "dd.sqlite")
    roll = play_team_deals(model, deals, dealer, vuln, torch.device("cpu"),
                           cache=cache, greedy=True)
    b = roll.n_deals
    assert not roll.diverged.any()
    assert np.allclose(roll.team_imps, 0.0)
    assert np.array_equal(roll.score_ns[:b], roll.score_ns[b:])
    # same DD solves as the single-table case: batch dedupes both tables
    assert cache.misses == b and cache.hits == 0
    r = roll.row_rewards(reward_scale=1.0, team_weight=1.0, par_weight=0.25)
    # par-only: equals signed per-row par term
    expect = 0.25 * roll.par_imps_ns[np.asarray(roll.buf.deal_idx)] * np.where(
        np.asarray(roll.buf.seat) % 2 == 0, 1.0, -1.0)
    assert np.allclose(r, expect)
    cache.close()


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_play_team_deals_league_seating_and_signs(tmp_path):
    """League: learner N-S at table 0, E-W at table 1; opponents elsewhere.
    Team reward is zero-sum across the two tables of a deal, and every
    learner row past divergence carries the same signed team term."""
    from bidding_dt.dd.solver import TableCache
    rng = np.random.default_rng(1)
    deals = random_deals(rng, 6)
    dealer = rng.integers(0, 4, 6)
    vuln = rng.integers(0, 4, 6)
    model = tiny_model(seed=7).eval()
    cache = TableCache(tmp_path / "dd.sqlite")
    roll = play_team_deals(model, deals, dealer, vuln, torch.device("cpu"),
                           cache=cache, opponent=UniformPolicy(), greedy=True)
    b = roll.n_deals
    buf = roll.buf
    di = np.asarray(buf.deal_idx)
    seat = np.asarray(buf.seat)
    # table 0 rows (di < b): learner sits N-S; table 1 rows: learner E-W
    assert (seat[di < b] % 2 == 0).all()
    assert (seat[di >= b] % 2 == 1).all()
    r = roll.row_rewards(reward_scale=1.0, team_weight=1.0, par_weight=0.0)
    # team term sign: +team_imps at table 0, +team_imps at table 1 (E-W view)
    team = roll.team_imps[di % b]
    gate = (np.asarray(buf.call_idx) >= roll.diverge[di % b]).astype(float)
    assert np.allclose(r, team * gate)
    cache.close()


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_play_team_deals_sampling_diverges(tmp_path):
    """Sampled self-play at temp > 1 with an untrained (near-uniform) policy
    produces different auctions at the two tables for most deals."""
    from bidding_dt.dd.solver import TableCache
    rng = np.random.default_rng(11)
    deals = random_deals(rng, 32)
    dealer = rng.integers(0, 4, 32)
    vuln = rng.integers(0, 4, 32)
    model = tiny_model(seed=13).eval()
    cache = TableCache(tmp_path / "dd.sqlite")
    torch.manual_seed(17)
    roll = play_team_deals(model, deals, dealer, vuln, torch.device("cpu"),
                           cache=cache, greedy=False, temp=1.2)
    assert roll.diverged.mean() > 0.5
    # diverged deals gate from the first differing call; identical deals
    # report diverge == auction length and zero swing
    same = ~roll.diverged
    if same.any():
        assert np.allclose(roll.team_imps[same], 0.0)
    cache.close()


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_play_team_deals_selfplay_all_seats_hot(tmp_path):
    """Self-play: every seat samples at temp and is recorded at BOTH tables
    (both partners of each pair play hot and learn), and the heat makes the
    tables diverge."""
    from bidding_dt.dd.solver import TableCache
    rng = np.random.default_rng(21)
    deals = random_deals(rng, 6)
    dealer = rng.integers(0, 4, 6)
    vuln = rng.integers(0, 4, 6)
    model = tiny_model(seed=23).eval()
    cache = TableCache(tmp_path / "dd.sqlite")
    torch.manual_seed(29)
    roll = play_team_deals(model, deals, dealer, vuln, torch.device("cpu"),
                           cache=cache, greedy=False, temp=1.3)
    b = roll.n_deals
    di = np.asarray(roll.buf.deal_idx)
    seat = np.asarray(roll.buf.seat)
    assert set(seat[di < b].tolist()) == {0, 1, 2, 3}
    assert set(seat[di >= b].tolist()) == {0, 1, 2, 3}
    assert roll.diverged.any()
    cache.close()


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_play_team_deals_league_both_partners_hot(tmp_path):
    """League: both learner-team seats sample at temp and are recorded --
    N+S at table 0 and E+W at table 1; only the frozen opponent's seats are
    missing from the buffer."""
    from bidding_dt.dd.solver import TableCache
    rng = np.random.default_rng(31)
    deals = random_deals(rng, 6)
    dealer = rng.integers(0, 4, 6)
    vuln = rng.integers(0, 4, 6)
    model = tiny_model(seed=37).eval()
    cache = TableCache(tmp_path / "dd.sqlite")
    torch.manual_seed(41)
    roll = play_team_deals(model, deals, dealer, vuln, torch.device("cpu"),
                           cache=cache, opponent=UniformPolicy(),
                           greedy=False, temp=1.3)
    b = roll.n_deals
    di = np.asarray(roll.buf.deal_idx)
    seat = np.asarray(roll.buf.seat)
    assert set(seat[di < b].tolist()) == {0, 2}
    assert set(seat[di >= b].tolist()) == {1, 3}
    cache.close()


def test_partner_greedy_dev_partner_means():
    """Partner deviation is the mean of the partner's per-call devs in the
    same auction; partners that never called (or legacy buffers without dev)
    count as 0."""
    buf = RolloutBuffer()
    buf.deal_idx = [0, 0, 0, 0, 5, 5]
    buf.seat = [0, 0, 2, 2, 1, 3]
    buf.dev = [1.0, 3.0, 0.0, 4.0, 2.0, 0.0]
    d = partner_greedy_dev(buf)
    # seat-0 rows: partner seat 2, mean (0+4)/2 = 2; seat-2 rows: partner
    # seat 0, mean (1+3)/2 = 2; row 4 (seat 1): partner seat 3 dev 0; row 5:
    # partner seat 1 dev 2
    assert np.allclose(d, [2.0, 2.0, 2.0, 2.0, 0.0, 2.0])
    buf2 = RolloutBuffer()
    buf2.deal_idx = [0]
    buf2.seat = [0]
    buf2.dev = [1.5]
    assert np.allclose(partner_greedy_dev(buf2), [0.0])   # partner silent
    assert partner_greedy_dev(RolloutBuffer()).shape == (0,)
    buf3 = _toy_buf([0], [1], [0])                        # legacy: no dev
    assert np.allclose(partner_greedy_dev(buf3), [0.0])


def test_team_row_rewards_partner_dev_scales_negatives_only():
    """Negative row rewards are scaled by exp(-partner_dev) in (0, 1];
    positive rewards and partner_dev=None leave them untouched."""
    buf = _toy_buf(deal_idx=[0, 0], seat=[0, 1], call_idx=[0, 0])
    kw = dict(n_deals=1, team_imps=np.array([-8.0]), diverge=np.array([0]),
              par_imps_ns=np.zeros(2), reward_scale=1.0,
              team_weight=1.0, par_weight=0.25)
    r0 = team_row_rewards(buf, **kw)
    assert np.allclose(r0, [-8.0, 8.0])        # N: -8; E: sign-flipped +8
    pd = np.full(2, np.log(4.0))
    r = team_row_rewards(buf, partner_dev=pd, **kw)
    # negative row scaled by exp(-ln 4) = 1/4 -> -2; positive row untouched
    assert np.allclose(r, [-2.0, 8.0])
    # dev 0 (greedy partner) -> full punishment
    assert np.allclose(team_row_rewards(buf, partner_dev=np.zeros(2), **kw),
                       [-8.0, 8.0])


def _toy_buffer(n, seed=0):
    """n synthetic buffer rows of mixed lengths (no endplay needed)."""
    buf = RolloutBuffer()
    rng = np.random.default_rng(seed)
    for r in range(n):
        L = int(rng.integers(3, 12))
        buf.tokens.append(rng.integers(0, VOCAB_SIZE, L).astype(np.int64))
        buf.hand.append(np.zeros(HAND_DIM, np.float32))
        buf.vuln.append(int(rng.integers(0, 4)))
        buf.row_len.append(L)
        m = np.zeros(VOCAB_SIZE, bool)
        m[CALL_PASS] = True
        m[4:12] = True
        buf.mask.append(m)
        buf.action.append(int(rng.choice(np.flatnonzero(m))))
        buf.logprob.append(-1.0)
        buf.value.append(0.0)
        buf.deal_idx.append(0)
        buf.seat.append(0)
        buf.call_idx.append(0)
    buf.finish(rng.normal(size=n).astype(np.float32))
    return buf


def _run_ppo(buf, model, ref, lr=1e-3, seed=3, rng_seed=7, **cfg_kw):
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    cfg = RLConfig(**cfg_kw)
    t = buf.to_torch(torch.device("cpu"))
    stats = ppo_update(m, ref, t, opt, torch.device("cpu"), cfg,
                       kl_beta=cfg.kl_beta, ent_coef=cfg.ent_coef,
                       rng=np.random.default_rng(rng_seed))
    return m, stats


def test_ppo_update_length_bucket_single_minibatch_equivalent():
    """mb >= n: bucketing only reorders rows within one minibatch, and every
    reduction is permutation-invariant -> same update as length_bucket=0
    (up to fp ordering)."""
    buf = _toy_buffer(24, seed=1)
    model, ref = tiny_model(seed=1), tiny_model(seed=2)
    for p in ref.parameters():
        p.requires_grad_(False)
    m0, s0 = _run_ppo(buf, model, ref, ppo_epochs=2, minibatch_size=64,
                      length_bucket=0)
    m8, s8 = _run_ppo(buf, model, ref, ppo_epochs=2, minibatch_size=64,
                      length_bucket=8)
    for k in s0:
        assert s0[k] == pytest.approx(s8[k], rel=1e-4, abs=1e-6), k
    for (n1, p1), (n2, p8) in zip(m0.named_parameters(), m8.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p8, atol=1e-5), n1


def test_ppo_update_length_bucket_multibatch_runs():
    """mb < n: bucketed path partitions rows into variable-size minibatches,
    truncates each to its longest row, and still trains."""
    buf = _toy_buffer(64, seed=2)
    model, ref = tiny_model(seed=1), tiny_model(seed=2)
    for p in ref.parameters():
        p.requires_grad_(False)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    m, stats = _run_ppo(buf, model, ref, ppo_epochs=2, minibatch_size=16,
                        length_bucket=4)
    assert all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(before[n], p.detach())
               for n, p in m.named_parameters())


def test_ppo_update_cpu_bf16_amp():
    """cpu_bf16 path: bf16 autocast on CPU produces a finite, training update."""
    buf = _toy_buffer(32, seed=3)
    model, ref = tiny_model(seed=1), tiny_model(seed=2)
    for p in ref.parameters():
        p.requires_grad_(False)
    torch.manual_seed(3)
    m = copy.deepcopy(model)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    cfg = RLConfig(ppo_epochs=1, minibatch_size=16)
    t = buf.to_torch(torch.device("cpu"))
    stats = ppo_update(m, ref, t, opt, torch.device("cpu"), cfg,
                       kl_beta=cfg.kl_beta, ent_coef=cfg.ent_coef,
                       rng=np.random.default_rng(7), use_amp=True)
    assert all(np.isfinite(v) for v in stats.values())
    assert stats["kl"] >= -1e-6
    assert any(not torch.equal(p0.detach(), p1.detach())
               for p0, p1 in zip(model.parameters(), m.parameters()))


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_play_team_deals_presolve_matches_inline(tmp_path):
    """A warm_cache future joined via presolve gives identical rollouts and
    rewards to inline solving, and scoring becomes pure cache hits."""
    from concurrent.futures import ThreadPoolExecutor
    from bidding_dt.dd.reward import warm_cache
    from bidding_dt.dd.solver import TableCache
    rng = np.random.default_rng(5)
    deals = random_deals(rng, 4)
    dealer = rng.integers(0, 4, 4)
    vuln = rng.integers(0, 4, 4)
    model = tiny_model(seed=9).eval()
    cache = TableCache(tmp_path / "dd.sqlite")
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(warm_cache, [d.hands for d in deals], cache)
    roll = play_team_deals(model, deals, dealer, vuln, torch.device("cpu"),
                           cache=cache, greedy=True, presolve=fut)
    pool.shutdown(wait=True)
    assert fut.result() == 4                    # each distinct deal solved once
    assert cache.misses == 4 and cache.hits == 4  # presolve missed, scoring hit
    ref = play_team_deals(model, deals, dealer, vuln, torch.device("cpu"),
                          cache=cache, greedy=True)
    assert np.array_equal(roll.env.calls, ref.env.calls)
    assert np.array_equal(roll.team_imps, ref.team_imps)
    assert np.array_equal(roll.par_imps_ns, ref.par_imps_ns)
    cache.close()


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_play_team_deals_cpu_amp(tmp_path):
    """bf16-autocast rollouts on CPU run end to end with finite logprobs."""
    from bidding_dt.dd.solver import TableCache
    rng = np.random.default_rng(6)
    deals = random_deals(rng, 6)
    dealer = rng.integers(0, 4, 6)
    vuln = rng.integers(0, 4, 6)
    model = tiny_model(seed=11).eval()
    cache = TableCache(tmp_path / "dd.sqlite")
    torch.manual_seed(13)
    roll = play_team_deals(model, deals, dealer, vuln, torch.device("cpu"),
                           cache=cache, greedy=False, temp=1.2, amp=True)
    assert roll.env.done.all()
    assert np.isfinite(roll.buf.logprob).all()
    assert np.isfinite(roll.buf.value).all()
    cache.close()


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_play_team_deals_partner_dev_discount(tmp_path):
    """Sampled team rollouts record per-call greedy deviations and discount
    negative row rewards by exp(-partner mean dev) in (0, 1]; positives and
    greedy rollouts are unchanged."""
    from bidding_dt.dd.solver import TableCache
    rng = np.random.default_rng(43)
    deals = random_deals(rng, 8)
    dealer = rng.integers(0, 4, 8)
    vuln = rng.integers(0, 4, 8)
    model = tiny_model(seed=47).eval()
    cache = TableCache(tmp_path / "dd.sqlite")
    torch.manual_seed(53)
    roll = play_team_deals(model, deals, dealer, vuln, torch.device("cpu"),
                           cache=cache, greedy=False, temp=1.4)
    buf = roll.buf
    dev = np.asarray(buf.dev)
    assert dev.shape == (buf.n,)
    assert np.isfinite(dev).all() and (dev >= 0.0).all()
    pd = roll.partner_dev
    assert pd.shape == (buf.n,) and np.isfinite(pd).all() and (pd >= 0.0).all()

    raw = team_row_rewards(buf, roll.n_deals, roll.team_imps, roll.diverge,
                           roll.par_imps_ns, reward_scale=1.0)
    r = roll.row_rewards(reward_scale=1.0)
    neg = raw < 0
    assert neg.any() and pd[neg].max() > 0.0     # scaling is exercised
    assert np.allclose(r[~neg], raw[~neg])       # positives untouched
    assert np.allclose(r[neg], raw[neg] * np.exp(-pd[neg]))
    assert (r[neg] <= 0.0).all() and (r[neg] >= raw[neg]).all()  # shrunk, sign kept
    cache.close()
