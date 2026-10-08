import copy

import numpy as np
import pytest
import torch

from bidding_dt.data.legal import legal_mask as ref_legal_mask
from bidding_dt.data.vocab import CALL_PASS, CALL_X, CALL_XX, VOCAB_SIZE, call_to_id
from bidding_dt.env.auction_env import AuctionBatch, sample_from_masks
from bidding_dt.env.deals import random_deals
from bidding_dt.hand.config import HandVQConfig
from bidding_dt.hand.data import rotated_tokens
from bidding_dt.hand.model import HandVQ
from bidding_dt.rl.config import RLConfig
from bidding_dt.rl.ppo import ppo_update
from bidding_dt.rl.rollout import UniformPolicy, play_team_deals
from bidding_dt.system.config import EvoConfig, SystemConfig, SystemModelConfig
from bidding_dt.system.model import SystemPolicy, load_system_policy
from bidding_dt.system.population import Population, SystemVariant, mutate
from bidding_dt.system.state import (DEALER_DIM, DENOM_DIM, LEGAL_DIM,
                                     LEVEL_DIM, N_SEATS, PASS_DIM, SEAT_DIM,
                                     SUMMARY_DIM, VULN_DIM, frame_vuln,
                                     legal_mask, seat_frames, summarize,
                                     summary_features)

HAS_ENDPLAY = True
try:
    import endplay  # noqa: F401
except ImportError:  # pragma: no cover
    HAS_ENDPLAY = False

DEV = torch.device("cpu")
ENC_CFG = HandVQConfig(d_model=32, d_ff=64, n_heads=4, n_enc_layers=1,
                       n_slot_layers=1, n_dec_layers=1, n_codes=4,
                       codebook_size=16, code_dim=8)
POL_CFG = SystemModelConfig(d_model=32, d_ff=64, n_layers=2, d_z=8)


def rows(calls_per_row, p_per_row=None):
    """Build supervised-layout token rows + row lengths from call lists."""
    p_per_row = p_per_row or [0] * len(calls_per_row)
    L = max(2 + p + len(c) for p, c in zip(p_per_row, calls_per_row))
    tokens = torch.zeros(len(calls_per_row), L, dtype=torch.long)
    row_len = torch.zeros(len(calls_per_row), dtype=torch.long)
    for r, (p, calls) in enumerate(zip(p_per_row, calls_per_row)):
        tokens[r, 2 + p:2 + p + len(calls)] = torch.tensor(calls, dtype=torch.long)
        row_len[r] = 2 + p + len(calls)
    return tokens, row_len


def encoder(seed=0):
    torch.manual_seed(seed)
    return HandVQ(ENC_CFG).eval()


def policy(n_systems=3, seed=0, enc=None, z_cache=1000):
    torch.manual_seed(seed)
    cfg = SystemModelConfig(**{**POL_CFG.__dict__, "z_cache": z_cache})
    p = SystemPolicy(cfg, z_in=ENC_CFG.n_codes * ENC_CFG.d_model,
                     n_systems=n_systems)
    p.add_encoder("e0", enc if enc is not None else encoder())
    return p


def env_inputs(n_deals=6, seed=0, steps=0):
    rng = np.random.default_rng(seed)
    deals = random_deals(rng, n_deals)
    dealer = rng.integers(0, 4, n_deals)
    vuln = rng.integers(0, 4, n_deals)
    env = AuctionBatch(deals, dealer, vuln)
    for _ in range(steps):
        if not env.any_active():
            break
        idx = env.active_idx()
        masks = env.legal_masks(idx)
        env.step(idx, sample_from_masks(masks, rng))
    idx = env.active_idx()
    tokens, hand, vuln_cls, hero, row_len = env.build_inputs(idx)
    return env, idx, hero, tokens, hand, vuln_cls, row_len


# -- state -------------------------------------------------------------------

def test_summary_dim_layout():
    assert SUMMARY_DIM == (VULN_DIM + DEALER_DIM + LEVEL_DIM + DENOM_DIM
                           + SEAT_DIM + 3 + SEAT_DIM + SEAT_DIM + PASS_DIM
                           + LEGAL_DIM) == 83


def test_summary_fields_known_auctions():
    h, ps, d, rd = call_to_id("1H"), CALL_PASS, CALL_X, CALL_XX
    tokens, row_len = rows([
        [h],                    # 1H by self, nothing else
        [h, ps, ps, d],         # 1H by self, doubled by LHO (slot 3)
        [h, d, rd, ps],         # doubled by RHO, redoubled by partner
        [ps, ps, ps],           # passed out; 3 frame pads -> dealer is LHO
    ], p_per_row=[0, 0, 0, 3])
    s = summarize(tokens, row_len)
    assert s.p.tolist() == [0, 0, 0, 3]
    assert s.n.tolist() == [1, 4, 4, 3]
    assert s.highest.tolist() == [h, h, h, 0]
    assert s.bid_role.tolist() == [0, 0, 0, -1]        # -1 = nobody has bid
    assert s.doubled.tolist() == [False, True, True, False]
    assert s.dbl_role.tolist() == [-1, 3, 1, -1]       # LHO / RHO
    assert s.redoubled.tolist() == [False, False, True, False]
    assert s.xx_role.tolist() == [-1, -1, 2, -1]       # partner
    assert s.pass_run.tolist() == [0, 0, 1, 3]


def test_legal_mask_matches_env_and_legal_py():
    """The mask rebuilt from the summary equals the env's and data/legal.py's."""
    env, idx, hero, tokens, hand, vuln_cls, row_len = env_inputs(steps=0)
    rng = np.random.default_rng(1)
    seen = 0
    while env.any_active() and seen < 12:
        idx = env.active_idx()
        tokens, hand, vuln_cls, hero, row_len = env.build_inputs(idx)
        want = env.legal_masks(idx)
        s = summarize(torch.from_numpy(tokens), torch.from_numpy(row_len))
        got = legal_mask(s).numpy()
        assert (got == want).all()
        # the env's own incremental bookkeeping is the reference for the rest
        assert s.n.tolist() == env.n_calls[idx].tolist()
        assert s.highest.tolist() == env._highest_bid[idx].tolist()
        assert s.doubled.tolist() == env._doubled[idx].tolist()
        assert s.redoubled.tolist() == env._redoubled[idx].tolist()
        assert s.pass_run.tolist() == np.minimum(env._pass_run[idx], 3).tolist()
        rel = (env._last_bid_seat[idx] - hero) % 4
        assert s.bid_role.tolist() == np.where(env._highest_bid[idx] > 0, rel,
                                               -1).tolist()
        for r, i in enumerate(idx):                    # and vs data/legal.py
            p = int(s.p[r])
            calls = np.asarray(env.calls[i, :int(env.n_calls[i])])
            assert (got[r] == ref_legal_mask(calls, role_offset=p)).all()
        env.step(idx, sample_from_masks(want, rng))
        seen += 1
    assert seen >= 4


def test_seat_frames_and_vuln_match_the_dataset_convention():
    env, idx, hero, tokens, hand, vuln_cls, row_len = env_inputs(n_deals=8, steps=2)
    t = torch.from_numpy(tokens)
    s = summarize(t, torch.from_numpy(row_len))
    frames, fmask, flen = seat_frames(t, s)
    fv = frame_vuln(torch.from_numpy(vuln_cls)).numpy()
    B = len(idx)
    assert frames.shape[0] == N_SEATS * B
    for r, i in enumerate(idx):
        n = int(env.n_calls[i])
        calls = env.calls[i, :n].tolist()
        for k in range(N_SEATS):
            seat = (int(hero[r]) + k) % 4
            want = rotated_tokens(int(env.dealer[i]), calls, seat)
            row = k * B + r
            got = frames[row, :int(flen[row])].numpy()
            assert np.array_equal(got, want)
            assert int(fmask[row].sum()) == len(want)
            from bidding_dt.data.dataset import BiddingDataset
            assert int(fv[row]) == BiddingDataset.vuln_class(int(env.vuln[i]), seat)


def test_summary_features_blocks():
    # partner deals (2 frame pads) and opens 3S, LHO doubles, everyone passes
    tokens, row_len = rows([[call_to_id("3S"), CALL_X, CALL_PASS, CALL_PASS]],
                           p_per_row=[2])
    vuln = torch.tensor([1])                            # we are vulnerable
    s = summarize(tokens, row_len)
    f = summary_features(s, vuln)
    assert f.shape == (1, SUMMARY_DIM)
    o = 0
    blocks = {}
    for name, dim in (("vuln", VULN_DIM), ("dealer", DEALER_DIM),
                      ("level", LEVEL_DIM), ("denom", DENOM_DIM),
                      ("bid_by", SEAT_DIM), ("status", 3), ("dbl_by", SEAT_DIM),
                      ("xx_by", SEAT_DIM), ("passes", PASS_DIM),
                      ("legal", LEGAL_DIM)):
        blocks[name] = f[0, o:o + dim]
        o += dim
    assert o == SUMMARY_DIM
    assert blocks["vuln"].argmax() == 1 and blocks["vuln"].sum() == 1
    assert blocks["dealer"].argmax() == 2               # 2 frame pads
    assert blocks["level"].argmax() == 3 and blocks["denom"].argmax() == 4  # 3S
    assert blocks["bid_by"].argmax() == 3               # slot 2 -> partner
    assert blocks["status"].argmax() == 1               # doubled
    assert blocks["dbl_by"].argmax() == 4               # slot 3 -> LHO
    assert blocks["xx_by"].argmax() == 0                # slot 0 = nobody
    assert blocks["passes"].argmax() == 2               # two trailing passes
    assert torch.equal(blocks["legal"] > 0, legal_mask(s)[0])
    # partner may redouble or bid on; the double cannot be doubled again
    assert blocks["legal"][CALL_XX] == 1 and blocks["legal"][CALL_X] == 0
    assert int(blocks["legal"].sum()) == 2 + (VOCAB_SIZE - 1 - call_to_id("3S"))


# -- policy ------------------------------------------------------------------

def test_policy_state_dim_and_forward():
    pol = policy()
    env, idx, hero, tokens, hand, vuln_cls, row_len = env_inputs()
    t, h, v, L = (torch.from_numpy(x) for x in (tokens, hand, vuln_cls, row_len))
    logits, values = pol.forward_last(t, h, v, L)
    assert logits.shape == (len(idx), VOCAB_SIZE) and values.shape == (len(idx),)
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()
    assert pol.state_dim == 52 + SUMMARY_DIM + N_SEATS * pol.cfg.d_z
    assert pol.state(t, h, v, L).shape == (len(idx), pol.state_dim)


def test_policy_ignores_trailing_padding():
    pol = policy().eval()
    env, idx, hero, tokens, hand, vuln_cls, row_len = env_inputs(steps=2)
    t, h, v, L = (torch.from_numpy(x) for x in (tokens, hand, vuln_cls, row_len))
    with torch.no_grad():
        a, av = pol.forward_last(t, h, v, L)
        pad = torch.cat([t, torch.zeros(len(t), 4, dtype=torch.long)], 1)
        b, bv = pol.forward_last(pad, h, v, L)
        trunc = [pol.forward_last(t[i:i + 1, :int(L[i])], h[i:i + 1],
                                 v[i:i + 1], L[i:i + 1]) for i in range(len(t))]
    assert torch.allclose(a, b, atol=1e-6) and torch.allclose(av, bv, atol=1e-6)
    for i, (li, vi) in enumerate(trunc):
        assert torch.allclose(a[i], li[0], atol=1e-5)
        assert torch.allclose(av[i], vi[0], atol=1e-5)


def test_z_matches_the_frozen_encoder():
    """The four z blocks are the hand encoder run on each seat's own frame."""
    enc = encoder()
    pol = policy(enc=enc).eval()
    env, idx, hero, tokens, hand, vuln_cls, row_len = env_inputs(steps=1)
    t, v, L = torch.from_numpy(tokens), torch.from_numpy(vuln_cls), \
        torch.from_numpy(row_len)
    s = summarize(t, L)
    frames, fmask, _ = seat_frames(t, s)
    with torch.no_grad():
        mem, mm = enc.memory(frames, frame_vuln(v), None, fmask)
        z_ref, aux = enc.quantize(mem, mm)
        codes = pol._codes(t, L, v)
        assert torch.equal(aux["idx"].view(N_SEATS, len(idx), -1).permute(1, 0, 2),
                           codes)
        z_got = pol._z_features(codes).reshape(len(idx), N_SEATS, -1)
        # straight-through z equals its code embedding up to float rounding
        adapter = pol.adapters[0]
        # z_ref comes back seat-major ([self(B), RHO(B), ...]): fold to B-major
        z_ref = z_ref.view(N_SEATS, len(idx), ENC_CFG.n_codes, -1)
        z_ref = z_ref.permute(1, 0, 2, 3).reshape(len(idx), N_SEATS, -1)
        z_want = adapter(z_ref)
        assert torch.allclose(z_got, z_want, atol=1e-5)
        assert pol._z_features(codes).shape[-1] == N_SEATS * pol.cfg.d_z


def test_z_cache_is_transparent():
    pol = policy(z_cache=64).eval()
    env, idx, hero, tokens, hand, vuln_cls, row_len = env_inputs(steps=1)
    t, h, v, L = (torch.from_numpy(x) for x in (tokens, hand, vuln_cls, row_len))
    with torch.no_grad():
        first = pol._codes(t, L, v)
        n_cached = len(pol.cache)
        second = pol._codes(t, L, v)
        assert torch.equal(first, second)
        assert len(pol.cache) == n_cached > 0           # pure hits
        assert torch.allclose(pol.forward_last(t, h, v, L)[0],
                              pol.forward_last(t, h, v, L)[0], atol=0)
    off = policy(z_cache=0).eval()
    with torch.no_grad():
        assert len(off.cache) == 0
        assert torch.equal(off._codes(t, L, v), off._codes(t, L, v))


def test_encoder_stays_out_of_the_policy():
    enc = encoder()
    pol = policy(enc=enc)
    assert not any(k.startswith("_encoder") for k in pol.state_dict())
    assert sum(p.numel() for p in pol.parameters()) == pol.num_params()
    assert all(not p.requires_grad for p in enc.parameters())
    pol.train()
    assert not enc.training                             # frozen stays in eval
    with pytest.raises(KeyError):
        pol.set_encoder("nope")


def test_z_blocks_reach_the_policy_head():
    """Swapping the frozen encoder changes the z block (and the logits), while
    everything else in the state is untouched and the cache does not leak codes
    between encoders."""
    env, idx, hero, tokens, hand, vuln_cls, row_len = env_inputs(steps=1)
    t, h, v, L = (torch.from_numpy(x) for x in (tokens, hand, vuln_cls, row_len))
    pol = policy(enc=encoder(seed=0), z_cache=256).eval()
    pol.add_encoder("e1", encoder(seed=1))
    z0 = slice(52 + SUMMARY_DIM, pol.state_dim)
    with torch.no_grad():
        s0 = pol.state(t, h, v, L)
        a = pol.forward_last(t, h, v, L)[0]
        pol.set_encoder("e1")
        s1 = pol.state(t, h, v, L)
        b = pol.forward_last(t, h, v, L)[0]
        pol.set_encoder("e0")
        s0_again = pol.state(t, h, v, L)
    assert not torch.allclose(s0[:, z0], s1[:, z0], atol=1e-9)   # z drives it
    assert torch.equal(s0[:, :z0.start], s1[:, :z0.start])       # rest is equal
    assert torch.equal(s0, s0_again)                             # per-encoder cache
    assert not torch.equal(a, b)                                 # reaches the head


# -- population / outer loop -------------------------------------------------

def test_mutate_is_gaussian_around_the_parent():
    torch.manual_seed(0)
    pol = policy(n_systems=2)
    src, dst = pol.adapters[0], pol.adapters[1]
    before = {k: v.clone() for k, v in src.state_dict().items()}
    rel = mutate(dst, src, sigma=0.01)
    diff = torch.cat([(d - before[k]).flatten()
                      for k, d in dst.state_dict().items()
                      if k == "net.0.weight"])           # the projection itself
    assert rel > 0
    assert float(diff.std()) == pytest.approx(0.01, rel=0.15)
    assert float(diff.mean()) == pytest.approx(0.0, abs=0.01)
    assert not torch.equal(dst.state_dict()["net.0.weight"],
                           before["net.0.weight"])


def test_population_evolve_keeps_elites_and_breeds_the_worst():
    pol = policy(n_systems=4)
    evo = EvoConfig(systems=4, keep_best=2, replace_frac=0.5, sigma=0.01)
    pop = Population(pol, evo, ["e0"])
    for v, score in zip(pop.variants, [1.0, -5.0, 3.0, -1.0]):
        pop.score(v, np.full(4, score))
    elite = {k: v.clone() for k, v in pol.adapters[2].state_dict().items()}
    torch.manual_seed(3)
    bred = pop.evolve()
    assert len(bred) == 2
    # the two worst (sys1 -5, sys3 -1) are replaced by children of the two
    # elites (sys2 3, sys0 1), best first
    assert {b["child"]: b["parent"] for b in bred} == {"sys3": "sys2",
                                                      "sys1": "sys0"}
    assert all(torch.equal(v, pol.adapters[2].state_dict()[k])
               for k, v in elite.items())               # elites untouched
    by_name = {v.name: v for v in pop.variants}
    assert by_name["sys1"].gen == 1 and by_name["sys1"].parent == "sys0"
    assert by_name["sys1"].evals == 0 and by_name["sys1"].score != \
        by_name["sys1"].score                            # NaN until re-evaluated
    assert not all(torch.equal(v, pol.adapters[1].state_dict()[k])
                   for k, v in elite.items())            # the child moved


def test_population_sampling_and_stats():
    pol = policy(n_systems=3)
    pop = Population(pol, EvoConfig(systems=3), ["e0"])
    rng = np.random.default_rng(0)
    drawn = {pop.sample(rng).name for _ in range(60)}
    assert drawn == set(pop.names())                     # uniform covers all
    v = pop.variants[0]
    pop.observe(v, np.array([2.0, 4.0]))
    assert v.iters == 1 and v.train_imps == pytest.approx(0.2 * 3.0)
    pop.score(v, np.array([2.0, -1.0]))
    assert v.score == pytest.approx(0.5) and v.evals == 1
    pop.activate(v)
    assert pol.active == v.adapter and pol.enc_name == v.encoder
    fit = Population(pol, EvoConfig(systems=3, sample_by="fitness"), ["e0"])
    for x, sc in zip(fit.variants, [-9.0, 0.0, 9.0]):
        pop.score(x, np.array([sc]))
    assert fit.sample(np.random.default_rng(1)).name == "sys2"


def test_clone_system_reproduces_the_parent():
    pol = policy(n_systems=3).eval()
    pop = Population(pol, EvoConfig(systems=3), ["e0"])
    for v, score in zip(pop.variants, [0.0, 1.0, -1.0]):
        pop.score(v, np.array([score]))
    best = pop.best(1)[0]
    clone = pop.clone_system(best)
    assert len(clone.adapters) == 1 and clone.cache.max == 0
    assert all(not p.requires_grad for p in clone.parameters())
    env, idx, hero, tokens, hand, vuln_cls, row_len = env_inputs(steps=2)
    t, h, v_, L = (torch.from_numpy(x) for x in (tokens, hand, vuln_cls, row_len))
    pop.activate(best)
    with torch.no_grad():
        a, av = pol.forward_last(t, h, v_, L)
        b, bv = clone.forward_last(t, h, v_, L)
    assert torch.allclose(a, b, atol=1e-6) and torch.allclose(av, bv, atol=1e-6)


def test_population_persistence_roundtrip(tmp_path):
    pol = policy(n_systems=3)
    pop = Population(pol, EvoConfig(systems=3, sigma=0.01), ["e0"])
    pop.score(pop.variants[0], np.array([1.5]))
    pop.observe(pop.variants[1], np.array([-2.0]))
    pop.save(tmp_path / "population")
    fresh = policy(n_systems=3, seed=5)
    loaded = Population.load(tmp_path / "population", fresh,
                             EvoConfig(systems=3, sigma=0.01), ["e0"])
    assert loaded.names() == pop.names()
    got = loaded.get("sys0")
    assert got.score == pytest.approx(1.5) and got.evals == 1
    assert loaded.get("sys1").iters == 1
    assert loaded.get("sys2").score != loaded.get("sys2").score   # NaN survives
    assert SystemVariant.from_json(got.to_json()) == got


def test_checkpoint_roundtrip(tmp_path):
    from bidding_dt.system.train import save_ckpt
    pol = policy(n_systems=2)
    pop = Population(pol, EvoConfig(systems=2), ["e0"])
    pop.score(pop.variants[1], np.array([2.0]))
    pol.set_system("e0", 1)
    opt = torch.optim.AdamW(pol.trainable_params(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    cfg = SystemConfig()
    cfg.evo.encoders = ["enc.pt"]
    save_ckpt(tmp_path / "last.pt", pol, pop, opt, sched, 2, 7, 1.25, cfg,
              {"e0": pol.encoder})
    ck = torch.load(tmp_path / "last.pt", weights_only=False)
    assert ck["gen"] == 2 and ck["iter"] == 7 and ck["best_imps"] == 1.25
    restored = load_system_policy(ck, {"e0": encoder()}, encoder="e0")
    assert restored.active == 1 and len(restored.adapters) == 2
    env, idx, hero, tokens, hand, vuln_cls, row_len = env_inputs(steps=1)
    t, h, v, L = (torch.from_numpy(x) for x in (tokens, hand, vuln_cls, row_len))
    with torch.no_grad():
        a = pol.eval().forward_last(t, h, v, L)[0]
        b = restored.forward_last(t, h, v, L)[0]
    assert torch.allclose(a, b, atol=1e-6)


# -- inner loop (needs the DD solver for rewards) -----------------------------

@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_ppo_iteration_updates_policy_not_adapters(tmp_path):
    pol = policy(n_systems=2, z_cache=4096)
    pop = Population(pol, EvoConfig(systems=2), ["e0"])
    pop.activate(pop.variants[0])
    trunk_before = copy.deepcopy(pol.policy_head.weight.detach())
    adapter_before = copy.deepcopy(pol.adapters[0].state_dict())
    rng = np.random.default_rng(0)
    deals = random_deals(rng, 4)
    dealer, vuln = rng.integers(0, 4, 4), rng.integers(0, 4, 4)
    from bidding_dt.dd.solver import TableCache
    cache = TableCache(tmp_path / "dd.sqlite")
    roll = play_team_deals(pol, deals, dealer, vuln, DEV, rng, cache=cache,
                           opponent=UniformPolicy(), temp=1.2)
    assert roll.buf.n > 0
    row_r = roll.row_rewards(0.2, 1.0, 0.25)
    roll.buf.finish(row_r, 1.0, 0.95)
    rl = RLConfig(minibatch_size=16, ppo_epochs=2, length_bucket=0)
    opt = torch.optim.AdamW(pol.trainable_params(adapter=False), lr=1e-3)
    stats = ppo_update(pol, None, roll.buf.to_torch(DEV), opt, DEV, rl,
                       0.0, 0.01, rng, temp=1.2)
    cache.close()
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"):
        assert np.isfinite(stats[key]), key
    assert not torch.equal(trunk_before, pol.policy_head.weight.detach())
    assert all(torch.equal(v, pol.adapters[0].state_dict()[k])
               for k, v in adapter_before.items())


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_evaluate_systems_scores_every_variant(tmp_path):
    from bidding_dt.system.train import evaluate_systems
    pol = policy(n_systems=2, z_cache=4096)
    pop = Population(pol, EvoConfig(systems=2, eval_deals=4), ["e0"])
    rng = np.random.default_rng(0)
    deals = random_deals(rng, 4)
    dealer, vuln = rng.integers(0, 4, 4), rng.integers(0, 4, 4)
    from bidding_dt.dd.solver import TableCache
    cache = TableCache(tmp_path / "dd.sqlite")
    scores = evaluate_systems(pol, pop, deals, dealer, vuln, DEV, cache,
                              UniformPolicy())
    cache.close()
    assert set(scores) == set(pop.names())
    assert all(v.evals == 1 for v in pop.variants)
    assert all(np.isfinite(v.score) for v in pop.variants)
