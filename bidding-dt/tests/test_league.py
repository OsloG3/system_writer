import numpy as np
import pytest
import torch

from bidding_dt.config import ModelConfig
from bidding_dt.env.deals import random_deals
from bidding_dt.model.transformer import BiddingDT
from bidding_dt.rl.league import League, parse_spec
from bidding_dt.rl.model import RLModel, build_from_ckpt, load_any_ckpt
from bidding_dt.rl.rollout import UniformPolicy, play_team_deals

HAS_ENDPLAY = True
try:
    import endplay  # noqa: F401
except ImportError:  # pragma: no cover
    HAS_ENDPLAY = False

TINY = ModelConfig.from_preset("tiny")
OTHER = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128)
DEV = torch.device("cpu")


def make_bc_ckpt(path, cfg=None, seed=0):
    cfg = cfg or TINY
    torch.manual_seed(seed)
    dt = BiddingDT(cfg)
    torch.save({"model": dt.state_dict(), "model_cfg": cfg.__dict__,
                "step": 0, "epoch": 0}, path)
    return dt


def make_rl_ckpt(path, cfg=None, seed=1):
    cfg = cfg or TINY
    torch.manual_seed(seed)
    m = RLModel(cfg)
    torch.save({"model": m.state_dict(), "model_cfg": cfg.__dict__,
                "iter": 3, "best_imps": 0.0}, path)
    return m


def test_parse_spec():
    assert parse_spec("random") == ("random", None)
    assert parse_spec("bc:runs/x/best.pt") == ("auto", "runs/x/best.pt")
    assert parse_spec("rl:runs/x/best.pt") == ("auto", "runs/x/best.pt")
    assert parse_spec("runs/x/best.pt") == ("auto", "runs/x/best.pt")


def test_build_from_ckpt_detects_kind(tmp_path):
    bc = tmp_path / "bc.pt"
    rl = tmp_path / "rl.pt"
    dt = make_bc_ckpt(bc)
    m = make_rl_ckpt(rl)
    ck = torch.load(bc, weights_only=False)
    got, kind = build_from_ckpt(ck)
    assert kind == "bc"
    assert torch.equal(got.dt.head.weight, dt.head.weight)
    ck = torch.load(rl, weights_only=False)
    got, kind = build_from_ckpt(ck)
    assert kind == "rl"
    assert torch.equal(got.value_head[-1].bias, m.value_head[-1].bias)
    # load_any_ckpt handles both and returns eval-mode RLModel
    for p in (bc, rl):
        got = load_any_ckpt(p)
        assert isinstance(got, RLModel) and not got.training


def test_league_add_spec_dedupe_and_hetero_arch(tmp_path):
    bc = tmp_path / "bc.pt"
    rl = tmp_path / "rl.pt"
    make_bc_ckpt(bc)
    make_rl_ckpt(rl, cfg=OTHER)
    lg = League(DEV)
    assert lg.sample(np.random.default_rng(0)) is None
    lg.add_spec("random")
    a = lg.add_spec(f"bc:{bc}", weight=2.0)
    b = lg.add_spec(str(bc), weight=3.0)        # same ckpt, no prefix
    c = lg.add_spec(f"rl:{rl}")
    assert a is b and a.weight == 3.0           # dedupe refreshes weight
    assert a.kind == "bc" and c.kind == "rl"
    assert len(lg) == 3
    # heterogeneous archs coexist: tiny BC + custom-arch RL
    assert a.model.dt.cfg.d_model == TINY.d_model
    assert c.model.dt.cfg.d_model == OTHER.d_model
    assert isinstance(lg.opponent(lg.get("random")), UniformPolicy)


def test_league_weighted_sampling(tmp_path):
    bc = tmp_path / "bc.pt"
    make_bc_ckpt(bc)
    lg = League(DEV)
    lg.add_spec("random", weight=3.0)
    lg.add_spec(f"bc:{bc}", weight=1.0)
    p = lg.probabilities()
    assert np.allclose(p, [0.75, 0.25])
    rng = np.random.default_rng(0)
    freq = {}
    for _ in range(4000):
        m = lg.sample(rng)
        freq[m.name] = freq.get(m.name, 0) + 1
    tot = sum(freq.values())
    assert abs(freq["random"] / tot - 0.75) < 0.05


def test_league_pfsp_prioritizes_hard_opponents(tmp_path):
    bc = tmp_path / "bc.pt"
    make_bc_ckpt(bc)
    lg = League(DEV, pfsp_alpha=1.0)
    easy = lg.add_spec("random", weight=1.0)
    hard = lg.add_spec(f"bc:{bc}", weight=1.0)
    assert np.allclose(lg.probabilities(), [0.5, 0.5])  # unplayed: neutral
    for _ in range(10):
        lg.update_result(easy, np.full(8, 5.0))    # learner always wins
        lg.update_result(hard, np.full(8, -5.0))   # learner always loses
    assert easy.winrate > 0.85 and hard.winrate < 0.15
    p = lg.probabilities()
    assert p[1] > 0.8                              # hard opponent dominates
    assert easy.iters == 10


def test_league_snapshot_materialize_and_evict(tmp_path):
    torch.manual_seed(0)
    model = RLModel(TINY)
    lg = League(DEV)
    m1 = lg.add_snapshot(model.state_dict(), TINY, name="s1", max_snapshots=2)
    m2 = lg.add_snapshot(model.state_dict(), TINY, name="s2", max_snapshots=2)
    m3 = lg.add_snapshot(model.state_dict(), TINY, name="s3", max_snapshots=2)
    assert lg.get("s1") is None and len(lg) == 2   # oldest evicted
    holder = RLModel(TINY)
    opp = lg.opponent(m3, holder)
    assert opp is holder
    for k, v in m3.state.items():
        assert torch.equal(dict(holder.state_dict())[k].cpu(), v)
    # without a holder the league builds and caches its own
    opp2 = lg.opponent(m2)
    assert isinstance(opp2, RLModel)
    assert lg.opponent(m2) is opp2
    # snapshots are frozen copies: mutating the learner does not touch them
    before = m3.state["dt.head.weight"].clone()
    with torch.no_grad():
        model.dt.head.weight.add_(1.0)
    assert torch.equal(m3.state["dt.head.weight"], before)


def test_league_persistence_roundtrip(tmp_path):
    bc = tmp_path / "bc.pt"
    make_bc_ckpt(bc)
    torch.manual_seed(0)
    model = RLModel(TINY)
    pdir = tmp_path / "league"
    lg = League(DEV, pfsp_alpha=0.5, persist_dir=pdir)
    lg.add_spec("random", weight=2.0)
    lg.add_spec(f"bc:{bc}")
    lg.add_snapshot(model.state_dict(), TINY, name="s1")
    lg.update_result(lg.get(f"bc:{bc}"), np.full(8, -3.0))
    lg.save_meta()

    lg2 = League.load(pdir, DEV, pfsp_alpha=0.5)
    assert lg2.names() == lg.names()
    for n in lg.names():
        a, b = lg.get(n), lg2.get(n)
        assert (a.kind, a.weight) == (b.kind, b.weight)
        assert (a.iters, a.winrate) == (b.iters, b.winrate)
    s = lg2.get("s1")
    for k, v in lg.get("s1").state.items():
        assert torch.equal(s.state[k], v)
    holder = RLModel(TINY)
    lg2.opponent(s, holder)                        # snapshot still usable
    # re-adding a restored spec is a no-op that keeps stats
    m = lg2.add_spec(f"bc:{bc}", weight=1.0)
    assert m.iters == lg.get(f"bc:{bc}").iters and len(lg2) == len(lg)
    # a member whose checkpoint vanished is skipped, not fatal
    bc.unlink()
    lg3 = League.load(pdir, DEV)
    assert f"bc:{bc}" not in lg3.names() and len(lg3) == 2


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_league_member_plays_team_match(tmp_path):
    """A heterogeneous external member (different arch) can hold the
    opponent seats in a two-table team rollout against the learner."""
    from bidding_dt.dd.solver import TableCache
    bc = tmp_path / "bc.pt"
    make_bc_ckpt(bc, cfg=OTHER, seed=3)
    lg = League(DEV)
    member = lg.add_spec(f"bc:{bc}")
    torch.manual_seed(0)
    learner = RLModel(TINY).eval()
    rng = np.random.default_rng(0)
    deals = random_deals(rng, 4)
    dealer = rng.integers(0, 4, 4)
    vuln = rng.integers(0, 4, 4)
    cache = TableCache(tmp_path / "dd.sqlite")
    roll = play_team_deals(learner, deals, dealer, vuln, DEV,
                           cache=cache, opponent=lg.opponent(member),
                           greedy=True, record=False)
    assert roll.team_imps.shape == (4,)
    assert np.isfinite(roll.team_imps).all()
    lg.update_result(member, roll.team_imps)
    assert member.iters == 1
    cache.close()


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_train_ppo_league_run_and_resume(tmp_path, capsys, monkeypatch):
    """End-to-end: warm start from a BC ckpt, train against a league that
    includes the init model + an external member + snapshots; resume
    restores the persisted league."""
    import json as _json
    import sys

    from bidding_dt.rl import train_ppo
    bc = tmp_path / "bc.pt"
    make_bc_ckpt(bc, seed=5)
    out = tmp_path / "run"
    cfgf = tmp_path / "rl.yaml"
    cfgf.write_text(f"""
rl:
  iters: 2
  deals_per_iter: 8
  eval_every: 1
  eval_deals: 4
  snapshot_every: 1
  league_prob: 1.0
  league_members: [random]
  minibatch_size: 32
  ppo_epochs: 1
  warmup_iters: 1
  rollout_temp: 1.2
  cache_path: {tmp_path / "dd.sqlite"}
""")
    base = ["train_ppo", "--config", str(cfgf), "--bc-ckpt", str(bc),
            "--out", str(out)]
    monkeypatch.setattr(sys, "argv", base)
    train_ppo.main()

    log = [_json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    init = log[0]
    assert init["init_kind"] == "bc"
    assert f"bc:{bc}" in init["league"] and "random" in init["league"]
    iters = [r for r in log if r["event"] == "iter"]
    assert len(iters) == 2
    assert all(r["league"] and r["league_member"] for r in iters)
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    meta = _json.loads((out / "league" / "members.json").read_text())
    kinds = [m["kind"] for m in meta["members"]]
    assert "snapshot" in kinds and "random" in kinds and "bc" in kinds

    # resume: league (incl. snapshots + stats) is restored, training continues
    monkeypatch.setattr(sys, "argv", base + ["--resume", "last", "--iters", "4"])
    train_ppo.main()
    captured = capsys.readouterr().out
    assert "league restored" in captured and "resumed from" in captured
    log = [_json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    iters = [r for r in log if r["event"] == "iter"]
    assert [r["iter"] for r in iters] == [0, 1, 2, 3]
    meta = _json.loads((out / "league" / "members.json").read_text())
    assert sum(m["kind"] == "snapshot" for m in meta["members"]) == 4
    played = [m for m in meta["members"] if m["iters"] > 0]
    assert played, "league stats should track faced opponents"
