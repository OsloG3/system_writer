r"""Self-play PPO training entrypoint.

Warm-starts from a BC checkpoint, then improves the bidding policy with a
two-table team reward: every deal is played twice (same dealer/vuln), and
the IMP swing between the tables -- credited only to calls at/after the
first point where the two auctions diverge -- is the primary signal, with
the double-dummy par-diff IMPs added at a quarter of the per-IMP weight.
In league mode the learner team sits N-S at one table and E-W at the other
against a frozen league opponent; otherwise both tables are pure self-play
(sampling temperature supplies the auction variance). A KL anchor to the
frozen BC policy and an entropy bonus are annealed to zero so the pair can
diverge from human systems and form its own agreements.

The league (rl/league.py) pools frozen snapshots of the learner with any
external models: BC (supervised) checkpoints, other RL runs, the random
baseline -- `--league bc:runs/small/best.pt --league rl:runs/old/best.pt`
(or rl.league_members in the YAML config). The warm-start checkpoint joins
the league automatically, so a run can be *further* trained from an earlier
RL checkpoint against the models of its league:

  python -m bidding_dt.rl.train_ppo --preset small \
      --bc-ckpt runs/rl_small/best.pt \
      --league bc:runs/small/best.pt --league rl:runs/rl_tiny/best.pt \
      --out runs/rl_small2
# --bc-ckpt accepts BC or RL checkpoints (auto-detected; arch is adopted)

CPU smoke:
  python -m bidding_dt.rl.train_ppo --preset tiny --bc-ckpt runs/smoke/best.pt \
      --config configs/rl_smoke.yaml --out runs/rl_smoke
GPU run:
  python -m bidding_dt.rl.train_ppo --config configs/rl_small.yaml \
      --preset small --bc-ckpt runs/small/best.pt --out runs/rl_small
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..env.deals import random_deals
from ..dd.solver import TableCache
from ..train import lr_lambda_fn, pick_device
from .config import RLTrainConfig
from .league import League
from .model import RLModel, build_from_ckpt
from .ppo import ppo_update
from .rollout import UniformPolicy, play_team_deals


def anneal(start: float, end: float, it: int, total: int) -> float:
    t = min(1.0, it / max(1, total - 1))
    return start + (end - start) * t


@torch.no_grad()
def evaluate_match(model, opp, deals, dealer, vuln, device, cache):
    """Team IMPs/board vs a fixed opponent: every deal is played at two
    tables (learner N-S at table 0, E-W at table 1); the DD swing between
    the tables cancels card-lie luck per board."""
    roll = play_team_deals(model, deals, dealer, vuln, device, cache=cache,
                           opponent=opp, greedy=True, record=False)
    x = roll.team_imps
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--preset", default=None, choices=["tiny", "small", "base", "large"])
    ap.add_argument("--bc-ckpt", "--init", dest="bc_ckpt", default=None,
                    help="warm start + KL anchor: BC or RL checkpoint (auto-detected; "
                         "the checkpoint's architecture overrides --preset/config)")
    ap.add_argument("--league", action="append", default=[], metavar="SPEC",
                    help="frozen league opponent, repeatable: random | bc:<ckpt> | "
                         "rl:<ckpt> | <ckpt> (merged with rl.league_members)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--deals-per-iter", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-deals", type=int, default=None)
    ap.add_argument("--no-league", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache", default=None, help="DD table sqlite path")
    ap.add_argument("--resume", default=None, help="checkpoint path or 'last'")
    args = ap.parse_args()

    cfg = RLTrainConfig.load(args.config, args.preset)
    rl = cfg.rl
    if args.iters is not None:
        rl.iters = args.iters
    if args.deals_per_iter is not None:
        rl.deals_per_iter = args.deals_per_iter
    if args.lr is not None:
        rl.lr = args.lr
    if args.eval_every is not None:
        rl.eval_every = args.eval_every
    if args.eval_deals is not None:
        rl.eval_deals = args.eval_deals
    if args.no_league:
        rl.league_prob = 0.0
    if args.cache is not None:
        rl.cache_path = args.cache

    out_dir = Path(args.out) if args.out else Path("runs") / ("rl-" + time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "log.jsonl", "a")

    def log(rec: dict):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    device = pick_device(args.device or rl.device)
    use_amp = device.type == "cuda"
    torch.manual_seed(rl.seed)
    rng = np.random.default_rng(rl.seed)

    # warm start from a BC (supervised) or RL checkpoint -- auto-detected;
    # the checkpoint's architecture wins over preset/config so any model can
    # be further trained. The same frozen policy is the KL anchor and (via
    # league_include_init) the league's first external member.
    ref_model = None
    init_kind = None
    if args.bc_ckpt:
        ck = torch.load(args.bc_ckpt, map_location="cpu", weights_only=False)
        init_model, init_kind = build_from_ckpt(ck, device=device)
        for p in init_model.parameters():
            p.requires_grad_(False)
        if init_model.cfg != cfg.model:
            print(f"adopting warm-start architecture from {args.bc_ckpt} "
                  f"(overrides preset/config)")
            cfg.model = init_model.cfg
        model = RLModel(init_model.cfg).to(device)
        model.load_state_dict(init_model.state_dict())
        model.eval()
        ref_model = init_model
        print(f"warm start + KL anchor from {args.bc_ckpt} ({init_kind})")
    else:
        model = RLModel(cfg.model).to(device)
        model.eval()

    opt = torch.optim.AdamW(model.parameters(), lr=rl.lr, weight_decay=rl.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_fn(rl.warmup_iters, rl.iters, rl.min_lr_frac))

    cache = TableCache(rl.cache_path)
    holder = RLModel(cfg.model).to(device).eval()  # materializes snapshots
    for p in holder.parameters():
        p.requires_grad_(False)
    uniform = UniformPolicy()

    start_iter = 0
    best_imps = -99.0
    if args.resume:
        path = out_dir / "last.pt" if args.resume == "last" else Path(args.resume)
        ck = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        start_iter = ck["iter"] + 1
        best_imps = ck.get("best_imps", -99.0)
        print(f"resumed from {path} at iter {start_iter}")

    # league: restore the persisted pool on resume, then add configured
    # external members (add_spec is idempotent, so restored ones keep their
    # winrate stats instead of being reloaded as duplicates).
    league_dir = out_dir / "league"
    league = None
    if args.resume and rl.league_persist and (league_dir / "members.json").exists():
        league = League.load(league_dir, device, pfsp_alpha=rl.pfsp_alpha)
        print(f"league restored from {league_dir}: {league.names()}")
    if league is None:
        league = League(device, pfsp_alpha=rl.pfsp_alpha,
                        persist_dir=league_dir if rl.league_persist else None)
    specs = list(rl.league_members) + list(args.league)
    if args.bc_ckpt and rl.league_include_init:
        specs.append(args.bc_ckpt)
    for spec in specs:
        try:
            m = league.add_spec(spec, weight=rl.member_weight)
            print(f"league member: {m.name} (kind={m.kind})")
        except (OSError, KeyError, RuntimeError) as ex:
            print(f"warning: league member {spec!r} not added: {ex}")
    league.save_meta()

    # fixed eval deals (team-scored: the learner sits N-S at one table and
    # E-W at the other on every deal, so seating is inherently balanced)
    eval_rng = np.random.default_rng(rl.seed + 1)
    eval_deals = random_deals(eval_rng, rl.eval_deals)
    eval_dealer = eval_rng.integers(0, 4, len(eval_deals))
    eval_vuln = eval_rng.integers(0, 4, len(eval_deals))

    def save(path, it):
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                    "sched": sched.state_dict(), "iter": it,
                    "best_imps": best_imps,
                    "model_cfg": model.cfg.__dict__,
                    "cfg": cfg.to_dict()}, path)
        league.save_meta()

    log({"event": "init", "params": model.num_params(), "device": str(device),
         "cfg": cfg.to_dict(), "bc_ckpt": args.bc_ckpt,
         "init_kind": init_kind, "league": league.names()})

    for it in range(start_iter, rl.iters):
        t0 = time.time()
        beta = anneal(rl.kl_beta, rl.kl_beta_end, it, rl.iters)
        ent = anneal(rl.ent_coef, rl.ent_coef_end, it, rl.iters)

        deals = random_deals(rng, rl.deals_per_iter)
        dealer = rng.integers(0, 4, len(deals))
        vuln = rng.integers(0, 4, len(deals))

        opp_model = None
        member = None
        league_on = bool((not args.no_league) and len(league)
                         and rng.random() < rl.league_prob)
        if league_on:
            member = league.sample(rng)
            opp_model = league.opponent(member, holder)

        roll = play_team_deals(model, deals, dealer, vuln, device, rng,
                               cache=cache, opponent=opp_model,
                               greedy=rl.greedy, temp=rl.rollout_temp)
        if member is not None:
            league.update_result(member, roll.team_imps)
        buf, env = roll.buf, roll.env
        row_r = roll.row_rewards(rl.reward_scale, rl.team_weight,
                                 rl.par_weight)
        buf.finish(row_r, rl.gamma, rl.lam)
        tensors = buf.to_torch(device)
        stats = ppo_update(model, ref_model, tensors, opt, device, rl,
                           beta, ent, rng, use_amp=use_amp,
                           temp=rl.rollout_temp)
        sched.step()

        if rl.snapshot_every and (it + 1) % rl.snapshot_every == 0:
            league.add_snapshot(model.state_dict(), cfg.model,
                                name=f"snapshot-{it + 1}",
                                weight=rl.snapshot_weight,
                                max_snapshots=rl.max_snapshots)

        n_calls = env.n_calls[env.done]
        passout = float(np.mean([(env.n_calls[i] == 4) and (env.calls[i, :4] == 1).all()
                                 for i in range(env.b)]))
        b = roll.n_deals
        # learner-perspective par IMPs: table 0 N-S, table 1 E-W (in self-play
        # both tables are the learner; the mean mixes both seats)
        learn_par = np.concatenate([roll.par_imps_ns[:b], -roll.par_imps_ns[b:]])
        first_diff = (float(roll.diverge[roll.diverged].mean())
                      if roll.diverged.any() else 0.0)
        rec = {"event": "iter", "iter": it,
               "reward_mean": round(float(row_r.mean()), 4),
               "reward_std": round(float(row_r.std()), 4),
               "team_imps": round(float(roll.team_imps.mean()), 4),
               "imps_vs_par": round(float(learn_par.mean()), 4),
               "diverge_rate": round(float(roll.diverged.mean()), 4),
               "first_diff_call": round(first_diff, 2),
               "unique_deals": int(len({d.key() for d in deals})),
               "decisions": buf.n,
               "calls_mean": round(float(n_calls.mean()), 2),
               "passout_rate": round(passout, 4),
               "league": bool(league_on),
               "league_member": member.name if member else None,
               "league_size": len(league),
               "member_winrate": round(member.winrate, 3) if member else None,
               "beta": round(beta, 4),
               "ent_coef": round(ent, 5),
               "lr": round(sched.get_last_lr()[0], 7),
               "wall": round(time.time() - t0, 1),
               **{k: round(v, 5) for k, v in stats.items()}}
        log(rec)

        if rl.eval_every and (it + 1) % rl.eval_every == 0:
            m_rand, se_rand = evaluate_match(model, uniform, eval_deals,
                                             eval_dealer, eval_vuln,
                                             device, cache)
            rec = {"event": "eval", "iter": it,
                   "imps_vs_random": round(m_rand, 3),
                   "se_vs_random": round(se_rand, 3)}
            if ref_model is not None:
                m_bc, se_bc = evaluate_match(model, ref_model, eval_deals,
                                             eval_dealer, eval_vuln,
                                             device, cache)
                rec["imps_vs_bc"] = round(m_bc, 3)
                rec["se_vs_bc"] = round(se_bc, 3)
                score = m_bc
            else:
                score = m_rand
            log(rec)
            if score > best_imps:
                best_imps = score
                save(out_dir / "best.pt", it)
            save(out_dir / "last.pt", it)

    save(out_dir / "last.pt", rl.iters - 1)
    cache.close()
    print(f"done. best eval imps/board: {best_imps:.3f}  checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
