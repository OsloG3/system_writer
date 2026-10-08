r"""Two-loop training: PPO inside, evolution over bidding systems outside.

The bidder here is `system/model.py`'s FFN: a fixed-length state (own hand,
vulnerability, dealer, last bid level/strain/by whom, double-redouble status and
by whom, consecutive passes, legal-action mask, and the frozen hand-inference
encoder's z for all four seats) -> SwiGLU MLP -> 39 call logits + a value. No
call sequence is read; the auction history arrives pre-compressed as
z_self/z_RHO/z_partner/z_LHO from `hand/model.py`.

    inner loop (per PPO iteration)
        sample system i from the population        (encoder_i + adapter_i)
        N-S play system i, opponents come from the league
        roll out two-table team matches            (rl/rollout.py, DD rewards)
        PPO-update the shared policy [and adapter] (rl/ppo.py)

    outer loop (per generation)
        evaluate every system on fixed deals vs a fixed opponent
        mutate/replace the weak ones:  child = parent + sigma * N(0,1)
        snapshot the strong ones into the league, so later generations
        play against systems that already worked

The KL anchor to a frozen BC/RL checkpoint (`--bc-ckpt`) doubles as the warm
start: the FFN begins life with no bidding knowledge, and a strong `kl_beta`
first pulls it toward the reference policy's calls before the IMP reward takes
over (anneal it down with rl.kl_beta_end).

  python -m bidding_dt.system.train --preset small --config configs/system_small.yaml \
      --encoder runs/hand_small/best.pt --bc-ckpt runs/small/best.pt \
      --league bc:runs/small/best.pt --out runs/sys_small
  # CPU smoke
  python -m bidding_dt.system.train --preset tiny --config configs/system_smoke.yaml \
      --encoder runs/hand_smoke/best.pt --bc-ckpt runs/smoke/best.pt --out runs/sys_smoke
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..dd.reward import warm_cache
from ..dd.solver import TableCache
from ..env.deals import random_deals
from ..hand.model import load_hand_model
from ..rl.league import League, parse_spec
from ..rl.model import build_from_ckpt
from ..rl.ppo import ppo_update
from ..rl.rollout import play_team_deals
from ..train import lr_lambda_fn, pick_device
from .config import SystemConfig
from .model import SystemPolicy
from .population import Population, SystemVariant


def league_member_for(league, spec: str, weight: float):
    """The league member for `spec`, added only if it is not there yet.

    `League.add_spec` re-adding an existing member resets its weight, which
    would silently re-weight a configured training opponent -- so look first.
    """
    kind, path = parse_spec(spec)
    for m in league.members:
        if (kind == "random" and m.kind == "random") or (path and m.path == str(path)):
            return m
    return league.add_spec(spec, weight=weight)


def anneal(start: float, end: float, it: int, total: int) -> float:
    t = min(1.0, it / max(1, total - 1))
    return start + (end - start) * t


def encoder_name(path: str, taken: set) -> str:
    name = Path(path).parent.name or Path(path).stem
    i = 1
    while name in taken:
        i += 1
        name = f"{Path(path).parent.name}{i}"
    taken.add(name)
    return name


def load_encoders(paths, device, taken=None) -> dict:
    """Checkpoint paths -> {name: frozen HandVQ} (z must be shape-compatible)."""
    taken = taken if taken is not None else set()
    out = {}
    for p in paths:
        name = encoder_name(str(p), taken)
        out[name] = load_hand_model(p, device)
        print(f"encoder {name}: k={out[name].cfg.n_codes} x "
              f"{out[name].cfg.codebook_size} codes, d={out[name].cfg.d_model} "
              f"({out[name].num_params()/1e6:.2f}M frozen params)")
    return out


@torch.no_grad()
def evaluate_systems(policy, population, deals, dealer, vuln, device, cache,
                     opponent, amp: bool = False) -> dict:
    """Score every system on the same fixed deals against the same opponent."""
    scores = {}
    for v in population.variants:
        population.activate(v)
        roll = play_team_deals(policy, deals, dealer, vuln, device, cache=cache,
                               opponent=opponent, greedy=True, record=False,
                               amp=amp)
        population.score(v, roll.team_imps)
        scores[v.name] = round(v.score, 3)
    return scores


def snapshot_systems(league, population, gen: int, top: int, added: list,
                     max_members: int, weight: float):
    """Clone the strongest systems into the league as frozen opponents."""
    for v in population.best(top):
        name = f"system:{v.name}@g{gen}"
        if league.get(name) is None:
            league.add_model(name, population.clone_system(v), kind="system",
                             weight=weight)
            added.append(name)
    while len(added) > max_members:
        old = added.pop(0)
        league.members = [m for m in league.members if m.name != old]
    league.save_meta()
    return [m.name for m in league.members if m.kind == "system"]


def save_ckpt(path, policy, population, opt, sched, gen, it, best_imps, cfg,
              encoders: dict):
    torch.save({"model": policy.state_dict(),
                "model_cfg": policy.cfg.__dict__,
                "z_in": policy.z_in,
                "active": policy.active,
                "enc_name": policy.enc_name,
                "n_systems": len(population),
                "encoders": {n: e.cfg.__dict__ for n, e in encoders.items()},
                "encoder_paths": cfg.evo.encoders,
                "optim": opt.state_dict(), "sched": sched.state_dict(),
                "gen": gen, "iter": it, "best_imps": best_imps,
                "variants": [v.to_json() for v in population.variants],
                "cfg": cfg.to_dict()}, path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--preset", default=None, choices=["tiny", "small", "base", "large"])
    ap.add_argument("--encoder", action="append", default=[], metavar="CKPT",
                    help="frozen hand-inference encoder (hand/train.py checkpoint), "
                         "repeatable; merged with evo.encoders")
    ap.add_argument("--bc-ckpt", default=None,
                    help="frozen BC/RL checkpoint: KL anchor + league member")
    ap.add_argument("--league", action="append", default=[], metavar="SPEC",
                    help="league opponent, repeatable: random | bc:<ckpt> | "
                         "rl:<ckpt> | <ckpt>")
    ap.add_argument("--out", default=None)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None,
                    help="inner PPO iterations per generation (rl.iters)")
    ap.add_argument("--systems", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--inner-target", default=None,
                    choices=["policy", "adapter", "both"])
    ap.add_argument("--deals-per-iter", type=int, default=None)
    ap.add_argument("--eval-deals", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-opponent", default=None)
    ap.add_argument("--no-league", action="store_true")
    ap.add_argument("--cache", default=None, help="DD table sqlite path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--cpu-bf16", action="store_true", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint path or 'last'")
    args = ap.parse_args()

    cfg = SystemConfig.load(args.config, args.preset)
    cfg.evo.encoders = list(cfg.evo.encoders) + list(args.encoder)
    rl, evo = cfg.rl, cfg.evo
    for value, obj, key in ((args.generations, evo, "generations"),
                            (args.iters, rl, "iters"),
                            (args.systems, evo, "systems"),
                            (args.sigma, evo, "sigma"),
                            (args.inner_target, evo, "inner_target"),
                            (args.deals_per_iter, rl, "deals_per_iter"),
                            (args.eval_deals, evo, "eval_deals"),
                            (args.eval_every, evo, "eval_every"),
                            (args.eval_opponent, evo, "eval_opponent"),
                            (args.cache, rl, "cache_path"),
                            (args.threads, rl, "threads"),
                            (args.cpu_bf16, rl, "cpu_bf16")):
        if value is not None:
            setattr(obj, key, value)
    if args.no_league:
        rl.league_prob = 0.0
    if not evo.encoders:
        ap.error("need at least one --encoder <hand checkpoint>")

    out_dir = Path(args.out) if args.out else \
        Path("runs") / ("sys-" + time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "log.jsonl", "a")

    def log(rec: dict):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    device = pick_device(args.device or rl.device)
    use_amp = device.type == "cuda" or (device.type == "cpu" and rl.cpu_bf16)
    rollout_amp = device.type == "cpu" and rl.cpu_bf16
    if device.type == "cpu" and rl.threads > 0:
        torch.set_num_threads(rl.threads)
    torch.manual_seed(evo.seed)
    rng = np.random.default_rng(evo.seed)

    # -- frozen encoders + policy + population --------------------------------
    encoders = load_encoders(evo.encoders, device)
    enc = next(iter(encoders.values()))
    policy = SystemPolicy(cfg.model, z_in=enc.cfg.n_codes * enc.cfg.d_model,
                          n_systems=evo.systems).to(device)
    for name, e in encoders.items():
        policy.add_encoder(name, e)
    population = Population(policy, evo, list(encoders))
    print(f"policy: {policy.num_params()/1e6:.3f}M trainable params "
          f"(state_dim {policy.state_dim}, {evo.systems} systems, "
          f"inner target {evo.inner_target})")

    # -- reference policy (KL anchor) and league -------------------------------
    ref_model = None
    if args.bc_ckpt:
        ck = torch.load(args.bc_ckpt, map_location="cpu", weights_only=False)
        ref_model, kind = build_from_ckpt(ck, device=device)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        print(f"KL anchor + league member from {args.bc_ckpt} ({kind})")
    league = League(device, pfsp_alpha=rl.pfsp_alpha,
                    persist_dir=out_dir / "league" if rl.league_persist else None)
    specs = list(rl.league_members) + list(args.league)
    if args.bc_ckpt and rl.league_include_init:
        specs.append(args.bc_ckpt)
    for spec in specs:
        try:
            m = league.add_spec(spec, weight=rl.member_weight)
            print(f"league member: {m.name} (kind={m.kind})")
        except (OSError, KeyError, RuntimeError) as ex:
            print(f"warning: league member {spec!r} not added: {ex}")
    # weight 0 when newly added: the evaluation reference is never drawn as a
    # training opponent, and an existing member keeps its configured weight
    eval_member = league_member_for(league, evo.eval_opponent, weight=0.0)
    league.save_meta()

    train_adapters = evo.inner_target in ("adapter", "both")
    opt = torch.optim.AdamW(policy.trainable_params(adapter=train_adapters),
                            lr=rl.lr, weight_decay=rl.weight_decay)
    total_iters = max(1, evo.generations * rl.iters)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_fn(rl.warmup_iters, total_iters, rl.min_lr_frac))

    start_gen = start_it = 0
    best_imps = -99.0
    if args.resume:
        path = out_dir / "last.pt" if args.resume == "last" else Path(args.resume)
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck["model_cfg"] != policy.cfg.__dict__ or ck["z_in"] != policy.z_in \
                or ck.get("n_systems", len(population)) != len(population):
            raise SystemExit(
                f"{path} was trained with a different architecture "
                f"(model_cfg/z_in/systems); pass the matching --config/--preset")
        policy.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        start_gen, start_it = ck["gen"], ck["iter"]
        best_imps = ck.get("best_imps", -99.0)
        policy.active = int(ck.get("active", 0))
        population.variants = [SystemVariant.from_json(v) for v in ck["variants"]]
        missing = {v.encoder for v in population.variants} - set(encoders)
        if missing:
            raise SystemExit(f"checkpoint systems need encoders {sorted(missing)}; "
                             f"pass them with --encoder")
        if ck.get("enc_name") in encoders:
            policy.set_encoder(ck["enc_name"])
        print(f"resumed from {path} at generation {start_gen}")

    cache = TableCache(rl.cache_path)
    eval_rng = np.random.default_rng(evo.seed + 1)
    eval_deals = random_deals(eval_rng, evo.eval_deals)
    eval_dealer = eval_rng.integers(0, 4, len(eval_deals))
    eval_vuln = eval_rng.integers(0, 4, len(eval_deals))
    warm_cache([d.hands for d in eval_deals], cache)     # fixed eval boards
    added_systems: list = []

    log({"event": "init", "params": policy.num_params(), "device": str(device),
         "state_dim": policy.state_dim, "encoders": list(encoders),
         "systems": population.names(), "league": league.names(),
         "cfg": cfg.to_dict()})

    dd_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dd")

    def draw_deals():
        d = random_deals(rng, rl.deals_per_iter)
        return d, rng.integers(0, 4, len(d)), rng.integers(0, 4, len(d))

    deals, dealer, vuln = draw_deals()
    presolve = dd_pool.submit(warm_cache, [d.hands for d in deals], cache)
    it = start_it
    for gen in range(start_gen, evo.generations):
        t0 = time.time()
        run = {"imps": [], "reward": [], "decisions": 0, "active": {},
               "stats": {}}
        for _ in range(rl.iters):
            variant = population.sample(rng)
            population.activate(variant)
            run["active"][variant.name] = run["active"].get(variant.name, 0) + 1

            member = None
            opp = None
            if len(league) and rng.random() < rl.league_prob:
                member = league.sample(rng)
                opp = league.opponent(member)
            beta = anneal(rl.kl_beta, rl.kl_beta_end, it, total_iters)
            ent = anneal(rl.ent_coef, rl.ent_coef_end, it, total_iters)

            roll = play_team_deals(policy, deals, dealer, vuln, device, rng,
                                   cache=cache, opponent=opp, greedy=rl.greedy,
                                   temp=rl.rollout_temp, amp=rollout_amp,
                                   presolve=presolve)
            if member is not None:
                league.update_result(member, roll.team_imps)
            population.observe(variant, roll.team_imps)

            next_deals = draw_deals()
            presolve = dd_pool.submit(warm_cache, [d.hands for d in next_deals[0]],
                                      cache)

            row_r = roll.row_rewards(rl.reward_scale, rl.team_weight,
                                     rl.par_weight)
            roll.buf.finish(row_r, rl.gamma, rl.lam)
            stats = ppo_update(policy, ref_model, roll.buf.to_torch(device), opt,
                               device, rl, beta, ent, rng, use_amp=use_amp,
                               temp=rl.rollout_temp)
            sched.step()
            it += 1
            run["imps"].append(float(roll.team_imps.mean()))
            run["reward"].append(float(row_r.mean()))
            run["decisions"] += roll.buf.n
            for k, v in stats.items():
                run["stats"][k] = run["stats"].get(k, 0.0) + v
            run["beta"], run["ent"] = beta, ent
            deals, dealer, vuln = next_deals

        rec = {"event": "generation", "gen": gen, "iters": it,
               "team_imps": round(float(np.mean(run["imps"])), 4),
               "reward_mean": round(float(np.mean(run["reward"])), 4),
               "decisions": run["decisions"],
               "active": run["active"],
               "lr": round(sched.get_last_lr()[0], 7),
               "z_cache": len(policy.cache),
               "beta": round(run.get("beta", 0.0), 4),
               "ent_coef": round(run.get("ent", 0.0), 5),
               "wall": round(time.time() - t0, 1),
               **{k: round(v / max(rl.iters, 1), 5)
                  for k, v in run["stats"].items()}}
        log(rec)

        if (gen + 1) % evo.eval_every == 0:
            t1 = time.time()
            scores = evaluate_systems(policy, population, eval_deals,
                                      eval_dealer, eval_vuln, device, cache,
                                      eval_member.model, amp=rollout_amp)
            top = population.best(1)
            best = top[0] if top else None
            rec = {"event": "eval", "gen": gen, "scores": scores,
                   "best": best.name if best else None,
                   "best_imps": round(best.score, 3) if best else None,
                   "train_imps": {v.name: round(v.train_imps, 3)
                                  for v in population.variants},
                   "wall": round(time.time() - t1, 1)}
            log(rec)
            if best is not None and best.score > best_imps:
                best_imps = best.score
                population.activate(best)
                save_ckpt(out_dir / "best.pt", policy, population, opt, sched,
                          gen, it, best_imps, cfg, encoders)
            if (gen + 1) % evo.snapshot_every == 0:
                rec = {"event": "snapshot", "gen": gen,
                       "systems": snapshot_systems(league, population, gen,
                                                   evo.snapshot_top,
                                                   added_systems,
                                                   rl.max_snapshots,
                                                   rl.snapshot_weight)}
                log(rec)
            bred = population.evolve()
            log({"event": "evolve", "gen": gen, "bred": bred,
                 "ranking": [(v.name, v.encoder, round(v.score, 3) if v.evals else None)
                             for v in population.ranked()]})
            if evo.persist:
                population.save(out_dir / "population")
            save_ckpt(out_dir / "last.pt", policy, population, opt, sched,
                      gen + 1, it, best_imps, cfg, encoders)

    save_ckpt(out_dir / "last.pt", policy, population, opt, sched,
              evo.generations, it, best_imps, cfg, encoders)
    dd_pool.shutdown(wait=True)
    cache.close()
    print(f"done. best system IMPs/board: {best_imps:.3f}  checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
