"""Standalone RL evaluation: team IMPs/board versus fixed opponents.

    python -m bidding_dt.rl.eval_rl --ckpt runs/rl_small/best.pt --opp random
    python -m bidding_dt.rl.eval_rl --ckpt runs/rl_small/best.pt \
        --opp bc:runs/small/best.pt --deals 4000

Every deal is played at TWO tables (a duplicate team match): the learner
team sits N-S at table 0 and E-W at table 1, the opponent holds the other
seats at both. The per-board score is the double-dummy IMP swing between
the tables,

    imps = IMPs( score_NS(table0) - score_NS(table1) ),

so card-lie luck cancels within each board -- no layout averaging needed
(the former --layout-avg option is gone; duplicate team scoring subsumes
it). Positive means the learner team out-bid the opponent on the same cards.

Opponents: `random` (uniform legal), `self` (both tables are self-play;
the swing is ~0 by symmetry, useful only as a sanity check), `bc:<ckpt>`,
`rl:<ckpt>` or a bare `<ckpt>` path (BC vs RL auto-detected). Both sides
play greedy unless --sample is given.
"""

import argparse
import json
import math

import numpy as np

from ..data.vocab import CALL_PASS
from ..dd.solver import TableCache
from ..env.deals import random_deals
from ..train import pick_device
from .model import load_any_ckpt, load_rl_ckpt
from .rollout import UniformPolicy, play_team_deals


def load_opponent(spec: str, device):
    if spec == "random":
        return UniformPolicy()
    if spec == "self":
        return None
    kind, sep, path = spec.partition(":")
    if not sep or kind not in ("bc", "rl", "auto"):
        path = spec            # bare checkpoint path; kind is auto-detected
    opp = load_any_ckpt(path, device=device)
    for p in opp.parameters():
        p.requires_grad_(False)
    return opp


def run_match(model, opp, deals, dealer, vuln, device, cache,
              greedy: bool, chunk: int = 256):
    """Two-table team match; returns (team IMPs/board from the learner-team
    perspective, passout count, total calls over both tables)."""
    out = np.zeros(len(deals))
    passout = calls_tot = 0
    for s in range(0, len(deals), chunk):
        e = min(s + chunk, len(deals))
        roll = play_team_deals(model, deals[s:e], dealer[s:e], vuln[s:e],
                               device, cache=cache, opponent=opp,
                               greedy=greedy, record=False)
        out[s:e] = roll.team_imps
        env = roll.env
        for i in range(env.b):
            n = int(env.n_calls[i])
            calls_tot += n
            if n == 4 and (env.calls[i, :4] == CALL_PASS).all():
                passout += 1
    return out, passout, calls_tot


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--opp", default="random",
                    help="random | self | bc:<path> | rl:<path> | <path>")
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", default="cache/dd_eval.sqlite")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, _ = load_rl_ckpt(args.ckpt, device=device)
    model.eval()
    opp = load_opponent(args.opp, device) if args.opp != "self" else None
    cache = TableCache(args.cache)
    rng = np.random.default_rng(args.seed)
    greedy = not args.sample

    deals = random_deals(rng, args.deals)
    dealer = rng.integers(0, 4, args.deals)
    vuln = rng.integers(0, 4, args.deals)

    signed, passout, calls_tot = run_match(model, opp, deals, dealer, vuln,
                                           device, cache, greedy)
    mean = float(signed.mean())
    se = float(signed.std(ddof=1) / math.sqrt(len(signed)))

    result = {
        "ckpt": args.ckpt, "opp": args.opp, "deals": args.deals,
        "greedy": greedy,
        "imps_per_board": round(mean, 4),
        "se": round(se, 4),
        "ci95": [round(mean - 1.96 * se, 4), round(mean + 1.96 * se, 4)],
        "passout_rate": round(passout / (2 * args.deals), 4),
        "calls_mean": round(calls_tot / (2 * args.deals), 2),
        "dd_cache": {"hits": cache.hits, "misses": cache.misses},
    }
    print(json.dumps(result, indent=2) if args.json else
          "\n".join(f"{k2:>14}: {v}" for k2, v in result.items()))
    cache.close()


if __name__ == "__main__":
    main()
