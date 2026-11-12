"""A/B harness for the card-play bot: whole hands, measured against par.

Plays N randomly dealt hands end to end with the production bot (a declarer-side
session plus one session per defender, exactly like the site drives them) and
scores each hand against its double-dummy table:

    declarer error = dd_tricks - achieved      (lower = better declarer play)
    defence error  = achieved - dd_tricks      (higher/closer to 0 = better defence)
    imps           = site metric, achieved score vs double-dummy par

Auctions are bid greedily by a bidding checkpoint (passed-out deals are
replaced), the deal's DD table comes from a persistent sqlite cache so repeat
runs pay it only once, and every PlayConfig knob can be overridden per side
from the command line, which is what makes an A/B an A/B: one arm moves only
the declarer-side flags, the other only the defender flags, on identical deals.

    # baseline arm
    python -m bidding_dt.play_ab --deals 16 --seed 1 --tag base --out /tmp/base.json
    # improved defence (declarer side stays baseline)
    python -m bidding_dt.play_ab --deals 16 --seed 1 --tag def \
        --def-set rank_objective=both,tie_prefer_spots=1 --out /tmp/def.json
    python -m bidding_dt.play_ab --compare /tmp/base.json /tmp/def.json

Per-deal rows (hands, auction, contract, tricks, timings) are kept in the JSON
so two arms can be diffed deal by deal afterwards.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .bid import auction_over, load_model, predict, random_deal, seat_to_act
from .data.hands import encode_hand, str_to_cards
from .data.vocab import call_to_id, id_to_call
from .dd import fastsolve
from .dd.reward import contract_from_auction
from .dd.scoring import contract_score, declarer_is_vul, points_to_imps
from .dd.solver import TableCache, dd_table, par_score
from .play_bot import PlayConfig, PlaySession, card_to_str, suit_of


def _cfg_value(old, raw: str):
    """'0.5' -> float when old is float, bool for 0/1/true/false, etc."""
    if isinstance(old, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(old, int):
        return int(raw)
    if isinstance(old, float):
        return float(raw)
    return raw


def parse_overrides(spec: str, base: PlayConfig) -> dict:
    out = {}
    for part in filter(None, (spec or "").split(",")):
        k, sep, v = part.partition("=")
        k = k.strip()
        if not sep or not hasattr(base, k):
            raise SystemExit(f"bad --set entry {part!r} (unknown field {k!r})")
        out[k] = _cfg_value(getattr(base, k), v.strip())
    return out


def bid_auction(model, dev, hands, dealer, vuln, rng, max_calls=36):
    """Greedy model auction; returns call-id list (may be a pass-out)."""
    hands4 = np.stack([encode_hand(h) for h in hands])
    calls: list[int] = []
    while not auction_over(calls) and len(calls) < max_calls:
        seat = seat_to_act(dealer, calls)
        picks = predict(model, dev, hands4, seat, dealer, vuln, calls, topk=1)
        calls.append(call_to_id(picks[0][0]))
    return calls


def play_hand(hands, dealer, vuln, calls, contract, dd_tricks,
              decl_cfg, def_cfg, hand_model, seed):
    """Play one deal to the end; returns (row dict)."""
    declarer = contract.declarer
    dummy = (declarer + 2) % 4
    def_seats = [s for s in range(4) if s not in (declarer, dummy)]
    ds = PlaySession("ab:d", dealer, vuln, calls, declarer, contract.denom,
                     {declarer: hands[declarer], dummy: hands[dummy]},
                     model=hand_model, cfg=decl_cfg,
                     rng=np.random.default_rng(seed * 100 + 1))
    xs = {s: PlaySession(f"ab:x{s}", dealer, vuln, calls, declarer,
                         contract.denom, {s: hands[s], dummy: hands[dummy]},
                         model=hand_model, cfg=def_cfg,
                         rng=np.random.default_rng(seed * 100 + 2 + s))
          for s in def_seats}
    rem = [set(str_to_cards(h).tolist()) for h in hands]
    plays: list[tuple[int, int]] = []
    to_act = (declarer + 1) % 4
    tricks_decl = 0
    t0 = time.time()
    decl_t = {"decisions": 0, "exhausted": 0, "nodes": 0, "search": 0.0,
              "pool": 0.0, "process": 0.0, "rank": 0.0, "weights": 0.0}
    def_t = {"decisions": 0, "rank": 0.0, "pool": 0.0, "process": 0.0,
             "weights": 0.0, "search": 0.0}
    for i in range(52):
        is_decl = to_act in (declarer, dummy)
        sess = ds if is_decl else xs[to_act]
        card = sess.choose(plays, to_act)
        tm = sess.last_timing
        acc = decl_t if is_decl else def_t
        acc["decisions"] += 1
        for k in ("search", "pool", "process", "rank", "weights"):
            acc[k] += float(tm.get(k, 0.0))
        if is_decl and sess.last_search:
            decl_t["exhausted"] += bool(sess.last_search.get("exhausted"))
            decl_t["nodes"] += int(sess.last_search.get("nodes", 0))
        assert card in rem[to_act], f"seat {to_act} played a card it lacks"
        if i % 4:
            lead = suit_of(plays[(i // 4) * 4][1])
            assert not any(suit_of(c) == lead for c in rem[to_act]) or \
                suit_of(card) == lead, f"seat {to_act} revoked"
        rem[to_act].discard(card)
        plays.append((to_act, card))
        if len(plays) % 4 == 0:
            trick = plays[-4:]
            w = (trick[0][0] + fastsolve.trick_winner(
                [c for _, c in trick], ds.trump_int)) % 4
            tricks_decl += w in (declarer, dummy)
            to_act = w
        else:
            to_act = (to_act + 1) % 4
    wall = time.time() - t0
    score = contract_score(contract.level, contract.denom, contract.penalty,
                           tricks_decl, declarer_is_vul(declarer, vuln))
    score_ns = score if declarer in (0, 2) else -score
    return {"tricks": tricks_decl, "dd_tricks": dd_tricks,
            "score_ns": score_ns, "wall": wall,
            "decl": decl_t, "def": def_t,
            "play": [[s, card_to_str(c)] for s, c in plays]}


def run(args):
    base = PlayConfig()
    decl_over = parse_overrides(args.decl_set, base)
    def_over = parse_overrides(args.def_set, base)
    decl_cfg = PlayConfig(**{**base.__dict__, **decl_over})
    def_cfg = PlayConfig(**{**base.__dict__, **def_over})

    bid_model, dev = load_model(args.bid_ckpt) if args.bid_ckpt else (None, None)
    hand_model = None
    if args.hand_ckpt:
        from .hand.model import load_hand_model
        hand_model = load_hand_model(args.hand_ckpt, "cpu")

    cache = TableCache(args.dd_cache)
    rng = np.random.default_rng(args.seed)
    deals = []
    guard = 0
    while len(deals) < args.deals and guard < args.deals * 20:
        guard += 1
        hands = random_deal(rng)
        dealer = int(rng.integers(4))
        vuln = int(rng.integers(4))
        if bid_model is None:
            calls = [call_to_id(t) for t in "1S P 4S P P P".split()]
        else:
            calls = bid_auction(bid_model, dev, hands, dealer, vuln, rng)
        contract = contract_from_auction(dealer, calls)
        if contract is None:
            continue                      # passed out: nothing to play
        deals.append((hands, dealer, vuln, calls, contract))
    if len(deals) < args.deals:
        print(f"warning: only found {len(deals)}/{args.deals} biddable deals")

    rows = []
    for di, (hands, dealer, vuln, calls, contract) in enumerate(deals):
        t0 = time.time()
        table = cache.get(" ".join(hands).encode())
        if table is None:
            table = dd_table(hands)
            cache.put(" ".join(hands).encode(), table)
        dd_tricks = int(table[contract.declarer, contract.denom])
        par_ns = par_score(table, vuln, dealer)
        row = play_hand(hands, dealer, vuln, calls, contract, dd_tricks,
                        decl_cfg, def_cfg, hand_model, args.seed * 1000 + di)
        row.update({"deal": di, "hands": hands, "dealer": dealer, "vuln": vuln,
                    "calls": [id_to_call(c) for c in calls],
                    "contract": str(contract), "par_ns": int(par_ns),
                    "imps": float(points_to_imps(row["score_ns"] - int(par_ns))),
                    "decl_err": dd_tricks - row["tricks"],
                    "setup_s": time.time() - t0})
        rows.append(row)
        print(f"deal {di:2d} {contract} dd={dd_tricks} got={row['tricks']:2d} "
              f"err={row['decl_err']:+d} imps={row['imps']:+6.1f} "
              f"wall={row['wall']:5.1f}s exh={row['decl']['exhausted']}"
              f"/{row['decl']['decisions']}", flush=True)

    n = len(rows)
    summary = {
        "tag": args.tag, "deals": n, "seed": args.seed,
        "decl_set": decl_over, "def_set": def_over,
        "mean_dd": float(np.mean([r["dd_tricks"] for r in rows])),
        "mean_tricks": float(np.mean([r["tricks"] for r in rows])),
        "mean_decl_err": float(np.mean([r["decl_err"] for r in rows])),
        "mean_def_err": float(np.mean([r["tricks"] - r["dd_tricks"]
                                       for r in rows])),
        "mean_imps_ns": float(np.mean([r["imps"] for r in rows])),
        "reached_dd": int(sum(r["tricks"] >= r["dd_tricks"] for r in rows)),
        "total_wall": float(sum(r["wall"] for r in rows)),
        "mean_wall": float(np.mean([r["wall"] for r in rows])),
        "decl_decisions": int(sum(r["decl"]["decisions"] for r in rows)),
        "decl_exhausted": int(sum(r["decl"]["exhausted"] for r in rows)),
        "decl_nodes": int(sum(r["decl"]["nodes"] for r in rows)),
        "decl_search_s": float(sum(r["decl"]["search"] for r in rows)),
        "decl_pool_s": float(sum(r["decl"]["pool"] for r in rows)),
        "def_pool_s": float(sum(r["def"]["pool"] for r in rows)),
        "def_rank_s": float(sum(r["def"]["rank"] for r in rows)),
        "solves": fastsolve.stats(),
    }
    out = {"summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in summary.items() if k != "solves"},
                     indent=1))


def compare(paths):
    arms = [json.loads(Path(p).read_text()) for p in paths]
    a, b = arms[0], arms[1]
    sa, sb = a["summary"], b["summary"]
    print(f"{'metric':<22}{sa['tag']:>14}{sb['tag']:>14}{'delta':>12}")
    for k in ("mean_decl_err", "mean_def_err", "mean_imps_ns", "reached_dd",
              "mean_tricks", "mean_wall", "total_wall", "decl_exhausted",
              "decl_nodes"):
        va, vb = sa.get(k), sb.get(k)
        if isinstance(va, float):
            print(f"{k:<22}{va:>14.3f}{vb:>14.3f}{vb - va:>+12.3f}")
        else:
            print(f"{k:<22}{va:>14}{vb:>14}{vb - va:>+12}")
    # paired per-deal diffs
    if len(a["rows"]) == len(b["rows"]):
        da = [rb["tricks"] - ra["tricks"]
              for ra, rb in zip(a["rows"], b["rows"])]
        wins = sum(d > 0 for d in da)
        losses = sum(d < 0 for d in da)
        print(f"paired tricks ({sb['tag']} vs {sa['tag']}): "
              f"+{wins} / -{losses} / ={len(da) - wins - losses}  "
              f"sum {sum(da):+d}")
        for ra, rb, d in zip(a["rows"], b["rows"], da):
            if d:
                print(f"  deal {ra['deal']:2d} {ra['contract']:<10} "
                      f"dd={ra['dd_tricks']:2d} {ra['tricks']:2d} -> "
                      f"{rb['tricks']:2d} ({d:+d})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deals", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="arm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bid-ckpt", default="../models/last.pt")
    ap.add_argument("--hand-ckpt", default="../models/hand_cache.pt")
    ap.add_argument("--dd-cache", default="cache/dd_ab.sqlite")
    ap.add_argument("--decl-set", default="",
                    help="PlayConfig overrides for the declarer side, k=v,...")
    ap.add_argument("--def-set", default="",
                    help="PlayConfig overrides for both defenders, k=v,...")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                    default=None)
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
        return
    if not args.out:
        args.out = f"/tmp/play_ab_{args.tag}.json"
    run(args)


if __name__ == "__main__":
    main()
