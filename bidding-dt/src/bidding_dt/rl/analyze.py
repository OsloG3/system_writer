"""Summarize how a model bids by letting it bid against itself.

    python -m bidding_dt.rl.analyze --ckpt runs/rl_small/best.pt \
        --deals 4000 --out runs/rl_small/tree.json.zst

The checkpoint plays every seat (pure self-play over fresh random deals, no
reward solving): all four seats sample at --temp, or bid deterministically
with --greedy. Every auction that happens is merged into one prefix tree
keyed by the call sequence from the dealer. The node reached by call c after
sequence s records, over all deals where the bot made that bid in that
sequence, how often it happened (n) plus the min/max HCP and the min/max
length in each suit of the hand it held. Because one policy sits at all four
seats the ranges pool over seats/dealer/vuln -- the position in the auction
is the tree depth (seat = dealer + depth, mod 4).

The console prints the tree indented, children sorted by frequency (prune
the print with --min-n / --depth; the JSON always holds the full tree):

    {"call": null, "n": 4000, "hcp": null, "suits": null, "children": {
       "P": {"call": "P", "n": 1891, "hcp": [0, 12],
             "suits": {"S": [0, 7], "H": [0, 6], "D": [0, 7], "C": [0, 6]},
             "children": {...}},
       ...}}

(root = the deals played; every other node = one bot bid in one sequence)
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch

from ..data.hands import SUITS
from ..data.vocab import id_to_call
from ..env.auction_env import AuctionBatch
from ..env.deals import random_deals
from ..train import pick_device
from .model import load_any_ckpt
from .rollout import RolloutBuffer, _run

_RANK_W = np.array([4, 3, 2, 1] + [0] * 9)   # A K Q J per suit, rest 0


def hand_hcp(encoded_hand: np.ndarray) -> int:
    """HCP from the (52,) card-indicator encoding: A=4 K=3 Q=2 J=1."""
    return int((encoded_hand.reshape(4, 13).astype(np.int64) * _RANK_W).sum())


def suit_lengths(encoded_hand: np.ndarray) -> np.ndarray:
    """(4,) int cards held per suit, S,H,D,C order (hands.py layout)."""
    return encoded_hand.reshape(4, 13).sum(axis=1).astype(np.int64)


def _new_node(call_id: int | None) -> dict:
    return {"call": call_id, "n": 0, "hcp_min": 99, "hcp_max": -1,
            "suit_min": [99] * 4, "suit_max": [-1] * 4, "children": {}}


def _observe(node: dict, hcp: int, lens: np.ndarray) -> None:
    node["n"] += 1
    node["hcp_min"] = min(node["hcp_min"], hcp)
    node["hcp_max"] = max(node["hcp_max"], hcp)
    for i in range(4):
        node["suit_min"][i] = min(node["suit_min"][i], int(lens[i]))
        node["suit_max"][i] = max(node["suit_max"][i], int(lens[i]))


def build_tree(deals, env: AuctionBatch, buf: RolloutBuffer) -> dict:
    """Merge every recorded decision into the auction prefix tree.

    Each buffer row is one bot decision (deal_idx, seat, call_idx, action):
    walk the calls already made in that auction (env.calls token ids, from
    the dealer) and update the child node of the call the row bid with the
    bidder's hand stats. Prefix nodes always exist because every call of
    every auction is itself a recorded decision.
    """
    root = _new_node(None)
    root["n"] = len(deals)
    for d_i, seat, k, act in zip(buf.deal_idx, buf.seat, buf.call_idx,
                                 buf.action):
        enc = deals[d_i].encoded[seat]
        hcp = hand_hcp(enc)
        lens = suit_lengths(enc)
        node = root
        for c in env.calls[d_i, :k]:
            node = node["children"].setdefault(int(c), _new_node(int(c)))
        child = node["children"].setdefault(int(act), _new_node(int(act)))
        _observe(child, hcp, lens)
    return root


@torch.no_grad()
def selfplay_tree(model, deals, dealer: np.ndarray, vuln: np.ndarray, device,
                  greedy: bool = False, temp: float = 1.0):
    """Play `model` against itself at all four seats; returns
    (tree, env, buf). No rewards are computed, so no DD solving/endplay."""
    env = AuctionBatch(deals, dealer, vuln)
    buf = RolloutBuffer()
    _run(env, model, device, None, None, greedy, buf, temp)
    return build_tree(deals, env, buf), env, buf


def node_to_json(node: dict) -> dict:
    """JSON view: call names, children sorted by frequency; the root (and
    any never-observed node) carries null hand stats."""
    seen = node["hcp_max"] >= 0
    kids = sorted(node["children"].values(), key=lambda c: -c["n"])
    return {"call": None if node["call"] is None else id_to_call(node["call"]),
            "n": node["n"],
            "hcp": [node["hcp_min"], node["hcp_max"]] if seen else None,
            "suits": ({s: [node["suit_min"][i], node["suit_max"][i]]
                       for i, s in enumerate(SUITS)} if seen else None),
            "children": {id_to_call(c["call"]): node_to_json(c) for c in kids}}


def print_tree(root: dict, max_depth: int | None = None,
               min_n: int = 1) -> None:
    def rec(node: dict, depth: int):
        if max_depth is not None and depth >= max_depth:
            return
        for c in sorted(node["children"].values(), key=lambda x: -x["n"]):
            if c["n"] < min_n:
                continue
            stats = (f"HCP {c['hcp_min']:>2}-{c['hcp_max']:<2}  "
                     + "  ".join(f"{s} {c['suit_min'][i]:>2}-{c['suit_max'][i]:<2}"
                                 for i, s in enumerate(SUITS)))
            print(f"{'    ' * depth}  {id_to_call(c['call']):<4} "
                  f"n={c['n']:<6} {stats}")
            rec(c, depth + 1)

    print(f"  (deal)      n={root['n']}")
    rec(root, 0)


def write_report(report: dict, out: str) -> Path:
    """Write the report JSON; .zst/.gz suffixes compress it transparently.

    The full tree is huge (hundreds of MB of JSON); zstd level 19 shrinks it
    ~100x and the Go alert loader (system_writer/alerts.go) decompresses both
    formats on the fly. Plain paths keep the indented, human-readable dump.
    """
    p = Path(out)
    if p.suffix == ".zst":
        import zstandard

        cctx = zstandard.ZstdCompressor(level=19, threads=-1)
        with p.open("wb") as f, cctx.stream_writer(f) as sw, \
                io.TextIOWrapper(sw, "utf-8") as tw:
            json.dump(report, tw, separators=(",", ":"))
    elif p.suffix == ".gz":
        import gzip

        with gzip.open(p, "wt", encoding="utf-8", compresslevel=9) as f:
            json.dump(report, f, separators=(",", ":"))
    else:
        p.write_text(json.dumps(report, indent=2))
    return p


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint to summarize (BC or RL, auto-detected)")
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--temp", type=float, default=1.0,
                    help="sampling temperature for every seat (self-play)")
    ap.add_argument("--greedy", action="store_true",
                    help="deterministic argmax bidding instead of sampling")
    ap.add_argument("--min-n", type=int, default=1,
                    help="print only nodes bid at least this often")
    ap.add_argument("--depth", type=int, default=None,
                    help="max print depth in calls from the dealer (default all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None,
                    help="write the full tree JSON here (.zst/.gz compress)")
    args = ap.parse_args()

    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    deals = random_deals(rng, args.deals)
    dealer = rng.integers(0, 4, args.deals)
    vuln = rng.integers(0, 4, args.deals)
    model = load_any_ckpt(args.ckpt, device)
    root, _, _ = selfplay_tree(model, deals, dealer, vuln, device,
                               greedy=args.greedy, temp=args.temp)

    mode = "greedy" if args.greedy else f"temp={args.temp}"
    print(f"\n=== self-play auction tree: {args.ckpt} "
          f"({args.deals} deals, {mode}, seed {args.seed}) ===")
    print_tree(root, max_depth=args.depth, min_n=args.min_n)
    if args.out:
        report = {"ckpt": args.ckpt, "deals": args.deals, "seed": args.seed,
                  "greedy": bool(args.greedy),
                  "temp": None if args.greedy else args.temp,
                  "tree": node_to_json(root)}
        write_report(report, args.out)
        print(f"\nfull tree written to {args.out}")


if __name__ == "__main__":
    main()
