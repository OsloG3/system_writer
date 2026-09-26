"""Interactive bidding assistant.

One-shot prediction:
  python -m bidding_dt.bid --ckpt runs/small/best.pt \
      --hand "AT7.KT943.A42.K8" --dealer N --vuln N-S --hero N \
      --auction "P 1D"

Spot-check against the training file (model vs actual, per decision):
  python -m bidding_dt.bid --ckpt runs/small/best.pt --check-file training.txt --check-deals 20

REPL (add --repl): commands are calls to append ("1C", "P X 2H"),
`pred` (model pick for seat to act), `auto` (greedy self-play to the end,
needs --deal), `undo`, `show`, `quit`.
"""

import argparse
import random
import sys

import numpy as np
import torch

from .config import ModelConfig
from .data.dataset import BiddingDataset
from .data.hands import HAND_SCALE, encode_deal, encode_hand
from .data.legal import legal_mask
from .data.parse import parse_file
from .data.vocab import (CALL_PAD, CALL_PASS, FIRST_BID, SEAT_TO_IDX, SEATS,
                         VULN_TO_IDX, bid_denom, id_to_call, call_to_id)
from .model.transformer import build_model
from .train import pick_device

VULN_NAMES = {v: k for k, v in VULN_TO_IDX.items()}


def load_model(ckpt_path: str, device=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dev = pick_device(device)
    state = ck["model"]
    if any(k.startswith("dt.") for k in state):  # RL checkpoint: trunk only
        state = {k[3:]: v for k, v in state.items() if k.startswith("dt.")}
    model = build_model(ModelConfig(**ck["model_cfg"])).to(dev)
    model.load_state_dict(state)
    model.eval()
    return model, dev


def auction_over(calls) -> bool:
    return len(calls) >= 4 and all(c == CALL_PASS for c in calls[-3:])


def seat_to_act(dealer: int, calls) -> int:
    return (dealer + len(calls)) % 4


@torch.no_grad()
def predict(model, dev, hands4: np.ndarray, seat: int, dealer: int, vuln: int,
            calls: list[int], topk: int = 5):
    """Top-k legal (call_str, prob) for `seat` given the auction so far."""
    o = (seat - dealer) % 4
    n = len(calls)
    if n % 4 != o % 4:
        raise ValueError(f"not {SEATS[seat]}'s turn (next to act: {SEATS[seat_to_act(dealer, calls)]})")
    p = (4 - o) % 4
    tokens = np.zeros(2 + p + n, dtype=np.int64)
    tokens[2 + p:] = calls
    hand = (np.asarray(hands4[seat], dtype=np.float32) * HAND_SCALE)[None]
    vuln_cls = np.array([BiddingDataset.vuln_class(vuln, seat)], dtype=np.int64)
    logits = model(torch.from_numpy(tokens)[None].to(dev),
                   torch.from_numpy(hand).to(dev),
                   torch.from_numpy(vuln_cls).to(dev))[0, -1].float()
    mask = torch.from_numpy(legal_mask(np.asarray(calls, dtype=np.int64), role_offset=p))
    logits = logits.masked_fill(~mask.to(dev), float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    k = min(topk, int(mask.sum()))
    top = probs.topk(k)
    return [(id_to_call(int(i)), float(v)) for v, i in zip(top.values, top.indices)]


def format_auction(dealer: int, calls) -> str:
    out = []
    for i, c in enumerate(calls):
        out.append(f"{SEATS[(dealer + i) % 4]}:{id_to_call(int(c))}")
    return " ".join(out) or "(empty)"


def do_auto(model, dev, hands4, dealer, vuln, calls, topk=1):
    """Greedy model self-play until the auction ends."""
    while not auction_over(calls) and len(calls) < 24:
        seat = seat_to_act(dealer, calls)
        picks = predict(model, dev, hands4, seat, dealer, vuln, calls, topk=1)
        call = call_to_id(picks[0][0])
        calls.append(call)
        print(f"  {SEATS[seat]}: {picks[0][0]}  (p={picks[0][1]:.2f})")
    print("  auction:", format_auction(dealer, calls))
    return calls


RANK_CHARS = "AKQJT98765432"
SUIT_CHARS = "SHDC"


def random_deal(rng: np.random.Generator):
    """Shuffle the deck; returns 4 hand strings in N,E,S,W order."""
    deck = rng.permutation(52)
    hands = []
    for h in range(4):
        per_suit = [[] for _ in range(4)]
        for card in deck[h * 13:(h + 1) * 13]:
            per_suit[int(card) % 4].append(RANK_CHARS[int(card) // 4])
        hands.append(".".join("".join(sorted(s, key=RANK_CHARS.index)) for s in per_suit))
    return hands


def summarize_contract(dealer: int, calls) -> str:
    """'4H by S (N-S)' or 'passed out'; declarer = first of the winning side
    to name the final denomination (see dd/reward.py)."""
    from .dd.reward import contract_from_auction
    from .data.vocab import DENOMS
    c = contract_from_auction(dealer, calls)
    if c is None:
        return "passed out"
    side = "N-S" if c.declarer in (0, 2) else "E-W"
    pen = ("", "X", "XX")[c.penalty]
    return f"{c.level}{DENOMS[c.denom]}{pen} by {SEATS[c.declarer]} ({side})"


def demo_random(model, dev, n_deals: int, seed: int = 0, topk: int = 3):
    """Generate random deals and let the model bid all four seats (greedy)."""
    rng = np.random.default_rng(seed)
    for d in range(n_deals):
        hand_strs = random_deal(rng)
        dealer = int(rng.integers(4))
        vuln = int(rng.integers(4))
        hands4 = np.stack([encode_hand(h) for h in hand_strs])
        print(f"\n=== random deal {d + 1} (dealer {SEATS[dealer]}, "
              f"vuln {VULN_NAMES[vuln]}) ===")
        for i, s in enumerate(SEATS):
            tag = " *" if i == dealer else "  "
            print(f" {tag}{s}: {hand_strs[i]}")
        calls: list[int] = []
        while not auction_over(calls) and len(calls) < 24:
            seat = seat_to_act(dealer, calls)
            picks = predict(model, dev, hands4, seat, dealer, vuln, calls, topk=topk)
            calls.append(call_to_id(picks[0][0]))
            alts = ", ".join(f"{c}:{p:.2f}" for c, p in picks[1:])
            print(f"   {SEATS[seat]}: {picks[0][0]:>3}  (p={picks[0][1]:.2f}"
                  + (f";  next: {alts}" if alts else "") + ")")
        print(f"   contract: {summarize_contract(dealer, calls)}")


def repl(model, dev, hands4, dealer, vuln, hero, calls):
    print("REPL: <call(s)> to append | pred | auto | undo | show | quit")
    while True:
        seat = seat_to_act(dealer, calls)
        try:
            line = input(f"[{SEATS[seat]} to act] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        cmd, *rest = line.split()
        low = cmd.lower()
        if low in ("q", "quit", "exit"):
            break
        elif low == "show":
            print(" ", format_auction(dealer, calls), "| over" if auction_over(calls) else "")
        elif low == "undo":
            if calls:
                calls.pop()
        elif low == "pred":
            if auction_over(calls):
                print("  auction is over")
                continue
            if hands4 is None or hands4[seat] is None:
                print(f"  hand unknown for {SEATS[seat]} (need --deal for all four)")
                continue
            try:
                picks = predict(model, dev, hands4, seat, dealer, vuln, calls, topk=5)
                print("  " + "  ".join(f"{c}:{p:.3f}" for c, p in picks))
            except ValueError as e:
                print(" ", e)
        elif low == "auto":
            if hands4 is None or any(h is None for h in hands4):
                print("  auto needs all four hands (--deal)")
                continue
            if auction_over(calls):
                print("  auction is over")
                continue
            do_auto(model, dev, hands4, dealer, vuln, calls)
        else:
            ok = True
            for tok in [cmd.upper()] + [t.upper() for t in rest]:
                try:
                    cid = call_to_id(tok)
                except ValueError:
                    print(f"  unknown call: {tok}")
                    ok = False
                    break
                s = seat_to_act(dealer, calls)
                o = (s - dealer) % 4
                p = (4 - o) % 4
                if not legal_mask(np.asarray(calls, dtype=np.int64), role_offset=p)[cid]:
                    print(f"  illegal for {SEATS[s]}: {tok}")
                    ok = False
                    break
                if auction_over(calls):
                    print("  auction is over")
                    ok = False
                    break
                calls.append(cid)
            if not ok:
                continue
    return calls


def spot_check(model, dev, path, n_deals, seed=0, show=5):
    """Replay real auctions; report model top-1 agreement per decision."""
    deals = []
    for hands, calls, dealer, vuln in parse_file(path):
        deals.append((hands, calls, dealer, vuln))
        if len(deals) >= max(n_deals * 20, 200):
            break
    random.Random(seed).shuffle(deals)
    deals = deals[:n_deals]
    agree = tot = 0
    shown = 0
    for hands, calls, dealer, vuln in deals:
        rows = []
        for i in range(len(calls)):
            if auction_over(calls[:i]):
                break
            seat = (dealer + i) % 4
            picks = predict(model, dev, hands, seat, dealer, vuln, list(calls[:i]), topk=1)
            pred = call_to_id(picks[0][0])
            ok = pred == int(calls[i])
            agree += ok
            tot += 1
            rows.append((seat, calls[i], picks[0], ok))
        if shown < show:
            shown += 1
            print(f"\ndeal (dealer {SEATS[dealer]}, vuln {VULN_NAMES[vuln]}):")
            for h, s in zip(str_hands(hands), "NESW"):
                print(f"  {s}: {h}")
            for seat, actual, (pcall, pprob), ok in rows:
                mark = "=" if ok else "!"
                print(f"  {SEATS[seat]}: actual {id_to_call(int(actual)):>3}  "
                      f"model {pcall:>3} (p={pprob:.2f}) {mark}")
    print(f"\nagreement: {agree}/{tot} = {agree/max(tot,1):.3f} over {len(deals)} deals")


def str_hands(hands_vec):
    """Render encoded hands back to readable 'S.H.D.C' strings (lossless)."""
    out = []
    for h in hands_vec:
        suits = []
        for s in range(4):
            suits.append("".join(RANK_CHARS[r] for r in range(13)
                                 if h[s * 13 + r]))
        out.append(".".join(suits))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hand", help="hero hand, 'S.H.D.C' e.g. AT7.KT943.A42.K8")
    ap.add_argument("--deal", help="all four hands N E S W, space separated")
    ap.add_argument("--dealer", default="N", choices=list(SEATS))
    ap.add_argument("--vuln", default="None", choices=list(VULN_TO_IDX))
    ap.add_argument("--hero", default="N", choices=list(SEATS))
    ap.add_argument("--auction", default="", help='calls in order, e.g. "P 1D X"')
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--repl", action="store_true")
    ap.add_argument("--check-file", default=None, help="training-format file to spot-check")
    ap.add_argument("--check-deals", type=int, default=10)
    ap.add_argument("--random-deals", type=int, default=None,
                    help="generate N random deals and show the model bidding all seats")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, dev = load_model(args.ckpt, args.device)
    print(f"loaded {args.ckpt} ({model.num_params()/1e6:.2f}M params) on {dev}")

    if args.random_deals:
        demo_random(model, dev, args.random_deals, args.seed, topk=args.topk)
        return

    if args.check_file:
        spot_check(model, dev, args.check_file, args.check_deals, args.seed)
        return

    if args.deal:
        hands4 = encode_deal(args.deal)
    elif args.hand:
        hands4 = [None, None, None, None]
        hands4[SEAT_TO_IDX[args.hero]] = encode_hand(args.hand)
    else:
        ap.error("need --hand or --deal (or use --check-file)")

    dealer = SEAT_TO_IDX[args.dealer]
    vuln = VULN_TO_IDX[args.vuln]
    hero = SEAT_TO_IDX[args.hero]
    calls = [call_to_id(c.upper()) for c in args.auction.split()]

    if auction_over(calls):
        print("auction is over:", format_auction(dealer, calls))
        return

    seat = seat_to_act(dealer, calls)
    if hands4[seat] is None:
        print(f"next to act is {SEATS[seat]} but only hero {SEATS[hero]}'s hand is known; "
              f"append their calls first, or use --deal/--repl")
    else:
        picks = predict(model, dev, hands4, seat, dealer, vuln, calls, topk=args.topk)
        print(f"auction: {format_auction(dealer, calls)}")
        print(f"{SEATS[seat]} to act:")
        for c, p in picks:
            print(f"  {c:>3}  {p:.4f}")

    if args.repl:
        repl(model, dev, hands4, dealer, vuln, hero, calls)


if __name__ == "__main__":
    main()
