"""Sample possible hands for one player from the public auction.

One auction:
  python -m bidding_dt.hand.sample --ckpt runs/hand_small/best.pt \
      --dealer N --vuln N-S --target E --auction "P 1D X 2H" \
      --mask "AT7.KT943.A42.K8" --n 8

`--mask` is repeatable and takes a 'S.H.D.C' hand -- yours, and dummy's once it
is tabled. Those cards are removed from the target's possible hands *and* shown
to the encoder as its KNOWN token, so the samples respect both the auction and
everything you can see. With no mask the target is drawn from the full deck.

Every sample is a complete, legal 13-card hand: the decoder draws card by card
and cannot repeat or contradict itself, and the VQ bottleneck is what keeps a
multi-meaning auction (a multi-2D: long hearts *or* long spades) from decoding
into the average of the two. `--codes` samples the codes themselves for extra
spread; `--temp` flattens the card draws.

Against a store's val split (generated or the human cache):
  python -m bidding_dt.hand.sample --ckpt runs/hand_small/best.pt \
      --source cache/gen --show 5
"""

import argparse

import numpy as np
import torch

from ..bid import format_auction, summarize_contract
from ..data.dataset import BiddingDataset
from ..data.hands import (COUNT_TO_CARD, HAND_DIM, cards_to_str, encode_hand,
                          hcp, indicator_from_cards, suit_lengths)
from ..data.vocab import SEATS, SEAT_TO_IDX, VULN_TO_IDX, call_to_id
from ..train import pick_device
from .data import AuctionStore, rotated_tokens
from .model import load_hand_model

VULN_NAMES = {v: k for k, v in VULN_TO_IDX.items()}


def describe(cards) -> str:
    """Card ids -> 'AK7.QT943.A42.K8  3-5-3-2  12 HCP'."""
    ind = indicator_from_cards(cards)
    shape = "-".join(str(int(x)) for x in suit_lengths(ind))
    return f"{cards_to_str(cards):<24} {shape:<7} {hcp(ind):>2} HCP"


@torch.no_grad()
def sample_hands(model, device, dealer: int, calls, target: int, vuln: int,
                 excl=None, n: int = 1, temp: float = 1.0, top_p=None,
                 greedy: bool = False, codes: bool = False,
                 code_temp: float = 1.0):
    """(dealer, calls, target seat, vuln) -> ((n,13) card ids, (n,k) codes)."""
    tokens = torch.from_numpy(rotated_tokens(dealer, calls, target))[None].to(device)
    v = torch.tensor([BiddingDataset.vuln_class(vuln, target)],
                     dtype=torch.long, device=device)
    e = None if excl is None else torch.as_tensor(
        np.asarray(excl, dtype=np.float32).reshape(1, HAND_DIM), device=device)
    cards, code_idx = model.sample(tokens, v, e, None, n=n, temp=temp,
                                   top_p=top_p, greedy=greedy, codes=codes,
                                   code_temp=code_temp)
    return cards[0].cpu().numpy(), code_idx[0].cpu().numpy()


def show_store(model, device, source: str, rows: int, seed: int,
               split: str = "val", **kw):
    """Print sampled vs actual hands for `rows` random `split`-views."""
    store = AuctionStore.load(source, split)
    rng = np.random.default_rng(seed)
    for j in range(rows):
        deal = int(rng.choice(store.deals))
        dealer = int(store.dealer[deal])
        vuln = int(store.vuln[deal])
        calls = [int(c) for c in store.calls[deal, :int(store.lengths[deal])]]
        k = int(rng.integers(0, len(calls) + 1))
        target = int(rng.integers(4))
        seen, code_idx = sample_hands(model, device, dealer, calls[:k], target,
                                      vuln, **kw)
        actual = np.flatnonzero(np.asarray(store.hands[deal, target]))
        print(f"\n--- {j + 1}  dealer {SEATS[dealer]}  vuln {VULN_NAMES[vuln]}")
        print(f"  auction : {format_auction(dealer, calls[:k])}"
              + (f"   [{summarize_contract(dealer, calls)}]" if k == len(calls)
                 and k >= 4 else ""))
        print(f"  target  : {SEATS[target]}")
        truth = np.sort(COUNT_TO_CARD[actual])
        print(f"  actual  : {describe(truth)}")
        for i, hand in enumerate(seen):
            hit = len(set(hand.tolist()) & set(truth.tolist()))
            print(f"  sample {i}: {describe(hand)}   {hit}/13 cards")
        print(f"  codes   : {code_idx[0].tolist()}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dealer", default="N", choices=list(SEATS))
    ap.add_argument("--vuln", default="None", choices=list(VULN_TO_IDX))
    ap.add_argument("--target", default="E", choices=list(SEATS),
                    help="seat whose hand is guessed")
    ap.add_argument("--auction", default="", help='calls in order, e.g. "P 1D X"')
    ap.add_argument("--mask", action="append", default=[], metavar="HAND",
                    help="'S.H.D.C' hand of cards the target cannot hold "
                         "(repeatable: yours, dummy's, ...)")
    ap.add_argument("--n", type=int, default=8, help="hands to draw")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--greedy", action="store_true",
                    help="argmax each draw (one hand, no spread)")
    ap.add_argument("--codes", action="store_true",
                    help="also sample the VQ codes (more spread between hands)")
    ap.add_argument("--code-temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--source", default=None, help="auction store for --show")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--show", type=int, default=0,
                    help="compare against n rows of --source/--split")
    args = ap.parse_args()

    device = pick_device(args.device)
    model = load_hand_model(args.ckpt, device)
    print(f"loaded {args.ckpt} ({model.num_params()/1e6:.2f}M params, "
          f"k={model.cfg.n_codes} x {model.cfg.codebook_size} codes) on {device}")
    torch.manual_seed(args.seed)

    kw = dict(n=args.n, temp=args.temp, top_p=args.top_p, greedy=args.greedy,
              codes=args.codes, code_temp=args.code_temp)
    if args.show:
        if not args.source:
            ap.error("--show needs --source <store dir>")
        show_store(model, device, args.source, args.show, args.seed,
                   split=args.split, **kw)
        return

    excl = np.zeros(HAND_DIM, dtype=np.uint8)
    for hand in args.mask:
        try:
            excl |= encode_hand(hand)
        except ValueError as ex:
            ap.error(f"--mask {hand!r}: {ex}")
    if excl.sum() > HAND_DIM - 13:
        ap.error(f"--mask excludes {int(excl.sum())} cards; at most 39 allowed")
    calls = [call_to_id(c.upper()) for c in args.auction.split()]
    dealer = SEAT_TO_IDX[args.dealer]
    target = SEAT_TO_IDX[args.target]
    vuln = VULN_TO_IDX[args.vuln]

    print(f"auction : {format_auction(dealer, calls)}")
    print(f"target  : {SEATS[target]}   vuln {args.vuln}"
          + (f"   masked {int(excl.sum())} cards" if excl.any() else ""))
    seen, code_idx = sample_hands(model, device, dealer, calls, target, vuln,
                                  excl=excl if excl.any() else None, **kw)
    for i, hand in enumerate(seen):
        print(f"  {i:>2}: {describe(hand)}")
    print(f"codes   : {code_idx[0].tolist()}")


if __name__ == "__main__":
    main()
