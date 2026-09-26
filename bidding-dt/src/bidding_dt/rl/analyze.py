"""Extract the emergent bidding system: opening/response frequencies by hand
type, comparing an RL checkpoint against its BC anchor and the human corpus.

    python -m bidding_dt.rl.analyze --ckpt runs/rl_small/best.pt \
        --bc runs/small/best.pt --hands 20000 --cache-dir cache

For every sampled hand the model is asked (greedy) for:
  - the opening call as dealer (empty auction, vuln None)
  - the response as opener's partner after each 1C/1D/1H/1S/1N opening
Hands are bucketed by HCP (computed from the card-indicator encoding).
The human column comes from the parsed cache (first call of each auction).
Per bucket we report each model's call distribution and the total-variation
distance RL-vs-BC and RL-vs-human: large TV = a genuinely different system,
i.e. new agreements rather than human imitation.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..data.hands import HAND_SCALE
from ..data.legal import legal_mask
from ..data.vocab import (CALL_PASS, DENOMS, call_to_id, id_to_call)
from ..env.deals import random_deals
from ..train import pick_device
from .model import load_any_ckpt

HCP_BUCKETS = [(0, 5), (6, 9), (10, 12), (13, 15), (16, 18), (19, 40)]
OPENINGS = ["P"] + [f"1{d}" for d in DENOMS]
RESPONSE_CTX = {f"1{d}": call_to_id(f"1{d}") for d in DENOMS}


def hcp_bucket(encoded_hand: np.ndarray) -> int:
    """HCP from the (52,) card-indicator encoding: A=4 K=3 Q=2 J=1."""
    v = encoded_hand.astype(np.int64)
    hcp = sum(w * v[s * 13 + r] for s in range(4)
              for r, w in enumerate((4, 3, 2, 1)))
    for i, (lo, hi) in enumerate(HCP_BUCKETS):
        if lo <= hcp <= hi:
            return i
    return len(HCP_BUCKETS) - 1


def bucket_name(i: int) -> str:
    lo, hi = HCP_BUCKETS[i]
    return f"{lo}-{hi}" if hi < 40 else f"{lo}+"


@torch.no_grad()
def predict_batch(model, tokens: np.ndarray, hand: np.ndarray,
                  vuln_cls: np.ndarray, row_len: np.ndarray,
                  device, mask: np.ndarray | None = None) -> np.ndarray:
    logits, _ = model.forward_last(
        torch.as_tensor(tokens, dtype=torch.long, device=device),
        torch.as_tensor(hand, dtype=torch.float32, device=device),
        torch.as_tensor(vuln_cls, dtype=torch.long, device=device),
        torch.as_tensor(row_len, dtype=torch.long, device=device))
    if mask is not None:
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0).expand_as(logits)
        logits = logits.float().masked_fill(~mask_t, float("-inf"))
    return logits.argmax(-1).cpu().numpy()


def system_profile(model, encoded_hands: np.ndarray, device) -> dict:
    """Greedy call distribution per HCP bucket: openings + 1X responses."""
    n = len(encoded_hands)
    buckets = np.array([hcp_bucket(h) for h in encoded_hands])
    prof: dict = {"openings": {}, "responses": {}}
    hand = encoded_hands.astype(np.float32) * HAND_SCALE

    # openings: hero = dealer, empty auction, vuln None -> [COND, HAND]
    tokens = np.zeros((n, 2), dtype=np.int64)
    vuln_cls = np.zeros(n, dtype=np.int64)  # vuln_class(0, 0) = 0
    row_len = np.full(n, 2)
    open_mask = legal_mask(np.zeros(0, dtype=np.int64), role_offset=0)
    acts = predict_batch(model, tokens, hand, vuln_cls, row_len, device,
                         mask=open_mask)
    for b in range(len(HCP_BUCKETS)):
        m = buckets == b
        if m.sum() == 0:
            continue
        prof["openings"][bucket_name(b)] = dist_of(acts[m])

    # responses: dealer N opens 1X, RHO passes, hero = S (partner).
    # Hero frame: o=2 -> p=2 pads, row = [COND, HAND, PAD, PAD, 1X, P], len 6
    resp_tokens = np.zeros((n, 6), dtype=np.int64)
    resp_tokens[:, 5] = CALL_PASS
    resp_row_len = np.full(n, 6)
    for name, op_id in RESPONSE_CTX.items():
        resp_tokens[:, 4] = op_id
        mask = legal_mask(np.array([op_id, CALL_PASS]), role_offset=2)
        acts = predict_batch(model, resp_tokens, hand, vuln_cls, resp_row_len,
                             device, mask=mask)
        prof["responses"][f"after_{name}"] = {}
        for b in range(len(HCP_BUCKETS)):
            m = buckets == b
            if m.sum() == 0:
                continue
            prof["responses"][f"after_{name}"][bucket_name(b)] = dist_of(acts[m])
    return prof


def dist_of(actions: np.ndarray) -> dict:
    c = Counter(int(a) for a in actions)
    tot = sum(c.values())
    out = {}
    for a, k in c.most_common():
        name = id_to_call(a)
        if name == "PAD":
            continue
        out[name] = round(k / tot, 4)
    return out


def human_openings(cache_dir: Path, max_deals=None):
    """Opening-call distribution per HCP bucket from the parsed corpus."""
    hands = np.load(cache_dir / "hands.npy", mmap_mode="r")
    calls = np.load(cache_dir / "calls.npy", mmap_mode="r")
    lengths = np.load(cache_dir / "lengths.npy", mmap_mode="r")
    dealer = np.load(cache_dir / "dealer.npy", mmap_mode="r")
    vuln = np.load(cache_dir / "vuln.npy", mmap_mode="r")
    idx = np.load(cache_dir / "train.npy")
    if max_deals is not None:
        idx = idx[:max_deals]
    per_bucket: dict = {bucket_name(b): Counter() for b in range(len(HCP_BUCKETS))}
    for d in idx:
        d = int(d)
        if lengths[d] == 0 or vuln[d] != 0:
            continue  # compare at vuln None only
        seat = int(dealer[d])
        act = int(calls[d, 0])
        b = hcp_bucket(np.asarray(hands[d, seat]))
        per_bucket[bucket_name(b)][act] += 1
    out = {}
    for name, c in per_bucket.items():
        tot = sum(c.values())
        if tot == 0:
            continue
        out[name] = {id_to_call(a): round(k / tot, 4)
                     for a, k in c.most_common()}
    return out


def total_variation(d1: dict, d2: dict) -> float:
    keys = set(d1) | set(d2)
    return 0.5 * sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in keys)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="RL (or BC) checkpoint")
    ap.add_argument("--bc", default=None, help="BC anchor checkpoint to diff against")
    ap.add_argument("--hands", type=int, default=20000)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--human-deals", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="write full profile JSON here")
    args = ap.parse_args()

    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)
    deals = random_deals(rng, max(1, args.hands // 4))
    encoded = np.concatenate([d.encoded for d in deals], axis=0)[: args.hands]

    model = load_any_ckpt(args.ckpt, device)
    prof = system_profile(model, encoded, device)
    report = {"ckpt": args.ckpt, "profile": prof, "tv": {}}

    if args.bc:
        bc = load_any_ckpt(args.bc, device)
        bc_prof = system_profile(bc, encoded, device)
        report["bc"] = args.bc
        report["tv"]["openings_vs_bc"] = {
            b: round(total_variation(prof["openings"][b], bc_prof["openings"][b]), 4)
            for b in prof["openings"]}
    cache_dir = Path(args.cache_dir)
    if (cache_dir / "hands.npy").exists():
        hum = human_openings(cache_dir, max_deals=args.human_deals)
        report["human_openings"] = hum
        report["tv"]["openings_vs_human"] = {
            b: round(total_variation(prof["openings"][b], hum[b]), 4)
            for b in prof["openings"] if b in hum}

    # console summary: openings table + TV distances
    print(f"\n=== opening calls by HCP bucket ({args.ckpt}) ===")
    for b, d in prof["openings"].items():
        top = ", ".join(f"{k}:{v:.2f}" for k, v in list(d.items())[:6])
        print(f" {b:>6}: {top}")
    for group, tvs in report["tv"].items():
        print(f"\n=== TV distance {group} ===")
        for b, v in tvs.items():
            print(f" {b:>6}: {v:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nfull profile written to {args.out}")


if __name__ == "__main__":
    main()
