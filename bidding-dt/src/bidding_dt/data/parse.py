"""Parse training.txt into a compact .npy cache.

File format (2 lines per deal):
  line 1: four hands in N,E,S,W order, each 'S.H.D.C' with chars AKQJT98765432
  line 2: dealer vuln call call ...   (vuln in None|N-S|E-W|Both, ends P P P)

Cache arrays (N = num deals):
  hands.npy   (N, 4, 52) uint8   encoded hands, seat order N,E,S,W
  calls.npy   (N, 36)    uint8   call token ids, PAD-padded (see vocab)
  lengths.npy (N,)       uint8   number of calls per auction
  dealer.npy  (N,)       uint8   seat idx of dealer (N=0,E=1,S=2,W=3)
  vuln.npy    (N,)       uint8   0=None 1=N-S 2=E-W 3=Both
  splits: train.npy val.npy test.npy (deal indices, seeded shuffle)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .hands import HAND_DIM, encode_hand
from .vocab import SEAT_TO_IDX, VULN_TO_IDX, call_to_id

MAX_CALLS = 36  # 9 full rounds; observed max auction length is 23

_CALL_MAP = {
    "P": 1, "X": 2, "XX": 3,
    **{f"{l}{d}": 4 + (l - 1) * 5 + "CDHSN".index(d)
       for l in range(1, 8) for d in "CDHSN"},
}


def parse_file(path: str | Path):
    """Stream the training file, yield (hands(4,52) u8, calls(L) u8, dealer, vuln)."""
    with open(path) as f:
        hands_line = None
        for lineno, line in enumerate(f, 1):
            if lineno % 2 == 1:
                hands_line = line
                continue
            meta = line.split()
            dealer = SEAT_TO_IDX[meta[0]]
            vuln = VULN_TO_IDX[meta[1]]
            raw_calls = meta[2:]
            if len(raw_calls) > MAX_CALLS:
                raise ValueError(f"deal at line {lineno-1}: {len(raw_calls)} calls > {MAX_CALLS}")
            hands = np.stack([encode_hand(h) for h in hands_line.split()])
            calls = np.fromiter(
                (_CALL_MAP[c] for c in raw_calls), dtype=np.uint8, count=len(raw_calls)
            )
            yield hands, calls, dealer, vuln


def build_cache(input_path: str | Path, cache_dir: str | Path,
                split=(0.98, 0.01, 0.01), seed=1337) -> dict:
    input_path = Path(input_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_lines = sum(1 for _ in open(input_path))
    n_deals = n_lines // 2

    hands = np.zeros((n_deals, 4, HAND_DIM), dtype=np.uint8)
    calls = np.zeros((n_deals, MAX_CALLS), dtype=np.uint8)
    lengths = np.zeros(n_deals, dtype=np.uint8)
    dealer = np.zeros(n_deals, dtype=np.uint8)
    vuln = np.zeros(n_deals, dtype=np.uint8)

    for i, (h, c, d, v) in enumerate(tqdm(parse_file(input_path), total=n_deals, unit="deal")):
        hands[i] = h
        calls[i, : len(c)] = c
        lengths[i] = len(c)
        dealer[i] = d
        vuln[i] = v
    assert i == n_deals - 1, f"expected {n_deals} deals, parsed {i+1}"

    np.save(cache_dir / "hands.npy", hands)
    np.save(cache_dir / "calls.npy", calls)
    np.save(cache_dir / "lengths.npy", lengths)
    np.save(cache_dir / "dealer.npy", dealer)
    np.save(cache_dir / "vuln.npy", vuln)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_deals)
    n_train = int(n_deals * split[0])
    n_val = int(n_deals * split[1])
    np.save(cache_dir / "train.npy", np.sort(perm[:n_train]))
    np.save(cache_dir / "val.npy", np.sort(perm[n_train:n_train + n_val]))
    np.save(cache_dir / "test.npy", np.sort(perm[n_train + n_val:]))

    meta = {
        "source": str(input_path),
        "num_deals": n_deals,
        "max_calls": MAX_CALLS,
        "hand_dim": HAND_DIM,
        "seed": seed,
        "split": list(split),
        "build_seconds": round(time.time() - t0, 1),
        "mean_calls": round(float(lengths.mean()), 3),
        "total_calls": int(lengths.sum()),
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="training.txt")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    build_cache(args.input, args.cache, seed=args.seed)


if __name__ == "__main__":
    main()
