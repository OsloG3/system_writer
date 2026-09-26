"""Benchmark DD solving + caching throughput on this machine.

    python -m bidding_dt.dd.bench --deals 200 --cache cache/dd.sqlite

Reports serial vs batched solve time (deals/s), par timing, and cache-hit
throughput. These numbers size the RL rollout budget: DDS is the bottleneck.
"""

import argparse
import time

import numpy as np

from ..env.deals import random_deals
from .solver import TableCache, dd_table, dd_tables_batch, par_score


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    deals = random_deals(rng, args.deals)
    hands = [d.hands for d in deals]

    dd_table(hands[0])  # warm up the library

    t0 = time.time()
    for h in hands[:20]:
        dd_table(h)
    t1 = time.time()
    print(f"serial  : {(t1 - t0) / 20 * 1000:7.1f} ms/deal  "
          f"({20 / (t1 - t0):.2f} deals/s)")

    t0 = time.time()
    tables = dd_tables_batch(hands)
    t1 = time.time()
    print(f"batched : {(t1 - t0) / len(hands) * 1000:7.1f} ms/deal  "
          f"({len(hands) / (t1 - t0):.2f} deals/s)")

    t0 = time.time()
    pars = [par_score(tables[i], int(rng.integers(4)), int(rng.integers(4)))
           for i in range(len(hands))]
    t1 = time.time()
    print(f"par     : {(t1 - t0) / len(hands) * 1000:7.3f} ms/deal  "
          f"mean |par| = {np.mean(np.abs(pars)):.0f}")

    if args.cache:
        cache = TableCache(args.cache)
        keys = [d.key() for d in deals]
        t0 = time.time()
        got = [cache.get(k) for k in keys]
        t1 = time.time()
        n_hit = sum(g is not None for g in got)
        cache.put_many(list(zip(keys, tables)))
        t0b = time.time()
        got2 = [cache.get(k) for k in keys]
        t1b = time.time()
        print(f"cache   : first pass {n_hit}/{len(keys)} hits in "
              f"{(t1 - t0) * 1000:.0f} ms; second pass "
              f"{sum(g is not None for g in got2)}/{len(keys)} hits in "
              f"{(t1b - t0b) * 1000:.0f} ms ({len(keys) / (t1b - t0b):.0f} deals/s)")
        cache.close()


if __name__ == "__main__":
    main()
