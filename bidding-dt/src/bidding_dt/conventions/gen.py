"""Manufacture convention demonstrations from the system file.

The RL system-teacher loss (rl/train_ppo.py --sys-weight) rewards and targets
system conventions in the learner's *own* auctions. This generator attacks the
other half of the problem: the model may never reach the convention auction at
all, so it has no full sequence to learn from. It plays random deals with a
fallback bidding model, but whenever the system file has a clean artificial
call available it substitutes that call with probability --conv-prob. The
result is a data/parse.py-compatible cache of auctions that actually use the
system's conventions, which can be mixed into BC training.

  python -m bidding_dt.conventions.gen \
      --system ../models/system.txt --deals 50000 \
      --policy bc:runs/bc_small/best.pt --out cache/system

Then, in the BC config:
  data:
    cache_dir: cache
    system_cache: cache/system   # mixed into the train split
    system_repeat: 3             # oversample the convention demonstrations

The system file's entries are checked with the same SystemEnforcer the trainer
uses, so the generated calls are exactly the file's conventions (hand shape,
K&R points, OR branches and the built-in RKCB fit/values gate all apply).
"""

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..data.hands import HAND_DIM
from ..data.parse import MAX_CALLS
from ..data.vocab import VOCAB_SIZE
from ..env.auction_env import AuctionBatch
from ..env.deals import random_deals
from ..hand.gen import policy_pool, save_store
from ..rl.rollout import _act
from ..train import pick_device
from .auction import AuctionReader
from .patterns import SystemEnforcer
from .rules import HandInfo
from .system_file import find_system_file, parse_system_file


def generate(out_dir, n_deals: int, specs, device, system_path: str,
             chunk: int = 256, conv_prob: float = 0.8, temp: float = 1.0,
             greedy: bool = False, seed: int = 4242, val_frac: float = 0.01,
             test_frac: float = 0.01, progress: bool = True,
             conv_min_prob: float = 0.0):
    """Roll random deals; substitute clean system conventions with conv_prob.

    `specs` are league policy specs (`bc:<ckpt>`, `rl:<ckpt>`, `random`, or a
    bare checkpoint path); the fallback policy is sampled per partnership.
    `system_path` is the conventions text file interpreted by the trainer.
    """
    spec = parse_system_file(system_path)
    enforcer = SystemEnforcer(spec, spec.config())
    policies, names, probs = policy_pool(specs, device)
    rng = np.random.default_rng(seed)

    hands = np.zeros((n_deals, 4, HAND_DIM), dtype=np.uint8)
    calls = np.zeros((n_deals, MAX_CALLS), dtype=np.uint8)
    lengths = np.zeros(n_deals, dtype=np.uint8)
    dealer = np.zeros(n_deals, dtype=np.uint8)
    vuln = np.zeros(n_deals, dtype=np.uint8)
    policy = np.zeros((n_deals, 4), dtype=np.int8)

    t0 = time.time()
    done = 0
    bar = tqdm(total=n_deals, unit="deal", disable=not progress)
    while done < n_deals:
        b = min(chunk, n_deals - done)
        deals = random_deals(rng, b)
        dl = rng.integers(0, 4, b)
        vn = rng.integers(0, 4, b)
        seat_policy = np.empty((b, 4), dtype=np.int64)
        # partners share a policy so the fallback side is a coherent pair
        seat_policy[:, 0] = seat_policy[:, 2] = rng.choice(
            len(policies), b, p=probs)
        seat_policy[:, 1] = seat_policy[:, 3] = rng.choice(
            len(policies), b, p=probs)

        env = AuctionBatch(deals, dl, vn)
        while env.any_active():
            idx = env.active_idx()
            tokens, hand, vuln_cls, hero, row_len = env.build_inputs(idx)
            masks = env.legal_masks(idx)
            actions = np.zeros(len(idx), dtype=np.int64)
            row_probs = np.zeros((len(idx), VOCAB_SIZE), dtype=np.float32)

            # fallback policy action for every active seat
            which = seat_policy[idx, hero]
            for pid in np.unique(which):
                rows = np.flatnonzero(which == pid)
                a, _, _, _, pr = _act(
                    policies[pid], tokens[rows], hand[rows], vuln_cls[rows],
                    row_len[rows], masks[rows], device, greedy, temp)
                actions[rows] = a.cpu().numpy()
                row_probs[rows] = pr.cpu().numpy()

            # substitute a clean system call, so the convention actually
            # happens and its follow-ups enter the data. The fallback policy's
            # probability keeps a wildcard/underspecified entry from forcing a
            # call the model considers essentially impossible.
            for k, i in enumerate(idx):
                if conv_prob <= 0.0 or rng.random() >= conv_prob:
                    continue
                seat = int(hero[k])
                ci = int(env.n_calls[i])
                reader = AuctionReader(int(env.dealer[i]), int(env.vuln[i]))
                for c in env.calls[i, :ci]:
                    reader.feed(int(c))
                hi = HandInfo(env.deals[i].hands[seat])
                clean = enforcer.clean_actions(
                    reader, seat, hi, masks[k], probs=row_probs[k],
                    min_prob=conv_min_prob)
                if clean:
                    actions[k] = int(clean[rng.integers(len(clean))])

            env.step(idx, actions)

        s = slice(done, done + b)
        hands[s] = np.stack([d.encoded for d in deals])
        calls[s] = env.calls
        lengths[s] = env.n_calls
        dealer[s] = dl
        vuln[s] = vn
        policy[s] = seat_policy
        done += b
        bar.update(b)
    bar.close()

    out_dir = save_store(
        out_dir, hands, calls, lengths, dealer, vuln, policy=policy,
        val_frac=val_frac, test_frac=test_frac, seed=seed,
        meta={"generated": True, "kind": "system-conventions",
              "system_file": str(system_path),
              "policies": names,
              "weights": [round(float(p), 4) for p in probs],
              "conv_prob": conv_prob, "temp": temp, "greedy": greedy,
              "seconds": round(time.time() - t0, 1)})
    print(f"generated {n_deals} convention deals -> {out_dir} "
          f"(mean {lengths.mean():.2f} calls, {time.time() - t0:.1f}s)")
    return out_dir


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="cache/system", help="store directory")
    ap.add_argument("--deals", type=int, default=50000)
    ap.add_argument("--policy", action="append", default=[], metavar="SPEC[@W]",
                    help="fallback policy, repeatable: random | bc:<ckpt> | "
                         "rl:<ckpt> | <ckpt>, optional @weight")
    ap.add_argument("--system", default=None,
                    help="system file (default: rl.system_file / "
                         "BIDDING_SYSTEM_FILE / models/system.txt / "
                         "../models/system.txt)")
    ap.add_argument("--conv-prob", type=float, default=0.8,
                    help="probability of substituting a clean system "
                         "convention at a decision where one is available")
    ap.add_argument("--conv-min-prob", type=float, default=0.0,
                    help="only substitute a convention the fallback policy "
                         "gives at least this probability (0 = no floor; "
                         "raise it if wildcard entries create absurd auctions)")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="fallback-policy sampling temperature")
    ap.add_argument("--greedy", action="store_true",
                    help="use the fallback policy's argmax instead of sampling")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        system_path = find_system_file(args.system)
    except FileNotFoundError as e:
        ap.error(str(e))
    if system_path is None:
        ap.error("no system file found (models/system.txt or --system)")
    specs = list(args.policy)
    if not specs:
        default = "bc:runs/bc_small/best.pt"
        specs = [default if Path("runs/bc_small/best.pt").is_file()
                 else "random"]
    device = pick_device(args.device)
    generate(args.out, args.deals, specs, device, system_path,
             chunk=args.chunk, conv_prob=args.conv_prob, temp=args.temp,
             greedy=args.greedy, seed=args.seed,
             conv_min_prob=args.conv_min_prob)


if __name__ == "__main__":
    main()