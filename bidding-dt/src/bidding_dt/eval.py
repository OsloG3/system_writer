"""Evaluate a checkpoint on a data split, with breakdowns.

  python -m bidding_dt.eval --ckpt runs/small/best.pt
"""

import argparse
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data.dataset import IGNORE, BiddingDataset, collate
from .data.vocab import CALL_PASS, CALL_X, CALL_XX, FIRST_BID
from .model.transformer import build_model
from .train import pick_device

VULN_NAMES = ["none", "we-only", "they-only", "both"]


class Bucket:
    __slots__ = ("n", "c1", "c3", "loss")

    def __init__(self):
        self.n = self.c1 = self.c3 = 0
        self.loss = 0.0

    def add(self, n, c1, c3, loss):
        self.n += n
        self.c1 += c1
        self.c3 += c3
        self.loss += loss

    @property
    def top1(self):
        return self.c1 / max(self.n, 1)

    @property
    def top3(self):
        return self.c3 / max(self.n, 1)


@torch.no_grad()
def evaluate_ckpt(ckpt_path, cache_dir="cache", split="test", batch_size=256,
                  device=None, max_deals=None, num_workers=2):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dev = pick_device(device)
    model = build_model(ModelConfig(**ck["model_cfg"])).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = BiddingDataset(cache_dir, split, max_deals=max_deals)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)

    overall = Bucket()
    buckets = defaultdict(Bucket)
    for batch in loader:
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        logits, loss = model(batch["tokens"], batch["hand"], batch["vuln"], batch["targets"])
        t = batch["targets"][:, :-1]
        m = t != IGNORE
        tgt = t[m]
        if tgt.numel() == 0:
            continue
        lg = logits[:, :-1][m].float()
        nll = torch.nn.functional.cross_entropy(lg, tgt, reduction="none")
        top3 = lg.topk(3, dim=-1).indices
        hit1 = (top3[:, 0] == tgt)
        hit3 = (top3 == tgt.unsqueeze(-1)).any(-1)

        overall.add(tgt.numel(), int(hit1.sum()), int(hit3.sum()), float(nll.sum()))

        dnos = batch["dnos"][:, :-1][m]
        vuln = batch["vuln"].unsqueeze(1).expand_as(t)[m]
        for j in dnos.unique().tolist():
            sel = dnos == j
            b = buckets[f"decision_{int(j)+1}"]
            b.add(int(sel.sum()), int(hit1[sel].sum()), int(hit3[sel].sum()),
                  float(nll[sel].sum()))
        for v in range(4):
            sel = vuln == v
            if int(sel.sum()):
                b = buckets[f"vuln_{VULN_NAMES[v]}"]
                b.add(int(sel.sum()), int(hit1[sel].sum()), int(hit3[sel].sum()),
                      float(nll[sel].sum()))
        for name, sel in (("target_pass", tgt == CALL_PASS),
                          ("target_bid", tgt >= FIRST_BID),
                          ("target_x/xx", (tgt == CALL_X) | (tgt == CALL_XX))):
            if int(sel.sum()):
                b = buckets[name]
                b.add(int(sel.sum()), int(hit1[sel].sum()), int(hit3[sel].sum()),
                      float(nll[sel].sum()))

    return overall, dict(buckets)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-deals", type=int, default=None)
    ap.add_argument("--json", default=None, help="write results to json file")
    args = ap.parse_args()

    overall, buckets = evaluate_ckpt(args.ckpt, args.cache, args.split,
                                     args.batch_size, args.device, args.max_deals)
    rows = [("OVERALL", overall)] + sorted(buckets.items())
    w = max(len(n) for n, _ in rows)
    print(f"{'bucket':<{w}}  {'n':>9}  {'top1':>7}  {'top3':>7}  {'nll':>7}")
    for name, b in rows:
        print(f"{name:<{w}}  {b.n:>9}  {b.top1:>7.4f}  {b.top3:>7.4f}  {b.loss/max(b.n,1):>7.4f}")
    if args.json:
        out = {"overall": {"n": overall.n, "top1": overall.top1, "top3": overall.top3},
               "buckets": {k: {"n": v.n, "top1": v.top1, "top3": v.top3}
                           for k, v in buckets.items()}}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
