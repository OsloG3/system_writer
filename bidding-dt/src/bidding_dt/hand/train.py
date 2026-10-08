r"""Hand-inference training entrypoint.

Phase 1 (data): roll random deals through the bidding models -- BC and/or RL
checkpoints, the random baseline, any mix -- and keep the auctions
(hand/gen.py). The hand model is trained on *those* auctions, so it learns the
conventions the bots actually play, including whatever an RL run invented for
itself; `--source cache` mixes in the human cache from training.txt.

Phase 2 (model): train the VQ encoder-decoder (hand/model.py) on
(public auction prefix, target seat's 13 cards) views, cross-entropy over the
13 card draws plus the VQ codebook/commitment terms. Dead codes are reseeded
from live encoder outputs every `model.revive_every` steps (a k x V product
codebook collapses otherwise), and the codebook is seeded from one batch before
training starts.

CPU smoke:
  python -m bidding_dt.hand.train --preset tiny --config configs/hand_smoke.yaml \
      --policy bc:runs/bc_small/best.pt --out runs/hand_smoke
Full run (mixed sources, bigger generation):
  python -m bidding_dt.hand.train --preset small --config configs/hand_small.yaml \
      --policy rl:runs/rl_small/best.pt --policy bc:runs/small/best.pt \
      --source cache --gen-deals 400000 --out runs/hand_small
Only existing stores (no rollouts):
  python -m bidding_dt.hand.train --gen-deals 0 --source cache/gen --source cache \
      --out runs/hand_small2 --resume last
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from ..data.hands import HAND_DIM, HCP_WEIGHTS, HAND_LEN
from ..train import lr_lambda_fn, pick_device
from .config import HandConfig
from .data import AuctionStore, HandDataset, collate
from .gen import load_or_generate
from .model import build_hand_model

N_SUITS = 4
_SHAPE_BASE = torch.tensor([1, 14, 196, 2744])   # suit lengths < 14 -> unique key


def param_groups(model, weight_decay: float):
    """AdamW groups: no weight decay on 1-D params, embeddings or the codebook
    (decaying the codebook shrinks every code toward zero and fights the VQ
    loss)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "codebook" in name or name.endswith("_emb.weight"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def indicator(cards) -> torch.Tensor:
    """(...,13) card ids -> (...,52) 0/1 float indicator (data/hands.py order)."""
    return torch.zeros(*cards.shape[:-1], HAND_DIM,
                       device=cards.device).scatter(-1, cards, 1.0)


def suit_shapes(ind: torch.Tensor) -> torch.Tensor:
    """(...,52) indicator -> (...,4) suit lengths (S,H,D,C)."""
    return ind.reshape(*ind.shape[:-1], N_SUITS, HAND_LEN).sum(-1)


def hcp(ind: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """(...,52) indicator -> (...) high-card points."""
    return (ind * weights).sum(-1)


@torch.no_grad()
def evaluate(model, loader, device, use_amp, max_batches=None, n_samples=8,
             code_temp=1.0):
    """Val stats: teacher-forced CE, greedy-decode accuracy, and sampling
    behaviour (n_samples hands per row).

    `sample_acc`/`best_acc` are the mean / best-over-n card recovery of drawn
    hands; `jaccard` is the mean pairwise overlap between draws of the same row
    (low = the decoder is genuinely exploring, i.e. not collapsing onto one
    average hand); `shapes` is the mean number of distinct suit shapes among
    the n draws.
    """
    raw = getattr(model, "_orig_mod", model)
    model.eval()
    w = torch.as_tensor(HCP_WEIGHTS, dtype=torch.float32, device=device)
    base = _SHAPE_BASE.to(device)
    tot = 0
    sums = dict.fromkeys(("ce", "ppl", "acc", "shape", "hcp", "sample_acc",
                          "best_acc", "jaccard", "shapes"), 0.0)
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        cards, excl = batch["cards"], batch["excl"]
        b = cards.shape[0]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = raw(batch["tokens"], batch["vuln"], cards, excl=excl,
                      mask=batch["mask"])
        tgt = indicator(cards)
        tgt_shape = suit_shapes(tgt)
        tgt_hcp = hcp(tgt, w)

        greedy, _ = raw.sample(batch["tokens"], batch["vuln"], excl,
                               batch["mask"], n=1, greedy=True)
        g = indicator(greedy[:, 0])
        sums["ce"] += out["ce"].item() * b
        sums["ppl"] += out["ppl"] * b
        sums["acc"] += float((g * tgt).sum(-1).mean()) * b / HAND_LEN
        sums["shape"] += float((suit_shapes(g) == tgt_shape).all(-1).float().mean()) * b
        sums["hcp"] += float((hcp(g, w) - tgt_hcp).abs().mean()) * b

        draws, _ = raw.sample(batch["tokens"], batch["vuln"], excl,
                              batch["mask"], n=n_samples, codes=True,
                              code_temp=code_temp)
        s = indicator(draws)                                  # (B,n,52)
        hit = (s * tgt[:, None, :]).sum(-1) / HAND_LEN         # (B,n)
        sums["sample_acc"] += float(hit.mean()) * b
        sums["best_acc"] += float(hit.max(-1).values.mean()) * b
        inter = torch.einsum("bij,bkj->bik", s, s)
        jac = inter / (2 * HAND_LEN - inter).clamp_min(1.0)
        n = jac.shape[-1]
        off = ~torch.eye(n, dtype=torch.bool, device=jac.device)
        if n > 1:
            sums["jaccard"] += float(jac[:, off].mean()) * b
        key = (suit_shapes(s) * base).sum(-1).sort(-1).values
        uniq = 1 + (key[:, 1:] != key[:, :-1]).sum(-1) if n > 1 else torch.ones(b)
        sums["shapes"] += float(uniq.float().mean()) * b
        tot += b
    model.train()
    if tot == 0:                                     # empty val split
        return dict.fromkeys(sums, 0.0) | {"ce": float("inf"), "n_rows": 0}
    return {k: v / max(tot, 1) for k, v in sums.items()} | {"n_rows": tot}


def save_ckpt(path, model, opt, sched, step, epoch, cfg, best_ce):
    raw = getattr(model, "_orig_mod", model)
    torch.save({"model": raw.state_dict(), "optim": opt.state_dict(),
                "sched": sched.state_dict(), "step": step, "epoch": epoch,
                "best_ce": best_ce, "model_cfg": raw.cfg.__dict__,
                "cfg": cfg.to_dict()}, path)


def build_loaders(paths, cfg, device):
    """Train/val loaders over every store (generated + `--source` extras)."""
    pin = device.type == "cuda"
    train_ds = ConcatDataset([
        HandDataset(AuctionStore.load(p, "train", cfg.data.max_deals),
                    prefix=cfg.train.prefix, mask_aug=cfg.train.mask_aug,
                    mask_cards=cfg.train.mask_cards) for p in paths])
    val_ds = ConcatDataset([
        HandDataset(AuctionStore.load(p, "val", cfg.data.max_deals),
                    prefix="all", mask_aug=cfg.train.mask_aug,
                    mask_cards=cfg.train.mask_cards) for p in paths])
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=cfg.data.num_workers,
                              collate_fn=collate, pin_memory=pin, drop_last=True,
                              persistent_workers=cfg.data.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=max(1, cfg.data.num_workers // 2),
                            collate_fn=collate, pin_memory=pin)
    return train_loader, val_loader, train_ds, val_ds


def resolve_sources(cfg, out_dir, device, use_amp):
    """Generate the store from the bidding models (unless gen.deals == 0) and
    append the configured extra sources; returns store paths in use order."""
    paths = []
    if cfg.gen.deals > 0:
        specs = list(cfg.gen.policies)
        gen_dir = Path(cfg.gen.dir)
        if not gen_dir.is_absolute():
            gen_dir = out_dir / gen_dir
        store = load_or_generate(gen_dir, cfg.gen.deals, specs, device,
                                 reuse=cfg.gen.reuse, chunk=cfg.gen.chunk,
                                 greedy=cfg.gen.greedy, temp=cfg.gen.temp,
                                 amp=use_amp and device.type == "cpu",
                                 val_frac=cfg.gen.val_frac,
                                 test_frac=cfg.gen.test_frac,
                                 seed=cfg.gen.seed)
        paths.append(store.path)
    paths += [str(p) for p in cfg.data.sources]
    if not paths:
        raise SystemExit("no training data: pass --policy (generate) or --source")
    return paths


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config")
    ap.add_argument("--preset", default=None, choices=["tiny", "small", "base", "large"])
    ap.add_argument("--policy", action="append", default=[], metavar="SPEC[@W]",
                    help="generator policy, repeatable: random | bc:<ckpt> | "
                         "rl:<ckpt> | <ckpt> (merged with gen.policies)")
    ap.add_argument("--source", action="append", default=[], metavar="DIR",
                    help="extra auction store to mix in (e.g. cache), repeatable")
    ap.add_argument("--gen-deals", type=int, default=None,
                    help="deals to roll out for training data (0 = none)")
    ap.add_argument("--reuse-gen", action="store_true", default=None,
                    help="keep an existing generated store instead of rolling again")
    ap.add_argument("--gen-temp", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-deals", type=int, default=None,
                    help="subsample each store (smoke runs)")
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--val-batches", type=int, default=None)
    ap.add_argument("--val-samples", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--mask-aug", type=float, default=None)
    ap.add_argument("--n-codes", type=int, default=None, help="VQ slots (k)")
    ap.add_argument("--codebook-size", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint path or 'last'")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--cpu-bf16", action="store_true", default=None)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    cfg = HandConfig.load(args.config, args.preset)
    cfg.gen.policies = list(cfg.gen.policies) + list(args.policy)
    cfg.data.sources = list(cfg.data.sources) + list(args.source)
    for key, section, value in (
            ("deals", "gen", args.gen_deals), ("reuse", "gen", args.reuse_gen),
            ("temp", "gen", args.gen_temp), ("batch_size", "train", args.batch_size),
            ("lr", "train", args.lr), ("epochs", "train", args.epochs),
            ("max_steps", "train", args.max_steps),
            ("max_deals", "data", args.max_deals),
            ("val_every", "train", args.val_every),
            ("val_batches", "train", args.val_batches),
            ("val_samples", "train", args.val_samples),
            ("num_workers", "data", args.workers),
            ("mask_aug", "train", args.mask_aug),
            ("n_codes", "model", args.n_codes),
            ("codebook_size", "model", args.codebook_size),
            ("threads", "gen", args.threads),
            ("cpu_bf16", "gen", args.cpu_bf16)):
        if value is not None:
            setattr(getattr(cfg, section), key, value)
    if args.compile:
        cfg.train.compile = True

    out_dir = Path(args.out) if args.out else \
        Path("runs") / ("hand-" + time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "log.jsonl", "a")

    def log(rec: dict):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    device = pick_device(args.device or cfg.train.device)
    use_amp = (cfg.train.bf16 and device.type == "cuda") or \
              (cfg.gen.cpu_bf16 and device.type == "cpu")
    if device.type == "cpu" and cfg.gen.threads > 0:
        torch.set_num_threads(cfg.gen.threads)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed % (2 ** 32))

    paths = resolve_sources(cfg, out_dir, device, use_amp)
    train_loader, val_loader, train_ds, val_ds = build_loaders(paths, cfg, device)
    if len(train_ds) == 0:
        raise SystemExit(f"no training rows in {paths}")

    model = build_hand_model(cfg.model).to(device)
    print(f"model params: {model.num_params()/1e6:.3f}M  device: {device}  "
          f"amp(bf16): {use_amp}")
    print(f"sources: {paths}")
    if len(val_ds) == 0:
        print("warning: no validation rows -- best.pt will not be tracked")
    log({"event": "init", "params": model.num_params(), "device": str(device),
         "sources": paths, "train_rows": len(train_ds), "val_rows": len(val_ds),
         "cfg": cfg.to_dict()})

    opt = torch.optim.AdamW(param_groups(model, cfg.train.weight_decay),
                            lr=cfg.train.lr,
                            betas=(cfg.train.beta1, cfg.train.beta2))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.train.max_steps or cfg.train.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_fn(cfg.train.warmup_steps, total_steps, cfg.train.min_lr_frac))

    start_step = start_epoch = 0
    best_ce = float("inf")
    if args.resume:
        path = out_dir / "last.pt" if args.resume == "last" else Path(args.resume)
        ck = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        start_step, start_epoch = ck["step"], ck["epoch"]
        best_ce = ck.get("best_ce", float("inf"))
        print(f"resumed from {path} at step {start_step}")
    else:
        # data-dependent codebook init: seed all V codes from one batch of
        # encoder outputs, otherwise every slot starts on the same few codes
        seed_loader = DataLoader(train_ds, batch_size=min(256, len(train_ds)),
                                 shuffle=True, num_workers=0, collate_fn=collate)
        batch = next(iter(seed_loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        n = model.init_codebook(batch["tokens"], batch["vuln"], batch["excl"],
                                batch["mask"])
        del seed_loader
        log({"event": "codebook_init", "codes": n})

    if cfg.train.compile:
        model = torch.compile(model)
    raw = getattr(model, "_orig_mod", model)

    def do_val(step, epoch):
        nonlocal best_ce
        m = evaluate(model, val_loader, device, use_amp, cfg.train.val_batches,
                     cfg.train.val_samples)
        if m["n_rows"] == 0:
            print("warning: no validation rows (val split empty)")
            return m
        log({"event": "val", "step": step, "epoch": epoch,
             **{k: round(v, 5) for k, v in m.items()}})
        if m["ce"] < best_ce:
            best_ce = m["ce"]
            save_ckpt(out_dir / "best.pt", model, opt, sched, step, epoch, cfg,
                      best_ce)
        return m

    step, epoch = start_step, start_epoch

    def new_run():
        return {"loss": 0.0, "ce": 0.0, "vq": 0.0, "commit": 0.0, "ppl": 0.0,
                "rows": 0, "revived": 0}

    run = new_run()
    t_last = time.time()
    model.train()
    stop = False
    for epoch in range(start_epoch, cfg.train.epochs):
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(batch["tokens"], batch["vuln"], batch["cards"],
                            excl=batch["excl"], mask=batch["mask"])
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.train.grad_clip)
            opt.step()
            sched.step()
            step += 1
            # after the update: reseeding touches codebook.weight in place
            if raw.cfg.revive_every and step % raw.cfg.revive_every == 0:
                run["revived"] += raw.revive_dead(out["z_pre"])
            b = batch["cards"].shape[0]
            run["loss"] += loss.item() * b
            run["ce"] += out["ce"].item() * b
            run["vq"] += out["vq"].item() * b
            run["commit"] += out["commit"].item() * b
            run["ppl"] += out["ppl"] * b
            run["rows"] += b
            if step % cfg.train.log_every == 0:
                now = time.time()
                n_rows = max(run["rows"], 1)
                log({"event": "train", "step": step, "epoch": epoch,
                     "loss": round(run["loss"] / n_rows, 4),
                     "ce": round(run["ce"] / n_rows, 4),
                     "vq": round(run["vq"] / n_rows, 5),
                     "commit": round(run["commit"] / n_rows, 5),
                     "ppl": round(run["ppl"] / n_rows, 1),
                     "revived": run["revived"],
                     "lr": round(sched.get_last_lr()[0], 6),
                     "rows_s": round(run["rows"] / (now - t_last)),
                     "wall": round(now - t_last, 1)})
                run = new_run()
                t_last = now
            if cfg.train.val_every and step % cfg.train.val_every == 0:
                do_val(step, epoch)
                save_ckpt(out_dir / "last.pt", model, opt, sched, step, epoch,
                          cfg, best_ce)
                model.train()
                t_last = time.time()
            if cfg.train.max_steps and step >= cfg.train.max_steps:
                stop = True
                break
        if stop:
            break
        do_val(step, epoch + 1)

    save_ckpt(out_dir / "last.pt", model, opt, sched, step, epoch + 1, cfg,
              best_ce)
    print(f"done. best val ce: {best_ce:.4f}  checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
