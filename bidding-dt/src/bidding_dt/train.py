"""Training entrypoint.

Local CPU smoke run:
  python -m bidding_dt.train --preset tiny --max-deals 20000 --epochs 1 \
      --batch-size 128 --out runs/smoke
GPU full run:
  python -m bidding_dt.train --config configs/small.yaml --out runs/small
TPU run (PyTorch/XLA; spawns one process per TPU core, data-parallel):
  python -m bidding_dt.train --config configs/small.yaml --device xla \
      --out runs/small
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import xla
from .config import Config
from .data.dataset import IGNORE, BiddingDataset, collate
from .model.transformer import build_model

XLA_DEVICE_NAMES = ("xla", "tpu")


def wants_xla(pref: str | None) -> bool:
    return (pref or "").lower() in XLA_DEVICE_NAMES


def pick_device(pref: str | None) -> torch.device:
    if pref:
        if wants_xla(pref):
            return torch.device(xla.device())
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_lambda_fn(warmup: int, total: int, min_frac: float):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t))
    return fn


@torch.no_grad()
def evaluate(model, loader, device, use_amp, max_batches=None):
    model.eval()
    tot = c1 = c3 = 0
    loss_sum = 0.0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits, loss = model(batch["tokens"], batch["hand"], batch["vuln"], batch["targets"])
        t = batch["targets"][:, :-1]
        m = t != IGNORE
        tgt = t[m]
        if tgt.numel() == 0:
            continue
        top3 = logits[:, :-1][m].float().topk(3, dim=-1).indices
        c1 += int((top3[:, 0] == tgt).sum())
        c3 += int((top3 == tgt.unsqueeze(-1)).any(-1).sum())
        tot += tgt.numel()
        loss_sum += loss.item() * tgt.numel()
    model.train()
    return {
        "loss": loss_sum / max(tot, 1),
        "top1": c1 / max(tot, 1),
        "top3": c3 / max(tot, 1),
        "n_targets": tot,
    }


def save_ckpt(path, model, opt, sched, step, epoch, cfg, best_top1):
    raw = getattr(model, "_orig_mod", model)
    xla.save({
        "model": raw.state_dict(),
        "optim": opt.state_dict(),
        "sched": sched.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_top1": best_top1,
        "model_cfg": raw.cfg.__dict__,
        "cfg": cfg.to_dict(),
    }, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config")
    ap.add_argument("--preset", default=None, choices=["tiny", "small", "base", "large"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-deals", type=int, default=None, help="subsample train split")
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--val-batches", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint path or 'last'")
    ap.add_argument("--device", default=None, help="'xla'/'tpu' for PyTorch/XLA")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config, args.preset)
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.max_deals is not None:
        cfg.data.max_deals = args.max_deals
    if args.val_every is not None:
        cfg.train.val_every = args.val_every
    if args.val_batches is not None:
        cfg.train.val_batches = args.val_batches
    if args.workers is not None:
        cfg.data.num_workers = args.workers
    if args.compile:
        cfg.train.compile = True

    out_dir = Path(args.out) if args.out else Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        cfg.train.device = args.device
    use_xla = wants_xla(cfg.train.device)
    if use_xla and not xla.available():
        raise SystemExit("--device xla needs torch_xla: uv sync --extra tpu")
    world = xla.world_size() if use_xla else 1
    if world > 1:
        # one data-parallel process per TPU core (gradients are averaged with
        # an all_reduce before every optimizer step)
        xla.spawn(_train, args=(cfg, args.resume, out_dir, use_xla, world),
                  nprocs=world)
    else:
        _train(0, cfg, args.resume, out_dir, use_xla, world)


def _train(rank, cfg, resume, out_dir, use_xla, world):
    master = rank == 0
    logf = open(out_dir / "log.jsonl", "a") if master else None

    def log(rec: dict):
        if not master:
            return
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    # identical seeds on every rank -> identical init; DDP-style averaging of
    # gradients then keeps parameters in sync without an explicit broadcast
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed % (2**32))
    device = xla.device() if use_xla else pick_device(cfg.train.device)
    use_amp = cfg.train.bf16 and device.type in ("cuda", "xla")

    train_ds = BiddingDataset(cfg.data.cache_dir, "train", max_deals=cfg.data.max_deals)
    val_ds = BiddingDataset(cfg.data.cache_dir, "val", max_deals=cfg.data.max_deals)
    pin = device.type == "cuda"
    train_sampler = (DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                        shuffle=True, drop_last=True)
                     if world > 1 else None)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=train_sampler is None,
                              sampler=train_sampler,
                              num_workers=cfg.data.num_workers, collate_fn=collate,
                              pin_memory=pin, persistent_workers=cfg.data.num_workers > 0,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=max(1, cfg.data.num_workers // 2), collate_fn=collate,
                            pin_memory=pin)

    model = build_model(cfg.model).to(device)
    n_params = model.num_params()
    if master:
        print(f"model params: {n_params/1e6:.3f}M  device: {device}  "
              f"amp(bf16): {use_amp}  ranks: {world}")
    log({"event": "init", "params": n_params, "device": str(device),
         "ranks": world,
         "train_seqs": len(train_ds), "val_seqs": len(val_ds), "cfg": cfg.to_dict()})

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            betas=(cfg.train.beta1, cfg.train.beta2),
                            weight_decay=cfg.train.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.train.max_steps or cfg.train.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_fn(cfg.train.warmup_steps, total_steps, cfg.train.min_lr_frac))

    start_step = start_epoch = 0
    best_top1 = 0.0
    if resume:
        resume_path = out_dir / "last.pt" if resume == "last" else Path(resume)
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        start_step = ck["step"]
        start_epoch = ck["epoch"]
        best_top1 = ck.get("best_top1", 0.0)
        if master:
            print(f"resumed from {resume_path} at step {start_step}")

    if cfg.train.compile and not use_xla:
        model = torch.compile(model)

    def do_val(step, epoch):
        nonlocal best_top1
        if master:
            m = evaluate(model, val_loader, device, use_amp, cfg.train.val_batches)
            log({"event": "val", "step": step, **{k: round(v, 5) if isinstance(v, float) else v
                                                  for k, v in m.items()}})
            if m["top1"] > best_top1:
                best_top1 = m["top1"]
                save_ckpt(out_dir / "best.pt", model, opt, sched, step, epoch, cfg, best_top1)
        if use_xla and world > 1:
            xla.barrier()

    step = start_step
    epoch = start_epoch
    # losses stay on-device between logs: .item() every step would force a
    # host sync per step, which is very expensive on TPUs
    run_loss = torch.zeros((), device=device)
    run_tok = 0
    t_last = time.time()
    model.train()
    stop = False
    for epoch in range(start_epoch, cfg.train.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(batch["tokens"], batch["hand"], batch["vuln"], batch["targets"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            if use_xla:
                xla.average_grads(model.parameters(), world)
                xla.optimizer_step(opt)
            else:
                opt.step()
            sched.step()
            step += 1

            n_tok = int((batch["targets"][:, :-1] != IGNORE).sum())
            run_loss += loss.detach().float() * n_tok
            run_tok += n_tok
            if step % cfg.train.log_every == 0:
                now = time.time()
                if master:
                    loss_f = float(run_loss.item())
                    log({"event": "train", "step": step, "epoch": epoch,
                         "loss": round(loss_f / max(run_tok, 1), 4),
                         "lr": round(sched.get_last_lr()[0], 6),
                         "tgt_tok_s": round(run_tok * world / (now - t_last)),
                         "wall": round(now - t_last, 1)})
                run_loss.zero_()
                run_tok = 0
                t_last = now
                if use_xla:
                    xla.mark_step()
            if cfg.train.val_every and step % cfg.train.val_every == 0:
                do_val(step, epoch)
                save_ckpt(out_dir / "last.pt", model, opt, sched, step, epoch, cfg, best_top1)
                model.train()
                t_last = time.time()
            if cfg.train.max_steps and step >= cfg.train.max_steps:
                stop = True
                break
        if stop:
            break
        do_val(step, epoch + 1)

    save_ckpt(out_dir / "last.pt", model, opt, sched, step, epoch + 1, cfg, best_top1)
    if master:
        print(f"done. best val top1: {best_top1:.4f}  checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
