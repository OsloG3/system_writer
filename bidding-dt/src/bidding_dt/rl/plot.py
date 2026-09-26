"""Plot RL training progress from a run's log.jsonl.

    python -m bidding_dt.rl.plot runs/rl_small              # -> runs/rl_small/progress.png
    python -m bidding_dt.rl.plot runs/a runs/b --out cmp.png   # overlay runs
    python -m bidding_dt.rl.plot runs/rl_small --smooth 25 --show

Reads the iter/eval records written by train_ppo.py and draws an eight-panel
overview: eval IMPs/board vs random and BC (+-1 SE), train reward (raw +
moving average), PPO losses, entropy, KL/clipfrac, auction behaviour,
annealing schedules, and per-iteration cost. A text summary is always
printed so progress is readable without opening the PNG. matplotlib is
imported lazily (it ships with the `rl` extra); parsing needs only numpy.
"""

import argparse
import json
from pathlib import Path

import numpy as np

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

EVAL_SERIES = [("imps_vs_random", "se_vs_random", "vs random"),
               ("imps_vs_bc", "se_vs_bc", "vs BC")]


def load_log(path) -> dict:
    """Parse a log.jsonl (or a run dir containing one).

    Returns {"init": last init record, <event>: {field: np.ndarray}} where
    every event dict carries "iter" as x; fields missing from some records
    become NaN.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "log.jsonl"
    out: dict = {}
    events: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        events.setdefault(rec.get("event", "?"), []).append(rec)
    for name, recs in events.items():
        if name == "init":
            out["init"] = recs[-1]
            continue
        cols = {}
        for f in sorted({k for r in recs for k in r if k != "event"}):
            try:
                cols[f] = np.array([float(r.get(f, np.nan)) for r in recs])
            except (TypeError, ValueError):
                continue  # non-numeric field
        out[name] = cols
    return out


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average (window w, clamped at edges); NaNs ignored."""
    y = np.asarray(y, float)
    if w <= 1 or len(y) == 0:
        return y
    out = np.empty_like(y)
    h = w // 2
    for i in range(len(y)):
        seg = y[max(0, i - h):i + h + 1]
        seg = seg[np.isfinite(seg)]
        out[i] = seg.mean() if len(seg) else np.nan
    return out


def summarize(run: str, log: dict) -> str:
    """One-line text digest of the latest state of a run."""
    it, ev = log.get("iter", {}), log.get("eval", {})
    parts = [f"[{run}]"]
    if "iter" in it:
        parts.append(f"iters {int(it['iter'][-1]) + 1}")
    for f in ("reward_mean", "entropy", "kl", "passout_rate"):
        if f in it:
            parts.append(f"{f} {it[f][-1]:g}")
    for f, _, _ in EVAL_SERIES:
        if f in ev:
            parts.append(f"{f} last {ev[f][-1]:g} / best {np.nanmax(ev[f]):g}")
    return "  ".join(parts)


def _plot_field(ax, logs: dict, event: str, field: str, ls="-", lw=1.4):
    """Plot logs[*][event][field] vs iter; True if anything was drawn."""
    drew = False
    multi = len(logs) > 1
    for i, (run, log) in enumerate(logs.items()):
        ev = log.get(event, {})
        if "iter" not in ev or field not in ev:
            continue
        ax.plot(ev["iter"], ev[field], ls, color=COLORS[i % len(COLORS)],
                lw=lw, label=f"{run} {field}" if multi else field)
        drew = True
    return drew


def make_figure(logs: dict, smooth_w: int = 10):
    """Build the progress figure. logs: run label -> load_log() output."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    multi = len(logs) > 1

    def finish(ax, title, drew, legend=True):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("iter", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        if not drew:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="0.6")
        elif legend:
            ax.legend(fontsize=7, loc="best")

    def draw(fields, target, default_ls):
        drew = False
        for item in fields:
            f, ls = item if isinstance(item, tuple) else (item, default_ls)
            drew |= _plot_field(target, logs, "iter", f, ls=ls)
        return drew

    def twin(ax, title, left_fields, right_fields):
        drew = draw(left_fields, ax, "-")
        drew_r = False
        ax2 = None
        if right_fields:
            ax2 = ax.twinx()
            ax2.grid(False)
            drew_r = draw(right_fields, ax2, "--")
        finish(ax, title, drew or drew_r, legend=False)
        if drew or drew_r:
            h, l = ax.get_legend_handles_labels()
            if ax2 is not None:
                h2, l2 = ax2.get_legend_handles_labels()
                h, l = h + h2, l + l2
            ax.legend(h, l, fontsize=7, loc="best")

    # eval: IMPs/board vs fixed opponents with +-1 SE bands
    ax = axes[0, 0]
    drew = False
    for i, (run, log) in enumerate(logs.items()):
        ev = log.get("eval", {})
        if "iter" not in ev:
            continue
        c = COLORS[i % len(COLORS)]
        for field, se_field, name in EVAL_SERIES:
            if field not in ev:
                continue
            x, y = ev["iter"], ev[field]
            ax.plot(x, y, "-o", color=c, lw=1.3, ms=3,
                    label=f"{run} {name}" if multi else name)
            if se_field in ev:
                ax.fill_between(x, y - ev[se_field], y + ev[se_field],
                                color=c, alpha=0.15, lw=0)
            drew = True
    ax.axhline(0.0, color="k", lw=0.7)
    finish(ax, "eval: IMPs/board vs fixed opponents (+-1 SE)", drew)

    # train reward: raw (faint) + moving average
    ax = axes[0, 1]
    drew = False
    for i, (run, log) in enumerate(logs.items()):
        it = log.get("iter", {})
        if "reward_mean" not in it:
            continue
        c = COLORS[i % len(COLORS)]
        x, y = it["iter"], it["reward_mean"]
        w = max(1, min(smooth_w, len(y)))
        ax.plot(x, y, color=c, lw=0.5, alpha=0.3)
        ax.plot(x, smooth(y, w), color=c, lw=1.6,
                label=f"{run} MA{w}" if multi else f"reward MA{w}")
        drew = True
    ax.axhline(0.0, color="k", lw=0.7)
    finish(ax, "train reward: signed IMPs vs par (faint = raw)", drew)

    twin(axes[1, 0], "PPO losses", ["value_loss"], ["policy_loss"])
    ax = axes[1, 1]
    finish(ax, "policy entropy", _plot_field(ax, logs, "iter", "entropy"))

    ax = axes[2, 0]
    drew = draw([("kl", "-"), ("approx_kl", "--"), ("clipfrac", ":")], ax, "-")
    finish(ax, "KL to BC / approx-KL / clipfrac", drew)

    twin(axes[2, 1], "auction behaviour", ["calls_mean"], ["passout_rate"])
    twin(axes[3, 0], "annealing schedule",
         ["beta", ("ent_coef", "--")], [("lr", ":")])
    twin(axes[3, 1], "iteration cost", ["wall"], ["decisions"])

    fig.suptitle("RL progress: " + ", ".join(logs), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+",
                    help="run dirs (containing log.jsonl) or log file paths")
    ap.add_argument("--out", default=None,
                    help="output PNG (default <run>/progress.png for one run dir)")
    ap.add_argument("--smooth", type=int, default=10,
                    help="moving-average window for the train reward")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--show", action="store_true",
                    help="also open an interactive window")
    args = ap.parse_args()

    paths = [Path(p) for p in args.runs]
    logs: dict[str, dict] = {}
    for p in paths:
        f = p / "log.jsonl" if p.is_dir() else p
        if not f.exists():
            raise SystemExit(f"no log file: {f}")
        label = p.name if p.is_dir() else (p.parent.name if p.name == "log.jsonl" else p.stem)
        base, k = label, 2
        while label in logs:
            label = f"{base}#{k}"
            k += 1
        logs[label] = load_log(f)
        if not logs[label].get("iter"):
            raise SystemExit(f"{label}: no iter events in {f}")
        print(summarize(label, logs[label]))

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    fig = make_figure(logs, smooth_w=args.smooth)

    if args.out:
        out = Path(args.out)
    elif len(paths) == 1 and paths[0].is_dir():
        out = paths[0] / "progress.png"
    else:
        out = Path("rl_progress.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"saved {out}")
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
