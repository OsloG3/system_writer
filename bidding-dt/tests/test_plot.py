import json
import sys

import numpy as np
import pytest
from pathlib import Path

from bidding_dt.rl import plot as rlplot

HAS_MPL = True
try:
    import matplotlib
except ImportError:  # pragma: no cover
    HAS_MPL = False


def write_log(path: Path, n_iter=6, n_eval=2):
    recs = [{"event": "init", "params": 1000, "device": "cpu", "cfg": {}}]
    for i in range(n_iter):
        recs.append({"event": "iter", "iter": i, "reward_mean": float(i),
                     "reward_std": 1.0, "imps_vs_par": float(i),
                     "decisions": 100 + i, "calls_mean": 9.0,
                     "passout_rate": 0.05, "league": i % 2 == 0,
                     "beta": 0.1, "ent_coef": 0.01, "lr": 1e-4, "wall": 2.5,
                     "policy_loss": 0.01, "value_loss": 1.0, "entropy": 0.5,
                     "kl": 0.02, "clipfrac": 0.1, "approx_kl": 0.03})
    for i in range(n_eval):
        rec = {"event": "eval", "iter": 2 * i + 1,
               "imps_vs_random": 3.0 + i, "se_vs_random": 1.0}
        if i:  # first eval lacks a BC anchor -> missing fields become NaN
            rec.update(imps_vs_bc=2.0 + i, se_vs_bc=1.0)
        recs.append(rec)
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_load_log(tmp_path):
    write_log(tmp_path / "log.jsonl")
    log = rlplot.load_log(tmp_path / "log.jsonl")
    assert log["init"]["params"] == 1000
    it = log["iter"]
    assert it["iter"].tolist() == list(range(6))
    assert it["league"].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    ev = log["eval"]
    assert ev["iter"].tolist() == [1, 3]
    assert np.isnan(ev["imps_vs_bc"][0]) and ev["imps_vs_bc"][1] == 3.0
    # a run dir resolves to the log.jsonl inside it
    run = tmp_path / "run"
    run.mkdir()
    write_log(run / "log.jsonl")
    assert rlplot.load_log(run)["iter"]["iter"].tolist() == list(range(6))


def test_smooth():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.allclose(rlplot.smooth(y, 1), y)
    assert np.allclose(rlplot.smooth(y, 3), [0.5, 1.0, 2.0, 3.0, 3.5])
    z = np.array([0.0, np.nan, 2.0])
    assert rlplot.smooth(z, 3)[1] == 1.0


def test_summarize(tmp_path):
    write_log(tmp_path / "log.jsonl")
    s = rlplot.summarize("run", rlplot.load_log(tmp_path / "log.jsonl"))
    assert "[run]" in s and "iters 6" in s
    assert "imps_vs_random last 4 / best 4" in s
    assert "imps_vs_bc last 3 / best 3" in s


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_make_figure(tmp_path):
    matplotlib.use("Agg")
    write_log(tmp_path / "log.jsonl")
    write_log(tmp_path / "noeval.jsonl", n_eval=0)
    logs = {"a": rlplot.load_log(tmp_path / "log.jsonl"),
            "b": rlplot.load_log(tmp_path / "noeval.jsonl")}
    fig = rlplot.make_figure(logs, smooth_w=3)
    assert len(fig.axes) >= 8  # 8 panels + twin axes
    out = tmp_path / "progress.png"
    fig.savefig(out)
    assert out.stat().st_size > 1000


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_main_cli(tmp_path, monkeypatch, capsys):
    matplotlib.use("Agg")
    run = tmp_path / "run"
    run.mkdir()
    write_log(run / "log.jsonl")

    monkeypatch.setattr(sys, "argv", ["plot", str(run)])
    rlplot.main()
    assert (run / "progress.png").exists()
    out = capsys.readouterr().out
    assert "[run]" in out and f"saved {run / 'progress.png'}" in out

    # two runs -> explicit --out, duplicate labels get disambiguated
    cmp_png = tmp_path / "cmp.png"
    monkeypatch.setattr(sys, "argv",
                        ["plot", str(run), str(run / "log.jsonl"),
                         "--out", str(cmp_png)])
    rlplot.main()
    assert cmp_png.exists()
    assert "run#2" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "argv", ["plot", str(tmp_path / "missing")])
        rlplot.main()
