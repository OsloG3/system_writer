"""League of frozen opponent policies for self-play RL training.

The league pools the opponents the learner can face in two-table team
iterations (with probability `league_prob` each iteration):

- **snapshots** of the learner itself, added every `snapshot_every` iters
  (classic league training against past selves, prevents strategy cycling);
- **external members**: frozen policies loaded from checkpoints -- supervised
  (BC) models, other RL runs, or the uniform-random baseline. Specs are
  `random` | `bc:<path>` | `rl:<path>` | `<path>` (the kind prefix is
  optional; checkpoint type and architecture are auto-detected, so league
  members may be any model size).

Members are sampled proportional to their base weights; with `pfsp_alpha > 0`
the sampling is additionally tilted toward members the learner currently
struggles against (prioritized fictitious self-play: weight is multiplied by
`(1 - winrate) ** pfsp_alpha`, winrate tracked as an EWMA of per-deal win
rates from `update_result`).

With a `persist_dir`, snapshot weights and member metadata (including
per-member winrate stats) are written to disk as they change, so a resumed
run restores the full league instead of restarting with an empty pool.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import ModelConfig
from .model import RLModel, build_from_ckpt, load_any_ckpt
from .rollout import UniformPolicy

META_FILE = "members.json"


def parse_spec(spec: str) -> tuple[str, str | None]:
    """'random' -> ('random', None); '[bc:|rl:]path' -> ('auto'|kind, path)."""
    if spec == "random":
        return "random", None
    kind, sep, path = spec.partition(":")
    if sep and kind in ("bc", "rl", "auto"):
        return "auto", path
    return "auto", spec


def _freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


@dataclass
class LeagueMember:
    name: str
    kind: str                     # random | bc | rl | snapshot
    weight: float = 1.0           # base sampling weight
    spec: str | None = None       # source spec, e.g. "rl:runs/x/best.pt"
    path: str | None = None       # resolved checkpoint path (external members)
    model: object = None          # frozen policy on device (external members)
    state: dict | None = None     # CPU state dict (snapshots)
    arch: dict | None = None      # ModelConfig dict (snapshots)
    file: str | None = None       # snapshot file name under persist_dir
    iters: int = 0                # league iterations faced the learner
    winrate: float = 0.5          # EWMA of the learner's per-deal winrate

    @property
    def is_snapshot(self) -> bool:
        return self.kind == "snapshot"


class League:
    def __init__(self, device, pfsp_alpha: float = 0.0, persist_dir=None,
                 ema: float = 0.2):
        self.device = device
        self.pfsp_alpha = pfsp_alpha
        self.ema = ema
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.members: list[LeagueMember] = []
        self._snap_counter = 0
        self._holders: dict = {}
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.members)

    def names(self) -> list:
        return [m.name for m in self.members]

    def get(self, name: str):
        return next((m for m in self.members if m.name == name), None)

    # -- membership ------------------------------------------------------------

    def add_spec(self, spec: str, weight: float = 1.0) -> LeagueMember:
        """Add an external member: 'random' | 'bc:<path>' | 'rl:<path>' |
        '<path>'. Idempotent by resolved name (re-adding refreshes weight)."""
        kind_hint, path = parse_spec(spec)
        if kind_hint == "random":
            name, member = "random", LeagueMember(
                name="random", kind="random", weight=weight, spec=spec,
                model=UniformPolicy())
        else:
            existing = next((m for m in self.members
                             if m.path == str(path)), None)
            if existing is not None:
                existing.weight = weight
                return existing
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model, kind = build_from_ckpt(ck, device=self.device)
            name = f"{kind}:{path}"
            member = LeagueMember(name=name, kind=kind, weight=weight,
                                  spec=spec, path=str(path),
                                  model=_freeze(model))
        existing = self.get(name)
        if existing is not None:
            existing.weight = weight
            return existing
        self.members.append(member)
        self.save_meta()
        return member

    def add_model(self, name: str, model, kind: str = "rl",
                  weight: float = 1.0, spec: str | None = None):
        """Add an already-built frozen policy (used for the warm-start model)."""
        existing = self.get(name)
        if existing is not None:
            existing.weight = weight
            return existing
        member = LeagueMember(name=name, kind=kind, weight=weight, spec=spec,
                              model=_freeze(model))
        self.members.append(member)
        self.save_meta()
        return member

    def add_snapshot(self, state: dict, arch, name: str | None = None,
                     weight: float = 1.0, max_snapshots: int | None = None):
        """Add a frozen copy of the learner's current weights (CPU-stored;
        materialized into a holder model on demand). Evicts oldest snapshots
        past max_snapshots."""
        state = {k: v.detach().cpu().clone() for k, v in state.items()}
        arch_d = arch.__dict__ if hasattr(arch, "__dict__") else dict(arch)
        self._snap_counter += 1
        name = name or f"snapshot-{self._snap_counter}"
        file = None
        if self.persist_dir:
            file = f"snap_{self._snap_counter:04d}.pt"
            torch.save({"model": state, "model_cfg": arch_d},
                       self.persist_dir / file)
        member = LeagueMember(name=name, kind="snapshot", weight=weight,
                              state=state, arch=arch_d, file=file)
        self.members.append(member)
        if max_snapshots:
            snaps = [m for m in self.members if m.is_snapshot]
            while len(snaps) > max_snapshots:
                old = snaps.pop(0)
                self.members.remove(old)
                if old.file and self.persist_dir:
                    (self.persist_dir / old.file).unlink(missing_ok=True)
        self.save_meta()
        return member

    # -- sampling ---------------------------------------------------------------

    def probabilities(self) -> np.ndarray:
        w = np.array([m.weight for m in self.members], dtype=np.float64)
        if self.pfsp_alpha > 0:
            w = w * np.array([(1.0 - m.winrate) ** self.pfsp_alpha
                              for m in self.members])
        s = w.sum()
        if s <= 0:
            return np.full(len(self.members), 1.0 / len(self.members))
        return w / s

    def sample(self, rng: np.random.Generator):
        if not self.members:
            return None
        i = int(rng.choice(len(self.members), p=self.probabilities()))
        return self.members[i]

    def opponent(self, member: LeagueMember, holder: RLModel | None = None):
        """Frozen policy for `member`. Snapshots are materialized into
        `holder` (an RLModel of matching arch, e.g. the train loop's reuse
        model); without one the league builds and caches its own."""
        if not member.is_snapshot:
            return member.model
        if holder is None:
            key = json.dumps(member.arch, sort_keys=True, default=str)
            holder = self._holders.get(key)
            if holder is None:
                holder = _freeze(RLModel(ModelConfig(**member.arch))
                                 .to(self.device))
                self._holders[key] = holder
        holder.load_state_dict(member.state)
        return holder

    # -- results ----------------------------------------------------------------

    def update_result(self, member: LeagueMember, team_imps):
        """Track the learner's per-deal winrate vs `member` (team_imps in
        learner-team view, as produced by play_team_deals league mode)."""
        wr = float((np.asarray(team_imps) > 0).mean())
        member.iters += 1
        member.winrate += self.ema * (wr - member.winrate)

    # -- persistence --------------------------------------------------------------

    def save_meta(self):
        if not self.persist_dir:
            return
        meta = {"snap_counter": self._snap_counter,
                "members": [{"name": m.name, "kind": m.kind,
                             "weight": m.weight, "spec": m.spec,
                             "path": m.path, "file": m.file,
                             "iters": m.iters, "winrate": m.winrate}
                            for m in self.members]}
        (self.persist_dir / META_FILE).write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, persist_dir, device, pfsp_alpha: float = 0.0,
             ema: float = 0.2) -> "League":
        """Restore a persisted league (external members reloaded from their
        checkpoint paths; unreadable members are skipped with a warning)."""
        persist_dir = Path(persist_dir)
        league = cls(device, pfsp_alpha=pfsp_alpha, ema=ema,
                     persist_dir=persist_dir)
        meta_path = persist_dir / META_FILE
        if not meta_path.exists():
            return league
        meta = json.loads(meta_path.read_text())
        league._snap_counter = meta.get("snap_counter", 0)
        for e in meta.get("members", []):
            kind = e["kind"]
            try:
                if kind == "snapshot":
                    ck = torch.load(persist_dir / e["file"],
                                    map_location="cpu", weights_only=False)
                    m = LeagueMember(name=e["name"], kind="snapshot",
                                     weight=e["weight"], state=ck["model"],
                                     arch=ck["model_cfg"], file=e["file"])
                elif kind == "random":
                    m = LeagueMember(name=e["name"], kind="random",
                                     weight=e["weight"], spec=e.get("spec"),
                                     model=UniformPolicy())
                else:
                    m = LeagueMember(name=e["name"], kind=kind,
                                     weight=e["weight"], spec=e.get("spec"),
                                     path=e.get("path"),
                                     model=_freeze(load_any_ckpt(e["path"],
                                                                 device)))
            except (OSError, KeyError, RuntimeError) as ex:
                print(f"warning: league member {e.get('name')!r} could not be "
                      f"restored ({ex}); skipping")
                continue
            m.iters = e.get("iters", 0)
            m.winrate = e.get("winrate", 0.5)
            league.members.append(m)
        return league
