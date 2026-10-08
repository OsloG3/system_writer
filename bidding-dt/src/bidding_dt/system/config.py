"""System-policy configs: FFN model, inner PPO (rl.config.RLConfig), outer evolution.

The inner loop is a stock PPO iteration, so it reuses `RLConfig` verbatim
(rollout size, reward weights, clip/vf/entropy, league probability, ...). What
is new is the outer loop (`EvoConfig`): how many systems live in the
population, how they are mutated and replaced, and how often they are
evaluated and snapshotted into the league.

`rl.iters` is the number of PPO iterations *per generation*; the LR schedule
spans `evo.generations * rl.iters`. `rl.eval_every` is unused -- the outer loop
evaluates once per generation (`evo.eval_every`).
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ..config import PRESETS
from ..data.hands import HAND_DIM
from ..data.vocab import VOCAB_SIZE
from ..rl.config import RLConfig
from .state import SUMMARY_DIM

TRUNK_LAYERS = {"tiny": 3, "small": 4, "base": 5, "large": 6}


@dataclass
class SystemModelConfig:
    d_model: int = 256
    d_ff: int = 680
    n_layers: int = 4           # residual SwiGLU blocks
    dropout: float = 0.0
    d_z: int = 32               # state features per seat's z
    z_hidden: int = 0           # >0: 2-layer adapter instead of a linear one
    n_actions: int = VOCAB_SIZE
    hand_dim: int = HAND_DIM
    summary_dim: int = SUMMARY_DIM
    z_cache: int = 50_000       # memoized encoder codes (0 = recompute always)

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "SystemModelConfig":
        if name not in PRESETS or name not in TRUNK_LAYERS:
            raise ValueError(f"unknown preset: {name}")
        wide = {k: v for k, v in PRESETS[name].items()
                if k in ("d_model", "d_ff", "dropout")}
        return cls(**wide, n_layers=TRUNK_LAYERS[name], **overrides)


@dataclass
class EvoConfig:
    generations: int = 20        # outer iterations
    systems: int = 8             # population size (adapter slots in the policy)
    encoders: list = field(default_factory=list)   # HandVQ checkpoints
    sigma: float = 0.002         # child = parent + sigma*N(0,1); ~10% of the
                                 # adapters' 0.02 init std, so a mutation moves
                                 # a system without re-randomizing it
    sigma_relative: bool = False  # true -> sigma is a fraction of each weight's
                                 # own std (scale-free across adapter shapes)
    keep_best: int = 4           # elites that are never replaced
    replace_frac: float = 0.5    # fraction of the population bred per generation
    sample_by: str = "uniform"   # uniform | fitness (softmax over eval scores)
    fitness_temp: float = 4.0    # IMPs per unit of sampling logit
    eval_every: int = 1          # generations between population evaluations
    eval_deals: int = 128
    eval_opponent: str = "random"   # league spec the systems are scored against
    snapshot_every: int = 2      # generations between league snapshots
    snapshot_top: int = 2
    inner_target: str = "policy"  # policy | adapter | both (what PPO updates)
    persist: bool = True         # keep the population under <out>/population
    ema: float = 0.2             # EWMA rate for a system's training IMPs
    seed: int = 1337


@dataclass
class SystemConfig:
    model: SystemModelConfig = field(default_factory=SystemModelConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    evo: EvoConfig = field(default_factory=EvoConfig)

    @classmethod
    def load(cls, path: str | Path | None = None,
             preset: str | None = None) -> "SystemConfig":
        cfg = cls()
        raw = {}
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text()) or {}
            for section in ("model", "rl", "evo"):
                for k, v in (raw.get(section) or {}).items():
                    obj = getattr(cfg, section)
                    if not hasattr(obj, k):
                        raise KeyError(f"unknown {section} option: {k}")
                    setattr(obj, k, v)
        if preset:
            cfg.model = SystemModelConfig.from_preset(preset)
            for k, v in (raw.get("model") or {}).items():   # yaml wins
                setattr(cfg.model, k, v)
        return cfg

    def to_dict(self):
        return asdict(self)
