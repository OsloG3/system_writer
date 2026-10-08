"""Population of bidding systems: (frozen encoder, mutable adapter) variants.

A *system* is what a partnership brings to the table: a hand-inference encoder
(the frozen `hand/model.py` HandVQ whose z blocks feed the policy) plus the
small adapter that turns those z's into state features. The shared FFN policy
is trained by PPO in the inner loop; the population of adapters is searched by
the outer loop:

    evaluate every system  ->  keep the elites  ->  replace the weakest with
    mutated children of the strongest:   child = parent + sigma * N(0, 1)

Children inherit their parent's encoder, so an encoder that keeps losing its
adapters' evaluations dies out -- that is the "select encoder variants" half of
the search, the mutation half being the Gaussian step on the adapter weights.
Strong systems are cloned (frozen, single adapter) into the league, so later
generations play against the systems that already worked.

Adapters live in `SystemPolicy.adapters` (one slot per system) so they are
covered by the optimizer, grad clipping and the policy checkpoint; this class
only holds the metadata and the search operators.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .config import EvoConfig
from .model import SystemPolicy

VARIANTS_FILE = "variants.json"


@dataclass
class SystemVariant:
    name: str
    encoder: str
    adapter: int               # index into SystemPolicy.adapters
    gen: int = 0
    parent: str | None = None
    iters: int = 0             # PPO iterations played with this system active
    train_imps: float = 0.0    # EWMA of team IMPs/board while active
    score: float = float("nan")  # last evaluation IMPs/board (nan = unevaluated)
    evals: int = 0
    wins: float = 0.5          # EWMA of per-deal winrate in evaluations

    def to_json(self) -> dict:
        d = asdict(self)
        if d["score"] != d["score"]:            # NaN is not valid JSON
            d["score"] = None
        return d

    @classmethod
    def from_json(cls, d: dict) -> "SystemVariant":
        d = dict(d)
        if d.get("score") is None:
            d["score"] = float("nan")
        return cls(**d)


@torch.no_grad()
def mutate(dst: torch.nn.Module, src: torch.nn.Module, sigma: float,
           relative: bool = False) -> float:
    """`dst = src + sigma * N(0,1)` (per-weight std scaling when relative).

    Returns the perturbation's L2 size relative to the parent's, which is what
    makes sigma comparable across adapter shapes and initializations.
    """
    n_params = 0
    src_norm = 0.0
    for d, s in zip(dst.parameters(), src.parameters()):
        noise = torch.randn_like(s)
        if relative:
            noise = noise * s.std()
        d.copy_(s + sigma * noise)
        n_params += s.numel()
        src_norm += float(s.pow(2).sum())
    for d, s in zip(dst.buffers(), src.buffers()):
        d.copy_(s)
    return float(sigma * math.sqrt(n_params) / math.sqrt(max(src_norm, 1e-12)))


class Population:
    def __init__(self, policy: SystemPolicy, cfg: EvoConfig, encoders: list,
                 seed: int | None = None, sync_adapters: bool = True):
        if not encoders:
            raise ValueError("population needs at least one encoder")
        if cfg.systems != len(policy.adapters):
            raise ValueError(f"policy has {len(policy.adapters)} adapter slots, "
                             f"evo.systems is {cfg.systems}")
        self.policy = policy
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed if seed is None else seed)
        # encoders are dealt round-robin so a pool of encoder checkpoints is
        # represented from generation 0; adapters all start from one draw, so
        # the first ranking is pure evaluation, not initialization luck
        self.variants = [SystemVariant(name=f"sys{i}",
                                       encoder=encoders[i % len(encoders)],
                                       adapter=i)
                         for i in range(cfg.systems)]
        if sync_adapters:
            with torch.no_grad():
                for v in self.variants[1:]:
                    for d, s in zip(policy.adapters[v.adapter].parameters(),
                                    policy.adapters[0].parameters()):
                        d.copy_(s)

    # -- access --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.variants)

    def names(self) -> list:
        return [v.name for v in self.variants]

    def get(self, name: str):
        return next((v for v in self.variants if v.name == name), None)

    def activate(self, v: SystemVariant):
        """Point the policy at this system (encoder + adapter slot)."""
        self.policy.set_system(v.encoder, v.adapter)

    def sample(self, rng: np.random.Generator | None = None) -> SystemVariant:
        """Draw the system N-S plays this iteration."""
        rng = rng or self.rng
        if self.cfg.sample_by == "fitness" and any(v.evals for v in self.variants):
            score = np.array([v.score if v.evals else 0.0 for v in self.variants])
            w = np.exp((score - score.max()) / max(self.cfg.fitness_temp, 1e-6))
            return self.variants[int(rng.choice(len(self.variants),
                                                p=w / w.sum()))]
        return self.variants[int(rng.integers(len(self.variants)))]

    def observe(self, v: SystemVariant, team_imps, ema: float | None = None):
        """Fold one inner iteration's result into the system's running stats."""
        ema = self.cfg.ema if ema is None else ema
        x = float(np.mean(team_imps))
        v.iters += 1
        v.train_imps += ema * (x - v.train_imps)

    def score(self, v: SystemVariant, team_imps):
        """Record an evaluation result (IMPs/board, per-deal winrate)."""
        arr = np.asarray(team_imps, dtype=np.float64)
        v.score = float(arr.mean())
        v.wins += self.cfg.ema * (float((arr > 0).mean()) - v.wins)
        v.evals += 1

    def ranked(self) -> list:
        """Best first; unevaluated systems sort last."""
        return sorted(self.variants,
                      key=lambda v: v.score if v.evals else float("-inf"),
                      reverse=True)

    def best(self, n: int = 1) -> list:
        return [v for v in self.ranked()[:n] if v.evals]

    # -- outer-loop operators --------------------------------------------------

    def evolve(self) -> list:
        """Replace the weakest systems with mutated children of the elites."""
        cfg = self.cfg
        ranked = self.ranked()
        elites = [v for v in ranked[:max(1, cfg.keep_best)] if v.evals]
        if not elites:
            return []                                  # nothing to breed from
        n_replace = max(1, int(round(len(self.variants) * cfg.replace_frac)))
        n_replace = min(n_replace, len(self.variants) - len(elites))
        bred = []
        for slot, victim in enumerate(ranked[len(ranked) - n_replace:]):
            if any(victim is e for e in elites):
                continue
            parent = elites[slot % len(elites)]
            rel = mutate(self.policy.adapters[victim.adapter],
                         self.policy.adapters[parent.adapter],
                         cfg.sigma, cfg.sigma_relative)
            victim.gen = parent.gen + 1
            victim.parent = parent.name
            victim.encoder = parent.encoder            # encoders are selected,
            victim.iters = 0                           # not mutated
            victim.train_imps = 0.0
            victim.score = float("nan")
            victim.evals = 0
            bred.append({"child": victim.name, "parent": parent.name,
                         "gen": victim.gen, "encoder": victim.encoder,
                         "sigma": cfg.sigma, "rel_noise": round(rel, 4)})
        return bred

    def clone_system(self, v: SystemVariant) -> SystemPolicy:
        """Frozen single-adapter copy for the league (shares the encoder)."""
        src = self.policy
        model = SystemPolicy(src.cfg, src.z_in, n_systems=1)
        state = {k: t for k, t in src.state_dict().items()
                 if not k.startswith("adapters.")}
        prefix = f"adapters.{v.adapter}."
        for k, t in src.state_dict().items():
            if k.startswith(prefix):
                state["adapters.0." + k[len(prefix):]] = t
        model.load_state_dict(state)
        model.add_encoder(v.encoder, src._encoders[v.encoder])
        model.set_system(v.encoder, 0)
        return model.freeze()

    # -- persistence -----------------------------------------------------------

    def save(self, directory):
        """Write the variant metadata (adapter weights ride in the policy ckpt)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / VARIANTS_FILE).write_text(json.dumps(
            {"cfg": asdict(self.cfg),
             "variants": [v.to_json() for v in self.variants]}, indent=1))

    @classmethod
    def load(cls, directory, policy: SystemPolicy, cfg: EvoConfig,
             encoders: list) -> "Population":
        """Restore metadata over an already-loaded policy (adapters included)."""
        raw = json.loads((Path(directory) / VARIANTS_FILE).read_text())
        # sync_adapters=False: the policy checkpoint already carries the
        # evolved adapter weights, they must not be reset to slot 0
        pop = cls(policy, cfg, encoders, sync_adapters=False)
        loaded = [SystemVariant.from_json(d) for d in raw["variants"]]
        if len(loaded) == len(pop.variants):
            pop.variants = loaded
        return pop
