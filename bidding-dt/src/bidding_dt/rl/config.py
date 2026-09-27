"""RL training config (model + rl sections), YAML-loadable like config.py."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ..config import PRESETS, ModelConfig


@dataclass
class RLConfig:
    # rollout
    iters: int = 500
    deals_per_iter: int = 256
    reward_scale: float = 0.2      # IMPs -> training units (+-24 IMP -> +-4.8)
    gamma: float = 1.0             # retained for signature compat; team reward
    lam: float = 0.95              # uses per-decision credit (see RolloutBuffer
                                   # .finish), so gamma/lam do not chain rewards
    greedy: bool = False           # sample during training rollouts
    # team (two-table) reward: r = team_weight * gated IMP swing between the
    # tables + par_weight * par-diff IMPs. The swing only reaches calls at/
    # after the first divergence of the two auctions.
    team_weight: float = 1.0
    par_weight: float = 0.25       # par term = 1/4 of the per-IMP team reward
    rollout_temp: float = 1.0      # >1 flattens rollout sampling so self-play
                                   # tables diverge (PPO recomputes logprobs
                                   # at the same temperature)
    # ppo
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_iters: int = 20
    min_lr_frac: float = 0.1
    grad_clip: float = 1.0
    clip_eps: float = 0.2
    ppo_epochs: int = 3
    minibatch_size: int = 512
    vf_coef: float = 0.5
    # length bucketing: rows are sorted by auction length within random
    # groups of length_bucket * minibatch_size so each minibatch's token
    # block can be truncated to its longest row (drops padded-column work
    # from every PPO forward/backward; 0 = purely random minibatches)
    length_bucket: int = 8
    # anchors (annealed linearly from *_start to *_end over iters)
    ent_coef: float = 0.01
    ent_coef_end: float = 0.0
    kl_beta: float = 0.1           # KL(pi || pi_BC) penalty; 0 disables
    kl_beta_end: float = 0.0
    # league: with prob league_prob the opposing pair is drawn from the
    # league -- frozen snapshots of the learner (added every snapshot_every
    # iters, capped at max_snapshots) plus external members: any BC (SL) or
    # RL checkpoint, or the random baseline. Specs: "random" | "bc:<path>"
    # | "rl:<path>" | "<path>" (kind + arch auto-detected). The warm-start
    # checkpoint joins automatically unless league_include_init is false.
    league_prob: float = 0.5
    snapshot_every: int = 25
    max_snapshots: int = 16
    league_members: list = field(default_factory=list)
    league_include_init: bool = True
    member_weight: float = 1.0     # base sampling weight per external member
    snapshot_weight: float = 1.0   # base sampling weight per snapshot
    league_persist: bool = True    # keep the league on disk under <out>/league
                                   # so --resume restores it (snapshots + stats)
    pfsp_alpha: float = 0.0        # >0: sample ∝ weight * (1-winrate)^alpha
                                   # (prioritize opponents that beat us)
    # eval / bookkeeping
    eval_every: int = 10
    eval_deals: int = 256
    log_every: int = 1
    seed: int = 1337
    device: str | None = None
    cache_path: str = "cache/dd.sqlite"
    # multi-core CPU: torch intra-op threads (0 = torch default). Capping
    # this (e.g. cores/2) leaves headroom for the background DD presolve
    # thread and libdds' own solver threads.
    threads: int = 0
    # bf16 autocast for the CPU rollout/update forwards: ~2x matmul speed on
    # AVX512-BF16/AMX cores (Zen4+, Intel Ice Lake+); slower on older CPUs.
    cpu_bf16: bool = False


@dataclass
class RLTrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    @classmethod
    def load(cls, path: str | Path | None = None,
             preset: str | None = None) -> "RLTrainConfig":
        cfg = cls()
        raw = {}
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text()) or {}
            for section in ("model", "rl"):
                for k, v in raw.get(section, {}).items():
                    obj = getattr(cfg, section)
                    if not hasattr(obj, k):
                        raise KeyError(f"unknown {section} option: {k}")
                    setattr(obj, k, v)
        if preset:
            if preset not in PRESETS:
                raise ValueError(f"unknown preset: {preset}")
            cfg.model = ModelConfig.from_preset(preset)
            for k, v in (raw.get("model") or {}).items():
                setattr(cfg.model, k, v)
        return cfg

    def to_dict(self):
        return asdict(self)
