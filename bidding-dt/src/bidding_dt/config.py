"""Config dataclasses + YAML loading."""

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from .data.dataset import MAX_SEQ, N_ROLES
from .data.hands import HAND_DIM
from .data.vocab import VOCAB_SIZE

PRESETS = {
    "tiny":  dict(d_model=128, n_layers=4, n_heads=4, d_ff=344),
    "small": dict(d_model=176, n_layers=6, n_heads=8, d_ff=448),
    "base":  dict(d_model=256, n_layers=8, n_heads=8, d_ff=680),
    "large": dict(d_model=384, n_layers=10, n_heads=8, d_ff=1024),
}


@dataclass
class ModelConfig:
    d_model: int = 176
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 448
    dropout: float = 0.0
    vocab_size: int = VOCAB_SIZE
    hand_dim: int = HAND_DIM
    max_seq_len: int = MAX_SEQ
    n_vuln: int = 4
    n_roles: int = N_ROLES

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "ModelConfig":
        return cls(**{**PRESETS[name], **overrides})


@dataclass
class TrainConfig:
    batch_size: int = 512
    lr: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 1000
    epochs: int = 1
    max_steps: int | None = None
    min_lr_frac: float = 0.1
    grad_clip: float = 1.0
    bf16: bool = True          # only active on CUDA
    compile: bool = False
    seed: int = 1337
    log_every: int = 100
    val_every: int = 2000
    val_batches: int | None = None  # None = full val split
    device: str | None = None       # None = auto


@dataclass
class DataConfig:
    cache_dir: str = "cache"
    num_workers: int = 4
    max_deals: int | None = None    # subsample train split (smoke runs)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def load(cls, path: str | Path | None = None, preset: str | None = None) -> "Config":
        cfg = cls()
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text()) or {}
            for section in ("model", "train", "data"):
                for k, v in raw.get(section, {}).items():
                    obj = getattr(cfg, section)
                    if not hasattr(obj, k):
                        raise KeyError(f"unknown {section} option: {k}")
                    setattr(obj, k, v)
        if preset:
            cfg.model = ModelConfig.from_preset(preset)
            if path is not None:  # yaml model opts still win over preset
                for k, v in (raw.get("model") or {}).items():
                    setattr(cfg.model, k, v)
        return cfg

    def to_dict(self):
        return asdict(self)
