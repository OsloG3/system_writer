"""Hand-inference configs (model / train / gen / data), YAML-loadable.

The model presets reuse config.PRESETS' widths (d_model, n_heads, d_ff) so a
`small` hand model costs about what a `small` bidder costs; the layer count is
split into encoder / slot-quantizer / decoder stacks.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ..config import PRESETS
from ..data.dataset import MAX_SEQ, N_ROLES
from ..data.hands import HAND_DIM, HAND_LEN
from ..data.vocab import VOCAB_SIZE

# encoder / VQ-slot / decoder layer counts per preset
LAYER_SPLITS = {
    "tiny": (4, 2, 4),
    "small": (6, 2, 6),
    "base": (8, 3, 8),
    "large": (10, 3, 10),
}


@dataclass
class HandVQConfig:
    d_model: int = 176
    d_ff: int = 448
    n_heads: int = 8
    n_enc_layers: int = 6
    n_slot_layers: int = 2
    n_dec_layers: int = 6
    dropout: float = 0.0
    # VQ bottleneck: z is n_codes codebook vectors out of codebook_size each
    n_codes: int = 16
    codebook_size: int = 1024
    code_dim: int = 64           # width of a code (projected from d_model)
    commit_beta: float = 0.25    # weight of the commitment term
    codebook_weight: float = 1.0
    usage_decay: float = 0.99    # EMA over per-code usage (perplexity, revival)
    revive_frac: float = 0.02    # codes won < frac/codebook_size of the time die
    revive_every: int = 200      # steps between dead-code reseeds (0 = never)
    train_code_temp: float = 0.0  # >0: sample near-tied codes while training
    # shapes / vocabularies
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ   # [COND, KNOWN, <=3 frame pads, <=36 calls]
    n_vuln: int = 4
    n_roles: int = N_ROLES
    n_cards: int = HAND_DIM
    hand_len: int = HAND_LEN

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "HandVQConfig":
        if name not in PRESETS or name not in LAYER_SPLITS:
            raise ValueError(f"unknown preset: {name}")
        enc, slots, dec = LAYER_SPLITS[name]
        wide = {k: v for k, v in PRESETS[name].items() if k != "n_layers"}
        return cls(**wide, n_enc_layers=enc, n_slot_layers=slots,
                   n_dec_layers=dec, **overrides)


@dataclass
class HandTrainConfig:
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 1000
    epochs: int = 1
    max_steps: int | None = None
    min_lr_frac: float = 0.1
    grad_clip: float = 1.0
    bf16: bool = True            # only active on CUDA (gen.cpu_bf16 for CPU)
    compile: bool = False
    seed: int = 1337
    log_every: int = 100
    val_every: int = 1000
    val_batches: int | None = None
    val_samples: int = 8         # hands drawn per row for the diversity metric
    device: str | None = None
    # prefix views: "random" draws one auction prefix per (deal, seat) row per
    # epoch, "all" enumerates every prefix (dataset grows ~10x)
    prefix: str = "random"
    # known-cards masking augmentation: with this probability a row also gets a
    # random exclusion mask (cards drawn from the 39 the target does not hold),
    # which is what teaches the model to use a caller-supplied mask at inference
    mask_aug: float = 0.5
    mask_cards: int = 13


@dataclass
class GenConfig:
    """Synthetic auction generation (hand/gen.py) -- the training data source."""
    deals: int = 100_000         # deals to generate (0 = use data.sources only)
    policies: list = field(default_factory=list)  # specs: random | bc:<p> | rl:<p> | <p>
    temp: float = 1.0            # rollout sampling temperature
    greedy: bool = False
    chunk: int = 256             # auctions rolled out per batch
    val_frac: float = 0.01
    test_frac: float = 0.01
    seed: int = 4242
    reuse: bool = False          # keep an existing store instead of regenerating
    dir: str = "gen"             # store dir, relative to the run's --out
    threads: int = 0             # torch CPU intra-op threads (0 = default)
    cpu_bf16: bool = False       # bf16 autocast for the rollout forwards


@dataclass
class HandDataConfig:
    sources: list = field(default_factory=list)  # extra auction stores to mix in
    num_workers: int = 4
    max_deals: int | None = None   # subsample every source (smoke runs)


@dataclass
class HandConfig:
    model: HandVQConfig = field(default_factory=HandVQConfig)
    train: HandTrainConfig = field(default_factory=HandTrainConfig)
    gen: GenConfig = field(default_factory=GenConfig)
    data: HandDataConfig = field(default_factory=HandDataConfig)

    @classmethod
    def load(cls, path: str | Path | None = None,
             preset: str | None = None) -> "HandConfig":
        cfg = cls()
        raw = {}
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text()) or {}
            for section in ("model", "train", "gen", "data"):
                for k, v in (raw.get(section) or {}).items():
                    obj = getattr(cfg, section)
                    if not hasattr(obj, k):
                        raise KeyError(f"unknown {section} option: {k}")
                    setattr(obj, k, v)
        if preset:
            cfg.model = HandVQConfig.from_preset(preset)
            for k, v in (raw.get("model") or {}).items():  # yaml wins over preset
                setattr(cfg.model, k, v)
        return cfg

    def to_dict(self):
        return asdict(self)
