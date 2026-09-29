import numpy as np
import torch

from bidding_dt.config import PRESETS, ModelConfig
from bidding_dt.data.dataset import IGNORE, MAX_SEQ, BiddingDataset, collate
from bidding_dt.model.transformer import build_model


def test_param_counts():
    n = build_model(ModelConfig.from_preset("small")).num_params()
    assert 1.9e6 < n < 2.6e6, n
    assert build_model(ModelConfig.from_preset("tiny")).num_params() < 1.2e6
    assert 5e6 < build_model(ModelConfig.from_preset("base")).num_params() < 8e6
    assert 15e6 < build_model(ModelConfig.from_preset("large")).num_params() < 22e6


def test_forward_shapes():
    torch.manual_seed(0)
    model = build_model(ModelConfig.from_preset("tiny"))
    B = 3
    tokens = torch.randint(0, 39, (B, MAX_SEQ))
    hand = torch.rand(B, 52)
    vuln = torch.randint(0, 4, (B,))
    logits = model(tokens, hand, vuln)
    assert logits.shape == (B, MAX_SEQ, 39)
    targets = torch.full((B, MAX_SEQ), IGNORE, dtype=torch.long)
    targets[0, 5] = 7
    logits, loss = model(tokens, hand, vuln, targets)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_variable_length_forward():
    model = build_model(ModelConfig.from_preset("tiny"))
    for L in (6, 12, MAX_SEQ):
        logits = model(torch.zeros(2, L, dtype=torch.long),
                       torch.zeros(2, 52), torch.zeros(2, dtype=torch.long))
        assert logits.shape == (2, L, 39)


def test_causality_and_cond_hand_positions():
    torch.manual_seed(0)
    model = build_model(ModelConfig.from_preset("tiny")).eval()
    ds_tokens = torch.randint(0, 39, (2, MAX_SEQ))
    hand = torch.rand(2, 52)
    vuln = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        base = model(ds_tokens, hand, vuln)
        # tokens at positions 0/1 are placeholders (COND/HAND built separately)
        t2 = ds_tokens.clone()
        t2[:, :2] = (t2[:, :2] + 1) % 39
        assert torch.allclose(base, model(t2, hand, vuln), atol=1e-6)
        # changing the final token cannot affect any earlier logits
        t3 = ds_tokens.clone()
        t3[:, -1] = (t3[:, -1] + 5) % 39
        out3 = model(t3, hand, vuln)
        assert torch.allclose(base[:, :-1], out3[:, :-1], atol=1e-6)


def test_overfit_one_batch(mini_cache):
    torch.manual_seed(0)
    np.random.seed(0)
    ds = BiddingDataset(mini_cache, "train")
    ds.deals = np.arange(4)
    batch = collate([ds[i] for i in range(len(ds))])  # 16 samples
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=160)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(300):
        _, loss = model(batch["tokens"], batch["hand"], batch["vuln"], batch["targets"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.15, loss.item()
    with torch.no_grad():
        logits = model(batch["tokens"], batch["hand"], batch["vuln"])
        t = batch["targets"][:, :-1]
        m = t != IGNORE
        acc = (logits[:, :-1][m].argmax(-1) == t[m]).float().mean()
    assert acc >= 0.8, acc
