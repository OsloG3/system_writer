import numpy as np
import pytest
import torch
from pathlib import Path

from bidding_dt.data.hands import encode_hand
from bidding_dt.env.deals import random_deals
from bidding_dt.rl.analyze import (bucket_name, hcp_bucket, human_openings,
                                    system_profile, total_variation)
from bidding_dt.rl.model import RLModel
from bidding_dt.config import ModelConfig


def test_hcp_bucket():
    # AKQJ of spades + blanks: 4+3+2+1 = 10 HCP -> bucket "10-12"
    h = encode_hand("AKQJ.432.432.432")
    assert hcp_bucket(h) == 2
    assert bucket_name(2) == "10-12"
    # four aces = 16 HCP -> bucket "16-18"
    v = np.zeros(52, dtype=np.uint8)
    v[[0, 13, 26, 39]] = 1  # A of each suit (4 suits x 13 ranks)
    assert hcp_bucket(v) == 4
    assert bucket_name(5) == "19+"
    # empty hand -> 0 HCP
    assert hcp_bucket(np.zeros(52, dtype=np.uint8)) == 0


def test_total_variation():
    assert total_variation({"P": 1.0}, {"P": 1.0}) == 0.0
    assert total_variation({"P": 1.0}, {"1C": 1.0}) == 1.0
    assert abs(total_variation({"P": 0.5, "1C": 0.5}, {"P": 1.0}) - 0.5) < 1e-9


def test_system_profile_legal_and_normalized():
    torch.manual_seed(0)
    model = RLModel(ModelConfig.from_preset("tiny")).eval()
    rng = np.random.default_rng(1)
    deals = random_deals(rng, 20)
    encoded = np.concatenate([d.encoded for d in deals], axis=0)
    prof = system_profile(model, encoded, torch.device("cpu"))
    assert "openings" in prof and "responses" in prof
    for dists in [prof["openings"], *prof["responses"].values()]:
        for bucket, d in dists.items():
            assert abs(sum(d.values()) - 1.0) < 1e-3, (bucket, d)
            # openings: X/XX are never legal; argmax must respect masks
            assert "X" not in d and "XX" not in d
    for ctx, dists in prof["responses"].items():
        for bucket, d in dists.items():
            assert "XX" not in d


@pytest.mark.skipif(not Path("cache/hands.npy").exists(),
                    reason="full corpus cache not built")
def test_human_openings():
    prof = human_openings(Path("cache"), max_deals=4000)
    assert prof, "expected openings in at least one bucket"
    for bucket, d in prof.items():
        assert abs(sum(d.values()) - 1.0) < 1e-3
    # humans pass nearly always with 0-5 HCP
    if "0-5" in prof:
        assert prof["0-5"].get("P", 0.0) > 0.9
