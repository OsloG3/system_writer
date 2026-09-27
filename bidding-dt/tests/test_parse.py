import json

import numpy as np

from bidding_dt.data.parse import MAX_CALLS, build_cache, parse_file
from bidding_dt.data.vocab import call_to_id


def test_parse_file(mini_file):
    deals = list(parse_file(mini_file))
    assert len(deals) == 4
    hands, calls, dealer, vuln = deals[0]
    assert hands.shape == (4, 52)
    assert dealer == 0          # N
    assert vuln == 2            # E-W
    assert calls.tolist() == [call_to_id(c) for c in ["1H", "P", "2H", "P", "P", "P"]]

    hands, calls, dealer, vuln = deals[1]
    assert dealer == 1          # E
    assert vuln == 1            # N-S
    assert len(calls) == 12


def test_build_cache(mini_cache):
    meta = json.loads((mini_cache / "meta.json").read_text())
    assert meta["num_deals"] == 4
    assert meta["total_calls"] == 6 + 12 + 10 + 6

    hands = np.load(mini_cache / "hands.npy")
    calls = np.load(mini_cache / "calls.npy")
    lengths = np.load(mini_cache / "lengths.npy")
    assert hands.shape == (4, 4, 52) and hands.dtype == np.uint8
    assert calls.shape == (4, MAX_CALLS)
    assert lengths.tolist() == [6, 12, 10, 6]
    # padding beyond length is PAD (0)
    assert calls[0, 6:].sum() == 0

    splits = [np.load(mini_cache / f"{s}.npy") for s in ("train", "val", "test")]
    all_idx = np.sort(np.concatenate(splits))
    assert all_idx.tolist() == [0, 1, 2, 3]
