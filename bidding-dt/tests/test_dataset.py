import numpy as np
import torch

from bidding_dt.data.dataset import IGNORE, MAX_SEQ, BiddingDataset, collate
from bidding_dt.data.hands import HAND_SCALE
from bidding_dt.data.vocab import call_to_id

DEAL0_CALLS = ["1H", "P", "2H", "P", "P", "P"]  # dealer N, vuln E-W


def ds_deal0(mini_cache):
    ds = BiddingDataset(mini_cache, "train")
    ds.deals = np.array([0])
    return ds


def test_hero_dealer_view(mini_cache):
    ds = ds_deal0(mini_cache)
    tokens, hand, vuln_cls, targets, dnos, seq_len = ds[0]  # hero N, dealer N
    assert seq_len == 2 + 6
    call_ids = [call_to_id(c) for c in DEAL0_CALLS]
    assert tokens[2:8].tolist() == call_ids
    # hero (dealer) decisions at positions 2 and 6 -> targets at 1 and 5
    # auction: N:1H E:P S:2H W:P N:P E:P
    assert targets[1] == call_to_id("1H") and dnos[1] == 0
    assert targets[5] == call_to_id("P") and dnos[5] == 1
    ignore = targets == IGNORE
    assert ignore.sum() == MAX_SEQ - 2
    assert vuln_cls == 2  # E-W vuln, hero on N-S -> they-only
    assert np.allclose(hand, np.asarray(ds.hands[0, 0], dtype=np.float32) * HAND_SCALE)


def test_rotated_view_pads(mini_cache):
    ds = ds_deal0(mini_cache)
    tokens, hand, vuln_cls, targets, dnos, seq_len = ds[1]  # hero E, dealer N
    # o=1 -> 3 leading pads, calls start at position 5
    assert seq_len == 2 + 3 + 6
    assert tokens[2:5].tolist() == [0, 0, 0]
    assert tokens[5:11].tolist() == [call_to_id(c) for c in DEAL0_CALLS]
    # E called P (index 1 -> pos 6) and P (index 5 -> pos 10)
    assert targets[5] == call_to_id("P") and dnos[5] == 0
    assert targets[9] == call_to_id("P") and dnos[9] == 1
    assert (targets != IGNORE).sum() == 2
    assert vuln_cls == 1  # E hero, E-W vuln -> we-only


def test_hero_positions_always_aligned(mini_cache):
    ds = BiddingDataset(mini_cache, "train")
    ds.deals = np.arange(4)
    for i in range(len(ds)):
        tokens, _, _, targets, dnos, seq_len = ds[i]
        idx = np.flatnonzero(targets != IGNORE)
        assert len(idx) > 0
        # hero call positions are 2 mod 4 -> target indices are 1 mod 4
        assert all(t % 4 == 1 for t in idx)
        # targets equal the token at the next position (teacher forcing)
        for t in idx:
            assert targets[t] == tokens[t + 1]
        assert int(dnos[idx].min()) == 0
        assert np.array_equal(dnos[idx], np.arange(len(idx)))
        assert seq_len <= MAX_SEQ


def test_w_hero_single_decision(mini_cache):
    ds = ds_deal0(mini_cache)
    tokens, _, vuln_cls, targets, dnos, seq_len = ds[3]  # hero W, dealer N
    # o=3 -> 1 pad; W called once (P at auction index 3 -> pos 6)
    assert seq_len == 2 + 1 + 6
    assert tokens[3] == call_to_id("1H")
    assert targets[5] == call_to_id("P")
    assert (targets != IGNORE).sum() == 1
    assert vuln_cls == 1  # W hero on E-W side, E-W vuln -> we-only


def test_collate(mini_cache):
    ds = BiddingDataset(mini_cache, "train")
    ds.deals = np.arange(4)
    batch = collate([ds[i] for i in range(8)])
    maxlen = max(int(ds[i][5]) for i in range(8))
    assert batch["tokens"].shape == (8, maxlen)
    assert batch["targets"].shape == (8, maxlen)
    assert batch["hand"].shape == (8, 52)
    assert batch["tokens"].dtype == torch.long
    assert batch["targets"].dtype == torch.long
    # padded tail of each row is IGNORE
    for k, (_, _, _, _, _, sl) in enumerate([ds[i] for i in range(8)]):
        if int(sl) < maxlen:
            assert (batch["targets"][k, int(sl) - 1:] == IGNORE).all()
