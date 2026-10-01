import json

import numpy as np
import torch

from bidding_dt.config import ModelConfig
from bidding_dt.data.hands import encode_hand
from bidding_dt.data.vocab import call_to_id
from bidding_dt.env.auction_env import AuctionBatch
from bidding_dt.env.deals import random_deals
from bidding_dt.rl.analyze import (build_tree, hand_hcp, node_to_json,
                                   selfplay_tree, suit_lengths)
from bidding_dt.rl.model import RLModel
from bidding_dt.rl.rollout import RolloutBuffer


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return RLModel(ModelConfig.from_preset("tiny"))


def test_hand_hcp_and_suit_lengths():
    h = encode_hand("AKQJ.432.432.432")
    assert hand_hcp(h) == 10                      # 4+3+2+1
    assert suit_lengths(h).tolist() == [4, 3, 3, 3]
    v = np.zeros(52, dtype=np.uint8)
    v[[0, 13, 26, 39]] = 1                        # the four aces
    assert hand_hcp(v) == 16
    assert suit_lengths(v).tolist() == [1, 1, 1, 1]
    assert hand_hcp(np.zeros(52, dtype=np.uint8)) == 0


def _fake_row(buf, d, s, k, a):
    buf.deal_idx.append(d)
    buf.seat.append(s)
    buf.call_idx.append(k)
    buf.action.append(a)


def test_build_tree_merges_prefixes_and_ranges():
    """Two deals sharing an opening merge into one prefix node; each node's
    ranges are the exact min/max over the hands that bid it there."""
    rng = np.random.default_rng(3)
    deals = random_deals(rng, 2)
    env = AuctionBatch(deals, np.zeros(2, np.int64), np.zeros(2, np.int64))
    c1c, cp, h1 = call_to_id("1C"), call_to_id("P"), call_to_id("1H")
    env.calls[0, :2] = [c1c, cp]
    env.n_calls[0] = 2
    env.calls[1, :2] = [c1c, h1]
    env.n_calls[1] = 2
    buf = RolloutBuffer()
    _fake_row(buf, 0, 0, 0, c1c)      # dealer N opens 1C in both deals
    _fake_row(buf, 1, 0, 0, c1c)
    _fake_row(buf, 0, 1, 1, cp)       # E passes in deal 0, bids 1H in deal 1
    _fake_row(buf, 1, 1, 1, h1)
    tree = build_tree(deals, env, buf)

    assert tree["n"] == 2 and set(tree["children"]) == {c1c}
    n1c = tree["children"][c1c]
    assert n1c["n"] == 2 and set(n1c["children"]) == {cp, h1}
    assert n1c["children"][cp]["n"] == 1 and n1c["children"][h1]["n"] == 1
    # the 1C node pools the two opening hands exactly
    n0, n1 = deals[0].encoded[0], deals[1].encoded[0]
    assert n1c["hcp_min"] == min(hand_hcp(n0), hand_hcp(n1))
    assert n1c["hcp_max"] == max(hand_hcp(n0), hand_hcp(n1))
    assert n1c["suit_min"] == np.minimum(suit_lengths(n0), suit_lengths(n1)).tolist()
    assert n1c["suit_max"] == np.maximum(suit_lengths(n0), suit_lengths(n1)).tolist()
    # the 1H-after-1C node saw only deal 1's East hand
    nh1 = n1c["children"][h1]
    assert nh1["hcp_min"] == nh1["hcp_max"] == hand_hcp(deals[1].encoded[1])
    assert nh1["suit_min"] == nh1["suit_max"] == suit_lengths(deals[1].encoded[1]).tolist()

    js = node_to_json(tree)
    assert js["call"] is None and js["hcp"] is None and js["suits"] is None
    assert set(js["children"]) == {"1C"}
    kid = js["children"]["1C"]
    assert kid["call"] == "1C" and kid["n"] == 2
    assert kid["hcp"] == [n1c["hcp_min"], n1c["hcp_max"]]
    assert set(kid["suits"]) == {"S", "H", "D", "C"}
    assert set(kid["children"]) == {"P", "1H"}
    json.dumps(js)                             # fully serializable


def test_selfplay_tree_covers_every_decision():
    """End-to-end self-play: every auction is a path in the tree, every
    recorded decision's hand lies inside its node's ranges, and greedy
    replays are deterministic."""
    model = tiny_model(seed=5).eval()
    rng = np.random.default_rng(7)
    deals = random_deals(rng, 8)
    dealer = rng.integers(0, 4, 8)
    vuln = rng.integers(0, 4, 8)
    torch.manual_seed(11)
    tree, env, buf = selfplay_tree(model, deals, dealer, vuln,
                                   torch.device("cpu"), temp=1.2)
    assert env.done.all()
    assert tree["n"] == 8
    # exactly one opening bid per deal
    assert sum(c["n"] for c in tree["children"].values()) == 8
    # each finished auction is a full path through the tree
    for d in range(8):
        node = tree
        for k in range(int(env.n_calls[d])):
            node = node["children"][int(env.calls[d, k])]
            assert node["n"] >= 1
    # every decision's hand stats are contained in (and helped form) its node
    for d_i, seat, k, act in zip(buf.deal_idx, buf.seat, buf.call_idx,
                                 buf.action):
        node = tree
        for c in env.calls[d_i, :k]:
            node = node["children"][int(c)]
        node = node["children"][int(act)]
        enc = deals[d_i].encoded[seat]
        hcp = hand_hcp(enc)
        lens = suit_lengths(enc)
        assert node["hcp_min"] <= hcp <= node["hcp_max"]
        assert all(node["suit_min"][i] <= int(lens[i]) <= node["suit_max"][i]
                   for i in range(4))
    # greedy self-play is deterministic: identical trees across replays
    t2, _, _ = selfplay_tree(model, deals, dealer, vuln, torch.device("cpu"),
                             greedy=True)
    t3, _, _ = selfplay_tree(model, deals, dealer, vuln, torch.device("cpu"),
                             greedy=True)
    assert (json.dumps(node_to_json(t2), sort_keys=True)
            == json.dumps(node_to_json(t3), sort_keys=True))
