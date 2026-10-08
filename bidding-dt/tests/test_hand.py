import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from bidding_dt.data.dataset import MAX_SEQ, BiddingDataset
from bidding_dt.data.parse import MAX_CALLS
from bidding_dt.data.hands import (CARD_TO_COUNT, COUNT_TO_CARD, HAND_DIM,
                                   cards_to_str, encode_hand, hand_cards, hcp,
                                   indicator_from_cards, str_to_cards,
                                   suit_lengths)
from bidding_dt.env.deals import random_deals
from bidding_dt.hand.config import HandVQConfig
from bidding_dt.hand.data import AuctionStore, HandDataset, collate, rotated_tokens
from bidding_dt.hand.gen import generate, policy_pool, split_weight
from bidding_dt.hand.model import (VectorQuantizer, build_hand_model,
                                   code_perplexity, top_p_filter)
from bidding_dt.hand.train import evaluate, indicator, param_groups

HANDS = ["AT7.KT943.A42.K8", "J6543.75.K8.AQJT", "K982.AQ6.T73.952",
         "Q.J82.QJ965.7643"]


def tiny_cfg(**kw):
    base = dict(d_model=64, d_ff=160, n_heads=4, n_enc_layers=2,
                n_slot_layers=1, n_dec_layers=2, n_codes=4, codebook_size=32,
                code_dim=16, revive_every=0)
    return HandVQConfig(**{**base, **kw})


def toy_batch(B=4, L=9, seed=0, excl_rows=0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, 39, (B, L), generator=g)
    tokens[:, :2] = 0
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, L - 2:] = False
    vuln = torch.randint(0, 4, (B,), generator=g)
    cards = torch.stack([torch.from_numpy(hand_cards(encode_hand(h)))
                         for h in HANDS[:B]])
    excl = torch.zeros(B, HAND_DIM)
    for r in range(excl_rows):
        free = np.flatnonzero(encode_hand(HANDS[r]) == 0)
        excl[r, free[:13]] = 1.0
    return {"tokens": tokens, "mask": mask, "vuln": vuln, "excl": excl,
            "cards": cards}


# -- card ids / hand helpers -------------------------------------------------

def test_card_id_permutations():
    assert np.array_equal(COUNT_TO_CARD[CARD_TO_COUNT], np.arange(HAND_DIM))
    assert np.array_equal(CARD_TO_COUNT[COUNT_TO_CARD], np.arange(HAND_DIM))


def test_hand_card_roundtrip():
    for h in HANDS:
        cards = hand_cards(encode_hand(h))
        assert len(cards) == 13 and len(set(cards.tolist())) == 13
        assert (np.diff(cards) > 0).all()          # canonical ascending order
        assert cards_to_str(cards) == h
        assert np.array_equal(str_to_cards(h), cards)
        assert np.array_equal(indicator_from_cards(cards), encode_hand(h))


def test_card_ids_match_deal_encoding():
    """hand_cards() on env/deals.py encodings gives that module's card ids."""
    deal = random_deals(np.random.default_rng(3), 1)[0]
    cards = np.flatnonzero(deal.cards == 1)        # seat E, card ids
    assert np.array_equal(indicator_from_cards(cards), deal.encoded[1])
    assert np.array_equal(hand_cards(deal.encoded[1]), np.sort(cards))


def test_hcp_and_shape():
    v = encode_hand("AKQ.AKQ.AKQ.AKQJ")
    assert hcp(v) == 4 * 4 + 3 * 4 + 2 * 4 + 1          # 4 aces/kings/queens, 1 jack
    assert suit_lengths(v).tolist() == [3, 3, 3, 4]
    assert hcp(encode_hand("987654.9876.98.9")) == 0
    assert suit_lengths(encode_hand("987654.9876.98.9")).tolist() == [6, 4, 2, 1]


# -- dataset views -----------------------------------------------------------

def test_rotated_tokens_match_supervised_dataset(mini_cache):
    store = AuctionStore.load(mini_cache)
    ds = BiddingDataset(mini_cache, "train")
    for i in range(len(ds)):
        tokens, hand, vuln, _tg, _dn, seq_end = ds[i]
        deal, seat = int(ds.deals[i >> 2]), i & 3
        calls = store.calls[deal, :int(store.lengths[deal])]
        mine = rotated_tokens(int(store.dealer[deal]), calls, seat)
        assert np.array_equal(mine, tokens[:seq_end])
        assert len(mine) == seq_end <= MAX_SEQ
        assert int(vuln) == BiddingDataset.vuln_class(int(store.vuln[deal]), seat)


def test_hand_dataset_all_prefixes(mini_cache):
    store = AuctionStore.load(mini_cache)
    ds = HandDataset(store, prefix="all")
    assert len(ds) == int((4 * (store.lengths[store.deals] + 1)).sum())
    # every (deal, seat, prefix length) appears exactly once
    rows = {(int(ds.deal[i]), int(ds.seat[i]), int(ds[i][4])) for i in range(len(ds))}
    assert len(rows) == len(ds)
    # the longest prefix of each (deal, seat) is the supervised row
    full = {(int(ds.deal[i]), int(ds.seat[i])): i for i in range(len(ds))
            if ds.k[i] == store.lengths[ds.deal[i]]}
    sup = BiddingDataset(mini_cache, "train")
    for i in range(len(sup)):
        deal, seat = int(sup.deals[i >> 2]), i & 3
        tokens, hand, vuln, _tg, _dn, seq_end = sup[i]
        t, cards, v, excl, L = ds[full[(deal, seat)]]
        assert L == seq_end and np.array_equal(t, tokens[:seq_end])
        assert int(v) == int(vuln) and not excl.any()
        assert np.array_equal(indicator_from_cards(cards), hand.astype(np.uint8))


def test_hand_dataset_random_prefix_and_mask_aug(mini_cache):
    store = AuctionStore.load(mini_cache)
    ds = HandDataset(store, prefix="random", mask_aug=1.0, mask_cards=13)
    assert len(ds) == 4 * len(store)
    lengths = set()
    for i in range(len(ds)):
        tokens, cards, vuln, excl, L = ds[i]
        deal, seat = int(ds.deal[i]), int(ds.seat[i])
        n = int(store.lengths[deal])
        lengths.add(L)
        p = (4 - (seat - int(store.dealer[deal])) % 4) % 4
        assert L == 2 + p + (L - 2 - p) and L - 2 - p <= n      # a real prefix
        assert (np.diff(cards) > 0).all() and len(cards) == 13
        assert np.array_equal(indicator_from_cards(cards),
                              np.asarray(store.hands[deal, seat]))
        assert int(excl.sum()) == 13                            # 13 known cards
        held = indicator_from_cards(cards).astype(bool)
        assert not (held & excl.astype(bool)).any()              # never a target
        assert np.array_equal(tokens[2 + p:], store.calls[deal, :L - 2 - p])
    assert len(lengths) > 1                                     # prefixes vary
    ds2 = HandDataset(store, prefix="random", mask_aug=1.0, mask_cards=26)
    assert int(ds2[0][3].sum()) == 26
    ds3 = HandDataset(store, prefix="random", mask_aug=0.0)
    assert not ds3[0][3].any()


def test_collate_pads_and_masks(mini_cache):
    store = AuctionStore.load(mini_cache)
    ds = HandDataset(store, prefix="all")
    batch = collate([ds[i] for i in range(8)])
    assert batch["tokens"].shape == batch["mask"].shape
    assert batch["cards"].shape == (8, 13)
    assert batch["excl"].shape == (8, HAND_DIM)
    for r in range(8):
        n = int(batch["mask"][r].sum())
        assert batch["mask"][r, :n].all() and not batch["mask"][r, n:].any()
        assert not batch["tokens"][r, n:].any()                 # padded with PAD


# -- model -------------------------------------------------------------------

def test_forward_shapes_and_grads():
    torch.manual_seed(0)
    model = build_hand_model(tiny_cfg())
    batch = toy_batch(excl_rows=2)
    out = model(batch["tokens"], batch["vuln"], batch["cards"],
                excl=batch["excl"], mask=batch["mask"])
    assert out["logits"].shape == (4, 13, HAND_DIM)
    assert out["codes"].shape == (4, 4)
    for key in ("loss", "ce", "vq", "commit"):
        assert out[key].dim() == 0 and torch.isfinite(out[key])
    out["loss"].backward()
    # gradient reaches the encoder through the straight-through estimator, the
    # codebook, and the decoder
    assert model.enc_blocks[0].attn.q.weight.grad.abs().sum() > 0
    assert model.slot_blocks[0].cross.kv.weight.grad.abs().sum() > 0
    assert model.vq.codebook.weight.grad.abs().sum() > 0
    assert model.dec_blocks[0].attn.q.weight.grad.abs().sum() > 0


def test_padding_is_ignored():
    """Trailing PADs masked out cannot change the loss (encoder key mask)."""
    torch.manual_seed(0)
    model = build_hand_model(tiny_cfg()).eval()
    batch = toy_batch(L=9)
    with torch.no_grad():
        base = model(batch["tokens"], batch["vuln"], batch["cards"],
                     mask=batch["mask"])
        tokens = torch.cat([batch["tokens"], torch.zeros(4, 5, dtype=torch.long)], 1)
        mask = torch.cat([batch["mask"], torch.zeros(4, 5, dtype=torch.bool)], 1)
        padded = model(tokens, batch["vuln"], batch["cards"], mask=mask)
    assert torch.allclose(base["ce"], padded["ce"], atol=1e-5)


def test_max_length_auction():
    """A 36-call auction behind a 3-slot frame offset still fits the encoder."""
    torch.manual_seed(0)
    model = build_hand_model(tiny_cfg()).eval()
    cards = torch.from_numpy(hand_cards(encode_hand(HANDS[0])))[None]
    vuln = torch.zeros(1, dtype=torch.long)
    assert model.roles.numel() == model.cfg.max_seq_len == MAX_SEQ
    for pads in range(4):
        tokens = torch.zeros(1, 2 + pads + MAX_CALLS, dtype=torch.long)
        tokens[0, 2 + pads:] = torch.randint(1, 39, (MAX_CALLS,))
        with torch.no_grad():
            out = model(tokens, vuln, cards)
            drawn, _ = model.sample(tokens, vuln, n=1)
        assert torch.isfinite(out["loss"]) and drawn.shape == (1, 1, 13)
    with pytest.raises(ValueError):
        model.memory(torch.zeros(1, MAX_SEQ + 1, dtype=torch.long), vuln)


def test_targets_are_never_banned():
    torch.manual_seed(0)
    model = build_hand_model(tiny_cfg())
    for excl_rows in (0, 4):
        batch = toy_batch(excl_rows=excl_rows)
        cards, excl = batch["cards"], batch["excl"]
        bos = torch.full((4, 1), HAND_DIM, dtype=torch.long)
        inp = torch.cat([bos, cards[:, :-1]], 1)
        banned = model._banned(inp, excl if excl.any() else None)
        assert int(banned.gather(2, cards[:, :, None]).sum()) == 0


def test_samples_are_valid_hands():
    torch.manual_seed(0)
    model = build_hand_model(tiny_cfg()).eval()
    model.init_codebook(**{k: toy_batch()[k] for k in ("tokens", "vuln", "excl",
                                                      "mask")})
    batch = toy_batch(excl_rows=1)
    cards, codes = model.sample(batch["tokens"], batch["vuln"], batch["excl"],
                                batch["mask"], n=3, codes=True)
    assert cards.shape == (4, 3, 13) and codes.shape == (4, 3, 4)
    known = batch["excl"][0].numpy().astype(bool)
    for b in range(4):
        for s in range(3):
            drawn = cards[b, s].numpy()
            assert len(set(drawn.tolist())) == 13
            assert (np.diff(drawn) > 0).all()
            assert drawn.min() >= 0 and drawn.max() < HAND_DIM
            if b == 0:
                assert not (indicator_from_cards(drawn).astype(bool) & known).any()


def test_masking_reaches_the_encoder():
    """The known-cards mask is an encoder input, not just a decoder filter."""
    torch.manual_seed(0)
    model = build_hand_model(tiny_cfg()).eval()
    batch = toy_batch()
    excl = torch.zeros(4, HAND_DIM)
    excl[:, np.flatnonzero(encode_hand(HANDS[0]) == 0)[:13]] = 1.0
    with torch.no_grad():
        a, _ = model.memory(batch["tokens"], batch["vuln"], None, batch["mask"])
        b, _ = model.memory(batch["tokens"], batch["vuln"], excl, batch["mask"])
    assert not torch.allclose(a, b, atol=1e-5)
    with pytest.raises(ValueError):
        full = torch.ones(1, HAND_DIM)
        full[:, :12] = 0                               # 40 cards gone
        model.generate(torch.zeros(1, 4, model.cfg.d_model), full)


def test_vq_usage_and_revival():
    torch.manual_seed(0)
    cfg = tiny_cfg(codebook_size=32, revive_frac=0.5)
    model = build_hand_model(cfg)
    model.train()
    batch = toy_batch()
    out = model(batch["tokens"], batch["vuln"], batch["cards"],
                excl=batch["excl"], mask=batch["mask"])
    assert model.vq.usage.sum() > 0                    # EMA tracks winners
    ppl = out["ppl"]
    assert 1.0 <= ppl <= cfg.codebook_size
    live = model.vq.codebook.weight.data.clone()
    revived = model.revive_dead(out["z_pre"])          # frac 0.5 -> most codes
    assert revived > 0
    assert not torch.equal(live, model.vq.codebook.weight.data)
    assert code_perplexity(torch.ones(8)) == pytest.approx(8.0, rel=1e-4)
    assert code_perplexity(torch.zeros(8)) == 0.0


def test_code_sampling_temperature():
    """Sampling codes is scale-free: a tiny temperature is argmin, temp=1 draws
    the near-tied runner-up exp(-1) as often as the winner."""
    torch.manual_seed(0)
    vq = VectorQuantizer(HandVQConfig(n_codes=1, codebook_size=8,
                                      code_dim=4)).eval()
    with torch.no_grad():
        vq.codebook.weight.copy_(torch.randn(8, 4) * 3)
    z = (vq.codebook.weight[0] + torch.tensor([0.3, 0.0, 0.0, 0.0])).view(1, 1, 4)
    winner = int(vq(z)[1]["idx"][0, 0])

    def draws(temp, n=400):
        return [int(vq(z, sample=True, temp=temp)[1]["idx"][0, 0])
                for _ in range(n)]

    cold = draws(1e-6)
    assert cold == [winner] * len(cold)               # -> argmin
    hot = draws(1.0)
    assert len(set(hot)) > 1                          # alternatives get drawn
    assert max(set(hot), key=hot.count) == winner     # winner stays the mode
    assert hot.count(winner) / len(hot) < 0.9
    assert 0.1 < hot.count(winner) / len(hot) < 0.9


def test_top_p_filter():
    logits = torch.tensor([[3.0, 2.0, 1.0, -5.0]])
    out = top_p_filter(logits, 0.9)                   # nucleus = the top two
    kept = out > torch.finfo(out.dtype).min / 2
    assert kept[0].tolist() == [True, True, False, False]
    assert torch.allclose(out[0, :2], logits[0, :2])
    assert torch.allclose(top_p_filter(logits, 1.0), logits)


def test_param_groups_skip_codebook():
    model = build_hand_model(tiny_cfg())
    groups = param_groups(model, 0.1)
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    decayed = {id(p) for p in groups[0]["params"]}
    assert id(model.vq.codebook.weight) not in decayed
    assert id(model.tok_emb.weight) not in decayed
    assert id(model.enc_blocks[0].ffn.w_in.weight) in decayed
    total = sum(len(g["params"]) for g in groups)
    assert total == sum(1 for p in model.parameters() if p.requires_grad)


def test_evaluate_metrics(mini_cache):
    torch.manual_seed(0)
    store = AuctionStore.load(mini_cache)
    ds = HandDataset(store, prefix="all", mask_aug=0.5)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate)
    model = build_hand_model(tiny_cfg())
    m = evaluate(model, loader, torch.device("cpu"), False, max_batches=2,
                 n_samples=3)
    for key in ("ce", "ppl", "acc", "shape", "hcp", "sample_acc", "best_acc",
                "jaccard", "shapes"):
        assert np.isfinite(m[key]), key
    assert m["n_rows"] == 16
    assert 0.0 <= m["acc"] <= 1.0 and 0.0 <= m["sample_acc"] <= 1.0
    assert m["best_acc"] >= m["sample_acc"] - 1e-6     # best of n >= mean of n
    assert 0.0 <= m["jaccard"] <= 1.0 and 1.0 <= m["shapes"] <= 3.0


def test_overfit_one_batch(mini_cache):
    """The bottleneck can carry a hand: memorize a few rows, then decode them."""
    torch.manual_seed(0)
    store = AuctionStore.load(mini_cache)
    ds = HandDataset(store, prefix="all")
    batch = collate([ds[i] for i in range(0, len(ds), 7)][:24])
    model = build_hand_model(tiny_cfg(n_codes=8, codebook_size=64, code_dim=32))
    model.init_codebook(batch["tokens"], batch["vuln"], batch["excl"],
                        batch["mask"])
    opt = torch.optim.AdamW(param_groups(model, 0.0), lr=2e-3)
    for _ in range(250):
        out = model(batch["tokens"], batch["vuln"], batch["cards"],
                    excl=batch["excl"], mask=batch["mask"])
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    assert out["ce"].item() < 0.5, out["ce"].item()
    model.eval()
    greedy, _ = model.sample(batch["tokens"], batch["vuln"], batch["excl"],
                             batch["mask"], n=1, greedy=True)
    overlap = (indicator(greedy[:, 0]) * indicator(batch["cards"])).sum()
    acc = float(overlap) / (13 * len(batch["cards"]))
    assert acc >= 0.7, acc


# -- generation from the bidding models --------------------------------------

def test_split_weight():
    assert split_weight("random") == ("random", 1.0)
    assert split_weight("random@0.25") == ("random", 0.25)
    assert split_weight("bc:runs/x/best.pt") == ("bc:runs/x/best.pt", 1.0)
    assert split_weight("bc:runs/x@2/best.pt") == ("bc:runs/x@2/best.pt", 1.0)


def test_policy_pool_random():
    models, names, probs = policy_pool(["random", "random@3"], torch.device("cpu"))
    assert len(models) == 1 and names == ["random"]     # idempotent by name
    assert probs.tolist() == [1.0]
    with pytest.raises(ValueError):
        policy_pool([], torch.device("cpu"))


def test_generate_store(tmp_path):
    out = tmp_path / "gen"
    store = generate(out, 8, ["random"], torch.device("cpu"), chunk=4,
                     val_frac=0.25, test_frac=0.25, seed=5, progress=False)
    assert len(store) == 8
    for name in ("hands", "calls", "lengths", "dealer", "vuln", "policy",
                 "train", "val", "test", "meta"):
        assert (out / f"{name}.npy").exists() or (out / f"{name}.json").exists()
    assert store.hands.shape == (8, 4, HAND_DIM)
    assert (store.lengths >= 4).all()                    # auctions run to 3 passes
    assert (store.lengths <= 36).all()
    assert np.array_equal(store.hands.sum(2), np.full((8, 4), 13))
    ds = HandDataset(store, prefix="random", mask_aug=0.5)
    tokens, cards, vuln, excl, L = ds[0]
    assert (np.diff(cards) > 0).all() and 0 <= int(vuln) <= 3 and L >= 2
    # every generated auction is legal-and-terminal: ends in three passes
    for d in range(8):
        n = int(store.lengths[d])
        assert n >= 4 and (store.calls[d, n - 3:n] == 1).all()
        assert (store.calls[d, n:] == 0).all()
