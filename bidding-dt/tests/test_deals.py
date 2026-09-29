import numpy as np

from bidding_dt.data.hands import encode_deal, encode_hand
from bidding_dt.env.deals import (RANK_CHARS, Deal, encode_deals,
                                  hands_from_cards, random_deals)


def cards_from_hands(hands4) -> np.ndarray:
    """(4,) hand strings -> (52,) card->seat map (inverse of hands_from_cards)."""
    cards = np.zeros(52, dtype=np.uint8)
    for seat, hand in enumerate(hands4):
        for s, suit_str in enumerate(hand.split(".")):
            for ch in suit_str:
                cards[RANK_CHARS.index(ch) * 4 + s] = seat
    return cards


DEAL1 = ("AT7.KT943.A42.K8", "J6543.75.K8.AQJT", "K982.AQ6.T73.952", "Q.J82.QJ965.7643")


def test_hands_from_cards_roundtrip():
    cards = cards_from_hands(DEAL1)
    assert hands_from_cards(cards) == DEAL1


def test_encode_deals_matches_encode_hand():
    rng = np.random.default_rng(0)
    deals = random_deals(rng, 50)
    cards = np.stack([cards_from_hands(DEAL1)] + [d.cards for d in deals])
    enc = encode_deals(cards)
    assert np.array_equal(enc[0], encode_deal(" ".join(DEAL1)))
    for i, d in enumerate(deals, start=1):
        assert np.array_equal(enc[i], d.encoded)
        for seat in range(4):
            assert np.array_equal(d.encoded[seat], encode_hand(d.hands[seat]))


def test_random_deals_valid():
    rng = np.random.default_rng(1)
    deals = random_deals(rng, 200)
    for d in deals:
        assert isinstance(d, Deal)
        bins = np.bincount(d.cards, minlength=4)
        assert list(bins) == [13, 13, 13, 13]
        for seat in range(4):
            assert sum(len(s) for s in d.hands[seat].split(".")) == 13
        # the indicator encoding agrees with the card map
        for seat in range(4):
            mine = np.flatnonzero(d.cards == seat)
            exp = np.zeros(52, dtype=np.uint8)
            exp[(mine % 4) * 13 + mine // 4] = 1
            assert np.array_equal(d.encoded[seat], exp)
    # distinct deals
    keys = {d.key() for d in deals}
    assert len(keys) == len(deals)


def test_uniform_marginals():
    """Each seat should hold each card ~1/4 of the time over many deals."""
    rng = np.random.default_rng(2)
    n = 4000
    deals = random_deals(rng, n)
    onehot = np.stack([np.eye(4, dtype=np.float64)[d.cards] for d in deals])
    freq = onehot.mean(axis=0)  # (52, 4)
    assert freq.shape == (52, 4)
    assert np.allclose(freq, 0.25, atol=4 * np.sqrt(0.25 * 0.75 / n))
