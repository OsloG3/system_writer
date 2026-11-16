"""Tests for the rule-based SystemBidder (conventions/bidder.py): it must open,
respond and rebid sensibly from the system file alone, never produce an
illegal call, and not run auctions away to the seven level.

Hands are 'S.H.D.C'; K&R values quoted were checked with conventions.kr_points.
"""

import numpy as np
import pytest

from bidding_dt.conventions import AuctionReader, HandInfo, SystemEnforcer
from bidding_dt.conventions.bidder import SystemBidder, load_bidder
from bidding_dt.data.vocab import (CALL_PASS, FIRST_BID, VOCAB_SIZE,
                                   call_to_id, id_to_call)
from bidding_dt.env.deals import random_deals

B = call_to_id
X = "X"


@pytest.fixture(scope="module")
def bidder():
    return load_bidder()


def feed(calls, dealer=0):
    r = AuctionReader(dealer, 0)
    for c in calls:
        r.feed(B(c))
    return r


def legal_after(reader):
    from bidding_dt.data.legal import legal_mask
    return np.flatnonzero(legal_mask(reader.calls,
                                     role_offset=(-reader.n) % 4)).tolist()


def choose(bidder, calls, hand, dealer=0):
    r = feed(calls, dealer)
    hero = (dealer + len(r.calls)) % 4
    return bidder.choose(r, hero, hand, legal_after(r))


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------

def test_opening_1nt(bidder):
    # 18 HCP balanced 4333, ~15.6 K&R -> 1NT
    assert choose(bidder, [], "QJ7.KQ6.AJ7.KQ84") == B("1N")


def test_opening_one_major(bidder):
    # 5 spades, opening values -> 1S
    assert choose(bidder, [], "AK982.54.AK42.K9") == B("1S")


def test_opening_strong_2c(bidder):
    # a huge hand -> strong 2C
    assert choose(bidder, [], "AKQ.AKQ.AKQ.AKQJ") == B("2C")


def test_opening_weak_two(bidder):
    # 6 hearts, weak -> 2H
    assert choose(bidder, [], "74.KQ8643.952.84") == B("2H")


def test_opening_pass_weak(bidder):
    # a yarborough -> pass
    assert choose(bidder, [], "743.8632.952.843") == CALL_PASS


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

def test_stayman_response_with_values(bidder):
    # over 1NT with 4 hearts and ~10 K&R -> Stayman 2C
    assert choose(bidder, ["1N", "P"], "K74.Q863.A52.K84") == B("2C")


def test_transfer_with_five_hearts(bidder):
    # over 1NT with 5 hearts, weak (< Stayman strength) -> transfer 2D
    assert choose(bidder, ["1N", "P"], "743.KQ864.952.84") == B("2D")


def test_no_garbage_stayman(bidder):
    # over 1NT with a bust and a 4-card major -> pass, not Stayman
    assert choose(bidder, ["1N", "P"], "743.8632.952.843") == CALL_PASS


def test_raise_partner_major(bidder):
    # 1H with 3-card support and ~7 K&R -> simple raise 2H
    assert choose(bidder, ["1H", "P"], "K74.Q83.952.KJ84") == B("2H")


def test_two_over_one_response(bidder):
    # 1S with a good hand and clubs -> 2C (2/1 GF)
    c = choose(bidder, ["1S", "P"], "74.K863.AKQ2.K84")
    assert c in (B("2C"), B("2D"), B("2H"))   # a 2/1 in a real suit


# ---------------------------------------------------------------------------
# Legality and sanity over many random deals
# ---------------------------------------------------------------------------

def test_bidder_never_illegal_and_terminates(bidder):
    rng = np.random.default_rng(0)
    deals = random_deals(rng, 60)
    max_len = 0
    for d in deals:
        dealer = int(rng.integers(0, 4))
        calls = bidder.bid_auction(d.hands, dealer)
        assert 1 <= len(calls) <= 40
        max_len = max(max_len, len(calls))
        # every call is a legal action id
        assert all(CALL_PASS <= c < VOCAB_SIZE for c in calls)
        # the auction must actually end (three trailing passes after a bid,
        # or four passes)
        r = AuctionReader(dealer, 0)
        for c in calls:
            r.feed(c)
        assert bidder._auction_over(r) or len(calls) >= 40
    # no runaway auctions
    assert max_len < 30


def test_bidder_auctions_are_mostly_consistent(bidder):
    """The bidder follows the file: its own calls rarely violate it."""
    enf = SystemEnforcer(bidder.spec, bidder.cfg)
    rng = np.random.default_rng(3)
    deals = random_deals(rng, 80)
    n_calls = n_viol = 0
    for d in deals:
        dealer = int(rng.integers(0, 4))
        r = AuctionReader(dealer, 0)
        hands = [HandInfo(h) for h in d.hands]
        for _ in range(40):
            if bidder._auction_over(r):
                break
            hero = r.next_seat()
            c = bidder.choose(r, hero, hands[hero], legal_after(r))
            n_calls += 1
            if enf.check(r, hero, hands[hero], c):
                n_viol += 1
            r.feed(c)
    rate = n_viol / max(1, n_calls)
    assert rate < 0.15, f"bidder violates its own system on {rate:.1%} of calls"


def test_bidder_reaches_reasonable_contracts(bidder):
    """Most deals end in a partscore or game, not a slam/freak."""
    rng = np.random.default_rng(7)
    deals = random_deals(rng, 120)
    high = 0
    for d in deals:
        dealer = int(rng.integers(0, 4))
        calls = bidder.bid_auction(d.hands, dealer)
        bids = [c for c in calls if c >= FIRST_BID]
        if not bids:
            continue
        lvl = (bids[-1] - FIRST_BID) // 5 + 1
        if lvl >= 6:
            high += 1
    assert high / len(deals) < 0.10, f"{high} slams in {len(deals)} deals"


# ---------------------------------------------------------------------------
# Continuations: transfers, stayman, quantitative, RKCB, advances, game tries
# ---------------------------------------------------------------------------

def test_completes_transfer(bidder):
    # opener must complete the heart transfer with any hand
    assert choose(bidder, ["1N", "P", "2D", "P"], "743.862.952.K843") == B("2H")
    # and the spade transfer
    assert choose(bidder, ["1N", "P", "2H", "P"], "743.862.952.K843") == B("2S")


def test_answers_stayman(bidder):
    # no 4-card major -> 2D denial
    assert choose(bidder, ["1N", "P", "2C", "P"], "743.862.A952.K84") == B("2D")
    # 4 hearts -> 2H
    assert choose(bidder, ["1N", "P", "2C", "P"], "743.Q862.A95.K84") == B("2H")
    # both majors -> hearts first
    assert choose(bidder, ["1N", "P", "2C", "P"], "Q743.Q862.A95.K8") == B("2H")


def test_stayman_continuation_ladder(bidder):
    # responder after the 2D denial: game values -> 3NT
    assert choose(bidder, ["1N", "P", "2C", "P", "2D", "P"],
                  "K74.Q86.A952.K84") == B("3N")
    # invite values -> 2NT
    assert choose(bidder, ["1N", "P", "2C", "P", "2D", "P"],
                  "K74.Q86.Q952.K84") == B("2N")
    # fit found (4H response) with game values -> 4H
    assert choose(bidder, ["1N", "P", "2C", "P", "2H", "P"],
                  "K74.Q863.A952.K8") == B("4H")


def test_transfer_continuation(bidder):
    # weak 5-card major: pass the completion (signoff)
    assert choose(bidder, ["1N", "P", "2D", "P", "2H", "P"],
                  "743.KQ864.952.84") == CALL_PASS
    # game values with 5: choice of games 3NT
    assert choose(bidder, ["1N", "P", "2D", "P", "2H", "P"],
                  "K74.AQ864.952.K8") == B("3N")


def test_quantitative_4nt(bidder):
    # responder with balanced ~15.4 K&R (18 HCP flat) invites with 4NT
    assert choose(bidder, ["1N", "P"], "KJ7.AQ6.QJ74.KQ8") == B("4N")
    # opener with a maximum accepts
    assert choose(bidder, ["1N", "P", "4N", "P"], "AQ7.KQ6.AJ7.KJ84") == B("6N")
    # opener with a minimum declines
    assert choose(bidder, ["1N", "P", "4N", "P"],
                  "KJ7.Q86.AQ74.KJ8") == CALL_PASS


def test_rkcb_step_response(bidder):
    # trump spades (1S-2C-2S-4NT): 1 keycard -> 5C (1430)
    assert choose(bidder, ["1S", "P", "2C", "P", "2S", "P", "4N", "P"],
                  "A743.863.952.K84") == B("5C")
    # 0 keycards -> 5D
    assert choose(bidder, ["1S", "P", "2C", "P", "2S", "P", "4N", "P"],
                  "743.863.K952.K84") == B("5D")
    # 2 without the queen -> 5H
    assert choose(bidder, ["1S", "P", "2C", "P", "2S", "P", "4N", "P"],
                  "AK43.863.952.K84") == B("5H")
    # 2 with the queen -> 5S
    assert choose(bidder, ["1S", "P", "2C", "P", "2S", "P", "4N", "P"],
                  "AKQ43.863.952.84") == B("5S")


def test_advance_after_takeout_double(bidder):
    # (1H)-X-(P): weak with spades -> 1S advance
    assert choose(bidder, ["1H", X, "P"], "KQ74.863.952.843", dealer=0) == B("1S")


def test_game_try_response(bidder):
    # 1H-2H, opener tries 3C; responder with the club ace accepts
    assert choose(bidder, ["1H", "P", "2H", "P", "3C", "P"],
                  "K74.Q863.A952.84") == B("4H")
    # with no help and a minimum, decline with 3H
    assert choose(bidder, ["1H", "P", "2H", "P", "3C", "P"],
                  "K74.Q863.952.J84") == B("3H")


def test_weak_two_feature_ask(bidder):
    # invitational+ hand over partner's 2H weak two -> 3H invite or better
    c = choose(bidder, ["2H", "P"], "KQ7.A63.KJ52.Q84")
    assert c in (B("2N"), B("3H"), B("4H"))
    # opener with a minimum completes 3H
    assert choose(bidder, ["2H", "P", "2N", "P"], "74.KQ8643.952.84") == B("3H")
