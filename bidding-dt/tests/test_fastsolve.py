"""Tests for dd/fastsolve: raw-struct batched solving + the position cache.

The contract fastsolve must honour: for any position, its {card id: tricks}
result equals what endplay's own solve_board returns for the equivalent
Deal, and its trick_winner equals endplay's. Everything else (packing,
dedupe, caching) must be invisible in the results.
"""

import numpy as np
import pytest

from bidding_dt.data.hands import str_to_cards
from bidding_dt.dd import fastsolve

HAS_ENDPLAY = True
try:
    import endplay.types as T
    from endplay.dds.solve import SolveMode, solve_board
    from endplay.utils.play import trick_winner as ep_trick_winner
except ImportError:  # pragma: no cover
    HAS_ENDPLAY = False

RANKS = "AKQJT98765432"
# N,E,S,W (same fixture deal as test_play_bot)
DEAL = ["AT7.KT943.A42.K8", "J6543.75.K8.AQJT", "K982.AQ6.T73.952", "Q.J82.QJ965.7643"]
HANDS = [set(str_to_cards(h).tolist()) for h in DEAL]
# project denom index (C,D,H,S,N) -> endplay denom int (S=0,H=1,D=2,C=3,NT=4)
PROJ_TO_EP = (3, 2, 1, 0, 4)


def cid(rank_ch, suit_ch):
    return RANKS.index(rank_ch) * 4 + "SHDC".index(suit_ch)


def ep_card(c):
    return T.Card(suit=T.Denom(c % 4), rank=T.Rank.find(RANKS[c // 4]))


def ep_deal(hands4, leader, trump_int, cur):
    """The endplay reference Deal for a position (hands exclude cur cards)."""
    from bidding_dt.data.hands import cards_to_str
    parts = [cards_to_str(sorted(hands4[s])) if hands4[s] else "-" for s in range(4)]
    deal = T.Deal.from_pbn("N:" + " ".join(parts))
    deal.trump = T.Denom(trump_int)
    deal.first = T.Player(leader)
    for c in cur:
        deal.play(ep_card(c), from_hand=False)
    return deal


def ep_solve(hands4, leader, trump_int, cur):
    """Reference {cid: tricks} via endplay's solve_board."""
    out = {}
    for card, tricks in solve_board(ep_deal(hands4, leader, trump_int, cur),
                                    SolveMode.Default):
        out[RANKS.index(card.rank.abbr) * 4 + int(card.suit)] = int(tricks)
    return out


# ---- packing ------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_pack_hand_matches_dds_remaincards():
    """The packed int must be exactly the four DDS suit masks, so keys and
    struct fills cannot drift apart."""
    packed = fastsolve.pack_hand(HANDS[0])
    ref = ep_deal(HANDS, 0, 1, [])          # spades trump, any leader
    for s in range(4):
        assert (packed >> (16 * s)) & 0xFFFF == int(ref._data.remainCards[0][s])


def test_position_key_distinguishes_everything():
    base = (HANDS[0], HANDS[1], HANDS[2], HANDS[3])
    k = fastsolve.position_key(base, 0, 1, [])
    assert k != fastsolve.position_key(base, 1, 1, [])          # leader
    assert k != fastsolve.position_key(base, 0, 4, [])          # trump
    assert k != fastsolve.position_key(base, 0, 1, [cid("6", "S")])   # trick
    moved = (HANDS[0] - {cid("A", "S")}, HANDS[1] | {cid("A", "S")},
             HANDS[2], HANDS[3])
    assert k != fastsolve.position_key(moved, 0, 1, [])         # hands


def test_long_cur_is_refused():
    with pytest.raises(ValueError):
        fastsolve.position_key(HANDS, 0, 1, [1, 2, 3, 4, 5])


def test_completed_trick_rotates_to_the_winner():
    """A 4-card cur means the trick just closed: the key must normalise to
    'winner leads, empty trick' (what endplay's Deal.play does on the 4th
    card), so both spellings of the position share one cache entry."""
    # E leads 6S, S 2S, W QS, N AS -> North wins. Hands exclude all four.
    four = [cid("6", "S"), cid("2", "S"), cid("Q", "S"), cid("A", "S")]
    hands_after = {0: HANDS[0] - {cid("A", "S")}, 1: HANDS[1] - {cid("6", "S")},
                   2: HANDS[2] - {cid("2", "S")}, 3: HANDS[3] - {cid("Q", "S")}}
    k_completed = fastsolve.position_key(hands_after, 1, PROJ_TO_EP[2], four)
    k_rotated = fastsolve.position_key(hands_after, 0, PROJ_TO_EP[2], [])
    assert k_completed == k_rotated


# ---- solving ------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_solve_position_matches_endplay_fresh_deal():
    fastsolve.clear_cache()
    for leader in range(4):
        for denom in range(5):
            trump = PROJ_TO_EP[denom]
            got = fastsolve.solve_position(HANDS, leader, trump, [])
            assert got == ep_solve(HANDS, leader, trump, [])


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_solve_position_matches_endplay_mid_trick():
    # trick one: E leads 6S, S plays 2S, W the QS; N to act (hands exclude cur)
    cur = [cid("6", "S"), cid("2", "S"), cid("Q", "S")]
    hands = [HANDS[0], HANDS[1] - {cid("6", "S")},
             HANDS[2] - {cid("2", "S")}, HANDS[3] - {cid("Q", "S")}]
    for denom in range(5):
        trump = PROJ_TO_EP[denom]
        got = fastsolve.solve_position(hands, 1, trump, cur)
        assert got == ep_solve(hands, 1, trump, cur)
        # the observed play's card is scored like every other legal card
        assert cid("A", "S") in got


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_solve_positions_batch_matches_individual():
    fastsolve.clear_cache()
    rng = np.random.default_rng(3)
    positions = []
    for _ in range(12):
        perm = rng.permutation(52)
        hands = {s: set(perm[s * 13:(s + 1) * 13].tolist()) for s in range(4)}
        positions.append((hands, int(rng.integers(4)), []))
    trump = PROJ_TO_EP[2]
    batch = fastsolve.solve_positions(positions, trump)
    fastsolve.clear_cache()
    solo = [fastsolve.solve_position(h, f, trump, c) for h, f, c in positions]
    assert batch == solo


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_batches_larger_than_the_dds_limit_are_chunked():
    # MAX_BOARDS (200) is the hard DDS batch cap; solve_positions must chunk
    rng = np.random.default_rng(5)
    hands = {s: set(HANDS[s]) for s in range(4)}
    n = fastsolve.MAX_BOARDS + 7
    positions = [(hands, i % 4, []) for i in range(n)]   # mostly duplicates
    out = fastsolve.solve_positions(positions, PROJ_TO_EP[2])
    assert len(out) == n
    ref = out[0]
    for i, o in enumerate(out):
        if i % 4 == 0:
            assert o == ref                    # same hands, same leader
    # a duplicate-free batch of the same size also works
    positions2 = []
    seen = set()
    while len(positions2) < n:
        perm = rng.permutation(52)
        h = {s: set(perm[s * 13:(s + 1) * 13].tolist()) for s in range(4)}
        k = frozenset(h[0])
        if k in seen:
            continue
        seen.add(k)
        positions2.append((h, 0, []))
    out2 = fastsolve.solve_positions(positions2, PROJ_TO_EP[2])
    assert len(out2) == n and all(o for o in out2)


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_in_batch_duplicates_are_solved_once_and_cached():
    fastsolve.clear_cache()
    hands = {s: set(HANDS[s]) for s in range(4)}
    before = fastsolve.stats()
    positions = [(hands, 1, [])] * 5
    out = fastsolve.solve_positions(positions, PROJ_TO_EP[2])
    mid = fastsolve.stats()
    assert all(o == out[0] for o in out)
    assert mid["misses"] - before["misses"] == 1     # one unique board solved
    # second call: everything is a cache hit, no new DDS work
    out2 = fastsolve.solve_positions(positions, PROJ_TO_EP[2])
    after = fastsolve.stats()
    assert out2 == out
    assert after["misses"] == mid["misses"]
    assert after["hits"] - mid["hits"] == 5


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_cache_returns_equal_results_not_stale_ones():
    """The cache is keyed by the full position, so a different trump/leader
    can never return a previous position's numbers."""
    fastsolve.clear_cache()
    hands = {s: set(HANDS[s]) for s in range(4)}
    hearts = fastsolve.solve_position(hands, 0, PROJ_TO_EP[2], [])
    spades = fastsolve.solve_position(hands, 0, PROJ_TO_EP[3], [])
    nt = fastsolve.solve_position(hands, 0, PROJ_TO_EP[4], [])
    assert hearts == ep_solve(hands, 0, PROJ_TO_EP[2], [])
    assert spades == ep_solve(hands, 0, PROJ_TO_EP[3], [])
    assert nt == ep_solve(hands, 0, PROJ_TO_EP[4], [])
    assert not (hearts == spades == nt)      # this deal genuinely differs


# ---- trick winner ---------------------------------------------------------------

@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_trick_winner_matches_endplay():
    rng = np.random.default_rng(9)
    for trump in (0, 1, 2, 3, fastsolve.NT_DENOM):
        for _ in range(200):
            cards = rng.choice(52, size=4, replace=False).tolist()
            leader = int(rng.integers(4))
            got = fastsolve.trick_winner(cards, trump)
            want = ep_trick_winner([ep_card(c) for c in cards],
                                   T.Player(leader), T.Denom(trump))
            assert (leader + got) % 4 == int(want)


def test_trick_winner_rules():
    s6, s2, sq, sa = cid("6", "S"), cid("2", "S"), cid("Q", "S"), cid("A", "S")
    h5 = cid("5", "H")
    ht = PROJ_TO_EP[2]                        # hearts trump = denom int 1
    # highest of the led suit wins
    assert fastsolve.trick_winner([s6, s2, sq, sa], ht) == 3
    # a trump beats a higher led-suit card; the top trump wins among trumps
    assert fastsolve.trick_winner([s6, h5, sq, sa], ht) == 1
    hk, hq = cid("K", "H"), cid("Q", "H")
    assert fastsolve.trick_winner([s6, hq, sa, hk], ht) == 3
    # off-suit discards never win; in NT only the led suit counts, and the
    # highest of it wins (SQ is the top spade here -> index 3)
    d9 = cid("9", "D")
    assert fastsolve.trick_winner([s6, d9, s2, sq], fastsolve.NT_DENOM) == 3
    # leading trumps: the top trump wins, discards are irrelevant
    assert fastsolve.trick_winner([hq, hk, s6, d9], ht) == 1
