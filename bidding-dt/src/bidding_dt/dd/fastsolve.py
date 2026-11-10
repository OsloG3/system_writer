"""Fast batched double-dummy solving for the card-play bot.

The play bot solves tens of thousands of positions per board. endplay's
convenience layer dominates that cost: `Deal.from_pbn` parses hand strings and
`Deal.play` builds Card objects (~150 us per deal), and every SolvedBoard
entry is re-materialised as a Card on the way out. This module talks to the
raw DDS structs instead:

  * positions are identified by four packed 64-bit suit-mask ints (one per
    hand) plus leader/trump/current trick -- the masks ARE the DDS
    remainCards encoding, so the cache key doubles as the struct fill source
    and no card set is ever walked twice;
  * batches go through `SolveAllBoardsBin` with exactly the parameters
    endplay's `solve_all_boards(deals, SolveMode.Default)` uses: target=0,
    solutions=3, mode=1. mode MUST stay 1 -- DDS's mode=0 ("same trump as
    the previous board") reuses thread memory and was measured to return
    wrong scores for mid-play positions;
  * `futureTricks` results are parsed into {card id: tricks} dicts by
    walking the rank bitmasks directly;
  * a process-wide LRU memoises solved positions. The same public position
    recurs constantly: a strategy-search lookahead for card c is solved
    again inside the child node c leads to, retro play-quality scoring
    revisits positions across decisions, identical pool candidates collapse
    within one batch, and in drills mode the bot and the dd-safety gate both
    solve the real deal. Hits cost microseconds instead of milliseconds of
    DDS time, which is what lets the search analyse far more layouts within
    the same wall-clock budget.

Position convention (matches what play_bot/play_strategy always built):
`hands[seat]` holds the cards the seat still holds EXCLUDING the current
trick's cards; `cur` lists those cards (ids, env/deals.py order:
card = rank*4 + suit) in play order starting at `first`.

Card-id suit order S,H,D,C = 0..3 equals the DDS Denom order (spades=0,
hearts=1, diamonds=2, clubs=3, nt=4), and a DDS rank bit is 1 << (14 - r)
for rank index r (A=0 .. 2=12), so ids convert with shifts only.

Returned dicts are shared cache values: callers must treat them as
read-only. Thread-safe; the HTTP sidecar solves from several threads.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

import endplay._dds as _dds

MAX_BOARDS = _dds.MAXNOOFBOARDS       # DDS batch limit (200)
NT_DENOM = 4                          # endplay Denom.nt

# rank index (A=0 .. 2=12) -> DDS rank bitmask bit
_RANK_BIT = tuple(1 << (14 - r) for r in range(13))
_MASK16 = 0xFFFF

# Process-wide LRU of solved positions. Each entry is a small dict, so this
# costs at most a few tens of MB; it lives for the lifetime of the sidecar.
_CACHE_MAX = 32768
_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()

# Reused per-thread DDS batch buffers (~20 KB + ~45 KB of ctypes structs):
# allocating them per call is measurable when batches are small.
_local = threading.local()

_stats_lock = threading.Lock()
_STATS = {"hits": 0, "misses": 0, "dds_seconds": 0.0}


def stats() -> dict:
    """Cumulative cache counters (diagnostics)."""
    with _stats_lock:
        return dict(_STATS)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def pack_hand(cards) -> int:
    """One hand's card ids -> four 16-bit suit masks packed into a single int.

    Suit s occupies bits [16s, 16s+16); within a suit, rank index r is bit
    1 << (14 - r) -- exactly the DDS remainCards encoding, so a packed hand
    doubles as the cache key material and the struct fill source.
    """
    m0 = m1 = m2 = m3 = 0
    for c in cards:
        bit = _RANK_BIT[c >> 2]
        s = c & 3
        if s == 0:
            m0 |= bit
        elif s == 1:
            m1 |= bit
        elif s == 2:
            m2 |= bit
        else:
            m3 |= bit
    return m0 | (m1 << 16) | (m2 << 32) | (m3 << 48)


def position_key(hands, first: int, trump: int, cur) -> tuple:
    """Hashable identity of a DDS position (the cache key). The first four
    elements are the packed hands, so the key also carries everything needed
    to fill a raw deal struct without touching the card sets again.

    A 4-card `cur` means the trick just completed (the strategy search
    evaluates the card that closes it): the position is normalised to the
    winner leading an empty trick -- exactly what endplay's `Deal.play` does
    on the fourth card -- so both spellings of the same position share one
    cache entry and one struct form. `hands` must already exclude every cur
    card (the standard convention), which makes the rotation lossless.
    """
    cur = tuple(cur)
    if len(cur) == 4:
        first = (int(first) + trick_winner(cur, trump)) % 4
        cur = ()
    elif len(cur) > 4:
        raise ValueError("a trick has at most four cards")
    return (pack_hand(hands[0]), pack_hand(hands[1]), pack_hand(hands[2]),
            pack_hand(hands[3]), int(first), int(trump), cur)


def _fill_from_key(d, key) -> None:
    """Fill a (possibly reused) `_dds.deal` struct straight from a key."""
    p0, p1, p2, p3, first, trump, cur = key
    d.trump = trump
    d.first = first
    rc = d.remainCards
    for s, packed in enumerate((p0, p1, p2, p3)):
        row = rc[s]
        row[0] = packed & _MASK16
        row[1] = (packed >> 16) & _MASK16
        row[2] = (packed >> 32) & _MASK16
        row[3] = (packed >> 48) & _MASK16
    cts, ctr = d.currentTrickSuit, d.currentTrickRank
    n = len(cur)
    for i in range(3):
        if i < n:
            c = cur[i]
            cts[i] = c & 3
            ctr[i] = 14 - (c >> 2)
        else:                        # stale buffer slots must not leak in
            cts[i] = 0
            ctr[i] = 0


def make_deal(hands, first: int, trump: int, cur):
    """A raw `_dds.deal` struct for one position (see module docstring)."""
    d = _dds.deal()
    _fill_from_key(d, position_key(hands, first, trump, cur))
    return d


def _parse_board(fut) -> dict:
    """futureTricks -> {card id: tricks}, expanding DDS 'equals' bitmasks
    (equivalent cards share the score, as endplay's SolvedBoard iterator
    does, but without building Card objects)."""
    out = {}
    suit, rank, equals, score = fut.suit, fut.rank, fut.equals, fut.score
    for j in range(fut.cards):
        sc = score[j]
        if sc == -1:
            continue
        su = suit[j]
        m = (1 << rank[j]) | equals[j]
        while m:
            bit = m & -m
            out[(15 - bit.bit_length()) * 4 + su] = sc
            m ^= bit
    return out


def _buffers():
    buf = getattr(_local, "buf", None)
    if buf is None:
        buf = (_dds.boards(), _dds.solvedBoards())
        _local.buf = buf
    return buf


def solve_positions(positions, trump: int) -> list:
    """Solve a batch of positions -> [{card id: tricks for the side to move}].

    `positions` is a list of (hands, first, cur) (see module docstring).
    Results are ordered like the input. Cache hits and repeats inside the
    batch itself (identical candidate hands are common in the pool) never
    reach DDS. Raises endplay's DDSError when the solver fails.
    """
    n = len(positions)
    if n == 0:
        return []
    results: list = [None] * n
    pending: dict = {}                      # key -> result indices needing it
    misses: list = []                       # unique keys to solve, in order
    hits = 0
    with _cache_lock:
        for i, (hands, first, cur) in enumerate(positions):
            key = position_key(hands, first, trump, cur)
            got = _cache.get(key)
            if got is not None:
                _cache.move_to_end(key)
                results[i] = got
                hits += 1
                continue
            idxs = pending.get(key)
            if idxs is None:
                pending[key] = [i]
                misses.append(key)
            else:
                idxs.append(i)              # in-batch duplicate: solved once
    if hits:
        with _stats_lock:
            _STATS["hits"] += hits
    if not misses:
        return results

    t0 = time.monotonic()
    bop, solvedp = _buffers()
    for start in range(0, len(misses), MAX_BOARDS):
        chunk = misses[start:start + MAX_BOARDS]
        for j, key in enumerate(chunk):
            _fill_from_key(bop.deals[j], key)
            bop.target[j] = 0
            bop.solutions[j] = 3
            bop.mode[j] = 1                 # never 0: see module docstring
        bop.noOfBoards = len(chunk)
        _dds.SolveAllBoardsBin(bop, solvedp)
        with _cache_lock:
            for j, key in enumerate(chunk):
                board = _parse_board(solvedp.solvedBoard[j])
                for i in pending[key]:
                    results[i] = board
                _cache[key] = board
                _cache.move_to_end(key)
            while len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)
    with _stats_lock:
        _STATS["misses"] += len(misses)
        _STATS["dds_seconds"] += time.monotonic() - t0
    return results


def solve_position(hands, first: int, trump: int, cur) -> dict:
    """Single-position convenience wrapper around solve_positions."""
    return solve_positions([(hands, first, cur)], trump)[0]


def trick_winner(cids, trump: int) -> int:
    """Index (into `cids`, play order) of the trick winner. Pure integer
    mirror of endplay's trick_winner: the highest card of the led suit wins
    unless a trump was played, in which case the highest trump wins. A
    lower card id is a higher rank (A=0). `trump` may be NT_DENOM, which no
    card's suit can match."""
    best = cids[0]
    best_i = 0
    best_suit = best & 3
    if best_suit == trump:                # led suit is trumps (or NT: never)
        for i in range(1, len(cids)):
            c = cids[i]
            if (c & 3) == best_suit and c < best:
                best, best_i = c, i
    else:
        for i in range(1, len(cids)):
            c = cids[i]
            s = c & 3
            if s == best_suit:
                if c < best:
                    best, best_i = c, i
            elif s == trump:
                best, best_i = c, i
                best_suit = trump
    return best_i
