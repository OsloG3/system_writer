"""Auction -> contract -> par-diff reward in IMPs.

Reward for N-S (zero-sum; E-W gets the negation):

    r = IMPs( score_NS(final contract, DD tricks) - par_score_NS(deal) )

Both terms are computed on the same 52 cards, so the card-lie luck largely
cancels and r measures only what the auction lost/gained versus double-dummy
optimal bidding. A passed-out deal scores 0 versus par.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..data.vocab import CALL_X, CALL_XX, FIRST_BID, bid_denom, bid_level, DENOMS
from .scoring import (PENALTY_DOUBLE, PENALTY_NONE, PENALTY_REDOUBLE,
                      contract_score, declarer_is_vul, points_to_imps)
from .solver import TableCache, dd_tables_batch, deal_key, par_score


@dataclass(frozen=True, slots=True)
class ContractInfo:
    level: int       # 1..7
    denom: int       # project CDHSN index
    declarer: int    # seat N=0 E=1 S=2 W=3
    penalty: int     # 0 undoubled, 1 doubled, 2 redoubled

    def __str__(self) -> str:
        pen = ("", "X", "XX")[self.penalty]
        return f"{self.level}{DENOMS[self.denom]}{pen} by {'NESW'[self.declarer]}"


@dataclass(frozen=True, slots=True)
class DealReward:
    reward_imps: float          # N-S perspective
    par_ns: int                 # par score, N-S perspective
    score_ns: int               # achieved score, N-S perspective
    contract: ContractInfo | None
    tricks: int | None          # DD tricks for the played contract


def contract_from_auction(dealer: int, calls: Sequence[int]) -> ContractInfo | None:
    """Recover the final contract from an auction in seat order (None=passout).

    Declarer = first player of the winning side to name the final denomination.
    Penalty state comes from X/XX after the last bid (intervening passes are
    irrelevant; legality guarantees X only applies to an opponent's bid).
    """
    last_bid_idx = -1
    doubled = redoubled = False
    first_denom_seat = {}  # (denom, side) -> first seat (absolute) to bid it
    for i, c in enumerate(calls):
        c = int(c)
        if c >= FIRST_BID:
            last_bid_idx = i
            doubled = redoubled = False
            seat = (dealer + i) % 4
            key = (bid_denom(c), seat % 2)
            if key not in first_denom_seat:
                first_denom_seat[key] = seat
        elif c == CALL_X:
            doubled = True
        elif c == CALL_XX:
            redoubled = True
    if last_bid_idx < 0:
        return None
    call = int(calls[last_bid_idx])
    denom = DENOMS.index(bid_denom(call))
    winning_side_bidder = (dealer + last_bid_idx) % 4
    declarer = first_denom_seat[(bid_denom(call), winning_side_bidder % 2)]
    penalty = (PENALTY_REDOUBLE if redoubled
               else PENALTY_DOUBLE if doubled else PENALTY_NONE)
    return ContractInfo(bid_level(call), denom, declarer, penalty)


def score_deal(table: np.ndarray, dealer: int, vuln: int,
               calls: Sequence[int]) -> DealReward:
    """Compute the DealReward given a solved (4,5) DD table."""
    par_ns = par_score(table, vuln, dealer)
    contract = contract_from_auction(dealer, calls)
    if contract is None:
        score_ns = 0
        tricks = None
    else:
        tricks = int(table[contract.declarer, contract.denom])
        s = contract_score(contract.level, contract.denom, contract.penalty,
                           tricks, declarer_is_vul(contract.declarer, vuln))
        score_ns = s if contract.declarer in (0, 2) else -s
    return DealReward(
        reward_imps=float(points_to_imps(score_ns - par_ns)),
        par_ns=par_ns, score_ns=score_ns, contract=contract, tricks=tricks,
    )


@dataclass(slots=True)
class RewardRequest:
    hands4: Sequence[str]   # 'S.H.D.C' strings, N,E,S,W
    dealer: int
    vuln: int
    calls: Sequence[int]


def deal_reward(hands4, dealer: int, vuln: int, calls: Sequence[int],
                cache: TableCache | None = None) -> DealReward:
    req = RewardRequest(hands4, dealer, vuln, calls)
    return deal_rewards([req], cache=cache)[0]


def warm_cache(hands_list, cache: TableCache | None) -> int:
    """Pre-solve DD tables for the given deals into cache; returns the number
    of newly solved deals.

    Used to overlap libdds solving with the neural rollout on multi-core CPUs:
    submit to a background thread at the start of an iteration (libdds releases
    the GIL and runs its own solver threads) and join before env.rewards(), so
    scoring becomes pure cache hits. Duplicate hands are solved once.
    """
    if cache is None:
        return 0
    uniq: dict[bytes, Sequence[str]] = {}
    for h in hands_list:
        uniq.setdefault(deal_key(h), h)
    miss = [(k, h) for k, h in uniq.items() if cache.get(k) is None]
    if not miss:
        return 0
    solved = dd_tables_batch([h for _, h in miss])
    cache.put_many([(k, solved[j]) for j, (k, _) in enumerate(miss)])
    return len(miss)


def deal_rewards(requests: Sequence[RewardRequest],
                 cache: TableCache | None = None) -> list[DealReward]:
    """Batch par-diff rewards; solves uncached deals together.

    Requests with identical hands (e.g. the same deal played at two tables)
    are grouped so each distinct deal is solved exactly once per batch.
    """
    tables: list[np.ndarray | None] = [None] * len(requests)
    groups: dict[bytes, list[int]] = {}
    for i, r in enumerate(requests):
        groups.setdefault(deal_key(r.hands4), []).append(i)
    miss: dict[bytes, list[int]] = {}
    if cache is not None:
        for k, rows in groups.items():
            t = cache.get(k)
            if t is None:
                miss[k] = rows
            else:
                for i in rows:
                    tables[i] = t
    else:
        miss = groups
    if miss:
        keys = list(miss)
        solved = dd_tables_batch([requests[miss[k][0]].hands4 for k in keys])
        items = []
        for j, k in enumerate(keys):
            for i in miss[k]:
                tables[i] = solved[j]
            items.append((k, solved[j]))
        if cache is not None:
            cache.put_many(items)
    return [score_deal(tables[i], r.dealer, r.vuln, r.calls)
            for i, r in enumerate(requests)]
