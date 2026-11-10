"""Bidding-system conventions: hand evaluation + consistency checks.

`kr.py` implements the Kaplan-Rubens point count (the 26-step evaluator from
rpbridge.net/8j19.htm, verified against Pavlicek's own CGI). `auction.py`
replays an auction and structurally classifies each call (opening, 2/1,
jump shift, splinter, overcall, XYZ, fourth suit, double meaning, per-side
game-force status). `rules.py` ties the two together: given the hero's hand,
the auction so far and a candidate call, `call_violations` reports the
system conventions that call would break, each with a weight, and
`call_penalty` sums them.

Used by the RL trainer (rl/rollout.py + rl/train_ppo.py, `rl.conv_weight`)
to penalize the decision-transformer policy for bids inconsistent with the
enforced system: 5-card majors, 15-17 1NT, strong 2C, 20-21 2NT, GF 2/1,
invitational jump shifts (6 cards, 10-11, also in competition), standard
overcalls, splinter double jumps below 4H, RKCB 4NT in the last suit the
partnership bid, XYZ, takeout doubles, fourth-suit GF and standard natural
bidding. See rules.py for the full list of rule codes and thresholds.
"""

from .auction import AuctionReader, CallInfo, is_game_or_above
from .kr import kr_points, kr_suit
from .rules import (DEFAULT_WEIGHTS, ConventionConfig, HandInfo, Violation,
                    call_penalty, call_violations, check_call)

__all__ = [
    "AuctionReader", "CallInfo", "is_game_or_above",
    "kr_points", "kr_suit",
    "DEFAULT_WEIGHTS", "ConventionConfig", "HandInfo", "Violation",
    "call_penalty", "call_violations", "check_call",
]
