"""Bidding-system conventions: hand evaluation + consistency checks.

`kr.py` implements the Kaplan-Rubens point count (the 26-step evaluator from
rpbridge.net/8j19.htm, verified against Pavlicek's own CGI). `auction.py`
replays an auction and structurally classifies each call (opening, 2/1,
jump shift, splinter, smolen, overcall, XYZ, fourth suit, RKCB ask/response,
double meaning, per-side game-force status). `rules.py` ties the two
together: given the hero's hand, the auction so far and a candidate call,
`call_violations` reports the system conventions that call would break, each
with a weight, `call_penalty` sums them, and `call_bonus` rewards one clean
use of an artificial convention. `alerts.py` renders the alert text of a
convention call (dumped into the self-play tree by rl/analyze.py and served
to the client by the Go side).

Used by the RL trainer (rl/rollout.py + rl/train_ppo.py, `rl.conv_weight`)
to penalize the decision-transformer policy for bids inconsistent with the
enforced system: 5-card majors, 15-17 1NT, strong 2C, 20-21 2NT, GF 2/1,
invitational jump shifts (6 cards, 10-11, also in competition), standard
overcalls (balancing 1NT = 12-15), splinter double jumps below 4H, smolen
after 1/2NT stayman, RKCB step responses (the 4NT ask itself is free),
support cuebids of the opponents' suit, XYZ, takeout doubles (power doubles
with 18+ need no shape), fourth-suit GF and standard natural bidding.
Correctly using XYZ, RKCB, splinter, smolen, fourth suit or a support
cuebid earns the small `conv_bonus` reward. See rules.py for the full list
of rule codes and thresholds.
"""

from .alerts import RKCB_STEPS, convention_alert
from .auction import AuctionReader, CallInfo, is_game_or_above
from .kr import kr_points, kr_suit
from .rules import (DEFAULT_WEIGHTS, ConventionConfig, HandInfo, Violation,
                    call_bonus, call_penalty, call_violations, check_call,
                    convention_bonus, convention_kind)

__all__ = [
    "RKCB_STEPS", "convention_alert",
    "AuctionReader", "CallInfo", "is_game_or_above",
    "kr_points", "kr_suit",
    "DEFAULT_WEIGHTS", "ConventionConfig", "HandInfo", "Violation",
    "call_bonus", "call_penalty", "call_violations", "check_call",
    "convention_bonus", "convention_kind",
]
