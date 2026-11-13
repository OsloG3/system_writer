"""Bidding-system conventions: hand evaluation + consistency checks.

`kr.py` implements the Kaplan-Rubens point count (the 26-step evaluator from
rpbridge.net/8j19.htm, verified against Pavlicek's own CGI). `auction.py`
replays an auction and structurally classifies each call (opening, 2/1,
jump shift, splinter, smolen, overcall, XYZ, fourth suit, RKCB ask/response,
double meaning, per-side game-force status). `rules.py` ties the two
together: given the hero's hand, the auction so far and a candidate call,
`call_violations` reports the built-in system conventions that call would
break, each with a weight, `call_penalty` sums them, and `call_bonus`
rewards one clean use of an artificial convention. `alerts.py` renders the
built-in alert texts (the fallback for tooling without a system file).

The system itself is WRITTEN IN A TEXT FILE -- models/system.txt -- in the
sequence-pattern language of `patterns.py`: one entry per call kind
(`1X-1Y-2Z; X<Z = reverse: 5+X 4+Z p>16 F @2.0`), with generic suits,
opponent calls in parentheses, clause conditions and OR-split descriptions
that are both the website's alert text and the training spec.
`system_file.py` parses the file (pattern entries + top-level config lines);
`patterns.SystemEnforcer` enforces the entries per call during RL training,
falling back to the built-in `rules.py` checks for calls no entry matches.
The Go server (system_writer/patterns.go + system_file.go) implements the
same grammar and matching over the same file to generate the website's bid
alerts -- the tests on both sides pin them to the same goldens.

Used by the RL trainer (rl/rollout.py + rl/train_ppo.py, `rl.conv_weight`)
to penalize the decision-transformer policy for bids inconsistent with the
system: 5-card majors, 15-17 1NT, strong 2C, 20-21 2NT, GF 2/1,
invitational jump shifts (6 cards, 10-11, also in competition), standard
overcalls (balancing 1NT = 12-15), splinter double jumps below 4H, smolen
after 1/2NT stayman, RKCB step responses (the 4NT ask itself is free),
support cuebids of the opponents' suit, XYZ, doubles with TO/PENALTY
partner discipline, fourth-suit GF, reverses and standard natural bidding.
Correctly using XYZ, RKCB, splinter, smolen, fourth suit or a support
cuebid earns the small `conv_bonus` reward. The enforced numbers live in
models/system.txt; rules.py holds the built-in defaults and rule codes.
"""

from .alerts import RKCB_STEPS, convention_alert
from .auction import AuctionReader, CallInfo, is_game_or_above
from .kr import kr_points, kr_suit
from .patterns import (SPEC_PARAMS, Entry, MatchState, SystemEnforcer,
                       call_id_of, fold_state, hand_violations, match_first,
                       parse_entry, render_desc, replay_state)
from .rules import (DEFAULT_WEIGHTS, ConventionConfig, HandInfo, Violation,
                    call_bonus, call_penalty, call_violations, check_call,
                    convention_bonus, convention_kind)
from .system_file import SystemSpec, find_system_file, parse_system_file

__all__ = [
    "RKCB_STEPS", "convention_alert",
    "AuctionReader", "CallInfo", "is_game_or_above",
    "kr_points", "kr_suit",
    "SPEC_PARAMS", "Entry", "MatchState", "SystemEnforcer", "call_id_of",
    "fold_state", "hand_violations", "match_first", "parse_entry",
    "render_desc", "replay_state",
    "DEFAULT_WEIGHTS", "ConventionConfig", "HandInfo", "Violation",
    "call_bonus", "call_penalty", "call_violations", "check_call",
    "convention_bonus", "convention_kind",
    "SystemSpec", "find_system_file", "parse_system_file",
]
