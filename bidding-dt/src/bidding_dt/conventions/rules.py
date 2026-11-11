"""System-convention consistency checks and convention-usage rewards.

The enforced system (all strength numbers are Kaplan-Rubens points, kr.py,
with a configurable tolerance on every range):

  * 5-card majors; 1D promises 3+, 1C 2+; open 12+ (rule-of-20 escape,
    lighter in 3rd/4th seat)
  * 1NT = 15-17 balanced; 2NT = 20-21 balanced
  * 2C = strong artificial (game force until opener rebids 2NT);
    2D/2H/2S = weak two (6 cards, ~5-10)
  * 2/1 responses are game forcing (~12+ and a real suit); passing below
    game after any GF bid (2/1, strong 2C, fourth suit, XYZ 2D) is a
    violation
  * invitational jump shift: a single jump to a new suit at the 2 or 3
    level shows exactly 6+ cards and 10-11 -- in competition too (jump
    overcalls included); opener's jump-shift rebid stays the standard
    strong one (~17+)
  * splinter: a double jump shift below 4H shows singleton/void in the bid
    suit, fit for partner's last suit, slam interest
  * smolen: after a 1NT/2NT opening, stayman (2C/3C) and the 2D/3D denial,
    a 3M bid by responder is GF showing 4 in the bid major and 5+ in the
    other major
  * standard overcalls: 5+ suit (4 only with ~13+ at the 1 level), values
    by level, lighter in balancing seat; direct 1NT = 15-18 + stopper,
    balancing 1NT = 12-15
  * takeout doubles: opening-ish values + 3+ in every unbid major (lighter
    in balancing seat); a power double (18+ K&R) needs no specific shape;
    negative doubles need values or an unbid 4-card major; doubles once our
    side has bid are penalty-oriented and unchecked
  * 4NT is RKCB in the last suit the partnership bid: the ask itself is not
    evaluated, but the asker's partner must answer with the step showing the
    correct keycard count (5C = 0/3, 5D = 1/4, 5H = 2 without the trump
    queen, 5S = 2 with it); with no suit bid 4NT stays quantitative
    (unchecked)
  * a cuebid of the opponents' suit shows invitational-or-better support
    for partner's last suit (with no suit bid by partner it is undefined
    and unchecked)
  * XYZ after an uncontested 1x-1y-1z: 2C = relay (any invite or a diamond
    signoff), 2D = artificial GF, opener completes the relay without extras
  * fourth suit is artificial and game forcing (~11+)
  * standard natural glue: 1NT/2NT/3NT response ranges, simple and limit
    raises (support + values), opener's 1NT/2NT rebid ranges, positive
    responses to 2C

Each violated rule emits a `Violation` whose weight (DEFAULT_WEIGHTS,
overridable per rule) is summed into the call's penalty; the RL loop
multiplies that by `rl.conv_weight` and subtracts it from the row reward
(see rl/rollout.py). Correctly using one of the artificial conventions
(XYZ 2C/2D, RKCB ask/response, splinter, smolen, fourth-suit GF, cuebid
raise) instead earns the small flat `conv_bonus` reward (convention_bonus).
Rules that a hand cannot violate (pass in a non-GF auction, redouble, game-
level natural bids, passed-hand jumps) deliberately emit nothing -- the
checker is lenient by design so it punishes system breaks, not style.
"""

from dataclasses import dataclass, field

import numpy as np

from ..data.hands import HAND_DIM, HCP_WEIGHTS, NUM_RANKS, encode_hand
from .auction import (DENOM_N, MAJORS, MINORS, AuctionReader, CallInfo,
                      bid_id, is_game_or_above)
from .kr import kr_from_matrix

# denom (C,D,H,S) -> row in the data/hands.py (S,H,D,C) indicator matrix
_DENOM_ROW = (3, 2, 1, 0)
_DENOMS = "CDHSN"

DEFAULT_WEIGHTS = {
    # --- the system the user specified (weight 1.0) ---
    "open_1nt_range": 1.0,      # 1NT = 15-17 balanced
    "open_2nt_range": 1.0,      # 2NT = 20-21 balanced
    "open_2c_strong": 1.0,      # 2C = strong artificial
    "open_major_5card": 1.0,    # 5-card majors
    "open_weak2": 0.1,          # 2D/2H/2S = weak two
    "two_one_gf": 1.0,          # 2/1 responses are GF
    "inv_jump_shift": 2.0,      # single jump: 6 cards, 10-11 (also in comp)
    "splinter": 2.0,            # double jump below 4H: shortness + fit
    "smolen": 1.0,              # 3M after a stayman denial: 4 + 5 majors, GF
    "overcall": 1.0,            # standard overcalls
    "takeout_double": 1.0,      # takeout doubles (power doubles shape-free)
    "rkcb_response": 1.0,       # step answer shows the right keycard count
    "cuebid_support": 1.0,      # their-suit cuebid: invite+ fit for partner
    "xyz_2c": 1.0,              # XYZ 2C = relay (invite or diamond signoff)
    "xyz_2d": 1.0,              # XYZ 2D = artificial GF
    "fourth_suit": 1.0,         # fourth suit is artificial and GF
    "gf_pass": 2.0,             # never pass below game once GF
    # --- standard natural glue (softer, weight 0.5) ---
    "open_values": 0.5,
    "open_minor_length": 0.5,
    "resp_1nt": 0.5,
    "resp_raise": 0.5,
    "resp_jump_raise": 0.5,
    "resp_2nt": 0.5,
    "resp_3nt": 0.5,
    "rebid_1nt": 0.5,
    "rebid_2nt": 0.5,
    "resp_2c_positive": 0.5,
    "strong_jump_rebid": 0.5,
    "negative_double": 1.5,
    "overcall_1nt": 0.5,
    "overcall_1nt_bal": 0.5,
    "xyz_relay": 2.5,
    "xyz_pass": 2.5,
}


@dataclass
class ConventionConfig:
    """Thresholds and per-rule weights for the convention checker.

    Every range is applied with +-kr_tolerance slack. The tolerance matters
    because K&R is its own scale, not HCP: it evaluates quack-heavy flat
    hands below their Milton-Work count (a 15-HCP 4333 of AJ74/KQ6/AJ7 is
    14.0 K&R) and distributional hands above it (a 10-HCP invitational jump
    shift with KQJT98 is ~12.8 K&R). The default 1.0 makes the stated ranges
    (e.g. 15-17 1NT) cover the normal hands a player would accept for them
    while still flagging clear breaks; verified against Pavlicek's evaluator.
    """
    kr_tolerance: float = 1.0
    # openings
    nt_range: tuple = (15.0, 17.0)
    nt2_range: tuple = (20.0, 21.0)
    two_c_min_kr: float = 20.5
    two_c_suit_len: int = 6          # long-strong alternative to flat values
    two_c_suit_min_kr: float = 16.5
    two_c_longsuit_len: int = 8
    weak2_range: tuple = (4.0, 12.0)   # K&R; ~5-10 HCP with a 6-card suit
    weak2_len: tuple = (5, 7)
    open_min_kr: float = 11.0
    open_min_kr_late: float = 9.5    # 3rd/4th seat
    open_rule20: bool = True         # HCP + two longest suits >= 20 also opens
    # responses
    resp_1nt_range: tuple = (5.0, 12.0)
    raise_range: tuple = (5.0, 12.5)
    raise_support_major: int = 3
    raise_support_minor: int = 4
    jump_raise_range: tuple = (10.0, 13.0)
    jump_raise_support_major: int = 3
    jump_raise_support_minor: int = 4
    resp_2nt_range: tuple = (10.5, 12.5)
    resp_3nt_range: tuple = (12.5, 15.5)
    resp_2c_pos_min_kr: float = 7.5
    resp_2c_pos_min_len: int = 5
    # rebids
    rebid_1nt_range: tuple = (11.5, 14.5)
    rebid_2nt_range: tuple = (17.5, 19.5)
    strong_js_min_kr: float = 17.0
    strong_js_alt_kr: float = 14.0   # distributional alternative
    strong_js_alt_len: int = 6
    # game-forcing bids
    two_one_min_kr: float = 11.5
    two_one_min_len: int = 4
    two_one_min_len_clubs: int = 3
    js_range: tuple = (10.5, 12.5)     # invitational jump shift; K&R already
    #                                  # includes the 6-card suit's shape, so
    #                                  # this is the "10-11 HCP" invite band
    js_min_len: int = 6
    splinter_min_kr: float = 12.0
    splinter_max_len: int = 1        # singleton or void in the bid suit
    splinter_fit_major: int = 4
    splinter_fit_minor: int = 5
    fourth_suit_min_kr: float = 10.0
    # smolen (3M after a NT opening - stayman - 2D/3D denial)
    smolen_bid_len: int = 4          # cards in the bid major
    smolen_other_len: int = 5        # cards in the other major
    # their-suit cuebids: invitational+ support for partner's last suit
    cuebid_min_kr: float = 10.0
    cuebid_fit_major: int = 3
    cuebid_fit_minor: int = 4
    # overcalls
    overcall1_min_kr: float = 7.0
    overcall2_min_kr: float = 10.0
    overcall3_min_kr: float = 12.5
    overcall1_bal_min_kr: float = 6.5
    overcall2_bal_min_kr: float = 8.0
    overcall_min_len: int = 5
    overcall3_min_len: int = 6
    overcall_bal1_min_len: int = 4
    overcall_short_ok_kr: float = 13.0   # 4-card 1-level overcall if strong
    overcall_nt_range: tuple = (14.5, 18.5)      # direct 1NT = 15-18
    overcall_nt_bal_range: tuple = (11.5, 15.5)  # balancing 1NT = 12-15
    # doubles
    takeout_min_kr: float = 10.0
    takeout_bal_min_kr: float = 8.0
    takeout_major_len: int = 3
    takeout_power_kr: float = 18.0   # power double: no shape requirement
    negative_min_kr: float = 5.5
    negative_values_kr: float = 8.5      # no unbid 4-card major -> need values
    # slam conventions (rkcb_min_kr/rkcb_min_fit now gate the ask's bonus;
    # the ask itself is not penalized -- only wrong step responses are)
    rkcb_min_kr: float = 11.0
    rkcb_min_fit: int = 3
    # XYZ
    xyz_2d_min_kr: float = 11.5
    xyz_2c_max_kr: float = 12.5          # GF values must bid 2D instead
    xyz_inv_min_kr: float = 9.5          # below this: diamond signoff only
    xyz_signoff_min_d: int = 5
    xyz_relay_break_kr: float = 15.5     # opener may bypass 2D with extras
    # small flat reward for correctly using an artificial convention
    # (XYZ 2C/2D, RKCB ask/response, splinter, smolen, fourth suit, cuebid
    # raise); 0 disables the bonus
    conv_bonus: float = 0.05
    # per-rule weights (0 disables a rule)
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def weight(self, code: str) -> float:
        return float(self.weights.get(code, 0.0))

    @classmethod
    def from_overrides(cls, rule_weights=None, params=None) -> "ConventionConfig":
        """Config with per-rule weight and threshold overrides applied."""
        cfg = cls()
        for code, w in (rule_weights or {}).items():
            if code not in DEFAULT_WEIGHTS:
                raise KeyError(f"unknown convention rule: {code!r}")
            cfg.weights[code] = float(w)
        for k, val in (params or {}).items():
            if k == "weights" or not hasattr(cfg, k):
                raise KeyError(f"unknown convention param: {k!r}")
            setattr(cfg, k, val)
        return cfg


@dataclass(frozen=True, slots=True)
class Violation:
    code: str
    weight: float
    detail: str = ""


class HandInfo:
    """Per-hand features shared by the rule checks."""
    __slots__ = ("m", "lengths", "kr", "hcp", "text")

    def __init__(self, hand):
        if isinstance(hand, HandInfo):
            self.m, self.lengths, self.kr, self.hcp, self.text = (
                hand.m, hand.lengths, hand.kr, hand.hcp, hand.text)
            return
        self.text = hand if isinstance(hand, str) else None
        vec = encode_hand(hand) if isinstance(hand, str) \
            else np.asarray(hand).astype(np.uint8)
        if vec.shape != (HAND_DIM,):
            raise ValueError(f"expected a ({HAND_DIM},) vector, got {vec.shape}")
        self.m = vec.reshape(4, NUM_RANKS) > 0        # rows S,H,D,C; A..2
        self.lengths = self.m.sum(axis=1)
        self.kr = kr_from_matrix(self.m.astype(np.int64))
        self.hcp = int((vec * HCP_WEIGHTS).sum())

    def length(self, denom: int) -> int:
        return int(self.lengths[_DENOM_ROW[denom]])

    def has(self, denom: int, rank: int) -> bool:
        """rank: 0=A 1=K 2=Q 3=J 4=T 5=9 6=8 (data/hands order)."""
        return bool(self.m[_DENOM_ROW[denom], rank])

    def top_honors(self, denom: int, k: int = 3) -> int:
        """Number of the top k honors (A,K,Q...) held in the suit."""
        return sum(int(self.has(denom, r)) for r in range(k))

    def stopper(self, denom: int) -> bool:
        n = self.length(denom)
        return (self.has(denom, 0) or (self.has(denom, 1) and n >= 2)
                or (self.has(denom, 2) and n >= 3)
                or (self.has(denom, 3) and n >= 4))

    def balanced(self) -> bool:
        s = tuple(sorted((int(x) for x in self.lengths), reverse=True))
        return s in ((4, 3, 3, 3), (4, 4, 3, 2), (5, 3, 3, 2))

    def longest_denom(self) -> int:
        row = int(np.argmax(self.lengths))
        return _DENOM_ROW.index(row)

    def two_longest(self) -> int:
        return int(sum(sorted((int(x) for x in self.lengths), reverse=True)[:2]))


def _in_range(kr: float, rng: tuple, tol: float) -> bool:
    return rng[0] - tol <= kr <= rng[1] + tol


def _check_opening(hi: HandInfo, info: CallInfo, cfg: ConventionConfig,
                   v) -> None:
    tol = cfg.kr_tolerance
    level, denom = info.level, info.denom
    if denom == DENOM_N:
        if level == 1:
            if not hi.balanced():
                v("open_1nt_range", "unbalanced")
            elif not _in_range(hi.kr, cfg.nt_range, tol):
                v("open_1nt_range", f"K&R {hi.kr:.2f} outside 15-17")
        elif level == 2:
            if not hi.balanced():
                v("open_2nt_range", "unbalanced")
            elif not _in_range(hi.kr, cfg.nt2_range, tol):
                v("open_2nt_range", f"K&R {hi.kr:.2f} outside 20-21")
        return
    if level == 2 and denom == 0:                       # strong 2C
        long_d = hi.longest_denom()
        strong = (hi.kr >= cfg.two_c_min_kr - tol
                  or hi.length(long_d) >= cfg.two_c_longsuit_len
                  or (hi.length(long_d) >= cfg.two_c_suit_len
                      and hi.top_honors(long_d) >= 2
                      and hi.kr >= cfg.two_c_suit_min_kr - tol))
        if not strong:
            v("open_2c_strong", f"K&R {hi.kr:.2f}, no long strong suit")
        return
    if level == 2:                                      # weak two
        n = hi.length(denom)
        lo, hi_ = cfg.weak2_range
        mn, mx = cfg.weak2_len
        if not (mn <= n <= mx):
            v("open_weak2", f"{n}-card {_DENOMS[denom]}")
        elif not (lo - tol <= hi.kr <= hi_ + tol):
            v("open_weak2", f"K&R {hi.kr:.2f} outside weak-two range")
        return
    if level == 1:
        if denom in MAJORS and hi.length(denom) < 5:
            v("open_major_5card", f"only {hi.length(denom)} cards")
        if denom in MINORS:
            need = 3 if denom == 1 else 2
            if hi.length(denom) < need:
                v("open_minor_length", f"only {hi.length(denom)} cards")
            elif any(hi.length(md) >= 5 and hi.length(md) > hi.length(denom)
                     for md in MAJORS):
                v("open_minor_length", "5+ major belongs in the major")
        late = info.n_prior_passes >= 2
        min_kr = (cfg.open_min_kr_late if late else cfg.open_min_kr) - tol
        rule20 = cfg.open_rule20 and hi.hcp + hi.two_longest() >= 20
        if hi.kr < min_kr and not rule20:
            v("open_values", f"K&R {hi.kr:.2f} is a pass")


def _check_double(reader: AuctionReader, hi: HandInfo, info: CallInfo,
                  cfg: ConventionConfig, v) -> None:
    tol = cfg.kr_tolerance
    ctx = info.dbl_ctx
    if ctx == "takeout":
        if hi.kr >= cfg.takeout_power_kr - tol:
            return                       # power double: 18+ needs no shape
        unbid_majors = [d for d in MAJORS if d not in reader.suits_bid]
        shape_ok = all(hi.length(d) >= cfg.takeout_major_len
                       for d in unbid_majors)
        if hi.kr < cfg.takeout_min_kr - tol or not shape_ok:
            v("takeout_double",
              f"K&R {hi.kr:.2f}, majors {[hi.length(d) for d in unbid_majors]}")
    elif ctx == "balancing_takeout":
        if hi.kr < cfg.takeout_bal_min_kr - tol:
            v("takeout_double", f"balancing: K&R {hi.kr:.2f}")
    elif ctx == "negative":
        unbid_majors = [d for d in MAJORS if d not in reader.suits_bid]
        has_major = any(hi.length(d) >= 4 for d in unbid_majors)
        if hi.kr < cfg.negative_min_kr - tol or (
                not has_major and hi.kr < cfg.negative_values_kr - tol):
            v("negative_double", f"K&R {hi.kr:.2f}, no unbid 4-card major")
    # penalty / nt_penalty doubles (our side has bid) are unchecked


def check_call(reader: AuctionReader, hero: int, hand, action: int,
               cfg: ConventionConfig | None = None) -> list[Violation]:
    """Violations committed by `hero` making `action` in `reader`'s auction.

    `hand` is a 'S.H.D.C' string, a (52,) indicator, or a HandInfo. One
    violation per rule code is reported (the first reason wins).
    """
    cfg = cfg or ConventionConfig()
    info = reader.classify(action, seat=hero)
    out: list[Violation] = []
    tol = cfg.kr_tolerance

    def v(code: str, detail: str = "") -> None:
        w = cfg.weight(code)
        if w > 0:
            out.append(Violation(code, w, detail))

    if info.kind == "redouble":
        return out

    if info.kind == "pass":
        if (info.gf_active
                and not is_game_or_above(reader.high_level, reader.high_denom)):
            v("gf_pass", "passed below game with a game force")
        if info.xyz_after_relay:
            # passing the forced 2D is the diamond signoff: only legal
            # without invitational values (a weak hand without diamonds was
            # already penalized when it bid 2C)
            hi = HandInfo(hand)
            if hi.kr >= cfg.xyz_inv_min_kr + tol:
                v("xyz_pass", f"invitational values (K&R {hi.kr:.2f}) "
                              f"passed the relay")
        return out

    if info.kind == "double":
        _check_double(reader, HandInfo(hand), info, cfg, v)
        return out

    return _check_bid(reader, HandInfo(hand), info, cfg, v, out)


def _check_bid(reader: AuctionReader, hi: HandInfo, info: CallInfo,
               cfg: ConventionConfig, v, out: list[Violation]) -> list[Violation]:
    tol = cfg.kr_tolerance
    level, denom = info.level, info.denom

    if info.opening:
        _check_opening(hi, info, cfg, v)
        return _dedupe(out)

    # strong-2C auctions run on their own agreements: only the positive
    # response structure is checked (GF passes are handled in the pass path)
    if info.side_strong2c:
        if info.strong2c and info.first_call:
            if denom == 1 and level == 2:
                pass                                  # 2D = waiting, any hand
            elif denom < DENOM_N:
                if (hi.kr < cfg.resp_2c_pos_min_kr - tol
                        or hi.length(denom) < cfg.resp_2c_pos_min_len):
                    v("resp_2c_positive",
                      f"K&R {hi.kr:.2f}, {hi.length(denom)} cards")
            elif level == 2:
                if hi.kr < cfg.resp_2c_pos_min_kr - tol:
                    v("resp_2c_positive", f"2NT positive with K&R {hi.kr:.2f}")
        return _dedupe(out)

    # XYZ (artificial meanings override the natural reading of 2C/2D)
    if info.xyz_2c:
        if hi.kr > cfg.xyz_2c_max_kr + tol:
            v("xyz_2c", f"GF values (K&R {hi.kr:.2f}) must bid 2D")
        elif (hi.kr < cfg.xyz_inv_min_kr - tol
              and hi.length(1) < cfg.xyz_signoff_min_d):
            v("xyz_2c", f"neither invitational nor a diamond signoff "
                        f"(K&R {hi.kr:.2f}, {hi.length(1)} diamonds)")
        return _dedupe(out)
    if info.xyz_2d:
        if hi.kr < cfg.xyz_2d_min_kr - tol:
            v("xyz_2d", f"artificial GF with K&R {hi.kr:.2f}")
        return _dedupe(out)
    if info.xyz_relay:
        if info.call_id != bid_id(2, 1) and hi.kr < cfg.xyz_relay_break_kr - tol:
            v("xyz_relay", f"broke the relay with K&R {hi.kr:.2f}")
        return _dedupe(out)

    if info.smolen:
        other = MAJORS[0] if denom == MAJORS[1] else MAJORS[1]
        bad = []
        if hi.length(denom) < cfg.smolen_bid_len:
            bad.append(f"only {hi.length(denom)} in the bid major")
        if hi.length(other) < cfg.smolen_other_len:
            bad.append(f"only {hi.length(other)} in the other major")
        if bad:
            v("smolen", "; ".join(bad))
        return _dedupe(out)

    if info.rkcb_response:
        trump = info.rkcb_trump
        keys = sum(int(hi.has(d, 0)) for d in range(4)) + int(hi.has(trump, 1))
        has_q = hi.has(trump, 2)
        step = info.rkcb_step
        ok = ((step == 1 and keys in (0, 3))
              or (step == 2 and keys in (1, 4))
              or (step == 3 and keys == 2 and not has_q)
              or (step == 4 and keys == 2 and has_q))
        if not ok:
            v("rkcb_response",
              f"{keys} keycards{' + trump Q' if has_q else ''} but step "
              f"{step} shows the wrong count")
        return _dedupe(out)

    if info.is_2over1:
        min_len = (cfg.two_one_min_len_clubs if denom == 0
                   else cfg.two_one_min_len)
        bad = []
        if hi.kr < cfg.two_one_min_kr - tol:
            bad.append(f"K&R {hi.kr:.2f}")
        if hi.length(denom) < min_len:
            bad.append(f"{hi.length(denom)} cards")
        if bad:
            v("two_one_gf", "; ".join(bad))
        return _dedupe(out)

    if info.fourth_suit:
        if hi.kr < cfg.fourth_suit_min_kr - tol:
            v("fourth_suit", f"artificial GF with K&R {hi.kr:.2f}")
        return _dedupe(out)

    if info.is_jump_shift:
        lo, hi_ = cfg.js_range
        bad = []
        if hi.length(denom) < cfg.js_min_len:
            bad.append(f"{hi.length(denom)} cards")
        if not (lo - tol <= hi.kr <= hi_ + tol):
            bad.append(f"K&R {hi.kr:.2f} outside {lo:.0f}-{hi_:.0f}")
        if bad:
            v("inv_jump_shift", "; ".join(bad))
        return _dedupe(out)

    if info.is_strong_js:
        ok = (hi.kr >= cfg.strong_js_min_kr - tol
              or (hi.length(denom) >= cfg.strong_js_alt_len
                  and hi.kr >= cfg.strong_js_alt_kr - tol))
        if not ok:
            v("strong_jump_rebid", f"K&R {hi.kr:.2f}")
        return _dedupe(out)

    if info.splinter:
        fit = info.partner_last_suit
        need = cfg.splinter_fit_major if fit in MAJORS else cfg.splinter_fit_minor
        bad = []
        if hi.length(denom) > cfg.splinter_max_len:
            bad.append(f"{hi.length(denom)} cards in the splinter suit")
        if hi.length(fit) < need:
            bad.append(f"only {hi.length(fit)}-card fit for partner")
        if hi.kr < cfg.splinter_min_kr - tol:
            bad.append(f"K&R {hi.kr:.2f}")
        if bad:
            v("splinter", "; ".join(bad))
        return _dedupe(out)

    # notrum bids
    if denom == DENOM_N:
        if level == 4:
            pass    # RKCB ask (its step responses are checked instead) or
            #       # quantitative 4NT: neither is evaluated
        elif info.overcall and level == 1:
            if info.balancing:
                if not _in_range(hi.kr, cfg.overcall_nt_bal_range, tol):
                    v("overcall_1nt_bal", f"K&R {hi.kr:.2f} outside 12-15")
            else:
                if not _in_range(hi.kr, cfg.overcall_nt_range, tol):
                    v("overcall_1nt", f"K&R {hi.kr:.2f} outside 15-18")
                elif (0 <= reader.high_denom < DENOM_N
                      and not hi.stopper(reader.high_denom)):
                    v("overcall_1nt", "no stopper in their suit")
        elif info.responder or info.advancer:
            if level == 1:
                if not _in_range(hi.kr, cfg.resp_1nt_range, tol):
                    v("resp_1nt", f"K&R {hi.kr:.2f} outside 6-10ish")
                elif info.responder:
                    for md in MAJORS:
                        if (hi.length(md) >= 4
                                and bid_id(1, md) > reader.highest):
                            v("resp_1nt", "biddable 4-card major withheld")
                            break
            elif level == 2 and info.jump == 1:
                if not hi.balanced():
                    v("resp_2nt", "unbalanced")
                elif not _in_range(hi.kr, cfg.resp_2nt_range, tol):
                    v("resp_2nt", f"K&R {hi.kr:.2f} outside 11-12")
            elif (level == 3 and info.jump == 2 and info.responder
                  and info.first_call):
                if not hi.balanced():
                    v("resp_3nt", "unbalanced")
                elif not _in_range(hi.kr, cfg.resp_3nt_range, tol):
                    v("resp_3nt", f"K&R {hi.kr:.2f} outside 13-15")
        elif info.opener_rebid:
            if level == 1 and not _in_range(hi.kr, cfg.rebid_1nt_range, tol):
                v("rebid_1nt", f"K&R {hi.kr:.2f} outside 12-14")
            elif (level == 2 and info.jump == 1
                  and not _in_range(hi.kr, cfg.rebid_2nt_range, tol)):
                v("rebid_2nt", f"K&R {hi.kr:.2f} outside 18-19")
        return _dedupe(out)

    if info.cuebid:
        # cuebid of the opponents' suit: invitational or better support for
        # partner's last suit; with no suit bid by partner it is a two-suited
        # / strong call and stays unchecked
        fit = info.partner_last_suit
        if fit >= 0:
            need = (cfg.cuebid_fit_major if fit in MAJORS
                    else cfg.cuebid_fit_minor)
            bad = []
            if hi.length(fit) < need:
                bad.append(f"only {hi.length(fit)}-card support for "
                           f"partner's {_DENOMS[fit]}")
            if hi.kr < cfg.cuebid_min_kr - tol:
                bad.append(f"K&R {hi.kr:.2f} below invitational")
            if bad:
                v("cuebid_support", "; ".join(bad))
        return _dedupe(out)

    if info.own_rebid:
        return _dedupe(out)         # natural rebid: unchecked

    if info.raise_bid:
        # descriptive raises only on the first chance to support (later
        # "raises" may be simple preferences); game raises are natural
        if info.first_call and not is_game_or_above(level, denom):
            fit = hi.length(denom)
            if info.jump == 0:
                need = (cfg.raise_support_major if denom in MAJORS
                        else cfg.raise_support_minor)
                if fit < need:
                    v("resp_raise", f"only {fit}-card support")
                elif not _in_range(hi.kr, cfg.raise_range, tol):
                    v("resp_raise", f"K&R {hi.kr:.2f} outside 6-10ish")
            elif info.jump == 1:
                need = (cfg.jump_raise_support_major if denom in MAJORS
                        else cfg.jump_raise_support_minor)
                if fit < need:
                    v("resp_jump_raise", f"only {fit}-card support")
                elif not _in_range(hi.kr, cfg.jump_raise_range, tol):
                    v("resp_jump_raise", f"K&R {hi.kr:.2f} outside 10-12")
        return _dedupe(out)

    if info.overcall and info.new_suit and info.jump == 0:
        if info.balancing:
            min_kr = (cfg.overcall1_bal_min_kr if level == 1
                      else cfg.overcall2_bal_min_kr)
            min_len = (cfg.overcall_bal1_min_len if level == 1
                       else cfg.overcall_min_len)
        else:
            min_kr = (cfg.overcall1_min_kr if level == 1 else
                      cfg.overcall2_min_kr if level == 2
                      else cfg.overcall3_min_kr)
            min_len = (cfg.overcall3_min_len if level >= 3
                       else cfg.overcall_min_len)
        if hi.kr < min_kr - tol or hi.length(denom) < min_len:
            # a 4-card 1-level overcall is fine with a strong hand
            if not (level == 1 and not info.balancing
                    and hi.length(denom) == 4
                    and hi.kr >= cfg.overcall_short_ok_kr - tol):
                v("overcall",
                  f"K&R {hi.kr:.2f}, {hi.length(denom)} cards at the "
                  f"{level} level")
    return _dedupe(out)


def _dedupe(out: list[Violation]) -> list[Violation]:
    seen = set()
    res = []
    for viol in out:
        if viol.code not in seen:
            seen.add(viol.code)
            res.append(viol)
    return res


def call_violations(hand, dealer: int, vuln: int, calls, action: int,
                    cfg: ConventionConfig | None = None) -> list[Violation]:
    """Convenience wrapper: replay `calls` (dealer order), check `action`."""
    reader = AuctionReader(dealer, vuln)
    for c in calls:
        reader.feed(int(c))
    hero = (dealer + len(calls)) % 4
    return check_call(reader, hero, hand, int(action), cfg)


def call_penalty(hand, dealer: int, vuln: int, calls, action: int,
                 cfg: ConventionConfig | None = None) -> float:
    """Total weighted penalty of `action` (0.0 = consistent with the system)."""
    return sum(v.weight for v in call_violations(hand, dealer, vuln, calls,
                                                 action, cfg))


# ---------------------------------------------------------------------------
# Convention-usage rewards
# ---------------------------------------------------------------------------

def convention_kind(info: CallInfo) -> str | None:
    """The rewarded convention a classified call uses (None for plain bids).

    The cuebid only counts as the convention when partner has bid a suit --
    that is the case the system defines (invitational+ support); with a
    silent partner the call is a two-suited/strong cuebid and earns nothing.
    """
    if info.xyz_2c:
        return "xyz_2c"
    if info.xyz_2d:
        return "xyz_2d"
    if info.smolen:
        return "smolen"
    if info.rkcb_ask:
        return "rkcb_ask"
    if info.rkcb_response:
        return "rkcb_response"
    if info.splinter:
        return "splinter"
    if info.fourth_suit:
        return "fourth_suit"
    if info.cuebid and info.partner_last_suit >= 0:
        return "cuebid_support"
    return None


def convention_bonus(info: CallInfo, hi: HandInfo, violations,
                     cfg: ConventionConfig) -> float:
    """Small reward for one clean use of an artificial convention.

    `info` is the classified call, `hi` the bidder's hand and `violations`
    the check_call result for the same call: any violation on it forfeits
    the bonus (a broken convention is punished, not rewarded). The RKCB ask
    itself is no longer rule-checked -- only its step responses are -- so
    its bonus instead requires the trump fit and slam values of a proper
    ask (rkcb_min_fit / rkcb_min_kr).
    """
    if cfg.conv_bonus <= 0 or violations:
        return 0.0
    kind = convention_kind(info)
    if kind is None:
        return 0.0
    if kind == "rkcb_ask" and (
            hi.length(info.rkcb_trump) < cfg.rkcb_min_fit
            or hi.kr < cfg.rkcb_min_kr - cfg.kr_tolerance):
        return 0.0
    return cfg.conv_bonus


def call_bonus(reader: AuctionReader, hero: int, hand, action: int,
               cfg: ConventionConfig | None = None) -> float:
    """Convenience wrapper: the convention bonus of `action` in `reader`'s
    auction (0.0 when it is not a correctly used convention)."""
    cfg = cfg or ConventionConfig()
    if cfg.conv_bonus <= 0:
        return 0.0
    info = reader.classify(action, seat=hero)
    hi = HandInfo(hand)
    return convention_bonus(info, hi,
                            check_call(reader, hero, hi, action, cfg), cfg)
