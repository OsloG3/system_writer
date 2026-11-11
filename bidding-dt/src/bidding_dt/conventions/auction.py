"""Sequential auction reader: the structural half of the convention checks.

`AuctionReader` is fed the calls of an auction in dealer order and maintains
everything the rules need to know about what a candidate call *means* in the
system: who opened/overcalled, whether the hero is responder/advancer/opener,
jump distance above the cheapest legal bid in the bid suit, raises vs new
suits vs cuebids, XYZ / fourth-suit / 2-over-1 contexts, Smolen (Stayman
state machine over 1NT-2C-2D / 2NT-3C-3D), RKCB asks and step responses,
double meanings (takeout / negative / penalty -- the penalty contexts are
structural only and no longer rule-checked) and per-side game-force status.

Classification is purely structural (no hands involved), so the same code
path both tracks agreements while replaying an auction prefix and describes a
candidate action at the decision point. rules.py attaches the hand-strength /
shape requirements and emits violations.

Jump counting is relative to the cheapest legal bid in the bid denomination
(`bid_id(lmin, denom) > highest so far`). One bridge-specific wrinkle: a
2-level new-suit response to a 1-level opening is a 2/1 response (never a
jump shift -- 1C-2D is the canonical 2/1), so a level-2 single "jump" only
stays a 2/1 when the opponents have not bid; with interference it becomes the
invitational jump shift (1C-(1D)-2S), which is what "applies in competition
as well" means for this system.

Denominations use the project order C=0 D=1 H=2 S=3 N=4 (data/vocab.DENOMS);
seats are absolute N=0 E=1 S=2 W=3, side = seat % 2 (0 = N-S).
"""

from dataclasses import dataclass, field
from copy import deepcopy

from ..data.vocab import CALL_PASS, CALL_X, CALL_XX, FIRST_BID

MAJORS = (2, 3)      # H, S in project denom order
MINORS = (0, 1)      # C, D
DENOM_N = 4


def bid_id(level: int, denom: int) -> int:
    """Call id of a bid (level 1-7, denom 0C 1D 2H 3S 4N); matches vocab."""
    return FIRST_BID + (level - 1) * 5 + denom


def is_bid(call: int) -> bool:
    return call >= FIRST_BID


def bid_level_of(call: int) -> int:
    return (call - FIRST_BID) // 5 + 1


def bid_denom_of(call: int) -> int:
    return (call - FIRST_BID) % 5


def is_game_or_above(level: int, denom: int) -> bool:
    """3NT+, 4M+, 5m+ (and any 6/7-level bid) fulfill a game force."""
    if level >= 6:
        return True
    if denom == DENOM_N:
        return level >= 3
    if denom in MAJORS:
        return level >= 4
    return level >= 5


@dataclass
class CallInfo:
    """Structural description of one (candidate) call in its auction."""
    seat: int = 0
    kind: str = "bid"             # pass | double | redouble | bid
    # bid-only fields
    level: int = 0
    denom: int = -1               # 0C 1D 2H 3S 4N
    call_id: int = 0
    jump: int = 0                 # levels above the cheapest legal bid in
    #                             # this denom (0 = no jump)
    new_suit: bool = False        # suit hero has not bid before
    raise_bid: bool = False       # partner's suit, hero has not bid it
    own_rebid: bool = False       # hero's own previously-bid suit
    cuebid: bool = False          # opponents' last-bid suit (invitational+
    #                             # support for partner's last suit)
    # role / context flags
    first_call: bool = False      # hero has not called before
    passed_hand: bool = False     # hero called before but never bid
    opening: bool = False         # first bid of the auction
    responder: bool = False       # partner opened; hero silent so far
    advancer: bool = False        # partner overcalled; hero silent so far
    opener_rebid: bool = False    # hero is the opener, calling again
    overcall: bool = False        # hero's side silent, opponents have bid
    balancing: bool = False       # 2+ passes since the last bid
    strong2c: bool = False        # partner opened the strong 2C
    side_strong2c: bool = False   # hero's side opened the strong 2C
    uncontested: bool = True      # no opponent bid/X/XX so far
    # system meanings
    is_2over1: bool = False       # GF 2-level response to a 1-level opening
    is_jump_shift: bool = False   # single jump, new suit, level 2-3
    #                             # (invitational in this system)
    is_strong_js: bool = False    # opener's jump-shift rebid (standard: ~19)
    splinter: bool = False        # double jump shift below 4H
    smolen: bool = False          # 3M after a NT-stayman denial: 4 in the bid
    #                             # major + 5 in the other, game forcing
    fourth_suit: bool = False     # completes the 4th suit for the partnership
    xyz: bool = False             # responder's call in a live XYZ auction
    xyz_2c: bool = False          # ... bid exactly 2C (relay: invite/signoff)
    xyz_2d: bool = False          # ... bid exactly 2D (artificial GF)
    xyz_relay: bool = False       # opener's bid after partner's XYZ 2C
    xyz_after_relay: bool = False  # responder's call after XYZ 2C-2D
    rkcb_ask: bool = False        # 4NT keycard ask in the side's last suit
    rkcb_response: bool = False   # step answer (5C..5S) to partner's RKCB ask
    rkcb_step: int = 0            # 1..4 = 5C..5S (0/3, 1/4, 2 no Q, 2 with Q)
    rkcb_trump: int = -1          # keycard suit of the ask / response
    # state snapshots at the moment of the call
    gf_active: bool = False       # hero's side is game-forced
    partner_last_suit: int = -1   # denom partner last bid (-1 none)
    side_last_suit: int = -1      # denom hero's side last bid (-1 none)
    dbl_ctx: str = ""             # takeout|balancing_takeout|negative|
    #                             # penalty|nt_penalty (the penalty contexts
    #                             # are structural only: not rule-checked)
    passes_since_bid: int = 0
    n_prior_passes: int = 0       # consecutive passes right before this call


@dataclass
class AuctionReader:
    """Feeds calls in dealer order; classifies any candidate call."""
    dealer: int = 0
    vuln: int = 0
    calls: list = field(default_factory=list)

    def __post_init__(self):
        self.n = 0
        self.highest = 0               # highest bid call id (0 = none)
        self.high_level = 0
        self.high_denom = -1
        self.high_seat = -1
        self.doubled = False
        self.redoubled = False
        self.passes_since_bid = 0
        self.gf = [False, False]       # per-side game-force status
        self.called = [False] * 4
        self.seat_suits = [set() for _ in range(4)]   # denoms bid per seat
        self.seat_nt = [False] * 4
        self.side_suits = [[], []]     # ordered (seat, denom) suit bids
        self.side_last_denom = [-1, -1]  # last suit each side bid (cuebids)
        self.side_has_bid = [False, False]
        self.suits_bid = set()         # union over both sides
        self.opener = -1
        self.open_level = 0
        self.open_denom = -1
        self.overcaller = -1
        self.opp_nonpass = 0           # opponent bid/X/XX calls so far
        self.opp_bids = 0              # opponent *bid* calls (2/1 exception)
        # strong 2C tracking
        self.strong2c_side = -1
        self.s2c_opener_calls = 0
        self.s2c_waiting = False       # responder's first call was 2D waiting
        # XYZ state machine: none -> s1 -> s2 -> s3 -> relay -> after_relay
        self.xyz_stage = "none"
        self.xyz_opener = -1
        self.xyz_for = -1              # responder seat with XYZ available
        self.xyz_2c_by = -1
        # stayman/Smolen state machine: none -> nt -> asked -> denied
        self.stayman_stage = "none"
        self.stayman_nt = 0            # NT opening level (1 or 2)
        self.stayman_opener = -1
        self.stayman_for = -1          # responder seat with Smolen available
        # RKCB: asker + trump latched from the 4NT ask until the partner's
        # next call consumes it (step responses are only checked undisturbed)
        self.rkcb_asker = -1
        self.rkcb_trump = -1
        self._pass_run = 0             # consecutive passes before next call

    # -- queries ------------------------------------------------------------

    def next_seat(self) -> int:
        return (self.dealer + self.n) % 4

    def _seat_bid(self, seat: int) -> bool:
        return bool(self.seat_suits[seat]) or self.seat_nt[seat]

    def _last_suit_of(self, side: int, seat: int = -1) -> int:
        for s, d in reversed(self.side_suits[side]):
            if seat < 0 or s == seat:
                return d
        return -1

    # -- classification -------------------------------------------------------

    def classify(self, call: int, seat: int | None = None) -> CallInfo:
        """Describe `call` made by `seat` (default: the seat to act).

        Reads the current state only -- nothing is mutated, so the same
        method describes a candidate action and (inside feed) a real one.
        """
        call = int(call)
        if not (CALL_PASS <= call <= bid_id(7, DENOM_N)):
            raise ValueError(f"bad call id: {call}")
        seat = self.next_seat() if seat is None else seat % 4
        side = seat % 2
        partner = (seat + 2) % 4
        info = CallInfo(seat=seat, call_id=call)
        info.first_call = not self.called[seat]
        info.passed_hand = self.called[seat] and not self._seat_bid(seat)
        info.gf_active = self.gf[side]
        info.uncontested = self.opp_nonpass == 0
        info.passes_since_bid = self.passes_since_bid
        info.n_prior_passes = self._pass_run
        info.partner_last_suit = self._last_suit_of(side, partner)
        info.side_last_suit = self._last_suit_of(side)
        info.balancing = self.passes_since_bid >= 2
        info.side_strong2c = self.strong2c_side == side

        side_bid = self.side_has_bid[side]
        partner_bid = self._seat_bid(partner)
        hero_bid = self._seat_bid(seat)

        if call == CALL_PASS:
            info.kind = "pass"
            # XYZ: responder passing opener's forced 2D -> diamond signoff
            if self.xyz_stage == "after_relay" and seat == self.xyz_2c_by:
                info.xyz_after_relay = True
            return info
        if call == CALL_X:
            info.kind = "double"
            info.dbl_ctx = self._double_ctx(side, hero_bid, partner_bid)
            return info
        if call == CALL_XX:
            info.kind = "redouble"
            return info

        # -- bids -------------------------------------------------------------
        level = bid_level_of(call)
        denom = bid_denom_of(call)
        info.level, info.denom = level, denom
        lmin = 1
        while bid_id(lmin, denom) <= self.highest:
            lmin += 1
        info.jump = level - lmin

        opp_last_denom = self.side_last_denom[1 - side]
        if denom < DENOM_N:
            info.cuebid = (denom == opp_last_denom
                           and denom not in self.seat_suits[seat]
                           and denom not in self.seat_suits[partner])
            info.own_rebid = denom in self.seat_suits[seat]
            info.raise_bid = (denom in self.seat_suits[partner]
                              and not info.own_rebid)
            info.new_suit = not info.own_rebid and not info.raise_bid

        # 4NT with a suit bid by our side is RKCB in that suit (the ask
        # itself is unchecked; rules.py evaluates the step responses)
        if denom == DENOM_N and level == 4 and info.side_last_suit >= 0:
            info.rkcb_ask = True
            info.rkcb_trump = info.side_last_suit

        # opening bid
        if self.highest == 0:
            info.opening = True
            return info

        # role
        if not side_bid:
            info.overcall = True       # hero's side has not bid yet
        elif not self.called[seat] and partner == self.opener:
            info.responder = True
        elif not self.called[seat] and partner_bid:
            info.advancer = True
        elif seat == self.opener:
            info.opener_rebid = True

        info.strong2c = (self.strong2c_side == side and info.responder)

        # XYZ (uncontested 1x-1y-1z): responder's 2C/2D are artificial, and
        # this overrides the natural raise/rebid reading of those bids, so it
        # is checked before the early returns below.
        if self.xyz_stage == "s3" and seat == self.xyz_for:
            info.xyz = True
            if call == bid_id(2, 0):
                info.xyz_2c = True
                return info
            if call == bid_id(2, 1):
                info.xyz_2d = True
                return info
        elif self.xyz_stage == "relay" and seat == self.xyz_opener:
            info.xyz_relay = True
        elif self.xyz_stage == "after_relay" and seat == self.xyz_2c_by:
            info.xyz_after_relay = True

        # Smolen: after a 1NT/2NT opening, stayman (2C/3C) and the 2D/3D
        # denial, responder's 3M shows 4 in the bid major + 5 in the other
        # and is game forcing. Intercepted here so 1NT-2C-2D-3M is not
        # misread as the invitational jump shift.
        if (self.stayman_stage == "denied" and seat == self.stayman_for
                and level == 3 and denom in MAJORS):
            info.smolen = True
            return info

        # RKCB step response: the asker's partner's next 5-level suit bid
        # (5C..5S = steps 1..4) answers the keycard ask.
        if (self.rkcb_asker >= 0 and seat == (self.rkcb_asker + 2) % 4
                and level == 5 and denom < DENOM_N):
            info.rkcb_response = True
            info.rkcb_step = denom + 1
            info.rkcb_trump = self.rkcb_trump
            return info

        if denom == DENOM_N or info.cuebid or info.own_rebid or info.raise_bid:
            return info              # NT / cuebid / rebid / raise: no shift logic

        # 2/1 game force: 2-level new-suit response to a 1-level opening by a
        # silent responder. A level-2 single "jump" only stays a 2/1 when the
        # opponents have not *bid* (doubles do not consume a level, so
        # 1C-(X)-2D is still a plain 2/1); with an intervening bid the jump
        # becomes the invitational jump shift.
        if (info.responder and info.first_call and level == 2
                and self.open_level == 1 and self.open_denom < DENOM_N
                and (info.jump == 0
                     or (info.jump == 1 and self.opp_bids == 0))):
            info.is_2over1 = True
            return info

        # fourth suit forcing (GF): partnership has bid exactly 3 distinct
        # suits, hero bids the missing one, hero has bid before, uncontested
        side_suits = {d for _, d in self.side_suits[side]}
        if (info.new_suit and info.uncontested and self._seat_bid(seat)
                and len(side_suits) == 3 and denom not in side_suits
                and self.xyz_stage == "none" and not info.side_strong2c):
            info.fourth_suit = True

        if not info.new_suit or info.side_strong2c:
            return info

        # single jump shift (level 2-3): invitational, 6 cards, 10-11 --
        # applies in competition too. Opener's jump-shift rebid is the
        # standard strong one instead; passed hands are undefined (skip).
        # In a live XYZ auction a direct level-2 jump is natural/non-forcing
        # (invites go through 2C), so the invitational reading is off there;
        # level-3 jumps keep it.
        if (info.jump == 1 and 2 <= level <= 3 and not info.passed_hand
                and not (info.xyz and level == 2)):
            if info.opener_rebid:
                info.is_strong_js = True
            elif side_bid or info.overcall:
                info.is_jump_shift = True

        # double jump shift below 4H: splinter (shortness + fit for
        # partner's last suit). Passed hands excluded above (cuebids too).
        if (info.jump == 2 and call < bid_id(4, MAJORS[0])
                and info.partner_last_suit >= 0 and not info.passed_hand):
            info.splinter = True
        return info

    def _double_ctx(self, side: int, hero_bid: bool, partner_bid: bool) -> str:
        if self.high_denom == DENOM_N:
            return "penalty" if hero_bid or partner_bid else "nt_penalty"
        if hero_bid:
            return "penalty"
        if partner_bid:
            return "negative"
        return "balancing_takeout" if self.passes_since_bid >= 2 else "takeout"

    # -- stepping ---------------------------------------------------------------

    def feed(self, call: int) -> CallInfo:
        """Classify the next call, then fold it into the auction state."""
        call = int(call)
        seat = self.next_seat()
        side = seat % 2
        info = self.classify(call, seat=seat)

        self.called[seat] = True
        if call == CALL_PASS:
            self.passes_since_bid += 1
            self._pass_run += 1
        elif call == CALL_X:
            self.doubled = True
            self._pass_run = 0
        elif call == CALL_XX:
            self.redoubled = True
            self._pass_run = 0
        else:
            level, denom = info.level, info.denom
            self.highest = call
            self.high_level, self.high_denom, self.high_seat = level, denom, seat
            self.doubled = self.redoubled = False
            self.passes_since_bid = 0
            self._pass_run = 0
            if denom == DENOM_N:
                self.seat_nt[seat] = True
            else:
                self.seat_suits[seat].add(denom)
                self.side_suits[side].append((seat, denom))
                self.side_last_denom[side] = denom
                self.suits_bid.add(denom)
            self.side_has_bid[side] = True
            if self.opener < 0:
                self.opener = seat
                self.open_level, self.open_denom = level, denom
                if level == 2 and denom == 0:      # strong 2C opening
                    self.strong2c_side = side
                    self.gf[side] = True
            elif side != self.opener % 2 and self.overcaller < 0:
                self.overcaller = seat

        # opponent activity (uncontested flag / 2/1 jump exception)
        if self.opener >= 0 and side != self.opener % 2 and call != CALL_PASS:
            self.opp_nonpass += 1
            if is_bid(call):
                self.opp_bids += 1

        # strong 2C: responder's 2D is the waiting bid (matters for whether
        # opener's 2NT rebid is invitational)
        if (self.strong2c_side == side and seat == (self.opener + 2) % 4
                and info.first_call and call == bid_id(2, 1)):
            self.s2c_waiting = True

        self._update_gf(seat, side, info)
        self._update_xyz(call, seat, info)
        self._update_stayman(call, seat, info)
        self._update_rkcb(call, seat, info)
        self.calls.append(call)
        self.n += 1
        return info

    def _update_gf(self, seat: int, side: int, info: CallInfo):
        if info.kind != "bid":
            return
        # strong 2C: after a 2D waiting response, opener's 2NT rebid is
        # invitational (22-24), not a game force
        if self.strong2c_side == side and seat == self.opener:
            self.s2c_opener_calls += 1
            if (self.s2c_opener_calls == 2 and self.s2c_waiting
                    and info.level == 2 and info.denom == DENOM_N):
                self.gf[side] = False
        if info.is_2over1 or info.fourth_suit or info.xyz_2d or info.smolen:
            self.gf[side] = True
        if is_game_or_above(info.level, info.denom):
            self.gf[side] = False      # obligation fulfilled

    def _update_xyz(self, call: int, seat: int, info: CallInfo):
        stage = self.xyz_stage
        if stage == "none":
            if info.opening and info.level == 1 and info.denom < DENOM_N:
                self.xyz_stage = "s1"
                self.xyz_opener = seat
            return
        # any opponent non-pass call kills basic XYZ
        if self.opp_nonpass > 0:
            self.xyz_stage = "none"
            return
        if stage == "s1":
            if info.kind == "pass":
                return                      # waiting for the responder
            if (seat == (self.xyz_opener + 2) % 4
                    and info.level == 1 and info.denom < DENOM_N):
                self.xyz_stage = "s2"
            else:
                self.xyz_stage = "none"
        elif stage == "s2":
            if info.kind == "pass":
                return                      # waiting for opener's rebid
            if (seat == self.xyz_opener
                    and info.level == 1 and info.denom < DENOM_N):
                self.xyz_stage = "s3"
                self.xyz_for = (self.xyz_opener + 2) % 4
            else:
                self.xyz_stage = "none"
        elif stage == "s3":
            if seat != self.xyz_for:
                if info.kind != "pass":
                    self.xyz_stage = "none"
                return
            if info.xyz_2c:
                self.xyz_stage = "relay"
                self.xyz_2c_by = seat
            else:
                self.xyz_stage = "none"   # bypassed, or GF 2D (gf already set)
        elif stage == "relay":
            if seat != self.xyz_opener:
                return                      # passes wait for the relay
            self.xyz_stage = ("after_relay"
                              if call == bid_id(2, 1) else "none")
        elif stage == "after_relay":
            if seat == self.xyz_2c_by:
                self.xyz_stage = "none"

    def _update_stayman(self, call: int, seat: int, info: CallInfo):
        """Stayman -> Smolen state machine (1NT-2C-2D / 2NT-3C-3D).

        Passes keep the stage; any opponent action or an unexpected call
        kills it. After the 2D/3D denial the responder's next non-pass call
        is the Smolen position (classify flags a 3M); anything else there is
        natural and ends the sequence.
        """
        stage = self.stayman_stage
        if stage == "none":
            if info.opening and info.denom == DENOM_N and info.level in (1, 2):
                self.stayman_stage = "nt"
                self.stayman_nt = info.level
                self.stayman_opener = seat
            return
        if self.opp_nonpass > 0:           # interference kills stayman
            self.stayman_stage = "none"
            return
        if info.kind == "pass":
            return
        ask = bid_id(self.stayman_nt + 1, 0)     # 2C over 1NT, 3C over 2NT
        deny = bid_id(self.stayman_nt + 1, 1)    # the matching 2D/3D denial
        if stage == "nt":
            if seat == (self.stayman_opener + 2) % 4 and call == ask:
                self.stayman_stage = "asked"
                self.stayman_for = seat
            else:
                self.stayman_stage = "none"
        elif stage == "asked":
            self.stayman_stage = ("denied"
                                  if seat == self.stayman_opener
                                  and call == deny else "none")
        elif stage == "denied":
            self.stayman_stage = "none"    # responder called (Smolen or not)

    def _update_rkcb(self, call: int, seat: int, info: CallInfo):
        """Latch the RKCB ask; the asker's partner's next call consumes it.

        An intervening opponent bid/double cancels the step-response window
        (DOPI-style agreements are not checked); passes keep it open.
        """
        if self.rkcb_asker >= 0:
            partner = (self.rkcb_asker + 2) % 4
            if seat == partner or call != CALL_PASS:
                self.rkcb_asker, self.rkcb_trump = -1, -1
        if info.rkcb_ask:
            self.rkcb_asker, self.rkcb_trump = seat, info.rkcb_trump

    def copy(self) -> "AuctionReader":
        """Independent deep copy of the whole state (rl/analyze.py walks the
        dumped auction tree with one reader per branch)."""
        return deepcopy(self)
