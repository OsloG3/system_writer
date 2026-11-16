"""A rule-based bidder that plays the system file directly.

`SystemBidder` turns the pattern entries of models/system.txt into actual bids:
given the auction so far, the hero's hand and the legal calls, it picks the call
a partnership following the file would make. It is the concrete "bot following
the rules" -- no neural net -- and serves two purposes:

  * it is a `system:` policy for conventions/gen.py and hand/gen.py, so the
    demonstration / hand-inference stores can be dealt from the FILE alone
    (the fallback-policy mix is still available for variety);
  * it is a sanity check on the file itself: if the file is detailed enough,
    these auctions are sensible bridge (they score ~par double-dummy).

Selection is layered:

  1. **Mandatory continuations** -- bids the system mechanically requires:
     completing a transfer or Stayman, the RKCB step response, completing the
     XYZ puppet, the Lebensohl 3C completion.
  2. **Position logic** -- openings, responses (major/minor/NT/2C/weak two/
     preempt), advances after partner's takeout double (incl. Lebensohl),
     overcalls, and rebids, all driven by K&R strength, shape and fit.
  3. **Continuations** -- after Stayman/transfers (invite/game/quantitative
     ladders), invite acceptance by the NT opener, game-try answers, weak-two
     feature asks, RKCB ask/signoff, and partner-range-aware raise decisions
     (the range is read FROM THE FILE entry describing partner's last bid).
  4. **Validation** -- the SystemEnforcer checks the chosen call against the
     file + built-in rules; if it would break the system the bidder falls back
     to the best *consistent* legal call ranked by bridge preference.

    python -m bidding_dt.conventions.bidder --hand "AT7.KT943.A42.K8" \
        --dealer N --auction "P 1D" --hero N
    python -m bidding_dt.conventions.bidder --selfplay --deals 20
"""

from dataclasses import dataclass

import numpy as np

from ..data.vocab import (CALL_PASS, CALL_X, CALL_XX, FIRST_BID, VOCAB_SIZE,
                          id_to_call)
from .auction import (DENOM_N, MAJORS, MINORS, AuctionReader, bid_id,
                      is_game_or_above)
from .patterns import SystemEnforcer
from .rules import ConventionConfig, HandInfo
from .system_file import SystemSpec, find_system_file, parse_system_file

# RKCB step responses in the file's order (1430): 5C=1/4, 5D=0/3, 5H=2-, 5S=2+
_RKCB_BRANCH = {0: (1, 4), 1: (0, 3), 2: (2,), 3: (2,)}


@dataclass
class BidContext:
    reader: AuctionReader
    hero: int
    hi: HandInfo
    legal: list

    @property
    def partner(self):
        return (self.hero + 2) % 4

    @property
    def side(self):
        return self.hero % 2


class SystemBidder:
    """Deterministic rule-based bidder over a parsed system file."""

    stochastic = False

    def __init__(self, spec: SystemSpec, cfg: ConventionConfig | None = None):
        self.spec = spec
        self.cfg = cfg or spec.config()
        self.enf = SystemEnforcer(spec, self.cfg)

    # -- public API -----------------------------------------------------------

    def choose(self, reader: AuctionReader, hero: int, hand, legal) -> int:
        """Pick a call for `hero` holding `hand` in `reader`'s auction.

        `legal` is an iterable of legal call ids (or a (VOCAB_SIZE,) bool mask).
        """
        if isinstance(legal, np.ndarray) and legal.dtype == bool:
            legal = np.flatnonzero(legal).tolist()
        legal = [int(c) for c in legal]
        if not legal:
            return CALL_PASS
        hi = hand if isinstance(hand, HandInfo) else HandInfo(hand)
        ctx = BidContext(reader, hero % 4, hi, legal)
        ideal = self._ideal(ctx)
        if ideal in legal and self._consistent(ctx, ideal):
            return ideal
        # fall back to the best consistent call by bridge preference
        best, best_key = None, None
        for c in legal:
            pen = self._penalty(ctx, c)
            pref = self._preference(ctx, c)
            key = (0 if pen == 0 else 1, pen, -pref)
            if best_key is None or key < best_key:
                best, best_key = c, key
        return best if best is not None else CALL_PASS

    def bid_auction(self, deal_hands, dealer: int, vuln: int = 0,
                    max_calls: int = 40) -> list:
        """Play a whole auction with all four seats following the file."""
        reader = AuctionReader(dealer, vuln)
        hands = [HandInfo(h) for h in deal_hands]
        calls = []
        for _ in range(max_calls):
            if self._auction_over(reader):
                break
            hero = reader.next_seat()
            legal = self._legal(reader)
            c = self.choose(reader, hero, hands[hero], legal)
            reader.feed(c)
            calls.append(c)
        return calls

    # -- consistency ----------------------------------------------------------

    def _consistent(self, ctx: BidContext, call: int) -> bool:
        return self._penalty(ctx, call) == 0

    def _penalty(self, ctx: BidContext, call: int) -> float:
        vs = self.enf.check(ctx.reader, ctx.hero, ctx.hi, call)
        return sum(v.weight for v in vs)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _legal(reader: AuctionReader) -> list:
        from ..data.legal import legal_mask
        mask = legal_mask(reader.calls, role_offset=(-reader.n) % 4)
        return np.flatnonzero(mask).tolist()

    @staticmethod
    def _auction_over(reader: AuctionReader) -> bool:
        calls = reader.calls
        n = len(calls)
        if n < 3:
            return False
        trailing = 0
        for c in reversed(calls):
            if c == CALL_PASS:
                trailing += 1
            else:
                break
        if any(c != CALL_PASS for c in calls):
            return trailing >= 3        # three passes after any bid/double
        return n >= 4                   # four passes: passed out

    @staticmethod
    def _lengths(hi: HandInfo):
        return [hi.length(d) for d in range(4)]   # C,D,H,S

    @staticmethod
    def _seat_calls(ctx: BidContext, seat: int) -> list:
        r = ctx.reader
        return [int(c) for i, c in enumerate(r.calls)
                if (r.dealer + i) % 4 == seat]

    def _hero_calls(self, ctx):
        return self._seat_calls(ctx, ctx.hero)

    def _partner_calls(self, ctx):
        return self._seat_calls(ctx, ctx.partner)

    def _last_call(self, ctx, seat):
        r = ctx.reader
        for i in range(len(r.calls) - 1, -1, -1):
            if (r.dealer + i) % 4 == seat:
                return int(r.calls[i])
        return None

    def _interference(self, ctx) -> bool:
        """Any non-pass call by the opponents (conventions are off)."""
        r = ctx.reader
        return any(int(c) != CALL_PASS and (r.dealer + i) % 4 != ctx.hero
                   and (r.dealer + i) % 4 != ctx.partner
                   for i, c in enumerate(r.calls))

    def _suit_order(self, hi: HandInfo):
        # longest first, majors preferred on ties
        return sorted(range(4), key=lambda d: (-hi.length(d), -(d in MAJORS),
                                               -d))

    def _cheapest(self, ctx: BidContext, denom: int):
        for lvl in range(1, 8):
            c = bid_id(lvl, denom)
            if c > ctx.reader.highest:
                return c
        return None

    def _cheapest_nt(self, ctx: BidContext):
        return self._cheapest(ctx, DENOM_N)

    # -- ideal bid by position ------------------------------------------------

    def _ideal(self, ctx: BidContext) -> int:
        r = ctx.reader
        if r.highest == 0:
            return self._opening(ctx)
        # mechanically required continuations first
        m = self._mandatory(ctx)
        if m is not None:
            return m
        if not r.called[ctx.hero]:
            stop = self._should_stop(ctx)
            if stop is not None:
                return stop
            if r.opener == ctx.partner:
                return self._respond_to_opening(ctx)
            if self._partner_doubled(ctx):
                return self._advance_takeout(ctx)
            if self._their_opened(ctx):
                return self._compete(ctx)
            return self._rebid(ctx)
        # hero has already called
        if not self._side_bid(ctx):
            # we passed and our side never bid: stay out unless balancing
            if r.passes_since_bid >= 2 and ctx.hi.kr >= 8 \
                    and not self._interference_locked(ctx):
                return self._compete(ctx)
            return CALL_PASS
        c = self._continuation(ctx)
        if c is not None:
            return c
        stop = self._should_stop(ctx)
        if stop is not None:
            return stop
        return self._rebid(ctx)

    def _interference_locked(self, ctx: BidContext) -> bool:
        """True when the opponents are in a live auction of their own (we
        should not balance into an active convention sequence)."""
        r = ctx.reader
        return r.xyz_stage not in ("none",) or r.rkcb_asker >= 0 \
            or r.strong2c_side == 1 - ctx.side

    def _side_bid(self, ctx: BidContext) -> bool:
        return ctx.reader.side_has_bid[ctx.side]

    def _their_opened(self, ctx: BidContext) -> bool:
        r = ctx.reader
        return r.opener >= 0 and r.opener % 2 != ctx.side

    def _partner_doubled(self, ctx: BidContext) -> bool:
        """Partner's most recent call is a double that is still the live
        call (nothing bid since) and our side has not bid."""
        r = ctx.reader
        return (self._last_call(ctx, ctx.partner) == CALL_X
                and not self._side_bid(ctx) and r.highest > 0)

    def _our_game(self, ctx: BidContext) -> bool:
        r = ctx.reader
        return (r.high_seat >= 0 and r.high_seat % 2 == ctx.side
                and is_game_or_above(r.high_level, r.high_denom))

    def _n_bids(self, ctx: BidContext, seat: int) -> int:
        r = ctx.reader
        return sum(1 for i, c in enumerate(r.calls)
                   if c >= FIRST_BID and (r.dealer + i) % 4 == seat)

    # -- mandatory continuations ----------------------------------------------

    def _mandatory(self, ctx: BidContext):
        """Bids the system mechanically requires (puppets and step answers)."""
        r, hi = ctx.reader, ctx.hi
        # RKCB step response (the reader keeps the window open only while the
        # response is live)
        if r.rkcb_asker >= 0 and ctx.hero == (r.rkcb_asker + 2) % 4:
            trump = r.rkcb_trump
            if trump >= 0:
                keys = sum(int(hi.has(d, 0)) for d in range(4))
                keys += int(hi.has(trump, 1))
                has_q = hi.has(trump, 2)
                if keys in (1, 4):
                    return bid_id(5, 0)
                if keys in (0, 3):
                    return bid_id(5, 1)
                return bid_id(5, 2) if not has_q else bid_id(5, 3)
        if self._interference(ctx):
            return None
        # complete partner's transfer over our NT opening
        if (r.opener == ctx.hero and r.open_denom == DENOM_N
                and r.open_level in (1, 2)
                and len(self._partner_calls(ctx)) == 1):
            L = r.open_level
            pl = self._last_call(ctx, ctx.partner)
            if pl == bid_id(L + 1, 1):          # 2D/3D -> hearts
                return bid_id(L + 1, 2)
            if pl == bid_id(L + 1, 2):          # 2H/3H -> spades
                return bid_id(L + 1, 3)
            if pl == bid_id(L + 1, 0) and r.seat_suits[ctx.partner] == {0}:
                # Stayman: show a 4-card major (hearts first with both)
                if hi.length(2) >= 4:
                    return bid_id(L + 1, 2)
                if hi.length(3) >= 4:
                    return bid_id(L + 1, 3)
                return bid_id(L + 1, 1)
        # complete the XYZ puppet (2C relay -> 2D) with a minimum
        if r.xyz_stage == "relay" and r.xyz_opener == ctx.hero:
            if hi.kr < 15.5:
                return bid_id(2, 1)
            # break with extras: a 4-card unbid major, else 2NT, else own suit
            for d in (3, 2):
                if (hi.length(d) >= 4 and d not in r.suits_bid
                        and bid_id(2, d) > r.highest):
                    return bid_id(2, d)
            if bid_id(2, DENOM_N) > r.highest:
                return bid_id(2, DENOM_N)
            return self._cheapest(ctx, r.open_denom)
        # complete the Lebensohl puppet after our takeout double of their 2M
        their = r.side_last_denom[1 - ctx.side]
        if (their in MAJORS and r.high_level == 2
                and self._last_call(ctx, ctx.hero) == CALL_X
                and self._last_call(ctx, ctx.partner) == bid_id(2, DENOM_N)):
            return bid_id(3, 0)
        return None

    # -- openings --------------------------------------------------------------

    def _opening(self, ctx: BidContext) -> int:
        hi = ctx.hi
        if self._strong_2c(hi):
            return bid_id(2, 0)
        if hi.balanced():
            if 14 <= hi.kr <= 18:
                return bid_id(1, DENOM_N)
            if 19 <= hi.kr <= 23:
                return bid_id(2, DENOM_N)
        preempt = self._preempt(hi)
        if preempt is not None:
            return preempt
        if self._opening_values(hi, ctx):
            return self._best_one_bid(hi)
        return CALL_PASS

    def _strong_2c(self, hi: HandInfo) -> bool:
        if hi.kr >= 22 - self.cfg.kr_tolerance:
            return True
        longest = hi.longest_denom()
        n = hi.length(longest)
        if n >= 8 and hi.top_honors(longest) >= 2:
            return True
        if n >= 6 and hi.top_honors(longest) >= 2 and hi.kr >= 16.5:
            return True
        return False

    def _opening_values(self, hi: HandInfo, ctx: BidContext) -> bool:
        late = ctx.reader._pass_run >= 2
        min_kr = (self.cfg.open_min_kr_late if late else self.cfg.open_min_kr)
        if hi.kr >= min_kr - self.cfg.kr_tolerance:
            return True
        return bool(self.cfg.open_rule20 and hi.hcp + hi.two_longest() >= 20)

    def _preempt(self, hi: HandInfo):
        lens = self._lengths(hi)
        for d in (3, 2, 1, 0):
            if lens[d] >= 8 and hi.kr < 12:
                return bid_id(4, d)
        for d in (3, 2, 1, 0):
            if lens[d] == 7 and hi.kr < 10:
                return bid_id(3, d)
        for d in (3, 2, 1):
            if lens[d] == 6 and 4 <= hi.kr <= 11:
                return bid_id(2, d)
        return None

    def _best_one_bid(self, hi: HandInfo) -> int:
        lens = self._lengths(hi)
        for d in (3, 2):                      # 5-card majors, higher first
            if lens[d] >= 5:
                return bid_id(1, d)
        if lens[1] > lens[0] and lens[1] >= 3:
            return bid_id(1, 1)
        if lens[0] >= lens[1]:
            # 3-3 (or 2-2 freak) minors open 1C; 4-4 opens 1D
            return bid_id(1, 1 if lens[1] >= 4 else 0)
        return bid_id(1, 1 if lens[1] >= 3 else 0)

    # -- responding to partner's opening --------------------------------------

    def _respond_to_opening(self, ctx: BidContext) -> int:
        r, hi = ctx.reader, ctx.hi
        od, ol = r.open_denom, r.open_level
        if ol == 2 and od == 0:
            return self._respond_2c(ctx)
        if od == DENOM_N:
            return self._respond_nt(ctx)
        if ol == 2 and od in MAJORS:
            return self._respond_weak2(ctx)
        if ol >= 2:
            return self._respond_preempt(ctx)
        if od in MAJORS:
            return self._respond_major(ctx)
        return self._respond_minor(ctx)

    def _respond_2c(self, ctx: BidContext) -> int:
        r, hi = ctx.reader, ctx.hi
        if hi.kr > 7.5:
            for d in (3, 2, 1, 0):
                if hi.length(d) >= 5:
                    c = self._cheapest(ctx, d)
                    if c is not None and bid_id_of_level(c) <= 3:
                        return c
            if hi.balanced():
                return bid_id(2, DENOM_N)
        return bid_id(2, 1)

    def _respond_nt(self, ctx: BidContext) -> int:
        r, hi = ctx.reader, ctx.hi
        L = r.open_level                       # 1 or 2
        # 5-card major -> transfer (any strength)
        if hi.length(2) >= 5:
            return bid_id(L + 1, 1)
        if hi.length(3) >= 5:
            return bid_id(L + 1, 2)
        # 4-card major with values -> Stayman
        if any(hi.length(d) >= 4 for d in MAJORS) \
                and hi.kr > (7.5 if L == 1 else 5.5):
            return bid_id(L + 1, 0)
        if L == 1:
            if hi.balanced():
                if 8 <= hi.kr <= 9.5:
                    return bid_id(2, DENOM_N)
                if hi.kr < 15.0:
                    return bid_id(3, DENOM_N) if hi.kr >= 9.5 else CALL_PASS
                if hi.kr <= 17.5:
                    return bid_id(4, DENOM_N)   # quantitative
                return bid_id(6, DENOM_N)
            if hi.kr >= 10:
                return bid_id(3, DENOM_N)       # long-minor game try-ish
            return CALL_PASS
        # over 2NT (20-22)
        if hi.kr < 4.5:
            return CALL_PASS
        if hi.kr < 10.5:
            return bid_id(3, DENOM_N)
        if hi.kr < 12.5:
            return bid_id(4, DENOM_N)           # quantitative
        return bid_id(6, DENOM_N)

    def _respond_weak2(self, ctx: BidContext) -> int:
        r, hi = ctx.reader, ctx.hi
        od = r.open_denom
        fit = hi.length(od)
        if fit >= 3:
            if hi.kr >= 13.5:
                return bid_id(4, od)
            if hi.kr >= 11:
                return bid_id(3, od)            # invitational raise
            return CALL_PASS
        if hi.kr >= 15.5 and hi.balanced():
            return bid_id(3, DENOM_N)
        for d in self._suit_order(hi):
            if d != od and hi.length(d) >= 5 and hi.kr >= 12:
                c = self._cheapest(ctx, d)
                if c is not None:
                    return c
        if hi.kr >= 12:
            return bid_id(2, DENOM_N)           # feature ask
        return CALL_PASS

    def _respond_preempt(self, ctx: BidContext) -> int:
        r, hi = ctx.reader, ctx.hi
        od = r.open_denom
        if od < DENOM_N and hi.length(od) >= 3 and hi.kr >= 13:
            g = bid_id(4, od) if od in MAJORS else bid_id(5, od)
            if g > r.highest:
                return g
        if hi.kr >= 12.5:
            for d in self._suit_order(hi):
                if d != od and hi.length(d) >= 5:
                    c = self._cheapest(ctx, d)
                    if c is not None and c < bid_id(4, DENOM_N):
                        return c
        return CALL_PASS

    def _respond_major(self, ctx: BidContext) -> int:
        r, hi = ctx.reader, ctx.hi
        od = r.open_denom
        fit = hi.length(od)
        kr = hi.kr
        # splinter: singleton/void, 4+ fit, slam values
        if fit >= 4 and kr > 12:
            for d in range(4):
                if d != od and hi.length(d) <= 1:
                    jump = self._splinter_call(ctx, d)
                    if jump is not None:
                        return jump
        if fit >= 4 and kr >= 13:
            return bid_id(4, od)
        if fit >= 3 and 10 <= kr <= 12.5:
            return bid_id(2, DENOM_N)           # Jordan invitational raise
        if fit >= 4 and kr < 7:
            return bid_id(3, od)                # preemptive raise
        if fit >= 3 and 5 <= kr <= 10:
            return bid_id(2, od)
        # 2/1 game force with a real suit
        if kr > 11.5:
            for d in self._suit_order(hi):
                if d == od:
                    continue
                need = 3 if d == 0 else 4
                if d == 2 and od == 3:
                    need = 5                    # 2H over 1S promises 5
                if hi.length(d) >= need:
                    c = bid_id(2, d)
                    if c > r.highest:
                        return c
        if od == 2 and hi.length(3) >= 4 and kr > 5.5:
            return bid_id(1, 3)
        if 5 <= kr <= 12:
            return bid_id(1, DENOM_N)
        return CALL_PASS

    def _respond_minor(self, ctx: BidContext) -> int:
        r, hi = ctx.reader, ctx.hi
        od = r.open_denom
        kr = hi.kr
        if hi.length(od) >= 4 and 5 <= kr <= 10:
            return bid_id(2, od)
        if hi.length(od) >= 5 and 10 <= kr <= 12.5:
            return bid_id(3, od)
        for d in (3, 2):
            if hi.length(d) >= 4 and kr > 5.5 and bid_id(1, d) > r.highest:
                return bid_id(1, d)
        if od == 0 and hi.length(1) >= 4 and kr > 5.5:
            return bid_id(1, 1)
        if kr > 11.5:
            for d in self._suit_order(hi):
                if d == od:
                    continue
                need = 3 if d == 0 else 4
                if hi.length(d) >= need and bid_id(2, d) > r.highest:
                    return bid_id(2, d)
        if 6 <= kr <= 10 and hi.balanced():
            return bid_id(1, DENOM_N)
        if 10 <= kr <= 12 and hi.balanced():
            return bid_id(2, DENOM_N)
        return CALL_PASS

    def _splinter_call(self, ctx: BidContext, short: int):
        r = ctx.reader
        for lvl in range(3, 5):
            c = bid_id(lvl, short)
            if c <= r.highest:
                continue
            info = r.classify(c, seat=ctx.hero)
            if info.jump == 2 and c < bid_id(4, MAJORS[0]):
                return c
        return None

    # -- competitive ----------------------------------------------------------

    def _compete(self, ctx: BidContext) -> int:
        r, hi = ctx.reader, ctx.hi
        balancing = r.passes_since_bid >= 2
        if r.high_level >= 3 and hi.kr < 12:
            return CALL_PASS
        their = r.side_last_denom[1 - ctx.side]
        # takeout double with shortness in their suit and support elsewhere
        if their >= 0 and r.high_level <= 2:
            min_kr = 8 if balancing else 10
            short = hi.length(their) <= 2
            unbid_ok = all(hi.length(d) >= 3 for d in MAJORS
                           if d not in r.suits_bid)
            if short and unbid_ok and hi.kr >= min_kr:
                return CALL_X
        # overcall: 5-card suit (4 only with strength), lighter when balancing
        for d in self._suit_order(hi):
            if d == their:
                continue
            n = hi.length(d)
            if n >= 5 and hi.kr > (5.5 if balancing else 7):
                c = bid_id(1, d)
                if c > r.highest:
                    return c
            if n >= 5 and hi.kr > (8 if balancing else 10):
                c = bid_id(2, d)
                if c > r.highest:
                    return c
            if n == 4 and d in MAJORS and hi.kr > (9 if balancing else 13):
                c = bid_id(1, d)
                if c > r.highest:
                    return c
        # 1NT overcall with a stopper
        if (their >= 0 and hi.balanced() and hi.stopper(their)
                and bid_id(1, DENOM_N) > r.highest):
            lo, up = (11.5, 15.5) if balancing else (14.5, 18.5)
            if lo <= hi.kr <= up:
                return bid_id(1, DENOM_N)
        return CALL_PASS

    def _advance_takeout(self, ctx: BidContext) -> int:
        """Advancer's bid after partner's takeout double."""
        r, hi = ctx.reader, ctx.hi
        their = r.side_last_denom[1 - ctx.side]
        if their < 0:
            return CALL_PASS
        stop = hi.stopper(their)
        unbid = [d for d in range(4) if d != their]
        longest = max(unbid, key=lambda d: (hi.length(d), d in MAJORS))
        # Lebensohl over their 2-level major opening
        if r.high_level == 2 and their in MAJORS:
            other = 3 if their == 2 else 2
            if hi.kr > 11:
                if hi.length(other) >= 4:
                    # cuebid: fast denies a stopper, slow (2NT) shows one
                    return (bid_id(2, DENOM_N) if stop
                            else bid_id(3, their))
                if stop:
                    return bid_id(2, DENOM_N)   # relay, then 3NT
                return bid_id(3, DENOM_N)       # fast denies
            if hi.length(longest) >= 5 and 8 <= hi.kr <= 12.5:
                return bid_id(3, longest)       # invitational (positive)
            if hi.kr < 8 and hi.length(longest) >= 5:
                return bid_id(2, DENOM_N)       # relay, then sign off
            if hi.length(their) >= 4:
                return CALL_PASS                # convert
            return bid_id(2, DENOM_N) if hi.kr < 8 else bid_id(3, longest)
        # standard advances
        if hi.length(their) >= 4 and hi.kr < 10:
            return CALL_PASS                    # penalty conversion
        if hi.kr >= 13:
            c = self._cheapest(ctx, their)      # cuebid ask
            if c is not None:
                return c
        if stop and hi.balanced():
            if 6 <= hi.kr <= 10 and bid_id(1, DENOM_N) > r.highest:
                return bid_id(1, DENOM_N)
            if 11 <= hi.kr <= 12 and bid_id(2, DENOM_N) > r.highest:
                return bid_id(2, DENOM_N)
            if hi.kr >= 13 and bid_id(3, DENOM_N) > r.highest:
                return bid_id(3, DENOM_N)
        if hi.length(longest) >= 4:
            if hi.kr < 10:
                c = self._cheapest(ctx, longest)
                if c is not None and c <= bid_id(2, longest):
                    return c
            if hi.length(longest) >= 5 and hi.kr >= 10:
                c = self._cheapest(ctx, longest)
                if c is not None:
                    jc = c + 5                  # one level higher, same suit
                    if jc in ctx.legal:
                        return jc
            if hi.kr >= 11:
                c = self._cheapest(ctx, their)  # cuebid covers the rest
                if c is not None:
                    return c
            c = self._cheapest(ctx, longest)
            if c is not None:
                return c
        return CALL_PASS

    # -- continuations (hero has already called) ------------------------------

    def _continuation(self, ctx: BidContext):
        r, hi = ctx.reader, ctx.hi
        if self._interference(ctx):
            return None
        L = r.open_level if r.open_denom == DENOM_N else None
        hero_calls = self._hero_calls(ctx)
        partner_calls = self._partner_calls(ctx)
        pl = self._last_call(ctx, ctx.partner)

        # -- responder after our Stayman ask
        if (r.opener == ctx.partner and L in (1, 2)
                and hero_calls == [bid_id(L + 1, 0)]
                and len(partner_calls) == 2 and pl is not None):
            resp = None
            if pl == bid_id(L + 1, 1):
                resp = None                     # denial
            elif pl in (bid_id(L + 1, 2), bid_id(L + 1, 3)):
                resp = (pl - FIRST_BID) % 5     # the shown major
            base = bid_id(1, DENOM_N) if L == 1 else bid_id(2, DENOM_N)
            if resp is not None and hi.length(resp) >= 4:
                # fit found
                if L == 2:
                    # 2NT shows 20-22: any values make game
                    return bid_id(4, resp) if hi.kr >= 4.5 else CALL_PASS
                if hi.kr >= 10:
                    return bid_id(4, resp)
                if hi.kr >= 8:
                    return bid_id(3, resp)      # invitational
                return CALL_PASS                # weak: play the 2-level fit
            # denial or no fit: NT ladder
            if L == 1:
                if hi.kr < 8:
                    return CALL_PASS
                if hi.kr <= 9.5:
                    return bid_id(2, DENOM_N)
                if hi.kr < 15.0:
                    return bid_id(3, DENOM_N)
                if hi.kr <= 17.5:
                    return bid_id(4, DENOM_N)
                return bid_id(6, DENOM_N)
            if hi.kr < 6:
                return CALL_PASS
            return bid_id(3, DENOM_N) if hi.kr < 13 else bid_id(6, DENOM_N)

        # -- responder after our transfer
        if r.opener == ctx.partner and L in (1, 2) and len(hero_calls) == 1:
            tmap = {bid_id(L + 1, 1): 2, bid_id(L + 1, 2): 3}
            if hero_calls[0] in tmap:
                major = tmap[hero_calls[0]]
                if pl == bid_id(L + 1, major) and len(partner_calls) == 2:
                    n = hi.length(major)
                    if L == 2:
                        # over 2NT (20-22): game with any values
                        if hi.kr < 4.5:
                            return CALL_PASS
                        return (bid_id(4, major) if n >= 6
                                else bid_id(3, DENOM_N))
                    if hi.kr < 8:
                        return CALL_PASS
                    if hi.kr <= 9.5:
                        return (bid_id(L + 2, major) if n >= 6
                                else bid_id(L + 1, DENOM_N))
                    if hi.kr < 15.0:
                        return (bid_id(L + 3, major) if n >= 6
                                else bid_id(L + 2, DENOM_N))
                    if hi.kr <= 17.5:
                        return bid_id(L + 3, DENOM_N)   # quantitative
                    return bid_id(L + 5, major) if n >= 6 \
                        else bid_id(L + 5, DENOM_N)

        # -- NT opener facing an invite / quantitative 4NT
        if r.opener == ctx.hero and r.open_denom == DENOM_N \
                and r.open_level in (1, 2):
            max_kr = 16.5 if r.open_level == 1 else 21.5
            if pl == bid_id(r.open_level + 1, DENOM_N) and hi.kr >= max_kr:
                return bid_id(3, DENOM_N)       # accept the 2NT invite
            if pl == bid_id(r.open_level + 1, DENOM_N):
                return CALL_PASS
            if pl == bid_id(4, DENOM_N):        # quantitative 4NT
                return bid_id(6, DENOM_N) if hi.kr >= max_kr else CALL_PASS
            # 3M invite after we showed the major (Stayman response or
            # transfer completion): accept with a maximum
            if pl is not None and pl >= FIRST_BID:
                d = (pl - FIRST_BID) % 5
                lvl = (pl - FIRST_BID) // 5 + 1
                if d in MAJORS and lvl == 3 and hi.length(d) >= 3 \
                        and self._last_call(ctx, ctx.hero) in (
                            bid_id(r.open_level + 1, d),):
                    return bid_id(4, d) if hi.kr >= max_kr else CALL_PASS

        # -- asker's action after our RKCB response
        if self._last_call(ctx, ctx.hero) == bid_id(4, DENOM_N) \
                and pl is not None and pl >= bid_id(5, 0) and pl < bid_id(6, 0):
            trump = self._trump_before_ask(ctx)
            if trump is not None:
                step = (pl - FIRST_BID) % 5
                branch = _RKCB_BRANCH.get(step, (0,))
                mine = sum(int(hi.has(d, 0)) for d in range(4)) \
                    + int(hi.has(trump, 1))
                if any(mine + b == 5 for b in branch):
                    g = bid_id(6, trump)
                    if g in ctx.legal:
                        return g
                s = bid_id(5, trump)
                return s if s in ctx.legal else CALL_PASS

        # -- Jordan 2NT limit raise of our major: opener places the contract
        if (r.opener == ctx.hero and r.open_level == 1
                and r.open_denom in MAJORS and pl == bid_id(2, DENOM_N)
                and len(partner_calls) == 1):
            od = r.open_denom
            if hi.kr >= 14.5:
                return bid_id(4, od)
            return bid_id(3, od)

        # -- game-try response (1X-2X-2Y/3Y -> accept or decline)
        gt = self._game_try_response(ctx)
        if gt is not None:
            return gt

        # -- weak-two feature ask: opener's feature response
        if (r.opener == ctx.hero and r.open_level == 2
                and r.open_denom in MAJORS and len(hero_calls) == 1
                and pl == bid_id(2, DENOM_N)):
            od = r.open_denom
            if hi.kr > 8.5 and hi.top_honors(od) >= 2:
                return bid_id(4, od)
            if hi.kr > 8:
                for d in (3, 2, 1, 0):
                    if d != od and (hi.has(d, 0) or hi.has(d, 1)):
                        c = bid_id(3, d)
                        if c in ctx.legal:
                            return c
            return bid_id(3, od)

        # -- feature asker placing the contract
        if (r.opener == ctx.partner and r.open_level == 2
                and r.open_denom in MAJORS
                and self._last_call(ctx, ctx.hero) == bid_id(2, DENOM_N)):
            od = r.open_denom
            if pl == bid_id(4, od):
                return CALL_PASS
            if hi.kr >= 13:
                return bid_id(4, od)
            c = bid_id(3, od)
            return c if c in ctx.legal else CALL_PASS

        # -- Lebensohl: after our 2NT relay over their 2M + partner's 3C
        their = r.side_last_denom[1 - ctx.side]
        if (their in MAJORS and r.high_level >= 2
                and self._last_call(ctx, ctx.hero) == bid_id(2, DENOM_N)
                and pl == bid_id(3, 0)):
            if hi.kr > 11:
                other = 3 if their == 2 else 2
                if hi.length(other) >= 4 and hi.stopper(their):
                    return bid_id(3, their)     # slow shows: 4-card + stopper
                if hi.stopper(their):
                    return bid_id(3, DENOM_N)
                return CALL_PASS                # should not happen
            # weak signoff: play 3C with clubs, else our 5+ suit at the 3 level
            if hi.length(0) >= 5:
                return CALL_PASS
            for d in (3, 2, 1):
                if d != their and hi.length(d) >= 5:
                    c = bid_id(3, d)
                    if c in ctx.legal:
                        return c
            return CALL_PASS

        # -- strong 2C auctions: second waiting / natural after opener's rebid
        if r.strong2c_side == ctx.side and r.s2c_waiting \
                and len(hero_calls) == 1 and r.opener == ctx.partner:
            pd = r._last_suit_of(ctx.side, ctx.partner)
            if hi.kr < 6:
                c = bid_id(2, DENOM_N)
                if c in ctx.legal:
                    return c
            if pd >= 0 and hi.length(pd) >= 3:
                lvl = self._raise_level(ctx, pd)
                if lvl is not None:
                    return bid_id(lvl, pd)
            for d in self._suit_order(hi):
                if hi.length(d) >= 5 and d != pd:
                    c = self._cheapest(ctx, d)
                    if c is not None:
                        return c
            c = self._cheapest_nt(ctx)
            return c if c is not None else CALL_PASS

        # -- XYZ after the completed relay: pass with weak diamonds, else invite
        if r.xyz_stage == "after_relay" and ctx.hero == r.xyz_2c_by:
            if hi.kr < 9.5 and hi.length(1) >= 4:
                return CALL_PASS
            if hi.balanced() and 10 <= hi.kr <= 12.5:
                c = bid_id(2, DENOM_N)
                if c in ctx.legal:
                    return c
            return None
        return None

    def _game_try_response(self, ctx: BidContext):
        """Responder's answer to opener's help-suit game try (1X-2X-2Y/3Y)."""
        r, hi = ctx.reader, ctx.hi
        if r.opener != ctx.partner or r.open_level != 1:
            return None
        od = r.open_denom
        if od == DENOM_N:
            return None
        pl = self._last_call(ctx, ctx.partner)
        if pl is None or pl < FIRST_BID:
            return None
        d = (pl - FIRST_BID) % 5
        lvl = (pl - FIRST_BID) // 5 + 1
        if d == DENOM_N or d == od or lvl not in (2, 3):
            return None
        # hero's last call must have been the simple raise of opener's suit
        if self._last_call(ctx, ctx.hero) != bid_id(2, od):
            return None
        help_suit = (hi.has(d, 0) or hi.has(d, 1) or hi.has(d, 2)
                     or hi.length(d) <= 1)
        accept = help_suit or hi.kr >= 9
        game = bid_id(4, od) if od in MAJORS else bid_id(5, od)
        if od in MINORS and accept and hi.balanced():
            game = bid_id(3, DENOM_N)
        if accept and game in ctx.legal:
            return game
        c = bid_id(3, od)
        return c if c in ctx.legal else CALL_PASS

    def _trump_before_ask(self, ctx: BidContext):
        """Our side's last suit before the hero's 4NT ask (the keycard suit)."""
        r = ctx.reader
        ask_idx = None
        for i in range(len(r.calls) - 1, -1, -1):
            if (r.dealer + i) % 4 == ctx.hero and int(r.calls[i]) == bid_id(4, DENOM_N):
                ask_idx = i
                break
        if ask_idx is None:
            return None
        trump = -1
        for i in range(ask_idx):
            seat = (r.dealer + i) % 4
            c = int(r.calls[i])
            if c >= FIRST_BID and seat % 2 == ctx.side:
                d = (c - FIRST_BID) % 5
                if d < DENOM_N:
                    trump = d
        return trump if trump >= 0 else None

    # -- partner's range from the file (for raise decisions) -------------------

    def _partner_range(self, ctx: BidContext):
        """(lo, hi) K&R range of partner's last bid per its file entry.

        hi is None when unbounded. (None, None) when the file gives no
        strength for the call.
        """
        r = ctx.reader
        idx = None
        for i in range(len(r.calls) - 1, -1, -1):
            seat = (r.dealer + i) % 4
            if seat == ctx.partner and int(r.calls[i]) >= FIRST_BID:
                idx = i
                break
        if idx is None:
            return None, None
        sub = AuctionReader(r.dealer, r.vuln)
        for c in r.calls[:idx]:
            sub.feed(int(c))
        e, _ = self.enf.match(r.dealer, list(sub.calls), int(r.calls[idx]))
        if e is None:
            return None, None
        lo = hi = None
        for br in e.branches:
            for t in br:
                if t.kind != "pts" or t.negated:
                    continue
                if t.op in (">", ">="):
                    b_lo, b_hi = t.lo, None
                elif t.op in ("<", "<="):
                    b_lo, b_hi = 0.0, t.lo
                elif t.op == "range":
                    b_lo, b_hi = t.lo, t.hi
                else:
                    b_lo = b_hi = t.lo
                lo = b_lo if lo is None else min(lo, b_lo)
                hi = None if (hi is None or b_hi is None) else max(hi, b_hi)
        if lo is None:
            return None, None
        if "GF" in e.flags:
            lo = max(lo, 11.5)
            hi = None
        return lo, hi

    # -- stop rules ------------------------------------------------------------

    def _should_stop(self, ctx: BidContext):
        """Return PASS when the auction should end for us, else None."""
        r, hi = ctx.reader, ctx.hi
        forced = self._forcing_auction(ctx)
        last_seat = (r.dealer + r.n - 1) % 4 if r.n else -1
        if self._our_game(ctx):
            fit = self._fit_suit(ctx)
            slam = hi.kr >= 16 and fit is not None and hi.length(fit) >= 4
            if not slam:
                return CALL_PASS
        if (r.high_seat >= 0 and r.high_seat % 2 != ctx.side
                and is_game_or_above(r.high_level, r.high_denom)):
            if hi.kr < 16:
                return CALL_PASS
        if r.high_level >= 5 and hi.kr < 16 and not forced:
            return CALL_PASS
        if self._n_bids(ctx, ctx.hero) >= 3 and hi.kr < 15 and not forced:
            return CALL_PASS
        if last_seat == ctx.partner and self._n_bids(ctx, ctx.hero) >= 1:
            if hi.kr < 12 and not forced:
                if not self._clear_continue(ctx):
                    return CALL_PASS
        return None

    def _fit_suit(self, ctx: BidContext):
        r, hi = ctx.reader, ctx.hi
        for d in MAJORS:
            if d in r.seat_suits[ctx.partner] and hi.length(d) >= 4:
                return d
        for d in MAJORS:
            if d in r.seat_suits[ctx.hero] and d in r.seat_suits[ctx.partner]:
                return d
        for d in MINORS:
            if d in r.seat_suits[ctx.partner] and hi.length(d) >= 5:
                return d
        return None

    def _forcing_auction(self, ctx: BidContext) -> bool:
        return ctx.reader.gf[ctx.side]

    def _clear_continue(self, ctx: BidContext) -> bool:
        r, hi = ctx.reader, ctx.hi
        fit = self._fit_suit(ctx)
        if fit is not None and hi.kr >= 10:
            return True
        for d in range(4):
            if hi.length(d) >= 6 and self._n_bids(ctx, ctx.hero) >= 1:
                if d in r.seat_suits[ctx.hero] \
                        and r._last_suit_of(ctx.side, ctx.hero) != d:
                    return True
        return False

    # -- rebids / later rounds ------------------------------------------------

    def _rebid(self, ctx: BidContext) -> int:
        r, hi = ctx.reader, ctx.hi
        forced = self._forcing_auction(ctx)
        # RKCB with a fit and slam values in a forced/game context
        fit = self._fit_suit(ctx)
        if fit is not None and hi.length(fit) >= 4 and hi.kr >= 16 \
                and (forced or self._our_game(ctx)) \
                and self._last_call(ctx, ctx.hero) != bid_id(4, DENOM_N):
            c = bid_id(4, DENOM_N)
            if c in ctx.legal:
                return c
        # raise partner's suit, using the file's range for partner's last bid
        pd = r._last_suit_of(ctx.side, ctx.partner)
        need = 3 if pd in MAJORS else 4
        pl = self._last_call(ctx, ctx.partner)
        if pd >= 0 and hi.length(pd) >= need and pl is not None:
            # partner bid 3 of the agreed suit: an invite -- unless it declines
            # our own game try, in which case we pass
            if pl == bid_id(3, pd):
                hl = self._last_call(ctx, ctx.hero)
                hero_tried = (hl is not None and hl >= FIRST_BID
                              and (hl - FIRST_BID) % 5 != pd
                              and bid_id(2, pd) in self._partner_calls(ctx))
                if hero_tried:
                    return CALL_PASS
                if hi.kr >= 8.5:
                    g = self._game_bid(ctx, pd)
                    if g is not None:
                        return g
                return CALL_PASS
        if pd >= 0 and hi.length(pd) >= need:
            lo, up = self._partner_range(ctx)
            if lo is not None:
                if hi.kr + lo >= 25:
                    g = self._game_bid(ctx, pd)
                    if g is not None:
                        return g
                if up is not None and hi.kr + up >= 24:
                    c = self._invite_bid(ctx, pd)
                    if c is not None:
                        return c
            if forced and (hi.kr + (lo or 11.5) >= 25
                           or self._n_bids(ctx, ctx.hero) >= 2):
                g = self._game_bid(ctx, pd)
                if g is not None:
                    return g
            if hi.kr >= 8 or forced:
                lvl = self._raise_level(ctx, pd)
                if lvl is not None:
                    return bid_id(lvl, pd)
        # balanced: rebid NT at the level our values justify
        if hi.balanced():
            c = self._nt_rebid(ctx)
            if c is not None:
                return c
        # rebid a long suit of our own
        for d in self._suit_order(hi):
            if hi.length(d) >= 5 and d in r.seat_suits[ctx.hero]:
                c = self._cheapest(ctx, d)
                if c is not None and r.classify(c, seat=ctx.hero).jump == 0:
                    return c
        # a new 4+ suit with values (never the opponents' suit -- that is a
        # cuebid, not a natural bid)
        their = r.side_last_denom[1 - ctx.side]
        if hi.kr >= 10 or forced:
            for d in self._suit_order(hi):
                if hi.length(d) >= 4 and d not in r.seat_suits[ctx.hero] \
                        and d != their:
                    c = self._cheapest(ctx, d)
                    if c is not None and r.classify(c, seat=ctx.hero).jump == 0:
                        return c
        if forced:
            c = self._cheapest_nt(ctx)
            if c is not None:
                return c
            for d in self._suit_order(hi):
                if d == their:
                    continue
                c = self._cheapest(ctx, d)
                if c is not None:
                    return c
        return CALL_PASS

    def _game_bid(self, ctx: BidContext, denom: int):
        r = ctx.reader
        if denom in MAJORS:
            c = bid_id(4, denom)
        elif ctx.hi.balanced() and ctx.hi.kr >= 13:
            c = bid_id(3, DENOM_N)
        else:
            c = bid_id(5, denom)
        return c if c in ctx.legal and c > r.highest else None

    def _invite_bid(self, ctx: BidContext, denom: int):
        r = ctx.reader
        c = bid_id(3, denom) if denom in MAJORS else bid_id(3, denom)
        if c in ctx.legal and c > r.highest \
                and not is_game_or_above((c - FIRST_BID) // 5 + 1, denom):
            return c
        return None

    def _nt_rebid(self, ctx: BidContext):
        r, hi = ctx.reader, ctx.hi
        if hi.kr < 10:
            return None
        # cheapest legal NT, and only a non-jump (a jump needs a specific
        # range the file describes elsewhere)
        for lvl in range(1, 4):
            c = bid_id(lvl, DENOM_N)
            if c > r.highest:
                info = r.classify(c, seat=ctx.hero)
                return c if info.jump == 0 else None
        return None

    def _raise_level(self, ctx: BidContext, denom: int):
        r, hi = ctx.reader, ctx.hi
        base = max(2, r.high_level)
        for lvl in range(base, base + 3):
            c = bid_id(lvl, denom)
            if c > r.highest and not is_game_or_above(lvl, denom):
                return lvl
            if c > r.highest and is_game_or_above(lvl, denom) and hi.kr >= 13:
                return lvl
        return None

    # -- preference ranking (fallback) ----------------------------------------

    def _preference(self, ctx: BidContext, call: int) -> float:
        r, hi = ctx.reader, ctx.hi
        if call == CALL_PASS:
            if self._our_game(ctx):
                return 50.0
            if hi.kr < 8:
                return 20.0
            if self._n_bids(ctx, ctx.hero) >= 2 and hi.kr < 14:
                return 15.0
            return -5.0
        if call == CALL_X:
            return 3.0
        if call == CALL_XX:
            return 1.0
        info = r.classify(call, seat=ctx.hero)
        d, lvl = info.denom, info.level
        score = 0.0
        if d < DENOM_N:
            score += hi.length(d) * 2.0
            if d in MAJORS:
                score += 3.0
            if info.raise_bid:
                score += 4.0
            if info.own_rebid and hi.length(d) >= 5:
                score += 2.0
        else:
            if hi.balanced():
                score += 6.0
        score -= (lvl - 1) * 1.5
        if hi.kr < 8 and lvl >= 3:
            score -= 6.0
        if is_game_or_above(lvl, d) and hi.kr < 13:
            score -= 10.0
        if lvl >= 5:
            score -= 15.0
        return score


def bid_id_of_level(call: int) -> int:
    return (call - FIRST_BID) // 5 + 1


def load_bidder(system_path: str | None = None) -> SystemBidder:
    path = system_path or find_system_file()
    if path is None:
        raise FileNotFoundError("no system file found (models/system.txt)")
    return SystemBidder(parse_system_file(path))


def main():
    import argparse
    from ..data.vocab import call_to_id, SEATS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", default=None)
    ap.add_argument("--hand", help="hero hand 'S.H.D.C'")
    ap.add_argument("--dealer", default="N", choices=list(SEATS))
    ap.add_argument("--hero", default=None, choices=list(SEATS))
    ap.add_argument("--auction", default="", help="calls so far, e.g. 'P 1D'")
    ap.add_argument("--selfplay", action="store_true",
                    help="deal random hands and print system-vs-system auctions")
    ap.add_argument("--deals", type=int, default=10)
    args = ap.parse_args()
    bidder = load_bidder(args.system)
    if args.selfplay:
        from ..env.deals import random_deals
        rng = np.random.default_rng(0)
        deals = random_deals(rng, args.deals)
        for i, deal in enumerate(deals):
            dealer = int(rng.integers(0, 4))
            calls = bidder.bid_auction(deal.hands, dealer)
            print(f"deal {i}: dealer {SEATS[dealer]}  "
                  + " ".join(id_to_call(c) for c in calls))
        return
    if not args.hand:
        ap.error("--hand is required (or use --selfplay)")
    dealer = SEATS.index(args.dealer)
    hero = SEATS.index(args.hero) if args.hero else dealer
    reader = AuctionReader(dealer, 0)
    for c in args.auction.split():
        reader.feed(call_to_id(c))
    legal = bidder._legal(reader)
    call = bidder.choose(reader, hero, args.hand, legal)
    print(id_to_call(call))


if __name__ == "__main__":
    main()
