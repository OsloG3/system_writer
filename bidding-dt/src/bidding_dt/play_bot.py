"""Card-play bot: choose a card by double-dummy simulation over a pool of
hidden-hand candidates.

One bot plays a *side*, not a seat: the declarer-side bot owns declarer AND
dummy (it sees both those hands and models the two defenders), and each defender
bot owns itself (it sees its own hand plus the tabled dummy and models declarer
and partner). Every bot keeps a pool of candidate deals for the hands it cannot
see, drawn from the auction with the hand VQ encoder/decoder (hand/model.py):

  * the encoder is run once per hidden player -- the auction never changes
    during the play, so its memory is cached and reused for every top-up;
  * candidates are dealt as whole 13-card hands at trick one and then *reused*
    trick to trick: a candidate survives as long as it stays legal, i.e. it
    still holds every card its owner actually played and never showed out of a
    suit it could have followed;
  * when the surviving pool drops below a minimum that shrinks as the hand
    advances, fresh candidates are topped up by sampling only the cards a seat
    still holds (hand/model.py generate(n_cards=R));
  * the declarer-side pool is larger than a defender's (declarer sees two hands
    and controls the play, so it can afford more simulations).

The card itself is picked by running endplay's double-dummy `solve_board` over
every surviving candidate and taking the card with the best expected tricks for
the acting side.

Defender leads are constrained: leading an honour above the nine denies the card
directly above it (from touching honours you lead the top one). The bot obeys
this when it leads, and every other bot folds it into hand generation by
blocking that higher card from the leader's candidates, which is a real
bridge inference (leading the K denies the A).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

try:  # endplay lives in the `rl` extra; play_server already requires it
    import endplay.types as ept
    from endplay.dds.solve import SolveMode, solve_all_boards, solve_board
    from endplay.utils.play import trick_winner
except ImportError:  # pragma: no cover - guarded by play_server's own check
    ept = None

from .data.dataset import BiddingDataset
from .data.hands import cards_to_str, indicator_from_cards, str_to_cards
from .dd.solver import _PROJ_DENOM_TO_ENDPLAY
from .hand.data import rotated_tokens

# Card ids follow env/deals.py: card = rank*4 + suit, rank A=0 .. 2=12, suit
# S,H,D,C = 0..3. A card *string* on the wire is rank+suit, e.g. "AS", "9S".
RANKS = "AKQJT98765432"
SUITS = "SHDC"
_RANK_IDX = {c: i for i, c in enumerate(RANKS)}
_SUIT_IDX = {c: i for i, c in enumerate(SUITS)}
ALL_CARDS = frozenset(range(52))


def card_to_str(cid: int) -> str:
    """card id -> 'AS', '9S', 'TD' (rank then suit)."""
    return RANKS[cid // 4] + SUITS[cid % 4]


def str_to_card(s: str) -> int:
    """'AS' / 'A S' / 'SA' -> card id. Accepts rank+suit and suit+rank."""
    s = s.strip().upper()
    if len(s) != 2:
        raise ValueError(f"bad card {s!r}")
    a, b = s[0], s[1]
    if a in _RANK_IDX and b in _SUIT_IDX:      # rank+suit ("AS")
        return _RANK_IDX[a] * 4 + _SUIT_IDX[b]
    if a in _SUIT_IDX and b in _RANK_IDX:      # suit+rank ("SA", endplay style)
        return _RANK_IDX[b] * 4 + _SUIT_IDX[a]
    raise ValueError(f"bad card {s!r}")


def suit_of(cid: int) -> int:
    return cid % 4


def rank_of(cid: int) -> int:
    return cid // 4


def card_above(cid: int) -> int | None:
    """The next higher card of the same suit, or None for an ace."""
    r = cid // 4
    return None if r == 0 else (r - 1) * 4 + cid % 4


def is_honour(cid: int) -> bool:
    """True for T,J,Q,K,A -- the ranks 'higher than the nine' the lead rule uses."""
    return cid // 4 <= 4


def ep_card(cid: int):
    return ept.Card(suit=ept.Denom(cid % 4), rank=ept.Rank.find(RANKS[cid // 4]))


def ep_to_card(card) -> int:
    return _RANK_IDX[card.rank.abbr] * 4 + int(card.suit)


def ep_denom(proj_denom: int):
    """Project CDHSN denom index -> endplay Denom."""
    return ept.Denom(_PROJ_DENOM_TO_ENDPLAY[proj_denom])


@dataclass
class PlayConfig:
    """Tunables for the card-play bot (the pool size is user-adjustable)."""
    decl_pool: int = 48        # candidate deals kept by the declarer-side bot
    def_pool: int = 30         # candidate deals kept by a defender bot
    min_floor: int = 6         # the pool minimum never drops below this
    sample_temp: float = 1.0   # decoder card-draw temperature
    code_temp: float = 1.0     # VQ code-sampling temperature
    codes: bool = True         # sample the VQ codes (more hand variety)
    topup_tries: int = 6       # model attempts per candidate before uniform
    seed: int = 0


@dataclass
class _Candidate:
    """One hypothesis: the still-held cards of each hidden seat."""
    hands: dict[int, set] = field(default_factory=dict)


class PlaySession:
    """Per-bot reasoning state for one board's play phase.

    `known` maps each seat the bot can see (its own, plus declarer/dummy) to
    that seat's whole original 'S.H.D.C' hand. The two remaining seats are
    hidden and modelled by the candidate pool.
    """

    def __init__(self, key, dealer, vuln, calls_ids, declarer, denom,
                 known: dict[int, str], model=None, device="cpu",
                 cfg: PlayConfig | None = None, rng=None):
        if ept is None:  # pragma: no cover
            raise RuntimeError("endplay is required for card play")
        self.key = key
        self.cfg = cfg or PlayConfig()
        self.dealer = int(dealer)
        self.vuln = int(vuln)
        self.calls_ids = [int(c) for c in calls_ids]
        self.declarer = int(declarer)
        self.dummy = (self.declarer + 2) % 4
        self.opener = (self.declarer + 1) % 4
        self.trump = ep_denom(int(denom))
        self.model = model
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng(self.cfg.seed)
        self._lock = threading.Lock()

        self.known_seats = sorted(int(s) for s in known)
        self.hidden_seats = [s for s in range(4) if s not in self.known_seats]
        if len(self.hidden_seats) != 2:
            raise ValueError("a play bot must see exactly two of the four hands")
        self.known_cards = {int(s): set(str_to_cards(h).tolist())
                            for s, h in known.items()}
        self.known_original = set().union(*self.known_cards.values()) \
            if self.known_cards else set()
        self.hidden_pool = ALL_CARDS - self.known_original   # the 26 unseen cards

        self.pool: list[_Candidate] = []
        self.processed = 0                                    # plays folded in
        self.showouts: dict[int, set] = {s: set() for s in self.hidden_seats}
        self.lead_denials: dict[int, set] = {s: set() for s in self.hidden_seats}
        self._enc = {}                                        # target -> (mem, mask)
        self._encode_hidden()

    # ---- encoder cache (runs once per hidden player) -----------------------

    def _encode_hidden(self):
        if self.model is None:
            return
        import torch
        base_excl = indicator_from_cards(sorted(self.known_original)).astype(np.float32)
        for target in self.hidden_seats:
            tokens = rotated_tokens(self.dealer, self.calls_ids, target)[None]
            vuln = np.array([BiddingDataset.vuln_class(self.vuln, target)],
                            dtype=np.int64)
            t = torch.from_numpy(tokens).to(self.device)
            v = torch.from_numpy(vuln).to(self.device)
            e = torch.from_numpy(base_excl).to(self.device)[None]
            with torch.no_grad():
                mem, mask = self.model.memory(t, v, e, None)
            self._enc[target] = (mem, mask)

    # ---- play analysis -----------------------------------------------------

    def _analyze(self, plays):
        """Order-preserving reconstruction of the position from the play list."""
        played_by_seat = {s: [] for s in range(4)}
        for seat, cid in plays:
            played_by_seat[int(seat)].append(int(cid))
        n = len(plays)
        n_done = n // 4
        cur = [(int(s), int(c)) for s, c in plays[4 * n_done:]]
        if cur:
            leader = cur[0][0]
        elif n_done == 0:
            leader = self.opener
        else:
            last = plays[4 * (n_done - 1):4 * n_done]
            winner = trick_winner([ep_card(int(c)) for _, c in last],
                                  ept.Player(int(last[0][0])), self.trump)
            leader = int(winner)
        return {
            "played_by_seat": played_by_seat,
            "n_done": n_done,
            "cur": cur,
            "leader": leader,
            "to_act": (leader + len(cur)) % 4,
            "played_all": [int(c) for _, c in plays],
        }

    def _apply_play(self, seat, cid, lead_suit, is_lead):
        """Fold one observed play into the pool, dropping inconsistent hands."""
        seat = int(seat)
        cid = int(cid)
        if seat in self.known_seats:
            # a hidden defender's honour lead denies the card above it
            return
        if is_lead and seat not in (self.declarer, self.dummy) and is_honour(cid):
            above = card_above(cid)
            if above is not None:
                self.lead_denials[seat].add(above)
        keep = []
        for cand in self.pool:
            rem = cand.hands[seat]
            if cid not in rem:
                continue                              # cannot have played it
            if lead_suit is not None and suit_of(cid) != lead_suit:
                if any(suit_of(x) == lead_suit for x in rem):
                    continue                          # held the suit, did not follow
                self.showouts[seat].add(lead_suit)      # showed out -> void
            nxt = _Candidate(hands=dict(cand.hands))
            nxt.hands[seat] = rem - {cid}
            keep.append(nxt)
        self.pool = keep

    def _process_new_plays(self, plays):
        with self._lock:
            for i in range(self.processed, len(plays)):
                seat, cid = int(plays[i][0]), int(plays[i][1])
                first = plays[(i // 4) * 4][1]
                lead_suit = None if i % 4 == 0 else suit_of(int(first))
                self._apply_play(seat, cid, lead_suit, i % 4 == 0)
            self.processed = len(plays)

    # ---- candidate generation ---------------------------------------------

    def _excl_for(self, seat, played):
        """Cards `seat` cannot still hold: seen, gone, voided or lead-denied."""
        bad = set(self.known_original) | set(played) | set(self.lead_denials[seat])
        for s in self.showouts[seat]:
            for cid in range(52):
                if suit_of(cid) == s:
                    bad.add(cid)
        return bad

    def _sample_target(self, target, excl_ids, R):
        """Draw one still-held hand (R cards) for `target` from the model."""
        import torch
        mem, mask = self._enc[target]
        excl = indicator_from_cards(sorted(excl_ids)).astype(np.float32)
        with torch.no_grad():
            z, _ = self.model.quantize(mem, mask, sample=self.cfg.codes,
                                       temp=self.cfg.code_temp)
            e = torch.from_numpy(excl).to(self.device)[None]
            cards = self.model.generate(z, e, temp=self.cfg.sample_temp, n_cards=R)
        return set(int(x) for x in cards[0].tolist())

    @staticmethod
    def _uniform_fill(free, R, forbid, rng):
        pool = [c for c in free if c not in forbid]
        if len(pool) < R:
            pool = list(free)                     # over-constrained: relax
        idx = rng.permutation(len(pool))[:R]
        return set(pool[i] for i in idx)

    def _uniform_partition(self, free, R, excl, rng):
        a, b = self.hidden_seats
        fa = excl[a] & free
        fb = excl[b] & free
        must_a = [c for c in free if c in fb and c not in fa]
        must_b = [c for c in free if c in fa and c not in fb]
        flex = [c for c in free if c not in fa and c not in fb]
        neither = [c for c in free if c in fa and c in fb]
        ha, hb = set(must_a), set(must_b)
        rng.shuffle(flex)
        need_a = max(0, R[a] - len(ha))
        ha |= set(flex[:need_a])
        rest = flex[need_a:]
        need_b = max(0, R[b] - len(hb))
        hb |= set(rest[:need_b])
        for c in neither + rest[need_b:]:
            if len(ha) < R[a]:
                ha.add(c)
            elif len(hb) < R[b]:
                hb.add(c)
        return {a: ha, b: hb}

    def _gen_candidate(self, info, rng):
        played = info["played_all"]
        free = self.hidden_pool - set(played)
        R = {s: 13 - len(info["played_by_seat"][s]) for s in self.hidden_seats}
        if sum(R.values()) != len(free):              # safety: sizes must match
            R = {s: max(0, R[s]) for s in R}
        a, b = self.hidden_seats
        excl = {s: self._excl_for(s, played) for s in self.hidden_seats}
        order = [a, b]
        rng.shuffle(order)
        for t in range(self.cfg.topup_tries):
            s = order[t % 2]
            o = b if s == a else a
            if R[s] <= 0:
                hands = {s: set(), o: set(free)}
            elif self.model is not None:
                try:
                    s_rem = self._sample_target(s, excl[s], R[s])
                except Exception:
                    s_rem = self._uniform_fill(free, R[s], excl[s], rng)
                if len(s_rem) != R[s] or (s_rem & excl[s]):
                    s_rem = self._uniform_fill(free, R[s], excl[s], rng)
                o_rem = free - s_rem
                hands = {s: s_rem, o: o_rem}
            else:
                hands = self._uniform_partition(free, R, excl, rng)
            o_rem = hands[o]
            if len(o_rem) != R[o] or (o_rem & (excl[o] & free)):
                continue                              # complement broke a rule
            if len(hands[s]) != R[s]:
                continue
            return _Candidate(hands=hands)
        # last resort: a plain partition ignoring soft inferences
        hands = self._uniform_partition(free, R, {s: set() for s in self.hidden_seats}, rng)
        return _Candidate(hands=hands)

    def _topup(self, need, info):
        rng = self.rng
        for _ in range(max(0, need)):
            self.pool.append(self._gen_candidate(info, rng))

    def _min_pool(self, n_done):
        base = self.cfg.decl_pool if self._is_decl_side else self.cfg.def_pool
        frac = max(0.0, (13 - n_done) / 13.0)
        return max(self.cfg.min_floor, int(round(base * frac)))

    @property
    def _is_decl_side(self):
        return self.declarer in self.known_seats

    def pool_size(self):
        return self.cfg.decl_pool if self._is_decl_side else self.cfg.def_pool

    # ---- decision ----------------------------------------------------------

    def _legal_cards(self, rem, cur):
        if not cur:
            return set(rem)
        lead = suit_of(cur[0][1])
        follow = {c for c in rem if suit_of(c) == lead}
        return follow if follow else set(rem)

    def _apply_lead_rule(self, legal, rem, is_defender, is_lead):
        if not (is_defender and is_lead):
            return legal
        allowed = set()
        for c in legal:
            if not is_honour(c):
                allowed.add(c)
                continue
            above = card_above(c)
            if above is None or above not in rem:
                allowed.add(c)                        # top of the touching run
        return allowed or legal

    def _build_deal(self, remaining, leader, cur):
        parts = []
        for s in range(4):
            cs = sorted(remaining[s])
            parts.append(cards_to_str(cs) if cs else "-")
        deal = ept.Deal.from_pbn("N:" + " ".join(parts))
        deal.trump = self.trump
        deal.first = ept.Player(leader)
        for _, cid in cur:
            deal.play(ep_card(cid), from_hand=False)
        return deal

    def choose(self, plays, to_act):
        """Return the card id `to_act` should play given the play so far."""
        plays = [(int(s), str_to_card(c) if isinstance(c, str) else int(c))
                 for s, c in plays]
        self._process_new_plays(plays)
        info = self._analyze(plays)
        to_act = int(to_act)
        with self._lock:
            target = self.pool_size()
            if len(self.pool) < self._min_pool(info["n_done"]):
                self._topup(target - len(self.pool), info)

            known_rem = {s: set(self.known_cards[s]) - set(info["played_by_seat"][s])
                         for s in self.known_seats}
            rem_act = known_rem[to_act]
            legal = self._legal_cards(rem_act, info["cur"])
            is_defender = to_act not in (self.declarer, self.dummy)
            legal = self._apply_lead_rule(legal, rem_act, is_defender,
                                          len(info["cur"]) == 0)
            if not legal:
                legal = self._legal_cards(rem_act, info["cur"]) or set(rem_act)
            if not self.pool:
                return min(legal)

            deals = []
            for cand in self.pool:
                remaining = dict(known_rem)
                remaining.update(cand.hands)
                deals.append(self._build_deal(remaining, info["leader"], info["cur"]))
            solved = solve_all_boards(deals, SolveMode.Default)
            total = {c: 0 for c in legal}
            count = {c: 0 for c in legal}
            for board in solved:
                for card, tricks in board:
                    cid = ep_to_card(card)
                    if cid in total:
                        total[cid] += tricks
                        count[cid] += 1
            best, best_key = None, None
            for c in sorted(legal):
                avg = total[c] / count[c] if count[c] else 0.0
                key = (-avg, c)
                if best_key is None or key < best_key:
                    best_key, best = key, c
            return best


class PlayBot:
    """Holds the per-board sessions for one sidecar (thread-safe)."""

    def __init__(self, model=None, device="cpu", cfg: PlayConfig | None = None,
                 ttl: float = 6 * 3600.0):
        self.model = model
        self.device = device
        self.cfg = cfg or PlayConfig()
        self.ttl = ttl
        self._sessions: dict[str, PlaySession] = {}
        self._last_used: dict[str, float] = {}
        self._lock = threading.Lock()

    def get_session(self, key, factory):
        import time
        now = time.time()
        with self._lock:
            self._evict(now)
            s = self._sessions.get(key)
            if s is None:
                s = factory()
                self._sessions[key] = s
            self._last_used[key] = now
            return s

    def _evict(self, now):
        stale = [k for k, t in self._last_used.items() if now - t > self.ttl]
        for k in stale:
            self._sessions.pop(k, None)
            self._last_used.pop(k, None)

    def drop(self, key):
        with self._lock:
            self._sessions.pop(key, None)
            self._last_used.pop(key, None)

    def drop_prefix(self, prefix):
        with self._lock:
            for k in [k for k in self._sessions if k.startswith(prefix)]:
                del self._sessions[k]
                self._last_used.pop(k, None)
