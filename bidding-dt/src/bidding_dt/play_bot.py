"""Card-play bot: choose a card by double-dummy simulation over a pool of
hidden-hand candidates, planned as a weighted strategy over merged layouts.

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
    still holds (hand/model.py generate(n_cards=R)) -- a whole cohort per
    hidden seat in ONE batched decoder call, with complements that break the
    other seat's constraints repaired by swapping instead of re-rolled
    (cfg.batch_topup; the per-candidate retry loop used to make the pool phase
    the single biggest latency source of a hand);
  * the declarer-side pool is larger than a defender's (declarer sees two hands
    and controls the play, so it can afford more simulations).

At every decision the pool is turned into a WEIGHTED set of MACRO LAYOUTS:
each candidate is re-scored under the hand model given the auction and every
exclusion the play has produced (cards seen, show-outs, honour leads denying
the card above), so layouts the play made implausible count less; layouts that
differ only in low cards that cannot beat anything the visible hands still
hold are merged into one macro layout whose weight is the sum (see
play_strategy.suit_relevance / merge_layouts). On top of that, every observed
hidden play is scored by its double-dummy cost under each hypothesis: a player
would not have made a clearly bad play from their hand, so layouts under which
the observed play gave away tricks are discounted by exp(-cost/play_temp).
That inference is kept cheap where it is free: a play that was FORCED (its
only legal card) carries no information and is never solved, and the
retro-scoring of fresh top-up candidates reviews recent plays NEWEST FIRST
under a wall-clock budget (cfg.play_review_time) instead of unconditionally
re-solving large early positions.

The card itself is picked from those weighted layouts:

  * the declarer side runs the beam strategy search in play_strategy.py -- a
    suit-combination-solver-style plan that chooses declarer's card at each
    reachable information set and is valued by how many (weighted) layouts the
    plan works on, within solve/time budgets;
  * defenders (and any fallback) rank each legal card by its expected
    double-dummy tricks weighted by layout likelihood plus, by default, the
    weighted likelihood that the card BEATS (defence) or MAKES (declarer
    fallback) the contract (cfg.rank_objective/rank_make_bonus): pure trick
    expectation happily trades a contract-killing play for an overtrick.

All double-dummy work goes through dd/fastsolve.py: positions are built as
raw DDS structs with integer maths and solved in memoised batches (the same
public position recurs constantly within and across decisions), which is what
keeps the per-card budgets affordable. Every decision also records its phase
times and DDS solve count in session.last_timing (surfaced on the wire as
"timing") for latency tuning.

Cards that tie in score are chosen AT RANDOM among the tied set (the bot used
to always play the highest, which is predictable and wastes honours), narrowed
first by the standard table preference: a spot doing an honour's job goes
first, and among tied honours the cheapest (cfg.tie_prefer_spots).

`choose_drill` is the variant the card-problems page uses: it ranks the options
the same way (with only the information a real game gives), but returns the
best-ranked card that also costs no trick double-dummy on the *real* deal,
which the drill server supplies.

Defender leads are constrained: leading an honour above the nine denies the card
directly above it (from touching honours you lead the top one). The bot obeys
this when it leads, and every other bot folds it into hand generation by
blocking that higher card from the leader's candidates -- and drops surviving
candidates that still hold a denied card -- which is a real bridge inference
(leading the K denies the A).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

try:  # endplay lives in the `rl` extra; play_server already requires it
    import endplay.types as ept
except ImportError:  # pragma: no cover - guarded by play_server's own check
    ept = None

from .data.dataset import BiddingDataset
from .data.hands import indicator_from_cards, str_to_cards
from .data.parse import MAX_CALLS
from .dd import fastsolve
from .dd.reward import contract_from_auction
from .dd.solver import _PROJ_DENOM_TO_ENDPLAY
from .hand.data import rotated_tokens
from .play_strategy import Layout, StrategySearch, merge_layouts, suit_relevance

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
    """Tunables for the card-play bot (the pool size is user-adjustable).

    The defaults were raised together with dd/fastsolve (raw-struct batched
    solving + a shared position cache): within the same per-decision
    wall-clock budget the raised search analyses ~3x more positions than the
    old one, and in a 12-deal A/B the declarer side reached its double-dummy
    trick count on 12/12 hands (worst hand exactly DD) versus 10/12 (worst
    -2 tricks) at the old caps.

    strat_time and play_review_time were then tuned DOWN for whole-hand
    latency (the pool-maintenance retro-scoring, not the search, was the
    biggest uncapped cost): a whole hand plays ~24% faster on an idle
    machine and more under CPU contention (the review budget bounds the
    top-up spikes that used to balloon there), while a 12-deal A/B kept the
    declarer side at its double-dummy trick count on 12/12 hands. Both knobs
    are CLI flags, so the quality-first settings (2.0 and 0) are one
    restart away.

    The speed group below (batch_topup, fast_onestep, endgame_skip, plus the
    _weight_cache in PlaySession) then roughly HALVED whole-hand wall time
    again at unchanged budgets (16-deal A/B: 103.8s -> 51.5s per hand; the
    strategy search explores ~2x more nodes inside the same strat_time), and
    the quality knobs (rank_objective, tie_prefer_spots for the defence;
    def_smart for the search's defence model) measured net-positive in
    isolated 16-deal arms (defence took +4 paired tricks against an unchanged
    declarer side; declarer reached its double-dummy count on 14/16 hands vs
    12/16). Everything is flag-gated for A/B runs via play_ab.py; the flags
    below are all one CLI switch away from the old behaviour.
    """
    decl_pool: int = 64        # candidate deals kept by the declarer-side bot
    def_pool: int = 40         # candidate deals kept by a defender bot
    min_floor: int = 6         # the pool minimum never drops below this
    sample_temp: float = 1.0   # decoder card-draw temperature
    code_temp: float = 1.0     # VQ code-sampling temperature
    codes: bool = True         # sample the VQ codes (more hand variety)
    topup_tries: int = 6       # model attempts per candidate before uniform
    seed: int = 0
    # ---- layout weighting (both sides) ----
    weight_temp: float = 1.0   # softmax temperature on model log-probs;
    #                          # 0 disables weighting (every layout counts 1)
    # ---- declarer-side strategy search (play_strategy.py) ----
    strategy: bool = True      # plan a conditional line, not a per-card vote
    strat_layouts: int = 20    # macro layouts the search evaluates per node
    strat_beam: int = 4        # declarer cards expanded per decision node
    strat_depth: int = 6       # tricks planned ahead with full branching;
    #                          # deeper positions get a clairvoyant DD leaf
    strat_def_depth: int = 3   # tricks where defenders are modelled DD-best
    #                          # (a cheap no-solve heuristic beyond that)
    strat_def_branch: int = 4  # defender branches expanded per node
    strat_solves: int = 12000  # safety cap on boards per decision (cache hits
    #                          # make boards cheap, so this was raised with the
    #                          # caps; the wall-clock budget below is the real
    #                          # limiter -- one root ranking already costs
    #                          # layouts x legal cards boards)
    strat_time: float = 1.5    # wall-clock seconds per decision. Lowered from
    #                          # 2.0 together with the fastsolve cache: a
    #                          # decision now fits ~2x the positions the old
    #                          # solver did per second, and quality A/B runs
    #                          # at 1.0-1.5s matched the unbudgeted bot's
    #                          # double-dummy results, so 1.5 keeps depth
    #                          # headroom while cutting whole-hand latency.
    strat_objective: str = "both"   # "ev" | "make" | "both": expected tricks,
    #                          # weighted share of layouts making the contract,
    #                          # or the sum (see play_strategy._objective)
    strat_make_bonus: float = 1.0   # weight of P(make) inside "both"
    fast_onestep: bool = True  # strategy search: one DDS solve per layout scores
    #                          # every declarer card (solutions=3), instead of
    #                          # one solve per (layout, card) -- identical values,
    #                          # ~|cards|x fewer boards inside the same budget
    def_smart: bool = True     # strategy search: model the no-solve defence
    #                          # beyond strat_def_depth with the take/cover/
    #                          # overtake table rules instead of always-lowest
    endgame_skip: bool = True  # strategy search: when a card's one-step value
    #                          # already proves declarer takes EVERY remaining
    #                          # trick, return it (among equals) instead of
    #                          # spending the budget confirming a run-out
    # ---- one-step ranking (defenders, and declarer-side fallbacks) ----
    rank_objective: str = "both"  # "ev" | "both": expected double-dummy tricks,
    #                          # or that plus the weighted likelihood of beating
    #                          # (defence) / making (declarer fallback) the
    #                          # contract -- the myopic mirror of strat_objective.
    #                          # A defence that only maximises expected tricks
    #                          # happily trades a contract-killing play for an
    #                          # overtrick; P(beat) makes that trade explicit.
    rank_make_bonus: float = 1.0  # weight of P(beat/make) inside "both"
    tie_prefer_spots: bool = True  # among exact score ties prefer spots over
    #                          # honours (and the cheapest honour among tied
    #                          # honours), still at random within the preference
    # ---- speed (same budgets, less wall-clock per decision) ----
    batch_topup: bool = True   # draw a whole top-up cohort in ONE batched decoder
    #                          # call per hidden seat and repair complement
    #                          # violations by swapping, instead of up to
    #                          # topup_tries sequential draws per candidate (the
    #                          # pool phase used to dominate whole-hand latency)
    # ---- play-quality inference ("a player wouldn't have made a bad play") ----
    play_temp: float = 1.0     # temperature turning each observed hidden play's
    #                          # double-dummy cost into a likelihood factor
    #                          # exp(-cost/play_temp); 0 disables the inference
    play_pen_cap: float = 2.0  # per-play cost cap in tricks (a 5-trick blunder
    #                          # should not annihilate a layout on its own)
    play_review: int = 8       # fresh top-up candidates are retro-scored against
    #                          # this many recent hidden plays
    play_review_time: float = 0.5  # wall-clock seconds one top-up's retro-scoring
    #                          # may spend, newest play first (0 = unbudgeted).
    #                          # Early-hand top-ups used to re-solve up to
    #                          # review x fresh-candidates LARGE past positions
    #                          # every decision -- the single biggest latency
    #                          # source in a hand; the budget bounds it while
    #                          # keeping the most relevant (recent) reviews.
    #                          # Skipped reviews are imputed at the session's
    #                          # mean penalty (solve-free, mean-preserving).


@dataclass
class _Candidate:
    """One hypothesis: the still-held cards of each hidden seat.

    `play_pen` accumulates the double-dummy cost (in tricks) of the plays its
    hidden seats actually made: a layout under which an observed play was a
    clear mistake is less likely than one under which it was fine (see
    PlaySession._quality_pens).
    """
    hands: dict[int, set] = field(default_factory=dict)
    play_pen: float = 0.0


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
        self.trump_int = int(self.trump)     # DDS denom int for fastsolve
        self.model = model
        # the VQ hand model reads auctions up to MAX_CALLS (its positional
        # embedding has no room beyond that); the site allows longer ones, so
        # for those fall back to the constraint-based sampler, which still
        # honours the play and the lead rules
        if self.model is not None and len(self.calls_ids) > MAX_CALLS:
            self.model = None
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
        # exact-match cache of _model_weights' per-seat decoder log-probs,
        # keyed (seat, hand size, exclusion set). Cleared whenever the pool's
        # hands change, so a hit means the torch pass would be bit-identical:
        # a known-seat play (declarer acting, then dummy, in one trick)
        # changes neither the pool nor any exclusion -- every played known
        # card is already excluded through known_original -- yet the weights
        # phase used to re-run both encoders for it every decision.
        self._weight_cache: dict = {}
        # running mean of the double-dummy cost per (observed play, candidate)
        # pair, accumulated from every real _quality_pens solve: the estimator
        # _retro_score imputes budget-skipped reviews with (guarded by _lock)
        self._pen_sum = 0.0
        self._pen_n = 0
        self._plays: list = []                # last normalised play list seen
        self.showouts: dict[int, set] = {s: set() for s in self.hidden_seats}
        self.lead_denials: dict[int, set] = {s: set() for s in self.hidden_seats}
        self.last_search: dict | None = None      # diagnostics of the last plan
        # per-phase wall-clock accounting (seconds): `timing` accumulates over
        # the session's life, `last_timing` describes the most recent decision;
        # both ride along in the sidecar response for latency tuning
        self.timing: dict = {"decisions": 0}
        self.last_timing: dict = {}
        # the contract's trick target (level + 6) drives the "works on the
        # highest number of layouts" objective of the strategy search
        self.target = None
        try:
            contract = contract_from_auction(self.dealer, self.calls_ids)
            if contract is not None:
                self.target = 6 + int(contract.level)
        except Exception:                            # unknown/passed-out form
            self.target = None
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
        tricks_decl = 0
        for t in range(n_done):
            trick = plays[4 * t:4 * t + 4]
            w = (int(trick[0][0]) + fastsolve.trick_winner(
                [int(c) for _, c in trick], self.trump_int)) % 4
            tricks_decl += w in (self.declarer, self.dummy)
        if cur:
            leader = cur[0][0]
        elif n_done == 0:
            leader = self.opener
        else:
            last = plays[4 * (n_done - 1):4 * n_done]
            leader = (int(last[0][0]) + fastsolve.trick_winner(
                [int(c) for _, c in last], self.trump_int)) % 4
        return {
            "played_by_seat": played_by_seat,
            "n_done": n_done,
            "cur": cur,
            "leader": leader,
            "to_act": (leader + len(cur)) % 4,
            "played_all": [int(c) for _, c in plays],
            "tricks_decl": tricks_decl,
        }

    def _apply_play(self, seat, cid, lead_suit, is_lead, pens=None):
        """Fold one observed play into the pool, dropping inconsistent hands.

        `pens` (optional) is a per-candidate list aligned with the pool: the
        double-dummy cost this play had under each hypothesis, added to the
        candidate's accumulated play penalty (see _quality_pens).
        """
        seat = int(seat)
        cid = int(cid)
        if seat in self.known_seats:
            # a hidden defender's honour lead denies the card above it
            return
        if is_lead and seat not in (self.declarer, self.dummy) and is_honour(cid):
            above = card_above(cid)
            if above is not None:
                self.lead_denials[seat].add(above)
        denied = self.lead_denials[seat]
        keep = []
        for j, cand in enumerate(self.pool):
            rem = cand.hands[seat]
            if cid not in rem:
                continue                              # cannot have played it
            if denied & rem:
                continue                              # holds a lead-denied card
            if lead_suit is not None and suit_of(cid) != lead_suit:
                if any(suit_of(x) == lead_suit for x in rem):
                    continue                          # held the suit, did not follow
                self.showouts[seat].add(lead_suit)      # showed out -> void
            nxt = _Candidate(hands=dict(cand.hands))
            nxt.hands[seat] = rem - {cid}
            nxt.play_pen = cand.play_pen + (pens[j] if pens is not None else 0.0)
            keep.append(nxt)
        self.pool = keep
        self._weight_cache.clear()          # pool hands changed under the cache

    def _quality_pens(self, seat, cid, idx, plays, hidden_hands):
        """The double-dummy cost of one observed hidden-seat play, per layout.

        For play index `idx` (seat `seat` played card `cid`) each candidate's
        pre-play position is solved; the penalty is how many tricks the played
        card costs versus the double-dummy best card from that same hand
        (both measured on the acting side, so the units cancel across
        layouts). The inference: a player would not have made a clearly bad
        play from their hand, so layouts under which the observed play was
        bad become less likely. `hidden_hands` is a list aligned with it
        giving each candidate's hidden-seat hands AT THAT MOMENT (the pool's
        current hands for live plays, reconstructed ones for retro-scoring
        fresh candidates). Returns a list of penalties, or None when the
        inference is off or not applicable.
        """
        if seat in self.known_seats or self.cfg.play_temp <= 0 or not hidden_hands:
            return None
        n_done = idx // 4
        cur = [(int(s), int(c)) for s, c in plays[4 * n_done:idx]]
        if cur:
            leader = cur[0][0]
        elif n_done == 0:
            leader = self.opener
        else:
            last = plays[4 * (n_done - 1):4 * n_done]
            leader = (int(last[0][0]) + fastsolve.trick_winner(
                [int(c) for _, c in last], self.trump_int)) % 4
        played_before = {s: set() for s in range(4)}
        for s, c in plays[:idx]:
            played_before[int(s)].add(int(c))
        # at a mid-trick position the seats that already played legitimately
        # hold one card fewer; anything else means an injected/synthetic pool
        base = 13 - n_done
        cur_seats = {s for s, _ in cur}
        cur_cids = [int(c) for _, c in cur]
        # the visible seats' hands are the same for every candidate
        known_now = {s: self.known_cards[s] - played_before[s]
                     for s in self.known_seats}
        is_defender = seat not in (self.declarer, self.dummy)
        cap = float(self.cfg.play_pen_cap)
        pens = [0.0] * len(hidden_hands)
        todo = []                     # (pen slot, allowed, remaining) to solve
        for j, hh in enumerate(hidden_hands):
            remaining = dict(known_now)
            remaining.update(hh)
            if any(len(v) != base - (s in cur_seats)
                   for s, v in remaining.items()):
                return None
            rem = hh[seat]
            legal = self._legal_cards(rem, cur)
            allowed = self._apply_lead_rule(legal, rem, is_defender, not cur)
            if allowed == {cid}:
                continue              # forced play: costs nothing by definition
            todo.append((j, allowed, remaining))
        if not todo:
            self._accrue_pens(pens)       # every play here was forced (pen 0)
            return pens
        try:
            solved = fastsolve.solve_positions(
                [(rem, leader, cur_cids) for _j, _a, rem in todo],
                self.trump_int)
        except Exception:
            return None                  # scoring must never break play folding
        for (j, allowed, _rem), scores in zip(todo, solved):
            obs = scores.get(int(cid))
            best = max((scores[c] for c in allowed if c in scores), default=None)
            if obs is None or best is None:
                continue                            # cannot judge: no penalty
            pens[j] = min(max(best - obs, 0.0), cap)
        self._accrue_pens(pens)
        return pens

    def _accrue_pens(self, pens):
        """Fold a scored penalty list into the session-wide running mean of
        double-dummy cost per (observed play, candidate) pair. _retro_score
        imputes budget-skipped reviews at this mean, so it draws on every
        play scored live across the whole hand -- including the informative
        early leads -- rather than only the (often forced, near-zero) recent
        plays a tight budget happened to review."""
        self._pen_sum += sum(pens)
        self._pen_n += len(pens)

    def _retro_score(self, fresh):
        """Score freshly topped-up candidates against the recent hidden plays.

        Live scoring only covers plays that happen while a candidate sits in
        the pool; without this, top-ups (which merely satisfy the hard
        constraints) would come out systematically likelier than penalised
        veterans. Bounded by cfg.play_review recent hidden plays AND a
        cfg.play_review_time wall-clock budget per top-up (0 = unbudgeted):
        plays are reviewed NEWEST FIRST, because the most recent plays are
        both the most decision-relevant and the cheapest to solve (fewer
        cards), so an exhausted budget drops the oldest, largest positions.
        Reviews the budget cuts are charged to the fresh cohort at the
        session-wide average penalty per (play, candidate) pair (see
        _accrue_pens) -- a mean-preserving completion that costs no solves
        and keeps top-ups from coming out systematically likelier than the
        penalised veterans; only the per-candidate discrimination of the
        skipped plays is lost. (Unbudgeted early-hand top-ups used to cost
        seconds of re-solved history every decision.)
        """
        plays = self._plays
        if (not fresh or not plays or self.cfg.play_temp <= 0
                or self.cfg.play_review <= 0):
            return
        hidden = set(self.hidden_seats)
        idxs = [i for i, (s, _) in enumerate(plays[:self.processed])
                if int(s) in hidden]
        budget = float(self.cfg.play_review_time)
        deadline = time.monotonic() + budget if budget > 0 else None
        review = idxs[-int(self.cfg.play_review):]
        done = 0
        for idx in reversed(review):
            if deadline is not None and done and time.monotonic() >= deadline:
                break
            seat, cid = int(plays[idx][0]), int(plays[idx][1])
            # each fresh candidate held, at idx, its current cards plus every
            # card its seat played from idx onwards
            after = {s: set() for s in self.hidden_seats}
            for s, c in plays[idx:self.processed]:
                if int(s) in after:
                    after[int(s)].add(int(c))
            hands = [{s: set(cand.hands[s]) | after[s] for s in self.hidden_seats}
                     for cand in fresh]
            pens = self._quality_pens(seat, cid, idx, plays, hands)
            if pens:
                for cand, p in zip(fresh, pens):
                    cand.play_pen += p
            done += 1
        # Mean-preserving completion: charge each skipped review at the
        # session-wide average penalty per (play, candidate) pair, so fresh
        # candidates are not left systematically under-penalised (over-weighted
        # vs the veterans) when the budget cuts reviews -- that bias measured
        # ~0.2 tricks/hand of declarer error. Costs no solves; only the
        # per-candidate discrimination of the skipped plays is lost.
        skipped = len(review) - done
        if skipped > 0 and fresh and self._pen_n > 0:
            add = skipped * (self._pen_sum / self._pen_n)
            if add > 0.0:
                for cand in fresh:
                    cand.play_pen += add

    def _process_new_plays(self, plays):
        with self._lock:
            self._plays = plays
            for i in range(self.processed, len(plays)):
                seat, cid = int(plays[i][0]), int(plays[i][1])
                first = plays[(i // 4) * 4][1]
                lead_suit = None if i % 4 == 0 else suit_of(int(first))
                pens = self._quality_pens(
                    seat, cid, i, plays, [c.hands for c in self.pool])
                self._apply_play(seat, cid, lead_suit, i % 4 == 0, pens)
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

    def _uniform_partition(self, free, R, excl, rng, first):
        """Split `free` between the two hidden seats honouring the exclusion
        sets. `first` is the seat filled before the other takes the remainder
        (it also wins any over-constrained leftovers), so callers alternate it
        to keep the pool free of a generation-order bias."""
        a, b = self.hidden_seats
        second = b if first == a else a
        fa = excl[first] & free
        fb = excl[second] & free
        must_a = [c for c in free if c in fb and c not in fa]
        must_b = [c for c in free if c in fa and c not in fb]
        flex = [c for c in free if c not in fa and c not in fb]
        neither = [c for c in free if c in fa and c in fb]
        ha, hb = set(must_a), set(must_b)
        rng.shuffle(flex)
        need_a = max(0, R[first] - len(ha))
        ha |= set(flex[:need_a])
        rest = flex[need_a:]
        need_b = max(0, R[second] - len(hb))
        hb |= set(rest[:need_b])
        for c in neither + rest[need_b:]:
            if len(ha) < R[first]:
                ha.add(c)
            elif len(hb) < R[second]:
                hb.add(c)
        return {first: ha, second: hb}

    def _gen_candidate(self, info, rng, first):
        """One hidden-hand hypothesis. `first` is the hidden seat generated
        up front (by the model, or uniformly); the other seat is then given
        the remaining cards. Callers alternate `first` so half the pool is
        built from each hidden seat's perspective."""
        played = info["played_all"]
        free = self.hidden_pool - set(played)
        R = {s: 13 - len(info["played_by_seat"][s]) for s in self.hidden_seats}
        if sum(R.values()) != len(free):              # safety: sizes must match
            R = {s: max(0, R[s]) for s in R}
        a, b = self.hidden_seats
        second = b if first == a else a
        excl = {s: self._excl_for(s, played) for s in self.hidden_seats}
        order = [first, second]
        for t in range(self.cfg.topup_tries):
            s = order[t % 2]
            o = second if s == first else first
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
                hands = self._uniform_partition(free, R, excl, rng, s)
            o_rem = hands[o]
            if len(o_rem) != R[o] or (o_rem & (excl[o] & free)):
                continue                              # complement broke a rule
            if len(hands[s]) != R[s]:
                continue
            return _Candidate(hands=hands)
        # last resort: a plain partition ignoring soft inferences
        hands = self._uniform_partition(
            free, R, {s: set() for s in self.hidden_seats}, rng, first)
        return _Candidate(hands=hands)

    def _sample_targets(self, seat, excl_ids, R, n):
        """Batched `_sample_target`: draw `n` still-held hands (R cards each)
        for `seat` in ONE quantize+decode pass over the cached auction memory.
        The decoder's banned mask guarantees each row is a valid R-subset of
        the non-excluded cards, exactly like the single draw."""
        import torch
        if n <= 0:
            return []
        if R <= 0:
            return [set() for _ in range(n)]
        mem, mask = self._enc[seat]
        excl = indicator_from_cards(sorted(excl_ids)).astype(np.float32)
        with torch.no_grad():
            z, _ = self.model.quantize(
                mem.expand(n, -1, -1),
                None if mask is None else mask.expand(n, -1, -1, -1),
                sample=self.cfg.codes, temp=self.cfg.code_temp)
            e = torch.from_numpy(excl).to(self.device)[None].expand(n, -1)
            cards = self.model.generate(z, e, temp=self.cfg.sample_temp,
                                        n_cards=R)
        return [set(int(x) for x in row.tolist()) for row in cards]

    @staticmethod
    def _repair_pair(s_rem, s, o, free, R, excl):
        """Split `free` between the hidden seats around a model draw for `s`.

        The draw already honours s's exclusions; its complement may still hold
        cards `o` cannot (those MUST sit with s). Each such violation is
        swapped for a drawn card `o` may hold, which keeps the sample
        model-shaped where the old per-candidate path re-rolled the whole hand
        and often ended in a uniform fill. Returns {seat: hand}, or None when
        unrepairable (a card neither seat may hold, or nothing swappable).
        """
        s_rem = set(s_rem)
        if len(s_rem) != R[s] or (s_rem & excl[s]) or not s_rem <= free:
            return None
        o_rem = free - s_rem
        if len(o_rem) != R[o]:
            return None
        for c in sorted(o_rem & excl[o]):
            if c in excl[s]:
                return None                    # neither seat may hold this card
            swaps = [x for x in sorted(s_rem) if x not in excl[o]]
            if not swaps:
                return None
            s_rem.discard(swaps[0])
            s_rem.add(c)
        return {s: s_rem, o: free - s_rem}

    def _batch_candidates(self, need, info):
        """Up to `need` candidates from ONE batched decoder call per hidden
        seat (two calls per round, a second round from the other seat's
        perspective for the draws whose complement was unrepairable). Each
        candidate keeps the old alternation: half are dealt from one hidden
        seat's perspective, half from the other's, so neither generator order
        dominates the pool. May return fewer than `need`; the caller tops the
        rest up through the per-candidate path."""
        played = info["played_all"]
        free = self.hidden_pool - set(played)
        R = {s: 13 - len(info["played_by_seat"][s]) for s in self.hidden_seats}
        if sum(R.values()) != len(free) or min(R.values()) < 0:
            return []                          # let the safe path handle it
        excl = {s: self._excl_for(s, played) for s in self.hidden_seats}
        a, b = self.hidden_seats
        firsts = [self.hidden_seats[(len(self.pool) + i) % 2]
                  for i in range(need)]
        out: list = [None] * need
        for attempt in (0, 1):
            todo = [i for i in range(need) if out[i] is None]
            if not todo:
                break
            for persp in (a, b):
                idxs = [i for i in todo
                        if (firsts[i] if attempt == 0 else
                            (b if firsts[i] == a else a)) == persp]
                if not idxs:
                    continue
                other = b if persp == a else a
                draws = self._sample_targets(persp, excl[persp], R[persp],
                                             len(idxs))
                for i, s_rem in zip(idxs, draws):
                    pair = self._repair_pair(s_rem, persp, other, free, R, excl)
                    if pair is not None:
                        out[i] = _Candidate(hands=pair)
        return [c for c in out if c is not None]

    def _topup(self, need, info):
        need = max(0, need)
        if not need:
            return
        fresh = []
        if self.model is not None and self.cfg.batch_topup:
            try:
                fresh = self._batch_candidates(need, info)
            except Exception:
                fresh = []                     # any batch failure -> old path
        while len(fresh) < need:
            # alternate which hidden seat is generated first: half the pool
            # is dealt from one seat with the other taking the remainder,
            # half the other way round, so neither seat's generator order
            # dominates the candidates
            first = self.hidden_seats[(len(self.pool) + len(fresh)) % 2]
            fresh.append(self._gen_candidate(info, self.rng, first))
        self.pool.extend(fresh)
        self._weight_cache.clear()          # pool grew under the cache
        self._retro_score(fresh)

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
        """An endplay Deal at this position, for callers that want endplay's
        own solve functions (the hot paths use dd/fastsolve directly). Built
        on the raw DDS struct -- 20x cheaper than Deal.from_pbn string
        parsing. Convention: `remaining` hands EXCLUDE the current trick's
        cards, which `cur` [(seat, cid), ...] puts on the table."""
        d = fastsolve.make_deal(remaining, leader, self.trump_int,
                                [int(c) for _, c in cur])
        deal = ept.Deal.__new__(ept.Deal)   # skip __init__'s struct churn
        deal._data = d
        return deal

    @staticmethod
    def _norm_plays(plays):
        return [(int(s), str_to_card(c) if isinstance(c, str) else int(c))
                for s, c in plays]

    def _ensure_pool(self, info):
        target = self.pool_size()
        if len(self.pool) < self._min_pool(info["n_done"]):
            self._topup(target - len(self.pool), info)

    def _legal_for(self, info, to_act):
        """The acting seat's still-held cards and its legal choices."""
        known_rem = {s: set(self.known_cards[s]) - set(info["played_by_seat"][s])
                     for s in self.known_seats}
        rem_act = known_rem[to_act]
        legal = self._legal_cards(rem_act, info["cur"])
        is_defender = to_act not in (self.declarer, self.dummy)
        legal = self._apply_lead_rule(legal, rem_act, is_defender,
                                      len(info["cur"]) == 0)
        if not legal:
            legal = self._legal_cards(rem_act, info["cur"]) or set(rem_act)
        return known_rem, legal

    # ---- layout weighting ----------------------------------------------------

    def _model_weights(self, info):
        """Per-candidate likelihood weights from the VQ hand model.

        The encoder is re-run with each hidden seat's CURRENT exclusion mask
        (seen, gone, voided and lead-denied cards), so the score reflects the
        auction *and* everything the play so far has shown: layouts the play
        made implausible come out low. The two hidden seats are scored as
        independent marginals (their hands are complementary, so this is an
        approximation) and the decoder log-probs of each candidate's
        still-held cards are summed. Candidates holding an excluded card get
        ~0 (the banned softmax slot). Returns a (len(pool),) float array.
        """
        import torch
        n = len(self.pool)
        logp = np.zeros(n)
        if n == 0 or self.model is None:
            return np.ones(max(n, 0))
        played = info["played_all"]
        dev = self.device
        n_cards = self.model.cfg.n_cards
        for s in self.hidden_seats:
            R = 13 - len(info["played_by_seat"][s])
            if R <= 0 or not all(len(c.hands[s]) == R for c in self.pool):
                continue                          # ragged pool: skip this seat
            excl_ids = sorted(self._excl_for(s, played))
            # exact-match cache: the decoder pass below is deterministic in
            # (seat, R, exclusions, pool hands), and the pool hands are pinned
            # by the clear-on-mutation discipline (see __init__), so a hit is
            # bit-identical to recomputing -- known-seat plays hit every time
            ck = (s, R, tuple(excl_ids))
            lp_np = self._weight_cache.get(ck)
            if lp_np is None:
                excl = indicator_from_cards(excl_ids).astype(np.float32)
                tokens = rotated_tokens(self.dealer, self.calls_ids, s)[None]
                vuln = np.array([BiddingDataset.vuln_class(self.vuln, s)],
                                dtype=np.int64)
                seqs = np.array([sorted(c.hands[s]) for c in self.pool],
                                dtype=np.int64)
                with torch.no_grad():
                    mem, mask = self.model.memory(
                        torch.from_numpy(tokens).to(dev),
                        torch.from_numpy(vuln).to(dev),
                        torch.from_numpy(excl).to(dev)[None], None)
                    z, _ = self.model.quantize(mem, mask, sample=False)
                    z = z.expand(n, -1, -1)
                    inp_t = torch.from_numpy(seqs).to(dev)
                    bos = torch.full((n, 1), n_cards, dtype=torch.long,
                                     device=dev)
                    inp = torch.cat([bos, inp_t[:, :-1]], dim=1)
                    excl_b = torch.from_numpy(excl).to(dev)[None].expand(n, -1)
                    logits = self.model._decode(z, inp).float().masked_fill(
                        self.model._banned(inp, excl_b, hand_len=R), -1e9)
                    lp = (torch.log_softmax(logits, dim=-1)
                          .gather(-1, inp_t[..., None]).squeeze(-1).sum(1))
                lp_np = lp.detach().cpu().numpy()
                self._weight_cache[ck] = lp_np
            logp += lp_np
        temp = float(self.cfg.weight_temp)
        if temp <= 0:
            return np.ones(n)
        scaled = (logp - logp.max()) / temp
        return np.exp(np.maximum(scaled, -60.0))

    def _weighted_layouts(self, info, known_rem):
        """The pool as merged, weighted macro layouts (play_strategy.Layout).

        Candidates are re-scored under the hand model (when one is loaded),
        zero-weight ones are dropped, layouts that differ only in irrelevant
        low cards are merged with their weights summed, and the weights are
        renormalised to sum to 1. The declarer-side search additionally caps
        the layout count per node (cfg.strat_layouts); the weighted one-step
        ranking uses the whole merged set.
        """
        n = len(self.pool)
        if n == 0:
            return []
        weights = np.ones(n)
        if self.model is not None and self.cfg.weight_temp > 0:
            try:
                weights = self._model_weights(info)
            except Exception:                  # any model failure -> flat pool
                weights = np.ones(n)
        if self.cfg.play_temp > 0:
            # "a player wouldn't have made a bad play from their hand": every
            # observed hidden play cost some tricks double-dummy under each
            # hypothesis; the accumulated cost discounts the layout
            pens = np.array([c.play_pen for c in self.pool], dtype=np.float64)
            weights = weights * np.exp(-pens / float(self.cfg.play_temp))
        known_live = set()
        for s in self.known_seats:
            known_live |= set(known_rem[s])
        ign = suit_relevance(known_live, info["played_all"])
        layouts = [Layout(rem=dict(c.hands), weight=float(w))
                   for c, w in zip(self.pool, weights) if w > 0.0]
        if not layouts:
            return []
        layouts = merge_layouts(layouts, self.hidden_seats, ign)
        layouts.sort(key=lambda L: -L.weight)         # stable: pool order breaks ties
        total = sum(L.weight for L in layouts) or 1.0
        for L in layouts:
            L.weight /= total
        return layouts

    def _make_need(self, info, to_act):
        """Board-trick threshold at which the acting side reaches its contract
        goal, or None when there is no goal to weigh. Solved boards cover the
        REMAINING tricks only, so the tricks already banked come off: the
        declarer side makes with `target` total tricks, the defence beats the
        contract by holding it to target-1, i.e. taking 14-target of its own."""
        if self.target is None or to_act is None:
            return None
        if to_act in (self.declarer, self.dummy):
            return self.target - int(info["tricks_decl"])
        def_won = int(info["n_done"]) - int(info["tricks_decl"])
        return (14 - self.target) - def_won

    def _pool_rank(self, info, known_rem, legal, layouts=None, to_act=None):
        """Weighted per-card score over the merged layouts -- the ranking
        defenders (and declarer-side fallbacks) pick by.

        The score is the expected double-dummy tricks for the acting side
        plus, when cfg.rank_objective is "both", the contract target is known
        and `to_act` is given, cfg.rank_make_bonus times the weighted
        likelihood that the card's tricks reach the acting side's goal (beat
        the contract for a defender, make it for the declarer side). `to_act`
        None keeps the plain expectation the old ranking returned.
        """
        if layouts is None:
            layouts = self._weighted_layouts(info, known_rem)
        if not layouts:
            return {c: 0.0 for c in legal}
        cur = [int(c) for _, c in info["cur"]]
        positions = []
        for lay in layouts:
            remaining = dict(known_rem)
            remaining.update(lay.rem)
            positions.append((remaining, info["leader"], cur))
        solved = fastsolve.solve_positions(positions, self.trump_int)
        need = self._make_need(info, to_act)
        both = (self.cfg.rank_objective == "both" and need is not None
                and float(self.cfg.rank_make_bonus) > 0)
        total = {c: 0.0 for c in legal}
        beat = {c: 0.0 for c in legal}
        count = {c: 0.0 for c in legal}
        for lay, board in zip(layouts, solved):
            for cid, tricks in board.items():
                if cid in total:
                    total[cid] += tricks * lay.weight
                    count[cid] += lay.weight
                    if both and tricks >= need:
                        beat[cid] += lay.weight
        out = {}
        for c in legal:
            if not count[c]:
                out[c] = 0.0
                continue
            out[c] = total[c] / count[c]
            if both:
                out[c] += float(self.cfg.rank_make_bonus) * beat[c] / count[c]
        return out

    def _strategy_card(self, info, known_rem, legal, layouts, to_act):
        """The declarer-side plan's first card, or None when the search cannot
        run (defender to act, empty pool, internal failure) -- callers then
        fall back to the weighted one-step ranking."""
        if not (self.cfg.strategy and self._is_decl_side and layouts):
            return None
        if to_act not in (self.declarer, self.dummy):
            return None
        try:
            search = StrategySearch(
                self.trump, self.declarer, self.dummy,
                self.known_seats, self.hidden_seats,
                max_layouts=self.cfg.strat_layouts, beam=self.cfg.strat_beam,
                depth=self.cfg.strat_depth, def_depth=self.cfg.strat_def_depth,
                def_branch=self.cfg.strat_def_branch,
                solve_budget=self.cfg.strat_solves, time_budget=self.cfg.strat_time,
                objective=self.cfg.strat_objective,
                make_bonus=self.cfg.strat_make_bonus, target=self.target,
                fast_onestep=self.cfg.fast_onestep,
                def_smart=self.cfg.def_smart,
                endgame_skip=self.cfg.endgame_skip,
                rng=self.rng)
            fresh = [Layout(rem=dict(L.rem), weight=L.weight) for L in layouts]
            card, diag = search.solve(
                known_rem, fresh, info["leader"], info["cur"],
                info["n_done"], info["tricks_decl"], to_act,
                legal, frozenset(info["played_all"]))
            self.last_search = diag
            return int(card)
        except Exception:
            return None

    def _best_ranked(self, avg, cards):
        """The best-scoring card, chosen AT RANDOM among score ties.

        With cfg.tie_prefer_spots the tied set is first narrowed the way a
        bridge player breaks ties: a spot card doing the same job as an
        honour is played and the honour is saved for a trick it alone can
        win, and among tied honours the cheapest goes. Equals within the
        preferred class stay random. Ties used to go to the lowest card id --
        the highest rank -- which made the bot predictable and happy to waste
        an honour where an equal low card did the same job.
        """
        cards = sorted(cards)
        if not cards:
            return None
        best = max(avg[c] for c in cards)
        tied = [c for c in cards if avg[c] >= best - 1e-9]
        if self.cfg.tie_prefer_spots and len(tied) > 1:
            spots = [c for c in tied if not is_honour(c)]
            if spots:
                tied = spots
            else:
                lowest = max(c // 4 for c in tied)   # rank index: bigger = lower
                tied = [c for c in tied if c // 4 == lowest]
        if len(tied) == 1 or self.rng is None:
            return tied[0]
        return tied[int(self.rng.integers(len(tied)))]

    def _dd_safe(self, info, legal, true_hands):
        """The legal cards that cost no trick double-dummy on the REAL deal.

        `true_hands` maps every seat to its whole original 'S.H.D.C' hand (the
        drill server knows all four hands); the returned set holds the cards
        whose double-dummy trick count on the actual deal equals the best any
        legal card achieves. Empty when the real deal cannot be solved.
        """
        try:
            remaining = {s: set(str_to_cards(true_hands[s]).tolist())
                         - set(info["played_by_seat"][s]) for s in range(4)}
            results = fastsolve.solve_position(
                remaining, info["leader"], self.trump_int,
                [int(c) for _, c in info["cur"]])
        except Exception:
            return set()
        dd = {}
        for cid, tricks in results.items():
            if cid in legal:
                dd[cid] = int(tricks)
        if not dd:
            return set()
        best = max(dd.values())
        return {c for c, t in dd.items() if t == best}

    def _record_timing(self, marks: dict):
        """Fold one decision's phase times into the session counters."""
        self.last_timing = marks
        t = self.timing
        t["decisions"] += 1
        for k, v in marks.items():
            if k != "decisions":
                t[k] = t.get(k, 0.0) + v

    def choose(self, plays, to_act):
        """Return the card id `to_act` should play given the play so far."""
        t0 = time.monotonic()
        m0 = fastsolve.stats()["misses"]
        plays = self._norm_plays(plays)
        self._process_new_plays(plays)
        t1 = time.monotonic()
        info = self._analyze(plays)
        to_act = int(to_act)
        with self._lock:
            self.last_search = None
            self._ensure_pool(info)
            t2 = time.monotonic()
            known_rem, legal = self._legal_for(info, to_act)
            if not self.pool:
                self._record_timing({"process": t1 - t0, "pool": t2 - t1,
                                     "weights": 0.0, "search": 0.0,
                                     "rank": 0.0, "total": t2 - t0,
                                     "solves": fastsolve.stats()["misses"] - m0})
                return min(legal)
            layouts = self._weighted_layouts(info, known_rem)
            t3 = time.monotonic()
            card = self._strategy_card(info, known_rem, legal, layouts, to_act)
            t4 = time.monotonic()
            if card is None:
                avg = self._pool_rank(info, known_rem, legal, layouts, to_act)
                card = self._best_ranked(avg, legal)
            t5 = time.monotonic()
            self._record_timing({"process": t1 - t0, "pool": t2 - t1,
                                 "weights": t3 - t2, "search": t4 - t3,
                                 "rank": t5 - t4, "total": t5 - t0,
                                 "solves": fastsolve.stats()["misses"] - m0})
            return card

    def choose_drill(self, plays, to_act, true_hands):
        """Drill variant of `choose`, for the card problems' bots-finish.

        Options are ranked exactly like the normal bot ranks them -- with only
        the information a real game gives (the pool of auction-consistent
        hidden hands) -- but the card actually returned is the best-ranked one
        that does not give away a trick double-dummy on the real deal, so the
        drill bots defend/declare honestly yet never sabotage the problem.
        """
        t0 = time.monotonic()
        plays = self._norm_plays(plays)
        self._process_new_plays(plays)
        t1 = time.monotonic()
        info = self._analyze(plays)
        to_act = int(to_act)
        with self._lock:
            self.last_search = None
            self._ensure_pool(info)
            t2 = time.monotonic()
            known_rem, legal = self._legal_for(info, to_act)
            safe = self._dd_safe(info, legal, true_hands) or set(legal)
            if not self.pool:
                self._record_timing({"process": t1 - t0, "pool": t2 - t1,
                                     "weights": 0.0, "search": 0.0,
                                     "rank": 0.0, "total": t2 - t0})
                return min(safe)
            layouts = self._weighted_layouts(info, known_rem)
            t3 = time.monotonic()
            card = self._strategy_card(info, known_rem, safe & legal, layouts, to_act)
            t4 = time.monotonic()
            if card is None:
                avg = self._pool_rank(info, known_rem, legal, layouts, to_act)
                card = self._best_ranked(avg, safe)
            t5 = time.monotonic()
            self._record_timing({"process": t1 - t0, "pool": t2 - t1,
                                 "weights": t3 - t2, "search": t4 - t3,
                                 "rank": t5 - t4, "total": t5 - t0})
            return card


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
