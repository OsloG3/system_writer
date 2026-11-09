"""Weighted-layout strategy search for the card-play bot.

The plain bot (play_bot.PlaySession) ranks each legal card by its expected
double-dummy tricks over a pool of candidate deals -- a myopic per-card vote
that re-decides every trick and never commits to a multi-trick plan. This
module adds a declarer-side *strategy* search over the same pool, closer to
how a suit-combination solver works:

  * layouts carry weights. Candidates are re-scored under the VQ hand model
    given the auction and everything the play has ruled out, and layouts that
    differ only in irrelevant low cards are merged (their weights summed), so
    the search sees a small set of weighted *macro layouts* rather than a flat
    list of raw deals;

  * a strategy is a conditional plan -- declarer's card at every information
    set the play can reach -- built by a depth- and beam-limited search over
    the public play. At a declarer decision the top-B cards (by a weighted
    double-dummy lookahead) are expanded; at a defender decision the layouts
    split into children by the card each layout's defence policy plays
    (double-dummy best near the root, a cheap heuristic deeper), because that
    card is public information declarer will get to see;

  * a plan is valued by aggregating declarer's total tricks over ALL layouts
    it is evaluated on: the weighted expectation ("ev"), the weighted share
    of layouts on which the contract target is reached ("make" -- the
    "works on the highest number of layouts" criterion), or both. A line that
    succeeds on more layouts therefore wins even when its single-layout trick
    average does not.

Deliberate approximations, to keep solving time manageable:

  * low cards below every still-live card of the visible hands can never beat
    a visible card, and among themselves they only decide *which defender*
    wins a trick -- never how many tricks the declaring side takes, only
    (rarely, via the on-lead tempo) second-order entries. Their identity is
    ignored: a macro layout keeps the relevant ranks plus the spot COUNT per
    hidden hand, and layouts agreeing on that are merged and evaluated through
    one representative;
  * the defence is modelled double-dummy-pessimistic for the first
    `def_depth` tricks and by a cheap heuristic beyond, and only the top
    `def_branch` defender branches per node are expanded recursively;
  * a solve-count and wall-clock budget cap everything; when they run out,
    nodes fall back to a naive estimate and the root falls back to the
    weighted one-step ranking, so a slow position degrades to exactly what
    the old bot did instead of timing out.

Only the declarer side uses the search (it controls both visible hands, which
is what makes a single plan meaningful); defenders keep the weighted
one-step ranking in play_bot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import endplay.types as ept
from endplay.dds.solve import SolveMode, solve_all_boards
from endplay.utils.play import trick_winner

from .data.hands import cards_to_str

# Card ids follow env/deals.py (card = rank*4 + suit, rank A=0 .. 2=12,
# suit S,H,D,C = 0..3). The tiny helpers below mirror play_bot's on purpose:
# play_bot imports this module, so importing them back would be circular.
RANKS = "AKQJT98765432"
SUITS = "SHDC"
_RANK_IDX = {c: i for i, c in enumerate(RANKS)}


def ep_card(cid: int):
    return ept.Card(suit=ept.Denom(cid % 4), rank=ept.Rank.find(RANKS[cid // 4]))


def ep_to_card(card) -> int:
    return _RANK_IDX[card.rank.abbr] * 4 + int(card.suit)


def card_above(cid: int) -> int | None:
    r = cid // 4
    return None if r == 0 else (r - 1) * 4 + cid % 4


def is_honour(cid: int) -> bool:
    """T,J,Q,K,A -- the ranks 'higher than the nine' the lead rule uses."""
    return cid // 4 <= 4


# ---- low-card canonicalization ---------------------------------------------

def suit_relevance(known_live, played) -> dict[int, set]:
    """Per suit, the hidden rank indices treated as interchangeable 'spots'.

    A live hidden rank is a spot when it ranks below EVERY card the visible
    (known) hands still hold in that suit: no visible card can ever be beaten
    by it, and among spots only the per-hand COUNT survives into a macro
    layout. `known_live`/`played` are iterables of card ids; the result maps
    suit -> set of ignorable rank indices (0=A .. 12=2).
    """
    known = [set() for _ in range(4)]
    gone = [set() for _ in range(4)]
    for cid in known_live:
        known[cid % 4].add(cid // 4)
    for cid in played:
        gone[cid % 4].add(cid // 4)
    out = {}
    for s in range(4):
        lowest_known = max(known[s]) if known[s] else -1
        out[s] = {r for r in range(13)
                  if r > lowest_known and r not in known[s] and r not in gone[s]}
    return out


@dataclass
class Layout:
    """One weighted hypothesis for the two hidden hands.

    `rem` maps each hidden seat to the card ids it still holds; `cur` is the
    search-internal trick-so-far as ACTUAL cards [(seat, cid), ...] (members
    of one macro class can differ in which spot they played).
    """
    rem: dict
    weight: float = 1.0
    cur: list = field(default_factory=list)


def macro_key(layout: Layout, hidden_seats, ignorable) -> tuple:
    """Canonical form: per hidden seat per suit, the relevant ranks held
    (sorted) plus the number of interchangeable spots."""
    parts = []
    for seat in hidden_seats:
        suits = []
        for s in range(4):
            rel, spots = [], 0
            for cid in layout.rem[seat]:
                if cid % 4 != s:
                    continue
                r = cid // 4
                if r in ignorable[s]:
                    spots += 1
                else:
                    rel.append(r)
            suits.append((tuple(sorted(rel)), spots))
        parts.append(tuple(suits))
    return tuple(parts)


def merge_layouts(layouts, hidden_seats, ignorable) -> list:
    """Merge layouts that agree on `macro_key`, summing their weights.

    The first member of each class is kept as the representative its tricks
    are simulated on (the documented low-card approximation).
    """
    groups: dict = {}
    for lay in layouts:
        k = macro_key(lay, hidden_seats, ignorable)
        g = groups.get(k)
        if g is None:
            groups[k] = Layout(rem={s: set(lay.rem[s]) for s in hidden_seats},
                               weight=lay.weight, cur=list(lay.cur))
        else:
            g.weight += lay.weight
    return list(groups.values())


# ---- search -----------------------------------------------------------------

@dataclass
class _Node:
    known_rem: dict          # visible seat -> cards left (before the current trick)
    layouts: list            # merged, weighted Layouts (each carries its own .cur)
    leader: int              # leader of the current trick
    tricks_done: int         # completed tricks
    tricks_decl: int         # completed tricks won by the declaring side
    to_act: int
    played: frozenset        # every card gone before this node (union over layouts)
    relevance: dict          # suit -> ignorable ranks at this node


class StrategySearch:
    """Beam search for the declarer-side card that anchors the best plan.

    Values are absolute weighted sums over the node's layouts (declarer-side
    total tricks, plus a P(make) bonus when configured), so expectation nodes
    (defender decisions) just add children up and choice nodes (declarer
    decisions) take the max.
    """

    def __init__(self, trump, declarer, dummy, known_seats, hidden_seats,
                 max_layouts: int = 12, beam: int = 3, depth: int = 4,
                 def_depth: int = 2, def_branch: int = 4,
                 solve_budget: int = 300, time_budget: float = 3.0,
                 objective: str = "both", make_bonus: float = 1.0,
                 target: int | None = None):
        self.trump = trump
        self.declarer = int(declarer)
        self.dummy = int(dummy)
        self.decl_seats = (self.declarer, self.dummy)
        self.known_seats = list(known_seats)
        self.hidden_seats = list(hidden_seats)
        self.max_layouts = max(1, int(max_layouts))
        self.beam = max(1, int(beam))
        self.depth = max(0, int(depth))
        self.def_depth = max(0, int(def_depth))
        self.def_branch = max(1, int(def_branch))
        self.solve_budget = int(solve_budget)
        self.objective = objective if objective in ("ev", "make", "both") else "both"
        self.make_bonus = float(make_bonus)
        self.target = int(target) if target is not None else None
        self.deadline = time.monotonic() + float(time_budget)
        self.solves = 0
        self.nodes = 0
        self.exhausted = False
        self.root_done = 0
        self._t0 = time.monotonic()

    # ---- public entry -------------------------------------------------------

    def solve(self, known_rem, layouts, leader, cur, tricks_done, tricks_decl,
              to_act, legal, played):
        """Pick the best card for `to_act` (a visible seat) and report stats.

        `layouts` are the (already macro-merged, weight-normalised) Layout
        objects from the session; `cur` is the actual trick-so-far
        [(seat, cid), ...]; `played` every card gone before the root.
        Returns (card, info-dict). Raises ValueError when it cannot act.
        """
        if to_act not in self.decl_seats:
            raise ValueError("the strategy search only plans for the declaring side")
        legal = sorted(legal)
        if not legal:
            raise ValueError("no legal card")
        if not layouts:
            raise ValueError("no layouts to evaluate")
        self.root_done = int(tricks_done)
        for lay in layouts:
            lay.cur = list(cur)
        root = self._node(known_rem, layouts, int(leader), int(tricks_done),
                          int(tricks_decl), int(to_act), frozenset(played))
        if len(legal) == 1:
            return legal[0], self._stats(len(root.layouts), None)

        onestep = self._onestep_decl_cards(root, legal)
        order = sorted(legal, key=lambda c: (-onestep[c], c))
        best_card, best_val = order[0], None
        for c in order[:self.beam]:
            val = 0.0
            for _w, child in self._advance_known(root, c):
                val += self._naive(child) if self._over() else self._search(child)
            if best_val is None or val > best_val + 1e-9:
                best_val, best_card = val, c
            if self._over():
                break
        value = float(best_val) if best_val is not None else float(onestep[best_card])
        return best_card, self._stats(len(root.layouts), value)

    def _stats(self, layouts: int, value):
        return {"layouts": layouts, "solves": self.solves, "nodes": self.nodes,
                "time": time.monotonic() - self._t0, "exhausted": self.exhausted,
                "value": value}

    # ---- budgets ------------------------------------------------------------

    def _over(self) -> bool:
        if self.solves >= self.solve_budget or time.monotonic() >= self.deadline:
            self.exhausted = True
            return True
        return False

    # ---- node construction ---------------------------------------------------

    def _node(self, known_rem, layouts, leader, tricks_done, tricks_decl,
              to_act, played) -> _Node:
        known_live = set()
        for s in self.known_seats:
            known_live |= set(known_rem.get(s, ()))
        ign = suit_relevance(known_live, played)
        if len(layouts) > 1:
            layouts = merge_layouts(layouts, self.hidden_seats, ign)
        if len(layouts) > self.max_layouts:
            layouts.sort(key=lambda L: -L.weight)      # stable: pool order breaks ties
            layouts = layouts[:self.max_layouts]
        return _Node(known_rem, layouts, leader, tricks_done, tricks_decl,
                     to_act, played, ign)

    def _complete(self, node: _Node, members, winner) -> _Node:
        """Child node after `members` finished the current trick, won by `winner`."""
        played_known: dict = {}
        allp = set(node.played)
        for m in members:
            for s, cid in m.cur:
                allp.add(cid)
                if s in node.known_rem:
                    played_known.setdefault(s, set()).add(cid)
            m.cur = []
        known_rem = {s: node.known_rem[s] - played_known.get(s, set())
                     for s in node.known_rem}
        return self._node(known_rem, members, winner, node.tricks_done + 1,
                          node.tricks_decl + (1 if winner in self.decl_seats else 0),
                          winner, frozenset(allp))

    def _advance_known(self, node: _Node, cid: int):
        """Apply a visible-seat card -> [(weight, child)] (usually one child)."""
        seat = node.to_act
        nxt = [Layout(rem=L.rem, weight=L.weight, cur=L.cur + [(seat, cid)])
               for L in node.layouts]
        if len(nxt[0].cur) == 4:
            groups: dict = {}
            for nl in nxt:
                w = int(trick_winner([ep_card(c) for _, c in nl.cur],
                                     ept.Player(node.leader), self.trump))
                groups.setdefault(w, []).append(nl)
            return [(sum(m.weight for m in ms), self._complete(node, ms, w))
                    for w, ms in groups.items()]
        child = self._node(node.known_rem, nxt, node.leader, node.tricks_done,
                           node.tricks_decl, (seat + 1) % 4, node.played)
        return [(sum(L.weight for L in child.layouts), child)]

    def _def_children(self, node: _Node):
        """Split a defender decision by the (macro class of the) card played."""
        seat = node.to_act
        picks = self._defence_picks(node)
        if picks is None:
            return []
        ign = node.relevance
        groups: dict = {}
        for L, cid in picks:
            nl = Layout(rem={s: set(L.rem[s]) for s in self.hidden_seats},
                        weight=L.weight, cur=L.cur + [(seat, cid)])
            nl.rem[seat].discard(cid)
            if len(nl.cur) == 4:
                w = int(trick_winner([ep_card(c) for _, c in nl.cur],
                                     ept.Player(node.leader), self.trump))
                key = (self._card_class(cid, ign), w)
            else:
                key = (self._card_class(cid, ign), -1)
            groups.setdefault(key, []).append(nl)
        out = []
        for (_cls, w), members in groups.items():
            if w >= 0:
                child = self._complete(node, members, w)
            else:
                child = self._node(node.known_rem, members, node.leader,
                                   node.tricks_done, node.tricks_decl,
                                   (seat + 1) % 4, node.played)
            out.append((sum(L.weight for L in child.layouts), child))
        out.sort(key=lambda p: -p[0])
        return out

    @staticmethod
    def _card_class(cid: int, ign) -> tuple:
        s, r = cid % 4, cid // 4
        return (s, -1 if r in ign[s] else r)

    # ---- legality / defence policy --------------------------------------------

    def _legal_known(self, node: _Node) -> set:
        """The acting visible seat's legal cards (identical across layouts)."""
        seat = node.to_act
        rem = set(node.known_rem[seat])
        cur = node.layouts[0].cur
        for s, cid in cur:
            if s == seat:
                rem.discard(cid)
        if not cur:
            return rem
        lead = cur[0][1] % 4
        follow = {c for c in rem if c % 4 == lead}
        return follow if follow else rem

    def _def_legal(self, node: _Node, lay: Layout) -> set:
        """A defender's legal cards in one layout, honour-lead rule included."""
        rem = lay.rem[node.to_act]
        if not lay.cur:
            allowed = set()
            for c in rem:
                if not is_honour(c):
                    allowed.add(c)
                    continue
                above = card_above(c)
                if above is None or above not in rem:
                    allowed.add(c)                      # top of a touching run
            return allowed or set(rem)
        lead = lay.cur[0][1] % 4
        follow = {c for c in rem if c % 4 == lead}
        return follow if follow else set(rem)

    def _defence_picks(self, node: _Node):
        """Per layout, the card its defender plays: double-dummy best for the
        first `def_depth` tricks, a cheap heuristic beyond."""
        solved = None
        if (node.tricks_done - self.root_done) < self.def_depth:
            deals = [self._deal_for(node, L) for L in node.layouts]
            solved = self._solve(deals)
        picks = []
        for i, L in enumerate(node.layouts):
            allowed = self._def_legal(node, L)
            if not allowed:
                return None
            c = None
            if solved is not None:
                best = None
                for x in sorted(allowed):
                    t = solved[i].get(x)
                    if t is None:
                        continue
                    if best is None or t > best:        # ties -> top of a run
                        best, c = t, x
            if c is None:
                c = self._cheap_pick(L, allowed)
            picks.append((L, c))
        return picks

    @staticmethod
    def _cheap_pick(lay: Layout, allowed: set) -> int:
        """No-solve defence: cheapest card following; leading, the cheapest
        card of the longest allowed suit."""
        if lay.cur:
            return max(allowed)                         # max id = lowest rank
        by_suit: dict = {}
        for c in allowed:
            by_suit.setdefault(c % 4, []).append(c)
        suit = max(sorted(by_suit), key=lambda s: len(by_suit[s]))
        return max(by_suit[suit])

    # ---- deal building / solving ----------------------------------------------

    def _deal_for(self, node: _Node, lay: Layout, extra=None):
        """An endplay Deal at this node's position for one layout (`extra` is
        an additional (seat, cid) play evaluated on top). Hands exclude the
        cards already played to the current trick, which go to currentTrick."""
        cur = lay.cur if extra is None else lay.cur + [extra]
        kplayed: dict = {}
        for s, cid in cur:
            if s in node.known_rem:
                kplayed.setdefault(s, set()).add(cid)
        parts = []
        for s in range(4):
            if s in node.known_rem:
                cs = node.known_rem[s].difference(kplayed.get(s, ()))
            else:
                cs = lay.rem[s]
            parts.append(cards_to_str(sorted(cs)) if cs else "-")
        deal = ept.Deal.from_pbn("N:" + " ".join(parts))
        deal.trump = self.trump
        deal.first = ept.Player(node.leader)
        for _, cid in cur:
            deal.play(ep_card(cid), from_hand=False)
        return deal

    def _solve(self, deals):
        """Batched solve -> [{card id: tricks for the side to move}]; None when
        the budget is spent (the soft cap lets one batch overshoot)."""
        if not deals or self._over():
            return None
        self.solves += len(deals)
        solved = solve_all_boards(deals, SolveMode.Default)
        return [{ep_to_card(c): int(t) for c, t in board} for board in solved]

    def _after_meta(self, node: _Node, lay: Layout, extra):
        """(tricks_done, tricks_decl, winner-side-is-decl) after `extra`."""
        cur = lay.cur + [extra]
        if len(cur) == 4:
            w = int(trick_winner([ep_card(c) for _, c in cur],
                                 ept.Player(node.leader), self.trump))
            return (node.tricks_done + 1,
                    node.tricks_decl + (1 if w in self.decl_seats else 0),
                    w in self.decl_seats)
        # a visible card was played, so (unless the trick ended) a defender is
        # to act and the solved tricks are the DEFENCE's share
        return node.tricks_done, node.tricks_decl, False

    # ---- objectives ------------------------------------------------------------

    def _objective(self, totals, weights) -> float:
        """Aggregate per-layout declarer trick totals into one node value."""
        ev = sum(w * t for w, t in zip(weights, totals))
        if self.objective == "ev" or self.target is None:
            return ev
        make = sum(w for w, t in zip(weights, totals) if t >= self.target)
        if self.objective == "make":
            return make
        return ev + self.make_bonus * make

    def _naive(self, node: _Node) -> float:
        """Budget-exhausted leaf: no solves, assume half the rest."""
        wsum = sum(L.weight for L in node.layouts)
        rem = 13 - node.tricks_done
        ev = wsum * (node.tricks_decl + 0.5 * rem)
        if self.objective == "ev" or self.target is None:
            return ev
        make = 0.5 * wsum
        return make if self.objective == "make" else ev + self.make_bonus * make

    def _onestep_decl_cards(self, node: _Node, cards) -> dict:
        """Objective per candidate visible-seat card, assuming double-dummy
        (clairvoyant) play afterwards -- the beam-ranking heuristic and the
        depth-limit leaf value."""
        deals, meta = [], []
        for c in cards:
            for i, L in enumerate(node.layouts):
                deals.append(self._deal_for(node, L, extra=(node.to_act, c)))
                meta.append((c, i))
        solved = self._solve(deals)
        if solved is None:
            return {c: self._naive(node) for c in cards}
        totals: dict = {c: [] for c in cards}
        weights: dict = {c: [] for c in cards}
        for (c, i), board in zip(meta, solved):
            L = node.layouts[i]
            if not board:
                continue
            # the solved entries belong to whoever acts AFTER `extra` (the
            # next defender, or -- when `extra` completed the trick -- the
            # trick winner leading afresh), so the position's double-dummy
            # value is the side-to-move's best entry, not a lookup of `c`
            t = max(board.values())
            done, decl, side = self._after_meta(node, L, (node.to_act, c))
            rem = 13 - done
            totals[c].append(decl + (t if side else rem - t))
            weights[c].append(L.weight)
        out = {}
        for c in cards:
            out[c] = (self._objective(totals[c], weights[c]) if totals[c]
                      else self._naive(node))
        return out

    def _onestep(self, node: _Node) -> float:
        if node.to_act in self.decl_seats:
            cards = sorted(self._legal_known(node))
            scores = self._onestep_decl_cards(node, cards)
            return max(scores.values()) if scores else self._naive(node)
        deals = [self._deal_for(node, L) for L in node.layouts]
        solved = self._solve(deals)
        if solved is None:
            return self._naive(node)
        rem = 13 - node.tricks_done
        totals, weights = [], []
        for i, L in enumerate(node.layouts):
            best = None
            for c in self._def_legal(node, L):
                t = solved[i].get(c)
                if t is None:
                    continue
                if best is None or t > best:            # defence plays DD-best
                    best = t
            if best is None:
                return self._naive(node)
            totals.append(node.tricks_decl + rem - best)
            weights.append(L.weight)
        return self._objective(totals, weights)

    # ---- recursion ---------------------------------------------------------------

    def _search(self, node: _Node) -> float:
        self.nodes += 1
        if not node.layouts:
            return 0.0
        if node.tricks_done == 13:
            # the play is over: every layout shares the public trick count
            return self._objective([node.tricks_decl] * len(node.layouts),
                                   [L.weight for L in node.layouts])
        if self._over():
            return self._naive(node)
        if node.tricks_done - self.root_done >= self.depth:
            return self._onestep(node)
        if node.to_act in self.decl_seats:
            cards = sorted(self._legal_known(node))
            if not cards:
                return self._naive(node)
            scores = self._onestep_decl_cards(node, cards)
            order = sorted(cards, key=lambda c: (-scores[c], c))
            best = None
            for c in order[:self.beam]:
                val = 0.0
                for _w, child in self._advance_known(node, c):
                    val += self._naive(child) if self._over() else self._search(child)
                if best is None or val > best:
                    best = val
                if self._over():
                    break
            return best if best is not None else self._naive(node)
        val = 0.0
        children = self._def_children(node)
        if not children:
            return self._naive(node)
        for i, (_w, child) in enumerate(children):
            if i >= self.def_branch or self._over():
                val += self._naive(child)
            else:
                val += self._search(child)
        return val
