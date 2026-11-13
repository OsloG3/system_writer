"""Auction-sequence patterns: the system file's definition language.

The whole system is written as pattern entries, one line each::

    PATTERN[; CLAUSES] = DESCRIPTION [@weight] [@!]

e.g.::

    1X-1Y-2Z; X<Z = reverse: 5+X 4+Z p>16 F @2.0
    1X-(1Y)-2Z; Z<Y, first = 2/1: 4+Z p>11.5 GF
    1N-2C = 4+H OR 4+S stayman ask @!
    ...-4N-(P)-5C; we_bid = RKCB response for {T}: kc=0/3 @!
    ...-X; we_xx OR dbls_by_us>=2 OR gf OR their_bid>=4S = PENALTY
    ...-X = TO

Pattern      steps joined by "-", matched against the auction from the
             dealer. A step is a call: "P", "X"/"DBL" (double), "XX"/"RDBL"
             (redouble), or level+denom ("1H", "2NT"). The denom is a literal
             suit/NT (C D H S N/NT) or one of the generic suit variables
             X Y Z W: different letters bind to different suits (and to
             different literals of the same pattern), the same letter to the
             same suit. The level may be "?" (any level: "?Y"). A step may be
             negated ("!2D": any of our bids except 2D), and "..." as a step
             absorbs any run of calls (anywhere in the pattern; a leading
             "..." makes the entry match any position). Parentheses mark an
             OPPONENT call ("(1Y)", "(P)"); bare steps are calls by the
             partnership of the described call (the candidate's side).
             Without a leading "...", the pattern is anchored at the start of
             the auction: leading passes are skipped (unless the first step
             is a pass) and opponent passes between steps are skipped
             implicitly -- own passes must be written out (or absorbed by
             "..."). The LAST step is the call described.

Clauses      after ";" (or ":"), OR-combined groups of AND atoms:
               suit order/comparison:  X<Z  Z>Y  Y!=C  T=P
                 refs are pattern variables, literals (C D H S N) and the
                 context suits T (our side's last suit), P (partner's last
                 suit), L (the opponents' last suit), O (the other major of
                 the candidate's major). < and > need both refs bound;
                 = / != treat an unbound ref as never equal.
               candidate shape:  jump=0  jump>=1  level>=3  below=4H
               state:  gf | uncontested | contested | balancing | passed |
                       first | opener | partnerbid | newsuit | webid |
                       notopening | noxyz | fourth | we_xx | they_xx |
                       dbls_by_us>=N | dbls_by_them>=N | bids_by_them=0 |
                       their_bid>=4S | our_bid>=1C
               any condition may be negated with a leading "!" (!passed).

Description  the alert text AND the training spec. " OR " (uppercase, own
             word) splits multiway alternatives: the call is consistent when
             the hand satisfies ALL checkable tokens of ANY branch; a branch
             with no checkable token is a wildcard (lenient by design).
             Tokens the trainer enforces (K&R points +- kr_tolerance):
               5+X  4-6H  6X  8+any   suit length; "any" binds the LONGEST
                                       suit for the rest of the branch
               2top            2 of the top 3 honours in the bound "any" suit
               p>16  p<10  p>=12  p15-17
               balanced
               openp           opening values: p>=open_min_kr (open_min_kr_late
                               after 2+ passes) or the rule of 20
               no4M            no 4-card major biddable at the 1 level
               noLong5M        no 5+ major longer than the bid suit
               stop            a stopper in the opponents' last suit
               kc=0/3 | kc=1/4 | kc=2- | kc=2+
                               RKCB keycards (aces + the {T} king; 2- / 2+
                               without / with the trump queen)
             Entry-level keywords: F (one-round force: partner passing it is
             penalized; an intervening opponent BID releases the force), GF
             (game force: passing below game is penalized), and for doubles
             TO / PENALTY. TO/PENALTY do NOT describe the doubler's hand --
             instead the PARTNER is penalized for passing a TO double without
             a trump stack (to_pass_min_len in their suit) or for pulling a
             PENALTY double with length there (penalty_pull_max_len).
             {X} {T} {P} {L} {O} in the text render as suit symbols.
             Everything else is free alert text.

Attributes   @N weight (default 1.0; @0 = alert-only: nothing is enforced or
             latched). A matched entry with weight > 0 REPLACES the built-in
             rule checks for that call. @! = artificial (the website shows
             the alert marker). A matched entry with an empty description
             suppresses the alert.

Resolution   entries are tried in FILE ORDER and the first match wins (put
             specific entries before general ones). Calls no entry matches
             fall back to the built-in rule checks (rules.check_call with the
             default weights) during training, and show no alert on the site.

The Go server (system_writer/patterns.go) implements the same grammar and
matching for the website's alerts; tests/test_patterns.py and the Go
TestPattern* goldens keep them in lockstep.
"""

import re
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from ..data.vocab import CALL_PASS, CALL_X, CALL_XX, FIRST_BID
from .auction import DENOM_N, MAJORS, AuctionReader, bid_id, is_game_or_above
from .rules import ConventionConfig, HandInfo, Violation, check_call

VAR_CHARS = "XYZW"
CTX_CHARS = "TPLO"                     # context suit refs (bound at match)
CONCRETE_DENOM = {"C": 0, "D": 1, "H": 2, "S": 3, "N": DENOM_N}
SYM = ("♣", "♦", "♥", "♠")             # project denom order C,D,H,S

_STEP_RE = re.compile(r"^([1-7?])(NT|N|[CDHS]|[" + VAR_CHARS + r"])$")
_LEN_RE = re.compile(r"^(\d+)(\+|-(\d+))?([CDHSN]|[" + VAR_CHARS + CTX_CHARS
                     + r"]|any)$")
_PTS_RE = re.compile(r"^p(>=|<=|>|<)?(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?$")
_KC_RE = re.compile(r"^kc=(0/3|1/4|2-|2\+)$")
_CMP_RE = re.compile(r"^([CDHSN" + VAR_CHARS + CTX_CHARS + r"])"
                     r"(>=|<=|!=|[<>]|=)"
                     r"([CDHSN" + VAR_CHARS + CTX_CHARS + r"])$")
_CNT_RE = re.compile(r"^(dbls|bids)_by_(us|them)(>=|=)(\d+)$")
_NUMATOM_RE = re.compile(r"^(jump|level)(>=|<=|=)(\d+)$")
_BIDCMP_RE = re.compile(r"^(their_bid|our_bid|below)(>=|=)"
                        r"([1-7](?:NT|N|[CDHS]))$")
_COND_WORDS = frozenset({
    "gf", "uncontested", "contested", "balancing", "passed", "first",
    "opener", "partnerbid", "newsuit", "we_bid", "notopening", "noxyz",
    "nostrong2c", "fourth", "we_xx", "they_xx",
})

_KEYWORD_TOKENS = {"F", "GF", "TO", "PENALTY", "balanced", "bal", "openp",
                   "no4M", "noLong5M", "stop", "2top"}

# enforcer-level config keys (not ConventionConfig fields)
SPEC_PARAMS = {
    "f_pass_weight": 2.0,        # passing partner's F bid
    "to_pass_weight": 1.0,       # passing partner's TO double without a stack
    "penalty_pull_weight": 1.0,  # pulling partner's PENALTY double with length
    "to_pass_min_len": 4,        # cards in their suit to pass a TO double
    "penalty_pull_max_len": 2,   # cards in their suit allowed to pull PENALTY
}


def call_id_of(name: str) -> int:
    """Call id of a literal call name (P / X / XX / DBL / RDBL / 1C..7NT)."""
    name = name.strip().upper()
    if name in ("DBL", "D"):
        return CALL_X
    if name in ("RDBL", "RD"):
        return CALL_XX
    if name.endswith("NT"):
        name = name[:-2] + "N"
    if name == "P":
        return CALL_PASS
    if name == "X":
        return CALL_X
    if name == "XX":
        return CALL_XX
    m = _STEP_RE.match(name)
    if not m or m.group(1) == "?" or m.group(2) in VAR_CHARS:
        raise ValueError(f"bad call name: {name!r}")
    return FIRST_BID + (int(m.group(1)) - 1) * 5 + CONCRETE_DENOM[m.group(2)]


# ---------------------------------------------------------------------------
# Parsed entry model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    them: bool          # parenthesized = opponent call
    kind: str           # bid | pass | dbl | rdbl | any ("...")
    level: int = 0      # 0 = any level ("?Y")
    denom: str = ""     # C D H S N or a variable letter (bid steps only)
    negated: bool = False   # "!2D": any of our bids except this one


@dataclass(frozen=True)
class Atom:
    kind: str           # cmp | cnt | num | bidcmp | below | cond
    a: str = ""
    op: str = ""
    b: str = ""
    n: int = 0
    call: int = -1
    negated: bool = False


@dataclass(frozen=True)
class Token:
    kind: str           # len | pts | kc | a keyword | text
    lo: float = 0.0
    hi: float = 0.0
    op: str = ""        # pts: > < >= <= range exact; kc: 0/3 1/4 2- 2+
    suit: str = ""      # len: var letter / literal / "any"
    text: str = ""


@dataclass
class Entry:
    raw: str                              # full source line
    pat: str                              # pattern text (violation code tail)
    steps: tuple
    anchored: bool                        # no leading "..." step
    atoms: tuple                          # OR-groups of AND-atoms
    desc: str
    branches: tuple                       # tuple[tuple[Token, ...], ...]
    flags: frozenset                      # F / GF / TO / PENALTY
    weight: float = 1.0
    artificial: bool = False

    @property
    def checkable(self) -> bool:
        """Whether any branch carries hand-checkable tokens."""
        return any(t.kind not in ("text", "F", "GF", "TO", "PENALTY")
                   for br in self.branches for t in br)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_step(tok: str) -> Step:
    them = False
    tok = tok.strip()
    if tok == "...":
        return Step(them=False, kind="any")
    if tok.startswith("(") and tok.endswith(")"):
        them, tok = True, tok[1:-1].strip()
    negated = False
    if tok.startswith("!"):
        negated, tok = True, tok[1:].strip()
    up = tok.upper()
    if up in ("P", "PASS"):
        return Step(them, "pass")
    if up in ("X", "DBL", "D"):
        return Step(them, "dbl")
    if up in ("XX", "RDBL", "RD"):
        return Step(them, "rdbl")
    m = _STEP_RE.match(up)
    if not m:
        raise ValueError(f"bad pattern step {tok!r}")
    d = m.group(2)
    if d == "NT":
        d = "N"
    level = 0 if m.group(1) == "?" else int(m.group(1))
    return Step(them, "bid", level, d, negated)


def _parse_atom(atom: str) -> Atom:
    negated = False
    if atom.startswith("!"):
        negated, atom = True, atom[1:]
    m = _CMP_RE.match(atom)
    if m:
        return Atom("cmp", m.group(1), m.group(2), m.group(3),
                    negated=negated)
    m = _CNT_RE.match(atom)
    if m:
        return Atom("cnt", m.group(1) + "_by_" + m.group(2), m.group(3), "",
                    int(m.group(4)), negated=negated)
    m = _NUMATOM_RE.match(atom)
    if m:
        return Atom("num", m.group(1), m.group(2), "", int(m.group(3)),
                    negated=negated)
    m = _BIDCMP_RE.match(atom)
    if m:
        which, op = m.group(1), m.group(2)
        cid = call_id_of(m.group(3))
        if which == "below":
            if op != "=":
                raise ValueError(f"bad clause atom {atom!r}")
            return Atom("below", call=cid, negated=negated)
        if op != ">=":
            raise ValueError(f"bad clause atom {atom!r}")
        return Atom("bidcmp", which, ">=", "", call=cid, negated=negated)
    if atom in _COND_WORDS:
        return Atom("cond", a=atom, negated=negated)
    raise ValueError(f"bad clause atom {atom!r}")


def _parse_atoms(clause: str) -> tuple:
    """OR-combined groups; atoms within a group are AND-ed."""
    groups = []
    for group in re.split(r"\s+OR\s+", clause.strip()):
        g = tuple(_parse_atom(a)
                  for a in re.split(r"[,\s]+", group.strip()) if a)
        if g:
            groups.append(g)
    if not groups:
        raise ValueError("empty clause")
    return tuple(groups)


def _parse_token(word: str) -> Token:
    word = word.strip(",:;.")
    m = _LEN_RE.match(word)
    if m:
        n = int(m.group(1))
        if m.group(2) == "+":
            lo, hi = n, 99
        elif m.group(2):
            lo, hi = n, int(m.group(3))
        else:
            lo = hi = n
        return Token("len", lo, hi, suit=m.group(4))
    m = _PTS_RE.match(word)
    if m:
        v = float(m.group(2))
        if m.group(3) is not None:
            return Token("pts", v, float(m.group(3)), op="range")
        return Token("pts", v, v, op=m.group(1) or "exact")
    m = _KC_RE.match(word)
    if m:
        return Token("kc", op=m.group(1))
    if word in ("balanced", "bal"):
        return Token("bal")
    if word in _KEYWORD_TOKENS:
        return Token(word)
    return Token("text", text=word)


def parse_entry(line: str) -> Entry:
    """Parse one pattern line, raising ValueError when malformed.

    The description separator is an "=" that is not part of a >=/<= clause
    atom; a line without one is a pattern with an empty description (which
    suppresses the alert for matching calls).
    """
    raw = line.strip()
    m = _desc_sep(raw)
    if m >= 0:
        lhs, desc = raw[:m].strip(), raw[m + 1:].strip()
    else:
        lhs, desc = raw, ""
    pat, clause = lhs, ""
    m = re.search(r"[;:]", lhs)
    if m:
        pat, clause = lhs[:m.start()].strip(), lhs[m.end():].strip()
    toks = [t for t in pat.split("-") if t.strip()]
    if not toks:
        raise ValueError(f"empty pattern: {raw!r}")
    steps = tuple(_parse_step(t) for t in toks)
    if all(s.kind == "any" for s in steps):
        raise ValueError(f"pattern is only wildcards: {raw!r}")
    anchored = steps[0].kind != "any"
    atoms = _parse_atoms(clause) if clause else ()

    weight, artificial = 1.0, False
    branches, cur = [], []
    for word in desc.split():
        if word.startswith("@"):
            attr = word[1:]
            if attr == "!":
                artificial = True
            else:
                weight = float(attr)
            continue
        if word.strip(",:;.") == "OR":
            branches.append(tuple(cur))
            cur = []
            continue
        cur.append(_parse_token(word))
    branches.append(tuple(cur))
    flags = frozenset(t.kind for br in branches for t in br
                      if t.kind in ("F", "GF", "TO", "PENALTY"))
    if "TO" in flags and "PENALTY" in flags:
        raise ValueError(f"entry is both TO and PENALTY: {raw!r}")
    return Entry(raw=raw, pat=pat, steps=steps, anchored=anchored,
                 atoms=atoms, desc=desc,
                 branches=tuple(branches), flags=flags, weight=weight,
                 artificial=artificial)


# The description separator: an "=" surrounded by whitespace (or ending the
# line). Clause atoms keep their tight comparisons (jump=0, their_bid>=4S,
# bids_by_them=0) precisely because those have no space before the "=".
_DESC_SEP_RE = re.compile(r"(?<=\s)=(?=\s|$)")


def _desc_sep(s: str) -> int:
    m = _DESC_SEP_RE.search(s)
    return m.start() if m else -1


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _call_matches(step: Step, call: int, binds: dict) -> bool:
    if step.kind == "pass":
        return call == CALL_PASS
    if step.kind == "dbl":
        return call == CALL_X
    if step.kind == "rdbl":
        return call == CALL_XX
    if call < FIRST_BID:
        return False                      # negated steps match bids only
    level = (call - FIRST_BID) // 5 + 1
    denom = (call - FIRST_BID) % 5
    if step.level not in (0, level):
        return False
    d = step.denom
    ok = False
    if d == "N":
        ok = denom == DENOM_N
    elif d in VAR_CHARS:
        if denom != DENOM_N:              # variables are suits, never NT
            prev = binds.get(d)
            if prev is None:
                if denom not in binds.values():
                    binds[d] = denom
                    ok = True
            else:
                ok = prev == denom
    else:
        ok = denom == CONCRETE_DENOM[d]
    return (not ok) if step.negated else ok


def _match_from(entry: Entry, dealer: int, seq: list, us: int, start: int,
                step_i: int = 0, pos: int = None, binds: dict = None):
    """Backtracking match of entry.steps[step_i:] onto seq[pos:], consuming
    exactly through the candidate (the last seq element)."""
    if binds is None:
        binds = {}
    if pos is None:
        pos = start
    if step_i == len(entry.steps):
        return dict(binds) if pos == len(seq) else None
    step = entry.steps[step_i]
    if step.kind == "any":
        # lazily absorb 0..n calls, then try the rest; each attempt gets a
        # fresh copy of the bindings so a failed branch cannot pollute the
        # next one
        for skip in range(0, len(seq) - pos + 1):
            r = _match_from(entry, dealer, seq, us, start,
                            step_i + 1, pos + skip, dict(binds))
            if r is not None:
                return r
        return None
    i = pos
    if not step.them and step.kind != "pass":
        while i < len(seq):               # implicit opponent-pass skip
            seat = (dealer + i) % 4
            if seat % 2 != us and seq[i] == CALL_PASS:
                i += 1
            else:
                break
    if i >= len(seq):
        return None
    seat = (dealer + i) % 4
    if step.them != (seat % 2 != us):
        return None
    if not _call_matches(step, seq[i], binds):
        return None
    return _match_from(entry, dealer, seq, us, start, step_i + 1, i + 1, binds)


def _denom_ref(ch: str, binds: dict) -> int:
    if ch in VAR_CHARS or ch in CTX_CHARS:
        return binds.get(ch, -1)
    return CONCRETE_DENOM.get(ch, -1)


@dataclass
class MatchState:
    """Auction state the clause conditions and double/force discipline read.

    Built by replaying the calls from the dealer, matching each one against
    the entries (first match wins) to collect its F/GF/TO/PENALTY flags, with
    a parallel AuctionReader for the structural context (roles, game force,
    XYZ/stayman stages, last suits).
    """
    gf: list = field(default_factory=lambda: [False, False])
    gf_file: list = field(default_factory=lambda: [False, False])
    f_pending: list = field(default_factory=lambda: [-1, -1])   # seat or -1
    dbl: list = field(default_factory=lambda: [None, None])     # (kind,denom,seat)
    dbls: list = field(default_factory=lambda: [0, 0])
    bids: list = field(default_factory=lambda: [0, 0])
    xx: list = field(default_factory=lambda: [False, False])
    high: list = field(default_factory=lambda: [0, 0])
    activity: list = field(default_factory=lambda: [0, 0])
    passes_since_bid: int = 0
    called: list = field(default_factory=lambda: [False] * 4)
    bid: list = field(default_factory=lambda: [False] * 4)
    high_denom: int = -1
    reader: object = None                # parallel AuctionReader

    def fresh(self, dealer: int) -> "MatchState":
        self.reader = AuctionReader(dealer, 0)
        return self


def fold_state(entries, dealer: int, st: MatchState, i: int, call: int) -> None:
    """Fold call #i (dealer order) into st; the entry match for the call uses
    the state BEFORE it (conditions are evaluated at decision time)."""
    call = int(call)
    seat = (dealer + i) % 4
    side = seat % 2
    if entries:
        entry, _ = match_first(entries, dealer, st.reader.calls, call, state=st)
    else:
        entry = None
    st.called[seat] = True
    live = entry is not None and entry.weight > 0
    if call == CALL_PASS:
        st.passes_since_bid += 1
        # partner's own call consumes a pending one-round force
        if st.f_pending[side] >= 0 and st.f_pending[side] != seat:
            st.f_pending[side] = -1
    elif call == CALL_X:
        st.dbls[side] += 1
        st.passes_since_bid = 0
        st.activity[side] += 1
        if st.f_pending[side] >= 0 and st.f_pending[side] != seat:
            st.f_pending[side] = -1
        if live and ("TO" in entry.flags or "PENALTY" in entry.flags):
            kind = "PENALTY" if "PENALTY" in entry.flags else "TO"
            st.dbl[side] = (kind, st.high_denom, seat)
    elif call == CALL_XX:
        st.xx[side] = True
        st.passes_since_bid = 0
        st.activity[side] += 1
        if st.f_pending[side] >= 0 and st.f_pending[side] != seat:
            st.f_pending[side] = -1
    else:
        denom = (call - FIRST_BID) % 5
        level = (call - FIRST_BID) // 5 + 1
        st.high[side] = max(st.high[side], call)
        st.high_denom = denom
        st.passes_since_bid = 0
        st.activity[side] += 1
        st.bids[side] += 1
        st.bid[seat] = True
        st.dbl = [None, None]             # any bid ends a double's pass window
        st.f_pending[1 - side] = -1       # an opponent bid releases their force
        if st.f_pending[side] >= 0 and st.f_pending[side] != seat:
            st.f_pending[side] = -1       # partner's own call consumes it
        if live:
            if "GF" in entry.flags:
                st.gf_file[side] = True
            if "F" in entry.flags:
                st.f_pending[side] = seat
        if is_game_or_above(level, denom):
            st.gf_file[side] = False      # obligation fulfilled
    st.reader.feed(call)
    st.gf = [st.gf_file[s] or st.reader.gf[s] for s in (0, 1)]


def replay_state(entries, dealer: int, calls) -> MatchState:
    """Fold `calls` (ids, dealer order) into a fresh MatchState."""
    st = MatchState().fresh(dealer)
    for i, call in enumerate(calls):
        fold_state(entries, dealer, st, i, call)
    return st


def _cond_ok(name: str, st: MatchState, info, reader, us: int,
             seat: int) -> bool:
    partner = (seat + 2) % 4
    if name == "gf":
        return st.gf[us]
    if name == "uncontested":
        return st.activity[1 - us] == 0
    if name == "contested":
        return st.activity[1 - us] > 0
    if name == "balancing":
        return st.passes_since_bid >= 2
    if name == "passed":
        return st.called[seat] and not st.bid[seat]
    if name == "first":
        return not reader.called[seat]
    if name == "opener":
        return reader.opener == seat
    if name == "partnerbid":
        return bool(reader.seat_suits[partner]) or reader.seat_nt[partner]
    if name == "newsuit":
        return (info.kind == "bid" and info.denom < DENOM_N
                and info.denom not in reader.seat_suits[seat])
    if name == "we_bid":
        return reader.side_has_bid[us]
    if name == "notopening":
        return reader.highest > 0
    if name == "noxyz":
        return reader.xyz_stage == "none"
    if name == "nostrong2c":
        return reader.strong2c_side != us
    if name == "fourth":
        return bool(info.fourth_suit)
    if name == "we_xx":
        return st.xx[us]
    if name == "they_xx":
        return st.xx[1 - us]
    return False


def _atoms_ok(groups, binds: dict, st: MatchState, info, reader,
              us: int, seat: int) -> bool:
    if not groups:
        return True
    for group in groups:                       # OR over groups
        ok = True
        for a in group:                        # AND within a group
            if a.kind == "cmp":
                x, y = _denom_ref(a.a, binds), _denom_ref(a.b, binds)
                if a.op in ("<", ">", ">=", "<="):
                    if x < 0 or y < 0:
                        r = False
                    elif a.op == "<":
                        r = x < y
                    elif a.op == ">":
                        r = x > y
                    elif a.op == ">=":
                        r = x >= y
                    else:
                        r = x <= y
                elif a.op == "=":
                    r = x >= 0 and y >= 0 and x == y
                else:                          # !=
                    r = x < 0 or y < 0 or x != y
                ok = (not r) if a.negated else r
            elif a.kind == "cnt":
                if a.a == "dbls_by_us":
                    n = st.dbls[us]
                elif a.a == "dbls_by_them":
                    n = st.dbls[1 - us]
                elif a.a == "bids_by_us":
                    n = st.bids[us]
                else:
                    n = st.bids[1 - us]
                r = n >= a.n if a.op == ">=" else n == a.n
                ok = (not r) if a.negated else r
            elif a.kind == "num":
                v = info.jump if a.a == "jump" else info.level
                if info.kind != "bid":
                    r = False
                elif a.op == ">=":
                    r = v >= a.n
                elif a.op == "<=":
                    r = v <= a.n
                else:
                    r = v == a.n
                ok = (not r) if a.negated else r
            elif a.kind == "bidcmp":
                h = st.high[1 - us] if a.a == "their_bid" else st.high[us]
                r = h >= a.call
                ok = (not r) if a.negated else r
            elif a.kind == "below":
                r = info.kind == "bid" and info.call_id < a.call
                ok = (not r) if a.negated else r
            else:                              # cond
                r = _cond_ok(a.a, st, info, reader, us, seat)
                ok = (not r) if a.negated else r
            if not ok:
                break
        if ok:
            return True
    return False


def _context_binds(binds: dict, reader, info, seat: int) -> dict:
    """Add the context suit refs T/P/L/O to a copy of the pattern binds."""
    side = seat % 2
    partner = (seat + 2) % 4
    out = dict(binds)
    t = reader._last_suit_of(side)
    if t >= 0:
        out["T"] = t
    p = reader._last_suit_of(side, partner)
    if p >= 0:
        out["P"] = p
    l = reader._last_suit_of(1 - side)
    if l >= 0:
        out["L"] = l
    if info.kind == "bid" and info.denom in MAJORS:
        out["O"] = MAJORS[0] if info.denom == MAJORS[1] else MAJORS[1]
    return out


def match_first(entries, dealer: int, calls, action: int,
                state: MatchState | None = None):
    """First entry (file order) matching `action` after `calls`:
    (entry, binds) or (None, None). binds carries the pattern variables plus
    the context refs T/P/L/O."""
    if not entries:
        return None, None
    seq = [int(c) for c in calls] + [int(action)]
    seat = (dealer + len(seq) - 1) % 4
    us = seat % 2
    if state is None:
        state = replay_state(entries, dealer, seq[:-1])
    reader = state.reader
    info = reader.classify(int(action), seat=seat)
    for e in entries:
        if e.anchored:
            if e.steps[0].kind == "pass":
                starts = (0,)
            else:
                s0 = next((i for i, c in enumerate(seq) if c != CALL_PASS), -1)
                starts = (s0,) if s0 >= 0 else ()
        else:
            starts = (0,)                     # leading "..." absorbs anything
        for s in starts:
            binds = _match_from(e, dealer, seq, us, s)
            if binds is None:
                continue
            full = _context_binds(binds, reader, info, seat)
            if _atoms_ok(e.atoms, full, state, info, reader, us, seat):
                return e, full
    return None, None


# ---------------------------------------------------------------------------
# Enforcement (training side)
# ---------------------------------------------------------------------------

@dataclass
class _Ctx:
    reader: object
    info: object
    cfg: ConventionConfig
    hero: int


def _longest_denom(hi: HandInfo) -> int:
    """The longest suit's denom (ties: S,H,D,C order -- rules.longest_denom)."""
    row = int(np.argmax(hi.lengths))          # rows are S,H,D,C
    return 3 - row


def _len_ok(tok: Token, binds: dict, hi: HandInfo, ctx: _Ctx) -> bool:
    ref = tok.suit
    denom = (_denom_ref(ref, binds) if ref in VAR_CHARS or ref in CTX_CHARS
             else CONCRETE_DENOM.get(ref, -1))
    if denom < 0 or denom == DENOM_N:
        return True                    # unbound/NT: not checkable here
    return tok.lo <= hi.length(denom) <= tok.hi


def _any_denom(hi: HandInfo) -> int:
    return _longest_denom(hi)


def _token_ok(t: Token, binds: dict, hi: HandInfo, ctx: _Ctx) -> bool:
    tol = ctx.cfg.kr_tolerance
    if t.kind == "len":
        if t.suit == "any":
            d = _any_denom(hi)
            return t.lo <= hi.length(d) <= t.hi
        return _len_ok(t, binds, hi, ctx)
    if t.kind == "pts":
        kr = hi.kr
        if t.op == "range":
            return t.lo - tol <= kr <= t.hi + tol
        if t.op in (">", ">="):
            return kr >= t.lo - tol
        if t.op in ("<", "<="):
            return kr <= t.lo + tol
        return abs(kr - t.lo) <= tol
    if t.kind == "bal":
        return hi.balanced()
    if t.kind == "openp":
        late = ctx.info.n_prior_passes >= 2
        min_kr = (ctx.cfg.open_min_kr_late if late
                  else ctx.cfg.open_min_kr)
        if hi.kr >= min_kr - tol:
            return True
        return bool(ctx.cfg.open_rule20
                    and hi.hcp + hi.two_longest() >= 20)
    if t.kind == "no4M":
        for md in MAJORS:
            if hi.length(md) >= 4 and bid_id(1, md) > ctx.reader.highest:
                return False
        return True
    if t.kind == "noLong5M":
        d = ctx.info.denom
        if not (0 <= d < DENOM_N):
            return True
        return not any(hi.length(m) >= 5 and hi.length(m) > hi.length(d)
                       for m in MAJORS)
    if t.kind == "stop":
        side = ctx.hero % 2
        their = ctx.reader.side_last_denom[1 - side]
        return True if their < 0 else hi.stopper(their)
    if t.kind == "2top":
        d = _any_denom(hi)
        return hi.top_honors(d, 3) >= 2
    if t.kind == "kc":
        trump = binds.get("T", -1)
        if not (0 <= trump < DENOM_N):
            return True                     # no trump suit: not checkable
        keys = sum(int(hi.has(d, 0)) for d in range(4)) + int(hi.has(trump, 1))
        has_q = hi.has(trump, 2)
        if t.op == "0/3":
            return keys in (0, 3)
        if t.op == "1/4":
            return keys in (1, 4)
        if t.op == "2-":
            return keys == 2 and not has_q
        return keys == 2 and has_q
    return True                              # keywords/text: not checkable


def _branch_ok(br, binds: dict, hi: HandInfo, ctx: _Ctx):
    """True/False = branch satisfied/violated; None = nothing checkable."""
    checks = [t for t in br
              if t.kind not in ("text", "F", "GF", "TO", "PENALTY")]
    if not checks:
        return None
    return all(_token_ok(t, binds, hi, ctx) for t in checks)


def hand_violations(entry: Entry, binds: dict, hi: HandInfo,
                    ctx: _Ctx) -> list:
    """Violations when the hand satisfies no OR branch of `entry`."""
    wildcard = False
    for br in entry.branches:
        r = _branch_ok(br, binds, hi, ctx)
        if r is None:
            wildcard = True
        elif r:
            return []
    if wildcard:
        return []
    return [Violation(f"pat:{entry.pat}", entry.weight,
                      f"hand ({hi.kr:.1f} K&R) matches no branch of "
                      f"{entry.raw!r}")]


class SystemEnforcer:
    """File-driven convention checks: the system file's pattern entries,
    with the built-in rule checks (rules.check_call, default weights) as the
    fallback for calls no entry matches.

    A matched entry with weight > 0 REPLACES the rule checks for that call
    (weight 0 = alert-only: rules keep enforcing); force/double discipline
    (f_pass, gf_pass beyond the structural one, to_pass, penalty_pull) is
    always file-driven and additive.
    """

    def __init__(self, spec, cfg: ConventionConfig | None = None):
        self.spec = spec
        self.cfg = cfg or ConventionConfig()
        p = spec.spec_params
        self.f_w = float(p.get("f_pass_weight", SPEC_PARAMS["f_pass_weight"]))
        self.to_w = float(p.get("to_pass_weight", SPEC_PARAMS["to_pass_weight"]))
        self.pull_w = float(p.get("penalty_pull_weight",
                                  SPEC_PARAMS["penalty_pull_weight"]))
        self.to_len = int(p.get("to_pass_min_len", SPEC_PARAMS["to_pass_min_len"]))
        self.pull_len = int(p.get("penalty_pull_max_len",
                                  SPEC_PARAMS["penalty_pull_max_len"]))
        self._states: dict = {}

    def state(self, dealer: int, calls) -> MatchState:
        """Memoized replay state; extends the parent prefix's state."""
        calls = tuple(int(c) for c in calls)
        key = (dealer, calls)
        st = self._states.get(key)
        if st is not None:
            return st
        if calls:
            st = deepcopy(self.state(dealer, calls[:-1]))
            fold_state(self.spec.entries, dealer, st, len(calls) - 1, calls[-1])
        else:
            st = MatchState().fresh(dealer)
        self._states[key] = st
        return st

    def match(self, dealer: int, calls, action: int):
        return match_first(self.spec.entries, dealer, calls, action,
                           state=self.state(dealer, calls))

    def check(self, reader: AuctionReader, hero: int, hi,
              action: int) -> list:
        calls = [int(c) for c in reader.calls]
        dealer = reader.dealer
        side = hero % 2
        info = reader.classify(action, seat=hero)
        st = self.state(dealer, calls)
        entry, binds = match_first(self.spec.entries, dealer, calls, action,
                                   state=st)
        hi = HandInfo(hi)
        ctx = _Ctx(reader, info, self.cfg, hero)
        out: list = []

        if info.kind == "pass":
            out += check_call(reader, hero, hi, action, self.cfg)
            if entry is not None and entry.weight > 0 and entry.checkable:
                out += hand_violations(entry, binds or {}, hi, ctx)
            if self.f_w > 0 and st.f_pending[side] >= 0 \
                    and st.f_pending[side] != hero:
                out.append(Violation("f_pass", self.f_w,
                                     "passed partner's forcing bid"))
            if (st.gf[side] and not reader.gf[side]
                    and not is_game_or_above(reader.high_level,
                                             reader.high_denom)):
                w = self.cfg.weight("gf_pass")
                if w > 0:
                    out.append(Violation("gf_pass", w,
                                         "passed below a file-defined GF"))
            out += self._to_pass(st, side, hero, hi)
            return out

        if info.kind == "redouble":
            return out + check_call(reader, hero, hi, action, self.cfg)

        if info.kind == "double":
            if entry is not None and entry.weight > 0:
                # TO/PENALTY: the file defines the meaning; the doubler's
                # hand is deliberately not described (and not checked)
                return out
            return out + check_call(reader, hero, hi, action, self.cfg)

        # -- bids --
        out += self._penalty_pull(st, side, hero, hi)
        if entry is not None and entry.weight > 0:
            if entry.checkable:
                out += hand_violations(entry, binds or {}, hi, ctx)
            return out
        return out + check_call(reader, hero, hi, action, self.cfg)

    # -- double discipline ---------------------------------------------------

    def _to_pass(self, st: MatchState, side: int, hero: int,
                 hi: HandInfo) -> list:
        d = st.dbl[side]
        if d is None or self.to_w <= 0:
            return []
        kind, denom, seat = d
        if kind != "TO" or seat == hero or not (0 <= denom < DENOM_N):
            return []
        if hi.length(denom) >= self.to_len:
            return []
        return [Violation("to_pass", self.to_w,
                          f"passed partner's takeout double with only "
                          f"{hi.length(denom)} in their suit")]

    def _penalty_pull(self, st: MatchState, side: int, hero: int,
                      hi: HandInfo) -> list:
        d = st.dbl[side]
        if d is None or self.pull_w <= 0:
            return []
        kind, denom, seat = d
        if kind != "PENALTY" or seat == hero or not (0 <= denom < DENOM_N):
            return []
        if hi.length(denom) <= self.pull_len:
            return []
        return [Violation("penalty_pull", self.pull_w,
                          f"pulled partner's penalty double holding "
                          f"{hi.length(denom)} in their suit")]


# ---------------------------------------------------------------------------
# Alert rendering (same mapping as the Go renderer)
# ---------------------------------------------------------------------------

_KC_TEXT = {"0/3": "0 or 3 keycards", "1/4": "1 or 4 keycards",
            "2-": "2 keycards without the trump queen",
            "2+": "2 keycards with the trump queen"}

_TOKEN_TEXT = {"F": "forcing", "GF": "game force", "TO": "takeout",
               "PENALTY": "penalty", "OR": "or", "openp": "opening values",
               "no4M": "no biddable 4-card major",
               "noLong5M": "no longer 5-card major",
               "stop": "stopper in their suit",
               "2top": "2 of the top 3 honours"}

_BRACE_RE = re.compile(r"\{([" + VAR_CHARS + CTX_CHARS + r"])\}")


def _suit_sym(ref: str, binds: dict) -> str:
    if ref == "any":
        return "any"
    d = _denom_ref(ref, binds)
    return SYM[d] if 0 <= d < DENOM_N else "any"


def _pretty_token(word: str, binds: dict) -> str:
    word = word.strip(",:;.")
    m = _LEN_RE.match(word)
    if m:
        sym = _suit_sym(m.group(4), binds)
        if m.group(2) == "+":
            return f"{m.group(1)}+ {sym}"
        if m.group(2):
            return f"{m.group(1)}-{m.group(3)} {sym}"
        return f"{m.group(1)} {sym}"
    m = _PTS_RE.match(word)
    if m:
        if m.group(3) is not None:
            return f"{m.group(2)}-{m.group(3)} pts"
        op = m.group(1)
        if op in (">", ">="):
            return f"{m.group(2)}+ pts"
        if op in ("<", "<="):
            return f"under {m.group(2)} pts"
        return f"{m.group(2)} pts"
    m = _KC_RE.match(word)
    if m:
        return _KC_TEXT[m.group(1)]
    if _BRACE_RE.search(word):
        return _BRACE_RE.sub(lambda mm: _suit_sym(mm.group(1), binds), word)
    return _TOKEN_TEXT.get(word, word)


def render_desc(entry: Entry, binds: dict) -> str:
    """The alert text of an entry: suit variables and {T}/{P}/{L}/{O} refs
    resolved to symbols, enforcement tokens prettified, @attributes dropped."""
    words = []
    for word in entry.desc.split():
        if word.startswith("@"):
            continue
        words.append(_pretty_token(word, binds or {}))
    return " ".join(words)
