"""Auction-sequence patterns: the flexible layer of the system file.

A pattern entry is one line of the system file (models/system.txt)::

    PATTERN[; CLAUSES] = DESCRIPTION [@weight] [@!]

e.g.::

    1X-1Y-2Z; X<Z = 5+X 4+Z p>16 F @2.0
    1X-(1Y)-2Z; Z<Y = 4+Z p>11.5 GF @0
    1NT-2C = 4+H OR 4+S stayman ask @!
    ...-X; we_xx OR dbls_by_us>=2 OR gf OR their_bid>=4S = PENALTY
    ...-X = TO

Pattern      sequence of steps joined by "-", matched against the auction
             from the dealer. A step is a call: "P", "X"/"DBL" (double),
             "XX"/"RDBL" (redouble), or level+denom ("1H", "2NT"). The denom
             is a literal suit/NT (C D H S N/NT) or one of the generic suit
             variables X Y Z W: different letters bind to different suits
             (and to different literals of the same pattern), the same letter
             to the same suit. Parentheses mark an OPPONENT call ("(1Y)",
             "(P)"); bare steps are calls by the partnership of the described
             call (the candidate's side). By default the pattern is anchored
             at the start of the auction (leading passes skipped unless the
             pattern itself starts with a pass) and opponent passes between
             steps are skipped implicitly -- own passes must be written out.
             A leading "..." step matches the pattern anywhere, ending at the
             candidate call. The pattern's LAST step is the call described.

Clauses      after ";" (or ":"), OR-combined groups of AND atoms:
             suit comparisons  X<Z, Z>Y   (order C<D<H<S<N; literals allowed)
             state conditions  gf | uncontested | contested | balancing |
                               passed | we_xx | they_xx |
                               dbls_by_us>=N | dbls_by_them>=N |
                               their_bid>=CALL | our_bid>=CALL

Description  the alert text AND the training spec. " OR " (uppercase, own
             word) splits multiway alternatives: the call is consistent when
             the hand satisfies ALL checkable tokens of ANY branch; a branch
             with no checkable token is a wildcard (lenient by design).
             Checkable tokens:
               5+X  4-6H  6X  8+any   suit length (bound var / literal / any)
               p>16  p<10  p>=12  p15-17   Kaplan-Rubens points (+- the
                                           file's kr_tolerance)
               balanced
             Entry-level keywords: F (one-round force: partner passing it is
             penalized; an intervening opponent BID releases the force), GF
             (game force: passing below game is penalized), and for doubles
             TO / PENALTY. TO/PENALTY do NOT describe the doubler's hand --
             instead the PARTNER is penalized for passing a TO double
             without a trump stack (to_pass_min_len in their suit) or for
             pulling a PENALTY double with length there
             (penalty_pull_max_len). Other words are free alert text.

Attributes   @N weight (0 = alert-only: nothing is enforced or latched and
             the rule-code sections keep their checks), @! = artificial (the
             website shows the alert marker). Default weight 1.0.

Resolution   entries are tried in FILE ORDER and the first match wins (put
             specific entries before general ones). A matched entry with
             weight > 0 REPLACES the rule-code checks for that call; the
             rule-code sections ([xyz_2c] & co) remain the fallback for
             calls no entry matches. A matched entry with an empty
             description suppresses the alert entirely.

The Go server (system_writer/patterns.go) implements the same grammar and
matching for the website's alerts; tests/test_patterns.py and the Go
TestPatternGoldens keep them in lockstep.
"""

import re
from copy import deepcopy
from dataclasses import dataclass, field

from ..data.vocab import CALL_PASS, CALL_X, CALL_XX, FIRST_BID
from .auction import DENOM_N, AuctionReader, is_game_or_above
from .rules import ConventionConfig, HandInfo, Violation, check_call

VAR_CHARS = "XYZW"
CONCRETE_DENOM = {"C": 0, "D": 1, "H": 2, "S": 3, "N": DENOM_N}
SYM = ("♣", "♦", "♥", "♠")             # project denom order C,D,H,S

_STEP_RE = re.compile(r"^([1-7])(NT|N|[CDHS]|[" + VAR_CHARS + r"])$")
_LEN_RE = re.compile(r"^(\d+)(\+|-(\d+))?([CDHSN]|[" + VAR_CHARS + r"]|any)$")
_PTS_RE = re.compile(r"^p(>=|<=|>|<)?(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?$")
_CMP_RE = re.compile(r"^([CDHSN" + VAR_CHARS + r"])([<>])([CDHSN" + VAR_CHARS + r"])$")
_CNT_RE = re.compile(r"^(dbls_by_us|dbls_by_them)>=(\d+)$")
_BIDCMP_RE = re.compile(r"^(their_bid|our_bid)>=([1-7](?:NT|N|[CDHS]))$")
_COND_WORDS = frozenset({"gf", "uncontested", "contested", "balancing",
                         "passed", "we_xx", "they_xx"})

# the description separator: an "=" that is not part of a >=/<= clause atom
_DESC_SEP = re.compile(r"(?<![<>])=")

# [general] keys consumed by the enforcer (not ConventionConfig fields)
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
    if not m or m.group(2) in VAR_CHARS:
        raise ValueError(f"bad call name: {name!r}")
    return FIRST_BID + (int(m.group(1)) - 1) * 5 + CONCRETE_DENOM[m.group(2)]


# ---------------------------------------------------------------------------
# Parsed entry model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    them: bool          # parenthesized = opponent call
    kind: str           # bid | pass | dbl | rdbl
    level: int = 0
    denom: str = ""     # C D H S N or a variable letter (bid steps only)


@dataclass(frozen=True)
class Atom:
    kind: str           # cmp | cnt | bidcmp | a condition word
    a: str = ""
    op: str = ""
    b: str = ""
    n: int = 0
    call: int = -1


@dataclass(frozen=True)
class Token:
    kind: str           # len | pts | bal | F | GF | TO | PENALTY | text
    lo: float = 0.0
    hi: float = 0.0
    op: str = ""        # pts: > < >= <= range exact
    suit: str = ""      # len: var letter / literal / "any"
    text: str = ""


@dataclass
class Entry:
    raw: str                              # full source line
    pat: str                              # pattern text (violation code tail)
    steps: tuple
    wildcard: bool
    atoms: tuple                          # OR-groups of AND-atoms
    concrete: frozenset                   # literal suit denoms in the pattern
    desc: str
    branches: tuple                       # tuple[tuple[Token, ...], ...]
    flags: frozenset                      # F / GF / TO / PENALTY
    weight: float = 1.0
    artificial: bool = False

    @property
    def checkable(self) -> bool:
        """Whether any branch carries hand-checkable tokens."""
        return any(t.kind in ("len", "pts", "bal")
                   for br in self.branches for t in br)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_step(tok: str) -> Step:
    them = False
    tok = tok.strip()
    if tok.startswith("(") and tok.endswith(")"):
        them, tok = True, tok[1:-1].strip()
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
    return Step(them, "bid", int(m.group(1)), d)


def _parse_atom(atom: str) -> Atom:
    m = _CMP_RE.match(atom)
    if m:
        return Atom("cmp", m.group(1), m.group(2), m.group(3))
    m = _CNT_RE.match(atom)
    if m:
        return Atom("cnt", m.group(1), ">=", "", int(m.group(2)))
    m = _BIDCMP_RE.match(atom)
    if m:
        return Atom("bidcmp", m.group(1), ">=", "",
                    call=call_id_of(m.group(2)))
    if atom in _COND_WORDS:
        return Atom(atom)
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
    if word in ("F", "GF", "TO", "PENALTY"):
        return Token(word)
    if word in ("balanced", "bal"):
        return Token("bal")
    return Token("text", text=word)


def parse_entry(line: str) -> Entry:
    """Parse one pattern line, raising ValueError when malformed.

    The description separator is an "=" that is not part of a >=/<= clause
    atom; a line without one is a pattern with an empty description (which
    suppresses the alert for matching calls).
    """
    raw = line.strip()
    m = _DESC_SEP.search(raw)
    if m:
        lhs, desc = raw[:m.start()].strip(), raw[m.end():].strip()
    else:
        lhs, desc = raw, ""
    pat, clause = lhs, ""
    m = re.search(r"[;:]", lhs)
    if m:
        pat, clause = lhs[:m.start()].strip(), lhs[m.end():].strip()
    toks = [t for t in pat.split("-") if t.strip()]
    if not toks:
        raise ValueError(f"empty pattern: {raw!r}")
    wildcard = False
    if toks[0].strip() == "...":
        wildcard = True
        toks = toks[1:]
    if not toks:
        raise ValueError(f"wildcard without steps: {raw!r}")
    steps = tuple(_parse_step(t) for t in toks)
    concrete = frozenset(
        CONCRETE_DENOM[s.denom] for s in steps
        if s.kind == "bid" and s.denom in CONCRETE_DENOM and s.denom != "N")
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
        if word == "OR":
            branches.append(tuple(cur))
            cur = []
            continue
        cur.append(_parse_token(word))
    branches.append(tuple(cur))
    flags = frozenset(t.kind for br in branches for t in br
                      if t.kind in ("F", "GF", "TO", "PENALTY"))
    if "TO" in flags and "PENALTY" in flags:
        raise ValueError(f"entry is both TO and PENALTY: {raw!r}")
    return Entry(raw=raw, pat=pat, steps=steps, wildcard=wildcard,
                 atoms=atoms, concrete=concrete, desc=desc.strip(),
                 branches=tuple(branches), flags=flags, weight=weight,
                 artificial=artificial)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _call_matches(step: Step, call: int, binds: dict,
                  concrete: frozenset) -> bool:
    if step.kind == "pass":
        return call == CALL_PASS
    if step.kind == "dbl":
        return call == CALL_X
    if step.kind == "rdbl":
        return call == CALL_XX
    if call < FIRST_BID:
        return False
    level = (call - FIRST_BID) // 5 + 1
    denom = (call - FIRST_BID) % 5
    if level != step.level:
        return False
    d = step.denom
    if d == "N":
        return denom == DENOM_N
    if d in VAR_CHARS:
        if denom == DENOM_N:
            return False                 # variables are suits, never NT
        prev = binds.get(d)
        if prev is None:
            if denom in binds.values() or denom in concrete:
                return False             # distinct from other vars/literals
            binds[d] = denom
            return True
        return prev == denom
    return denom == CONCRETE_DENOM[d]


def _match_from(entry: Entry, dealer: int, seq: list, us: int, start: int):
    """Bind `entry.steps` onto seq[start:], consuming exactly through the
    candidate (the last seq element); None when no alignment works."""
    binds = {}
    i = start
    for step in entry.steps:
        if not step.them and step.kind != "pass":
            while i < len(seq):          # implicit opponent-pass skip
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
        if not _call_matches(step, seq[i], binds, entry.concrete):
            return None
        i += 1
    return binds if i == len(seq) else None


def _denom_ref(ch: str, binds: dict) -> int:
    if ch in VAR_CHARS:
        return binds.get(ch, -1)
    return CONCRETE_DENOM.get(ch, -1)


@dataclass
class MatchState:
    """Auction state the clause conditions and double/force discipline read.

    Built by replaying the calls from the dealer, matching each one against
    the entries (first match wins) to collect its F/GF/TO/PENALTY flags, with
    a parallel AuctionReader for the structural game force.
    """
    gf: list = field(default_factory=lambda: [False, False])
    gf_file: list = field(default_factory=lambda: [False, False])
    f_pending: list = field(default_factory=lambda: [-1, -1])   # seat or -1
    dbl: list = field(default_factory=lambda: [None, None])     # (kind,denom,seat)
    dbls: list = field(default_factory=lambda: [0, 0])
    xx: list = field(default_factory=lambda: [False, False])
    high: list = field(default_factory=lambda: [0, 0])
    activity: list = field(default_factory=lambda: [0, 0])
    passes_since_bid: int = 0
    called: list = field(default_factory=lambda: [False] * 4)
    bid: list = field(default_factory=lambda: [False] * 4)
    high_denom: int = -1
    reader: object = None                # parallel AuctionReader (structural gf)

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


def _atoms_ok(groups, binds: dict, st: MatchState, us: int, seat: int) -> bool:
    if not groups:
        return True
    for group in groups:                       # OR over groups
        ok = True
        for a in group:                        # AND within a group
            if a.kind == "cmp":
                x, y = _denom_ref(a.a, binds), _denom_ref(a.b, binds)
                ok = x >= 0 and y >= 0 and (x < y if a.op == "<" else x > y)
            elif a.kind == "gf":
                ok = st.gf[us]
            elif a.kind == "uncontested":
                ok = st.activity[1 - us] == 0
            elif a.kind == "contested":
                ok = st.activity[1 - us] > 0
            elif a.kind == "balancing":
                ok = st.passes_since_bid >= 2
            elif a.kind == "passed":
                ok = st.called[seat] and not st.bid[seat]
            elif a.kind == "we_xx":
                ok = st.xx[us]
            elif a.kind == "they_xx":
                ok = st.xx[1 - us]
            elif a.kind == "cnt":
                n = st.dbls[us] if a.a == "dbls_by_us" else st.dbls[1 - us]
                ok = n >= a.n
            elif a.kind == "bidcmp":
                h = st.high[1 - us] if a.a == "their_bid" else st.high[us]
                ok = h >= a.call
            if not ok:
                break
        if ok:
            return True
    return False


def match_first(entries, dealer: int, calls, action: int,
                state: MatchState | None = None):
    """First entry (file order) matching `action` after `calls`:
    (entry, binds) or (None, None)."""
    if not entries:
        return None, None
    seq = [int(c) for c in calls] + [int(action)]
    seat = (dealer + len(seq) - 1) % 4
    us = seat % 2
    if state is None:
        state = replay_state(entries, dealer, seq[:-1])
    for e in entries:
        if e.wildcard:
            starts = range(len(seq))
        elif e.steps[0].kind == "pass":
            starts = (0,)
        else:
            s0 = next((i for i, c in enumerate(seq) if c != CALL_PASS), -1)
            starts = (s0,) if s0 >= 0 else ()
        for s in starts:
            binds = _match_from(e, dealer, seq, us, s)
            if binds is not None and _atoms_ok(e.atoms, binds, state, us, seat):
                return e, binds
    return None, None


# ---------------------------------------------------------------------------
# Enforcement (training side)
# ---------------------------------------------------------------------------

def _len_ok(tok: Token, binds: dict, hi: HandInfo) -> bool:
    ref = tok.suit
    if ref == "any":
        return any(tok.lo <= hi.length(d) <= tok.hi for d in range(4))
    denom = binds.get(ref, -1) if ref in VAR_CHARS else CONCRETE_DENOM.get(ref, -1)
    if denom < 0 or denom == DENOM_N:
        return True                    # unbound/NT: not checkable here
    return tok.lo <= hi.length(denom) <= tok.hi


def _pts_ok(tok: Token, hi: HandInfo, tol: float) -> bool:
    kr = hi.kr
    if tok.op == "range":
        return tok.lo - tol <= kr <= tok.hi + tol
    if tok.op in (">", ">="):
        return kr >= tok.lo - tol
    if tok.op in ("<", "<="):
        return kr <= tok.lo + tol
    return abs(kr - tok.lo) <= tol     # exact


def _branch_ok(br, binds: dict, hi: HandInfo, tol: float):
    """True/False = branch satisfied/violated; None = nothing checkable."""
    checks = [t for t in br if t.kind in ("len", "pts", "bal")]
    if not checks:
        return None
    for t in checks:
        if t.kind == "len" and not _len_ok(t, binds, hi):
            return False
        if t.kind == "pts" and not _pts_ok(t, hi, tol):
            return False
        if t.kind == "bal" and not hi.balanced():
            return False
    return True


def hand_violations(entry: Entry, binds: dict, hi: HandInfo,
                    tol: float) -> list:
    """Violations when the hand satisfies no OR branch of `entry`."""
    wildcard = False
    for br in entry.branches:
        r = _branch_ok(br, binds, hi, tol)
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
    """File-driven convention checks: pattern entries first, rule-code
    checks (rules.check_call) as the fallback.

    `check` returns the full Violation list for one candidate call. A matched
    entry with weight > 0 REPLACES the rule checks for that call (weight 0 =
    alert-only: rules keep enforcing); force/double discipline (f_pass,
    gf_pass beyond the structural one, to_pass, penalty_pull) is always
    file-driven and additive.
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
        out: list = []

        if info.kind == "pass":
            out += check_call(reader, hero, hi, action, self.cfg)
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
                out += hand_violations(entry, binds or {}, HandInfo(hi),
                                       self.cfg.kr_tolerance)
            return out
        return out + check_call(reader, hero, hi, action, self.cfg)

    # -- double discipline ---------------------------------------------------

    def _to_pass(self, st: MatchState, side: int, hero: int, hi) -> list:
        d = st.dbl[side]
        if d is None or self.to_w <= 0:
            return []
        kind, denom, seat = d
        if kind != "TO" or seat == hero or not (0 <= denom < DENOM_N):
            return []
        hi = HandInfo(hi)
        if hi.length(denom) >= self.to_len:
            return []
        return [Violation("to_pass", self.to_w,
                          f"passed partner's takeout double with only "
                          f"{hi.length(denom)} in their suit")]

    def _penalty_pull(self, st: MatchState, side: int, hero: int, hi) -> list:
        d = st.dbl[side]
        if d is None or self.pull_w <= 0:
            return []
        kind, denom, seat = d
        if kind != "PENALTY" or seat == hero or not (0 <= denom < DENOM_N):
            return []
        hi = HandInfo(hi)
        if hi.length(denom) <= self.pull_len:
            return []
        return [Violation("penalty_pull", self.pull_w,
                          f"pulled partner's penalty double holding "
                          f"{hi.length(denom)} in their suit")]


# ---------------------------------------------------------------------------
# Alert rendering (same mapping as the Go renderer)
# ---------------------------------------------------------------------------

def _suit_sym(ref: str, binds: dict) -> str:
    if ref == "any":
        return "any"
    d = binds.get(ref, -1) if ref in VAR_CHARS else CONCRETE_DENOM.get(ref, -1)
    return SYM[d] if 0 <= d < DENOM_N else "any"


def _pretty_token(word: str, binds: dict) -> str:
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
    return {"F": "forcing", "GF": "game force", "TO": "takeout",
            "PENALTY": "penalty", "OR": "or"}.get(word, word)


def render_desc(entry: Entry, binds: dict) -> str:
    """The alert text of an entry: variables resolved to suit symbols and
    the enforcement tokens prettified; @attributes dropped."""
    words = []
    for word in entry.desc.split():
        if word.startswith("@"):
            continue
        words.append(_pretty_token(word, binds))
    return " ".join(words)
