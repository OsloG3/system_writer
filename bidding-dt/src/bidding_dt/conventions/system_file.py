"""The bot's system file: conventions as one text file (models/system.txt).

The file is the single source of truth for the system the bots follow:

  * training interprets it at startup -- `SystemSpec.config` folds every
    section's `weight` into the convention-penalty rule weights and every
    numeric key into the matching ConventionConfig threshold (rl/train_ppo.py
    builds its conv_cfg from it, so editing a range in the file re-tunes what
    the RL loop enforces), and the sequence-pattern entries (patterns.py)
    are enforced per call by patterns.SystemEnforcer (rl/rollout.py);
  * the website generates its bid alerts from the same file --
    `SystemSpec.alert_for_call` resolves a call in its auction first against
    the pattern entries, then via `section_for` against the rule-code
    sections, and renders the match's alert text. The Go server
    (system_writer/system_file.go + patterns.go) implements the same
    resolution over the same file; keep the two in lockstep.

Format (INI-like, # and ; comments). Rule-code sections::

    [section]            rule code (rules.DEFAULT_WEIGHTS) or one of the
                         alert-only kinds (general, rkcb_ask, resp_2c_waiting,
                         balancing_takeout, penalty_double)
    alert = text         website alert text ({call} {suit} {other} {trump}
                         {partner_suit} {shows} {level} placeholders)
    artificial = yes     the call carries the alert (spade) marker on the site
    weight = N           convention-penalty weight (rule-code sections only)
    steps = a | b | c | d   RKCB step texts ({shows})
    <param> = value      ConventionConfig threshold: number, "lo-hi" range
                         or yes/no boolean ([general] also takes the
                         enforcer's SPEC_PARAMS keys, e.g. to_pass_min_len)

Any other `lhs = rhs` line is a sequence-pattern entry -- the flexible layer
that describes bids by the auction sequence itself (generic suits X/Y/Z/W,
opponent calls in parentheses, rank/state clauses, OR branches, F/GF, TO/
PENALTY doubles). See patterns.py for the full grammar and semantics.
"""

import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from .auction import DENOM_N, MAJORS, CallInfo, is_game_or_above
from .patterns import (_DESC_SEP, SPEC_PARAMS, MatchState, fold_state,
                       match_first, parse_entry, render_desc)
from .rules import DEFAULT_WEIGHTS, ConventionConfig

SYM = ("♣", "♦", "♥", "♠")          # project denom order C,D,H,S

# sections that carry alert text but no penalty rule of their own
ALERT_ONLY = frozenset({"general", "rkcb_ask", "resp_2c_waiting",
                        "balancing_takeout", "penalty_double"})

DEFAULT_CANDIDATES = ("models/system.txt", "../models/system.txt")

_RANGE = re.compile(r"^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$")
_INT = re.compile(r"^-?\d+$")

_PARAM_FIELDS = None                 # ConventionConfig field names, lazily


def _param_fields() -> frozenset:
    global _PARAM_FIELDS
    if _PARAM_FIELDS is None:
        import dataclasses

        _PARAM_FIELDS = frozenset(
            f.name for f in dataclasses.fields(ConventionConfig)
            if f.name != "weights")
    return _PARAM_FIELDS


# key = value lines the parser knows (everything else is a pattern entry)
_KV_KEYS = None

# bare identifiers are key = value lines -- unless they are a call token,
# which makes them a (single-step) pattern entry like "P = ..."
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CALL_STEP_TOKENS = {"P", "PASS", "X", "DBL", "D", "XX", "RDBL", "RD"}


def _is_kv_line(lhs: str) -> bool:
    return (bool(_IDENT.fullmatch(lhs))
            and lhs.upper() not in _CALL_STEP_TOKENS)


def _kv_keys() -> frozenset:
    global _KV_KEYS
    if _KV_KEYS is None:
        _KV_KEYS = (frozenset({"alert", "artificial", "weight", "steps"})
                    | _param_fields() | frozenset(SPEC_PARAMS))
    return _KV_KEYS


def _number(s: str):
    return float(s) if "." in s else int(s)


def parse_value(s: str):
    """Parse a right-hand side: bool, "lo-hi" range tuple, int or float."""
    s = s.strip()
    if s in ("yes", "true"):
        return True
    if s in ("no", "false"):
        return False
    m = _RANGE.match(s)
    if m:
        return (_number(m.group(1)), _number(m.group(2)))
    if _INT.match(s):
        return int(s)
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"not a number, range or boolean: {s!r}") from None


@dataclass
class Section:
    name: str
    alert: str | None = None
    artificial: bool = False
    weight: float | None = None
    steps: list = field(default_factory=list)
    params: dict = field(default_factory=dict)


@dataclass
class SystemSpec:
    """Parsed system file: rule-code sections keyed by name, plus the
    sequence-pattern entries (patterns.py) in file order."""
    path: str | None = None
    sections: dict = field(default_factory=dict)   # name -> Section
    entries: list = field(default_factory=list)    # pattern entries, file order
    _states: dict = field(default_factory=dict)    # replay-state memo

    # -- config for training -------------------------------------------------

    @property
    def spec_params(self) -> dict:
        """Enforcer-level params (SPEC_PARAMS keys, e.g. to_pass_min_len),
        conventionally set in [general]; later sections win."""
        out = dict(SPEC_PARAMS)
        for sec in self.sections.values():
            for k, v in sec.params.items():
                if k in SPEC_PARAMS:
                    out[k] = v
        return out

    def config(self, rule_overrides: dict | None = None,
               param_overrides: dict | None = None) -> ConventionConfig:
        """ConventionConfig with the file's weights and thresholds applied.

        YAML/CLI overrides (rl.conv_rules / rl.conv_params) layer on top of
        the file, exactly as they used to layer on the code defaults. The
        enforcer-level SPEC_PARAMS keys belong to the pattern layer
        (spec_params) and are filtered out here.
        """
        weights, params = {}, {}
        for sec in self.sections.values():
            for k, v in sec.params.items():
                if k not in SPEC_PARAMS:
                    params[k] = v
            if sec.weight is not None:
                weights[sec.name] = sec.weight
        weights.update(rule_overrides or {})
        params.update(param_overrides or {})
        return ConventionConfig.from_overrides(weights, params)

    # -- alerts for the website ----------------------------------------------

    def section_for(self, info: CallInfo) -> str | None:
        return section_for(info)

    def alert_for(self, info: CallInfo) -> str | None:
        """Rendered alert text for a classified call, or None when the system
        file defines no meaning for it (rule-code sections only)."""
        name = section_for(info)
        if name is None:
            return None
        sec = self.sections.get(name)
        if sec is None or sec.alert is None:
            return None
        return render(sec.alert, info, sec.steps)

    def replay(self, dealer: int, calls):
        """Memoized pattern-layer state of an auction prefix; extends the
        parent prefix's state (one fold per new call)."""
        calls = tuple(int(c) for c in calls)
        key = (dealer, calls)
        st = self._states.get(key)
        if st is not None:
            return st
        if calls:
            st = deepcopy(self.replay(dealer, calls[:-1]))
            fold_state(self.entries, dealer, st, len(calls) - 1, calls[-1])
        else:
            st = MatchState().fresh(dealer)
        self._states[key] = st
        return st

    def match_call(self, dealer: int, calls, action: int):
        """(entry, binds) of the first pattern entry matching `action` after
        `calls`, or (None, None)."""
        if not self.entries:
            return None, None
        return match_first(self.entries, dealer, calls, action,
                           state=self.replay(dealer, calls))

    def alert_for_call(self, reader, action: int) -> str | None:
        """Alert text for `action` in `reader`'s auction: the pattern layer
        first (an entry whose rendered description is empty suppresses the
        alert), then the rule-code sections."""
        if self.entries:
            entry, binds = self.match_call(reader.dealer, list(reader.calls),
                                           int(action))
            if entry is not None:
                return render_desc(entry, binds) or None
        info = reader.classify(int(action))
        return self.alert_for(info)


def parse_system_file(path: str | Path) -> SystemSpec:
    """Parse a system file, raising ValueError on anything malformed."""
    path = Path(path)
    spec = SystemSpec(path=str(path))
    cur: Section | None = None
    for lineno, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        where = f"{path}:{lineno}"
        if line.startswith("["):
            if not line.endswith("]"):
                raise ValueError(f"{where}: unterminated section header")
            name = line[1:-1].strip()
            if name not in DEFAULT_WEIGHTS and name not in ALERT_ONLY:
                raise ValueError(f"{where}: unknown section [{name}]")
            if name in spec.sections:
                raise ValueError(f"{where}: duplicate section [{name}]")
            cur = Section(name)
            spec.sections[name] = cur
            continue
        m = _DESC_SEP.search(line)
        lhs = line[:m.start()].strip() if m else line.strip()
        if _is_kv_line(lhs):
            if m is None:
                raise ValueError(f"{where}: expected key = value")
            if lhs not in _kv_keys():
                raise ValueError(f"{where}: unknown parameter {lhs!r}")
            val = line[m.end():].strip()
            if cur is None:
                raise ValueError(f"{where}: key outside any section")
            if lhs == "alert":
                cur.alert = val
            elif lhs == "artificial":
                cur.artificial = parse_value(val) is True
            elif lhs == "weight":
                if cur.name not in DEFAULT_WEIGHTS:
                    raise ValueError(
                        f"{where}: [{cur.name}] is alert-only and has no "
                        f"weight")
                cur.weight = float(val)
            elif lhs == "steps":
                cur.steps = [s.strip() for s in val.split("|")]
            else:
                cur.params[lhs] = parse_value(val)
            continue
        # not a key = value line: a sequence-pattern entry (patterns.py)
        try:
            spec.entries.append(parse_entry(line))
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
    if not spec.sections and not spec.entries:
        raise ValueError(f"{path}: no sections or pattern entries found")
    return spec


def find_system_file(explicit: str | None = None) -> str | None:
    """Resolve the system file: explicit path, then BIDDING_SYSTEM_FILE, then
    the well-known candidates (models/system.txt next to the website, and
    ../models/system.txt from a nested bidding-dt checkout). Returns None when
    nothing is found; a missing *explicit* path raises FileNotFoundError."""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"system file not found: {explicit}")
        return explicit
    env = os.environ.get("BIDDING_SYSTEM_FILE", "").strip()
    candidates = (env,) if env else DEFAULT_CANDIDATES
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None


# ---------------------------------------------------------------------------
# Call -> section resolution (mirrors the check order of rules._check_bid and
# the Go server's system_file.go sectionFor; keep them in lockstep)
# ---------------------------------------------------------------------------

_DOUBLE_SECTIONS = {
    "takeout": "takeout_double",
    "balancing_takeout": "balancing_takeout",
    "negative": "negative_double",
    "penalty": "penalty_double",
    "nt_penalty": "penalty_double",
}


def section_for(info: CallInfo) -> str | None:
    """The system-file section describing `info`, or None when the file has
    nothing to say about the call (natural rebids, preempts, redoubles...)."""
    if info.kind == "pass":
        return "xyz_pass" if info.xyz_after_relay else None
    if info.kind == "double":
        return _DOUBLE_SECTIONS.get(info.dbl_ctx)
    if info.kind != "bid":
        return None

    if info.opening:
        if info.denom == DENOM_N:
            return {1: "open_1nt_range", 2: "open_2nt_range"}.get(info.level)
        if info.level == 2:
            return "open_2c_strong" if info.denom == 0 else "open_weak2"
        if info.level == 1:
            return ("open_major_5card" if info.denom in MAJORS
                    else "open_minor_length")
        return None                            # 3+ preempts: undefined
    if info.side_strong2c:
        if info.strong2c and info.first_call:
            if info.denom == 1 and info.level == 2:
                return "resp_2c_waiting"
            if info.denom < DENOM_N or info.level == 2:
                return "resp_2c_positive"
        return None
    if info.xyz_2c:
        return "xyz_2c"
    if info.xyz_2d:
        return "xyz_2d"
    if info.xyz_relay:
        return "xyz_relay"
    if info.smolen:
        return "smolen"
    if info.rkcb_ask:
        return "rkcb_ask"
    if info.rkcb_response:
        return "rkcb_response"
    if info.is_2over1:
        return "two_one_gf"
    if info.fourth_suit:
        return "fourth_suit"
    if info.is_jump_shift:
        return "inv_jump_shift"
    if info.is_strong_js:
        return "strong_jump_rebid"
    if info.splinter:
        return "splinter"

    if info.denom == DENOM_N:
        if info.level == 4:
            return None                        # quantitative 4NT: undefined
        if info.overcall and info.level == 1:
            return ("overcall_1nt_bal" if info.balancing else "overcall_1nt")
        if info.responder or info.advancer:
            if info.level == 1:
                return "resp_1nt"
            if info.level == 2 and info.jump == 1:
                return "resp_2nt"
            if (info.level == 3 and info.jump == 2 and info.responder
                    and info.first_call):
                return "resp_3nt"
            return None
        if info.opener_rebid:
            if info.level == 1:
                return "rebid_1nt"
            if info.level == 2 and info.jump == 1:
                return "rebid_2nt"
        return None

    if info.cuebid:
        return "cuebid_support" if info.partner_last_suit >= 0 else None
    if info.own_rebid:
        return None
    if info.raise_bid:
        if info.first_call and not is_game_or_above(info.level, info.denom):
            return {0: "resp_raise", 1: "resp_jump_raise"}.get(info.jump)
        return None
    if info.overcall and info.new_suit and info.jump == 0:
        return "overcall"
    return None


# ---------------------------------------------------------------------------
# Alert rendering
# ---------------------------------------------------------------------------

def _sym(denom: int) -> str:
    return SYM[denom] if 0 <= denom < DENOM_N else "NT"


def pretty_call(info: CallInfo) -> str:
    if info.kind == "pass":
        return "Pass"
    if info.kind == "double":
        return "Double"
    if info.kind == "redouble":
        return "Redouble"
    return f"{info.level}{_sym(info.denom)}"


def render(template: str, info: CallInfo, steps: list | None = None) -> str:
    """Fill the placeholders of one section's alert text from the call."""
    out = template
    if "{call}" in out:
        out = out.replace("{call}", pretty_call(info))
    if "{level}" in out:
        out = out.replace("{level}", str(info.level))
    if "{suit}" in out:
        out = out.replace("{suit}", _sym(info.denom))
    if "{other}" in out:
        other = MAJORS[0] if info.denom == MAJORS[1] else MAJORS[1]
        out = out.replace("{other}", _sym(other))
    if "{trump}" in out:
        out = out.replace("{trump}", _sym(info.rkcb_trump))
    if "{partner_suit}" in out:
        out = out.replace("{partner_suit}", _sym(info.partner_last_suit))
    if "{shows}" in out:
        shows = "an unspecified keycard count"
        if steps and 1 <= info.rkcb_step <= len(steps):
            shows = steps[info.rkcb_step - 1]
        out = out.replace("{shows}", shows)
    return out
