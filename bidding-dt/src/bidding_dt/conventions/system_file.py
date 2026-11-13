"""The bot's system file: conventions as one text file (models/system.txt).

The file is the single source of truth for the system the bots follow:

  * training interprets it at startup -- `SystemSpec.config` folds every
    section's `weight` into the convention-penalty rule weights and every
    numeric key into the matching ConventionConfig threshold (rl/train_ppo.py
    builds its conv_cfg from it, so editing a range in the file re-tunes what
    the RL loop enforces);
  * the website generates its bid alerts from the same file -- `section_for`
    resolves a classified call (auction.py CallInfo) to the one section whose
    agreement it is, and `SystemSpec.alert_for` renders that section's alert
    template ({call} {suit} {other} {trump} {partner_suit} {shows} {level}
    placeholders). The Go server (system_writer/system_file.go) implements the
    same resolution over the same file; keep the two in lockstep.

Format (INI-like, # and ; comments)::

    [section]            rule code (rules.DEFAULT_WEIGHTS) or one of the
                         alert-only kinds (general, rkcb_ask, resp_2c_waiting,
                         balancing_takeout, penalty_double)
    alert = text         website alert text (placeholders above)
    artificial = yes     the call carries the alert (spade) marker on the site
    weight = N           convention-penalty weight (rule-code sections only)
    steps = a | b | c | d   RKCB step texts ({shows})
    <param> = value      ConventionConfig threshold: number, "lo-hi" range
                         or yes/no boolean
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .auction import DENOM_N, MAJORS, CallInfo, is_game_or_above
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
    """Parsed system file: sections in file order, keyed by name."""
    path: str | None = None
    sections: dict = field(default_factory=dict)   # name -> Section

    # -- config for training -------------------------------------------------

    def config(self, rule_overrides: dict | None = None,
               param_overrides: dict | None = None) -> ConventionConfig:
        """ConventionConfig with the file's weights and thresholds applied.

        YAML/CLI overrides (rl.conv_rules / rl.conv_params) layer on top of
        the file, exactly as they used to layer on the code defaults.
        """
        weights, params = {}, {}
        for sec in self.sections.values():
            params.update(sec.params)
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
        file defines no meaning for it."""
        name = section_for(info)
        if name is None:
            return None
        sec = self.sections.get(name)
        if sec is None or sec.alert is None:
            return None
        return render(sec.alert, info, sec.steps)


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
        if "=" not in line:
            raise ValueError(f"{where}: expected key = value")
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if cur is None:
            raise ValueError(f"{where}: key outside any section")
        if key == "alert":
            cur.alert = val
        elif key == "artificial":
            cur.artificial = parse_value(val) is True
        elif key == "weight":
            if cur.name not in DEFAULT_WEIGHTS:
                raise ValueError(
                    f"{where}: [{cur.name}] is alert-only and has no weight")
            cur.weight = float(val)
        elif key == "steps":
            cur.steps = [s.strip() for s in val.split("|")]
        else:
            if key not in _param_fields():
                raise ValueError(f"{where}: unknown parameter {key!r}")
            cur.params[key] = parse_value(val)
    if not spec.sections:
        raise ValueError(f"{path}: no sections found")
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
