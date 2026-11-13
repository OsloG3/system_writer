"""The bot's system file: conventions as one text file (models/system.txt).

The file is the single source of truth for the system the bots follow. It is
written entirely in the sequence-pattern language (patterns.py) plus a small
top-level configuration block:

  * `key = value` lines (bare identifiers: kr_tolerance, conv_bonus,
    open_min_kr, to_pass_min_len, ...) tune the ConventionConfig thresholds
    and the pattern-layer discipline params;
  * every other `lhs = rhs` line is a pattern entry describing one call kind
    by its auction sequence, clauses and meaning.

Two consumers, one file:

  * training interprets it at startup -- `SystemSpec.config` folds the config
    lines into a ConventionConfig (rl/train_ppo.py builds its conv_cfg from
    it, with rl.conv_rules / rl.conv_params layering on top), and the pattern
    entries are enforced per call by patterns.SystemEnforcer (rl/rollout.py);
  * the website generates its bid alerts from the same file --
    `SystemSpec.alert_for_call` resolves a call in its auction against the
    pattern entries (first match wins) and renders the match's description.
    The Go server (system_writer/system_file.go + patterns.go) implements the
    same resolution over the same file; keep the two in lockstep.
"""

import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from .patterns import (SPEC_PARAMS, MatchState, _desc_sep, fold_state,
                       match_first, parse_entry, render_desc)
from .rules import ConventionConfig

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


# config keys the file may set: ConventionConfig fields + the enforcer params
_CONFIG_KEYS = None


def _config_keys() -> frozenset:
    global _CONFIG_KEYS
    if _CONFIG_KEYS is None:
        _CONFIG_KEYS = _param_fields() | frozenset(SPEC_PARAMS)
    return _CONFIG_KEYS


# bare identifiers are config lines -- unless they are a call token, which
# makes them a (single-step) pattern entry like "P = ..."
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CALL_STEP_TOKENS = {"P", "PASS", "X", "DBL", "D", "XX", "RDBL", "RD"}


def _is_config_line(lhs: str) -> bool:
    return (bool(_IDENT.fullmatch(lhs))
            and lhs.upper() not in _CALL_STEP_TOKENS)


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
class SystemSpec:
    """Parsed system file: top-level settings plus the pattern entries in
    file order."""
    path: str | None = None
    settings: dict = field(default_factory=dict)   # key = value lines
    entries: list = field(default_factory=list)    # pattern entries
    _states: dict = field(default_factory=dict)    # replay-state memo

    # -- config for training -------------------------------------------------

    @property
    def spec_params(self) -> dict:
        """Enforcer-level params (SPEC_PARAMS keys, e.g. to_pass_min_len),
        with the defaults filled in."""
        out = dict(SPEC_PARAMS)
        for k, v in self.settings.items():
            if k in SPEC_PARAMS:
                out[k] = v
        return out

    def config(self, rule_overrides: dict | None = None,
               param_overrides: dict | None = None) -> ConventionConfig:
        """ConventionConfig with the file's settings applied.

        YAML/CLI overrides (rl.conv_rules / rl.conv_params) layer on top of
        the file, exactly as they used to layer on the code defaults. The
        enforcer-level SPEC_PARAMS keys belong to the pattern layer
        (spec_params) and are filtered out here. Rule weights are the code
        defaults (they only back calls no pattern entry matches); override
        them per run with rl.conv_rules.
        """
        params = {k: v for k, v in self.settings.items()
                  if k not in SPEC_PARAMS}
        params.update(param_overrides or {})
        return ConventionConfig.from_overrides(rule_overrides, params)

    # -- alerts for the website ----------------------------------------------

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
        """Alert text for `action` in `reader`'s auction: the first matching
        pattern entry's rendered description, or None when no entry matches.
        An entry whose rendered description is empty suppresses the alert."""
        entry, binds = self.match_call(reader.dealer, list(reader.calls),
                                       int(action))
        if entry is None:
            return None
        return render_desc(entry, binds) or None


def parse_system_file(path: str | Path) -> SystemSpec:
    """Parse a system file, raising ValueError on anything malformed."""
    path = Path(path)
    spec = SystemSpec(path=str(path))
    for lineno, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "#;" or line.startswith("["):
            if line.startswith("["):
                raise ValueError(
                    f"{path}:{lineno}: INI sections are gone -- write "
                    f"pattern entries and top-level key = value config")
            continue
        where = f"{path}:{lineno}"
        sep = _desc_sep(line)
        lhs = line[:sep].strip() if sep >= 0 else line
        if _is_config_line(lhs):
            if sep < 0:
                raise ValueError(f"{where}: expected key = value")
            if lhs not in _config_keys():
                raise ValueError(f"{where}: unknown config key {lhs!r}")
            spec.settings[lhs] = parse_value(line[sep + 1:].strip())
            continue
        # a sequence-pattern entry (patterns.py)
        try:
            spec.entries.append(parse_entry(line))
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
    if not spec.entries and not spec.settings:
        raise ValueError(f"{path}: no entries found")
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
