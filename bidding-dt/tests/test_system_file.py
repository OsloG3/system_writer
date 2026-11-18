"""Tests for the system file (conventions text file): config lines, pattern
entries, the training ConventionConfig built from them, alert resolution and
file lookup. The pattern language itself is tested in test_patterns.py.

Seat arithmetic as in test_conventions.py: with dealer N the caller after k
fed calls is seat k % 4, so auctions are spelled out to the hero's decision.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from bidding_dt.conventions import (AuctionReader, ConventionConfig, HandInfo,
                                    SystemEnforcer, find_system_file,
                                    parse_system_file)
from bidding_dt.data.vocab import call_to_id

B = call_to_id
P, X = "P", "X"

SHIPPED = Path(__file__).resolve().parents[2] / "models" / "system.txt"


def feed_all(calls, dealer=0):
    r = AuctionReader(dealer, 0)
    for c in calls:
        r.feed(B(c))
    return r


MINI = """
# a tiny system file
kr_tolerance = 0.5
conv_bonus = 0.1
to_pass_min_len = 5

1X-1Y-2Z; X<Z = reverse: 5+X 4+Z p>16 F @2.0
1N = 12-14 balanced
...-X = TO -- takeout
"""


def parse_text(tmp_path, text):
    f = tmp_path / "system.txt"
    f.write_text(text)
    return parse_system_file(f)


def test_parse_mini(tmp_path):
    spec = parse_text(tmp_path, MINI)
    assert spec.settings["kr_tolerance"] == 0.5
    assert spec.settings["conv_bonus"] == 0.1
    assert spec.settings["to_pass_min_len"] == 5
    assert len(spec.entries) == 3
    e = spec.entries[0]
    assert e.pat == "1X-1Y-2Z" and e.weight == 2.0
    assert e.flags == frozenset({"F"})
    assert spec.entries[2].flags == frozenset({"TO"})


def test_parse_errors(tmp_path):
    with pytest.raises(ValueError, match="unknown config key"):
        parse_text(tmp_path, "not_a_field = 1\n")
    with pytest.raises(ValueError, match="expected key = value"):
        parse_text(tmp_path, "kr_tolerance\n")
    with pytest.raises(ValueError, match="INI sections are gone"):
        parse_text(tmp_path, "[general]\nkr_tolerance = 1.0\n")
    with pytest.raises(ValueError, match="bad pattern step"):
        parse_text(tmp_path, "1Q-2X = nope\n")
    with pytest.raises(ValueError, match="no entries"):
        parse_text(tmp_path, "# nothing here\n")


def test_config_from_file(tmp_path):
    cfg = parse_text(tmp_path, MINI).config()
    assert cfg.kr_tolerance == 0.5
    assert cfg.conv_bonus == 0.1
    # SPEC_PARAMS keys never reach the ConventionConfig
    assert not hasattr(cfg, "to_pass_min_len")
    # untouched fields keep their defaults
    assert cfg.nt_range == ConventionConfig().nt_range
    # YAML/CLI overrides layer on top of the file
    cfg2 = parse_text(tmp_path, MINI).config(None, {"kr_tolerance": 2.0})
    assert cfg2.kr_tolerance == 2.0


def test_spec_params(tmp_path):
    p = parse_text(tmp_path, MINI).spec_params
    assert p["to_pass_min_len"] == 5          # from the file
    assert p["f_pass_weight"] == 2.0          # default kept


# ---------------------------------------------------------------------------
# The shipped file
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SHIPPED.is_file(), reason="models/system.txt absent")
def test_shipped_file():
    spec = parse_system_file(SHIPPED)
    assert len(spec.entries) >= 60
    assert spec.settings["kr_tolerance"] == 1.0
    cfg = spec.config()
    assert cfg.kr_tolerance == 1.0 and cfg.open_min_kr == 11.0


@pytest.mark.skipif(not SHIPPED.is_file(), reason="models/system.txt absent")
@pytest.mark.parametrize("calls,call,want", [
    ([], "1N", "15-17 pts balanced"),
    ([], "2H", "weak two"),
    ([], "2C", "game force"),
    ([], "3S", "preempt"),
    (["1C", P, "1H", P], "2D", "reverse"),
    (["1N", P], "2C", "stayman ask"),
    (["2C", P], "2D", "waiting"),
    (["2C", P], "2H", "positive"),
    (["1C", P, "1D", P, "1H", P], "2C", "XYZ relay"),
    (["1C", P, "1D", P, "1H", P], "2D", "artificial"),
    (["1C", P, "1D", P, "1H", P, "2C", P], "2D", "completes the XYZ relay"),
    (["1S", P, "2S", P, "4N", P], "5C", "1 or 4 keycards"),
    (["1S", P, "2S", P], "4N", "RKCB ask for ♠"),
    (["1S", P], "4D", "splinter in ♦"),
    (["1H", P], "2H", "raise"),
    (["1H", P], "3H", "preemptive raise"),
    (["1H", P], "2C", "2/1"),
    (["1C", "1S"], "2D", "2/1 game force"),
    (["1H"], X, "takeout"),
    ([P, "4S"], X, "penalty"),
    (["1H", P, P], "1S", "balancing overcall"),
    (["1H"], "1S", "overcall"),
    (["1H"], "1N", "1NT overcall"),
    (["1N", P, "2C", P, "2D", P], "3S", "smolen"),
    (["1C", P, "1H", P], "2N", "2NT rebid"),
    (["1C", P, "1H", P], "1N", "opener's 1NT rebid"),
    (["1H", P], "1N", "no biddable 4-card major"),
    (["1C", P, "1H", P, "2D", P], "2S", "fourth suit"),
])
def test_shipped_file_alerts(calls, call, want):
    spec = parse_system_file(SHIPPED)
    got = spec.alert_for_call(feed_all(calls), B(call))
    assert got and want in got, f"{' '.join(calls)} -> {call}: {got!r}"
    assert "{" not in got, f"unfilled placeholder: {got!r}"


@pytest.mark.skipif(not SHIPPED.is_file(), reason="models/system.txt absent")
def test_shipped_file_uncovered_calls():
    spec = parse_system_file(SHIPPED)
    # a plain pass carries no alert
    assert spec.alert_for_call(feed_all([P]), B(P)) is None
    # a strength redouble IS described now (responding to a doubled opening)
    got = spec.alert_for_call(feed_all(["1H", X]), B("XX"))
    assert got and "redouble" in got
    # opener's plain own-suit rebid IS described now (the file is bid-complete):
    # a minimum 5+ card rebid in the opened suit
    got = spec.alert_for_call(feed_all(["1C", P, "1H", P]), B("2C"))
    assert got and "minimum rebid" in got


# ---------------------------------------------------------------------------
# find_system_file
# ---------------------------------------------------------------------------

def test_find_system_file(tmp_path, monkeypatch):
    monkeypatch.delenv("BIDDING_SYSTEM_FILE", raising=False)
    f = tmp_path / "system.txt"
    f.write_text(MINI)
    assert find_system_file(str(f)) == str(f)
    with pytest.raises(FileNotFoundError):
        find_system_file(str(tmp_path / "missing.txt"))
    monkeypatch.setenv("BIDDING_SYSTEM_FILE", str(f))
    assert find_system_file() == str(f)
    monkeypatch.setenv("BIDDING_SYSTEM_FILE", str(tmp_path / "gone.txt"))
    assert find_system_file() is None
    monkeypatch.delenv("BIDDING_SYSTEM_FILE")
    # from the repo root the shipped file is found through the candidates
    old = os.getcwd()
    os.chdir(SHIPPED.parents[1])
    try:
        assert Path(find_system_file()).resolve() == SHIPPED.resolve()
    finally:
        os.chdir(old)


# ---------------------------------------------------------------------------
# Regressions: the system file must be a bid-complete 2/1 GF system without
# the bugs that made RL degrade (garbage Stayman, inverted Lebensohl, the
# shadowed Jordan raise, the unenforced 0/3 RKCB step, the dead pass pattern).
# ---------------------------------------------------------------------------

def _enf():
    spec = parse_system_file(SHIPPED)
    return SystemEnforcer(spec, spec.config()), spec


def test_stayman_requires_strength(tmp_path):
    """Garbage Stayman must be PENALIZED, not a free bid. The file is a pure
    constraint: the old positive clean_use/target gating is gone, so what
    remains is the hand-description penalty on the entry itself."""
    f = tmp_path / "system.txt"
    f.write_text("1N = p15-17 balanced\n"
                 "1N-2C = p>7.5 4+H OR p>7.5 4+S stayman ask @!\n")
    spec = parse_system_file(f)
    enf = SystemEnforcer(spec, spec.config())
    r = feed_all(["1N", P])
    bust = HandInfo("743.8632.952.843")     # 4 hearts, ~-0.5 K&R
    good = HandInfo("K74.Q863.A52.K84")     # 4 hearts, ~11 K&R
    # a bust Stayman is a violation ...
    assert [v.code for v in enf.check(r, 2, bust, B("2C"))] == ["pat:1N-2C"]
    # ... a real Stayman hand is clean
    assert enf.check(r, 2, good, B("2C")) == []


@pytest.mark.skipif(not SHIPPED.is_file(), reason="models/system.txt absent")
def test_lebensohl_cuebid_wants_no_stopper():
    """(2H)-X-3H shows 4 spades and DENIES a heart stopper (was inverted)."""
    enf, _ = _enf()
    r = feed_all(["2H", X, P], dealer=1)
    no_stop = HandInfo("KQ74.863.A52.K84")   # 4 spades, no heart stopper
    with_stop = HandInfo("KQ74.A63.A52.K84")  # 4 spades, heart ace = stopper
    assert enf.check(r, 0, no_stop, B("3H")) == []
    assert [v.code for v in enf.check(r, 0, with_stop, B("3H"))] == \
        ["pat:(2H)-X-3H"]


@pytest.mark.skipif(not SHIPPED.is_file(), reason="models/system.txt absent")
def test_jordan_2nt_raise_reachable():
    """1M-2NT is the artificial invitational raise, not a natural balanced
    invite -- an unbalanced limit raise with support must be clean."""
    enf, spec = _enf()
    r = feed_all(["1H", P])
    jordan = HandInfo("K74.QJ83.5.KQ964")   # 4 hearts, singleton, ~11.8 K&R
    entry, _ = enf.match(0, list(r.calls), B("2N"))
    assert entry is not None and entry.artificial
    assert enf.check(r, 2, jordan, B("2N")) == []


@pytest.mark.skipif(not SHIPPED.is_file(), reason="models/system.txt absent")
def test_rkcb_zero_three_step_enforced():
    """The 0/3 step (5D) is checkable -- `kc=0/3`, not the misspelled `3/0`."""
    enf, _ = _enf()
    r = feed_all(["1S", P, "2C", P, "2S", P, "4N", P])
    zero = HandInfo("743.863.K952.K84")     # 0 keycards
    one = HandInfo("A743.863.952.K84")      # 1 keycard (SA)
    assert enf.check(r, 0, zero, B("5D")) == []          # 0/3 correct
    assert [v.code for v in enf.check(r, 0, one, B("5D"))] == \
        ["pat:...-4N-(P)-5D"]                            # 1 keycard wrong


@pytest.mark.skipif(not SHIPPED.is_file(), reason="models/system.txt absent")
def test_passing_opening_with_values_penalized():
    """`1X-P` is no longer a dead pattern: passing an opening with values is
    a violation; a weak pass is clean."""
    enf, _ = _enf()
    r = feed_all(["1S", P])
    strong = HandInfo("K74.K863.952.K84")   # ~8.5 K&R
    weak = HandInfo("743.8632.952.843")     # ~4 K&R
    assert [v.code for v in enf.check(r, 2, strong, B("P"))] == ["pat:1X-P"]
    assert enf.check(r, 2, weak, B("P")) == []


@pytest.mark.skipif(not SHIPPED.is_file(), reason="models/system.txt absent")
def test_penalty_double_after_our_notrump():
    """A double after our 1NT opening is penalty, so partner may pass it."""
    enf, spec = _enf()
    r = feed_all(["1N", "2H"])
    entry, _ = enf.match(0, list(r.calls), B("X"))
    assert entry is not None and "PENALTY" in entry.flags
    # partner of the doubler passing with a trump stack is fine (no to_pass)
    r2 = feed_all(["1N", "2H", X, P])
    assert enf.check(r2, 0, HandInfo("QJ63.K5.874.AJ63"), B("P")) == []


@pytest.mark.skipif(not SHIPPED.is_file(), reason="models/system.txt absent")
def test_notrump_transfers_present():
    """The file describes Jacoby transfers over 1NT (5-card major)."""
    enf, spec = _enf()
    r = feed_all(["1N", P])
    entry, _ = enf.match(0, list(r.calls), B("2D"))
    assert entry is not None and entry.artificial
    hearts = HandInfo("74.KQ863.952.K84")   # 5 hearts
    assert enf.check(r, 2, hearts, B("2D")) == []
