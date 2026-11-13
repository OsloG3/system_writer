"""Tests for the system file (conventions text file): config lines, pattern
entries, the training ConventionConfig built from them, alert resolution and
file lookup. The pattern language itself is tested in test_patterns.py.

Seat arithmetic as in test_conventions.py: with dealer N the caller after k
fed calls is seat k % 4, so auctions are spelled out to the hero's decision.
"""

import os
from pathlib import Path

import pytest

from bidding_dt.conventions import (AuctionReader, ConventionConfig,
                                    find_system_file, parse_system_file)
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
    ([], "2C", "strong artificial"),
    ([], "3S", "preempt"),
    (["1C", P, "1H", P], "2D", "reverse"),
    (["1N", P], "2C", "stayman ask"),
    (["2C", P], "2D", "waiting"),
    (["2C", P], "2H", "positive"),
    (["1C", P, "1D", P, "1H", P], "2C", "XYZ relay"),
    (["1C", P, "1D", P, "1H", P], "2D", "artificial"),
    (["1C", P, "1D", P, "1H", P, "2C", P], "2D", "completes the XYZ relay"),
    (["1S", P, "2S", P, "4N", P], "5C", "0 or 3 keycards"),
    (["1S", P, "2S", P], "4N", "RKCB ask for ♠"),
    (["1S", P], "4D", "splinter in ♦"),
    (["1H", P], "2H", "raise"),
    (["1H", P], "3H", "limit raise"),
    (["1H", P], "2C", "2/1"),
    (["1C", "1S"], "2D", "2/1 over their bid"),
    (["1H"], X, "takeout"),
    ([P, "4S"], X, "penalty"),
    (["1H", P, P], "1S", "balancing overcall"),
    (["1H"], "1S", "overcall"),
    (["1H"], "1N", "1NT overcall"),
    (["1N", P, "2C", P, "2D", P], "3S", "smolen"),
    (["1C", P, "1H", P], "2N", "opener's jump 2NT rebid"),
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
    # a redouble neither
    assert spec.alert_for_call(feed_all(["1H", X]), B("XX")) is None
    # opener's plain own-suit rebid is not described
    assert spec.alert_for_call(feed_all(["1C", P, "1H", P]), B("2C")) is None


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
