"""Tests for the system file (conventions text file): parsing, the training
ConventionConfig it builds, the call -> section resolution shared with the Go
alert server, and the alert-text rendering.

Seat arithmetic as in test_conventions.py: with dealer N the caller after k
fed calls is seat k % 4, so auctions are spelled out to the hero's decision.
"""

import os
from pathlib import Path

import pytest

from bidding_dt.conventions import (AuctionReader, ConventionConfig,
                                    convention_alert, find_system_file,
                                    parse_system_file, section_for)
from bidding_dt.conventions.system_file import parse_value, render
from bidding_dt.data.vocab import call_to_id

B = call_to_id
P, X = "P", "X"

SHIPPED = Path(__file__).resolve().parents[2] / "models" / "system.txt"


def feed_all(calls, dealer=0):
    r = AuctionReader(dealer, 0)
    for c in calls:
        r.feed(B(c))
    return r


def classify(calls, call, seat=None):
    r = feed_all(calls)
    return r.classify(B(call), seat=seat)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_value():
    assert parse_value("15-17") == (15.0, 17.0)
    assert parse_value("10.5-12.5") == (10.5, 12.5)
    assert parse_value("5-7") == (5, 7)
    assert parse_value("3") == 3
    assert parse_value("1.5") == 1.5
    assert parse_value("yes") is True
    assert parse_value("no") is False
    with pytest.raises(ValueError):
        parse_value("maybe")


MINI = """
# a tiny system file
[general]
kr_tolerance = 0.5
conv_bonus = 0.1

[open_1nt_range]
alert = {call}: 12-14 balanced
artificial = yes
weight = 2.0
nt_range = 12-14

[rkcb_response]
alert = {call}: RKCB for {trump} -- {shows}
weight = 0.0
steps = zero or three | one or four | two without | two with
"""


def test_parse_mini(tmp_path):
    f = tmp_path / "system.txt"
    f.write_text(MINI)
    spec = parse_system_file(f)
    assert set(spec.sections) == {"general", "open_1nt_range", "rkcb_response"}
    sec = spec.sections["open_1nt_range"]
    assert sec.alert == "{call}: 12-14 balanced"
    assert sec.artificial and sec.weight == 2.0
    assert sec.params == {"nt_range": (12.0, 14.0)}
    assert spec.sections["general"].params == {"kr_tolerance": 0.5,
                                               "conv_bonus": 0.1}
    assert spec.sections["rkcb_response"].steps == [
        "zero or three", "one or four", "two without", "two with"]


def test_parse_errors(tmp_path):
    def parse(text):
        f = tmp_path / "s.txt"
        f.write_text(text)
        return parse_system_file(f)

    with pytest.raises(ValueError, match="unknown section"):
        parse("[nope]\nalert = x\n")
    with pytest.raises(ValueError, match="unknown parameter"):
        parse("[general]\nnot_a_field = 1\n")
    with pytest.raises(ValueError, match="alert-only"):
        parse("[rkcb_ask]\nweight = 1.0\n")
    with pytest.raises(ValueError, match="duplicate section"):
        parse("[general]\n[general]\n")
    with pytest.raises(ValueError, match="expected key = value"):
        parse("[general]\nkr_tolerance\n")
    with pytest.raises(ValueError, match="outside any section"):
        parse("kr_tolerance = 1.0\n")
    with pytest.raises(ValueError, match="no sections"):
        parse("# nothing here\n")


def test_config_from_file(tmp_path):
    f = tmp_path / "system.txt"
    f.write_text(MINI)
    cfg = parse_system_file(f).config()
    assert cfg.kr_tolerance == 0.5
    assert cfg.conv_bonus == 0.1
    assert cfg.nt_range == (12.0, 14.0)
    assert cfg.weights["open_1nt_range"] == 2.0
    assert cfg.weights["rkcb_response"] == 0.0
    # untouched rules keep their defaults
    assert cfg.weights["gf_pass"] == ConventionConfig().weights["gf_pass"]
    # YAML/CLI overrides layer on top of the file
    cfg2 = parse_system_file(f).config({"open_1nt_range": 0.0},
                                       {"kr_tolerance": 2.0})
    assert cfg2.weights["open_1nt_range"] == 0.0
    assert cfg2.kr_tolerance == 2.0


# ---------------------------------------------------------------------------
# The shipped file
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SHIPPED.is_file(), reason="models/system.txt absent")
def test_shipped_file_matches_builtin_defaults():
    """The shipped system.txt mirrors the code defaults exactly, so the
    config it builds equals a fresh ConventionConfig (modulo float/int)."""
    spec = parse_system_file(SHIPPED)
    cfg = spec.config()
    ref = ConventionConfig()
    assert cfg.weights == ref.weights
    import dataclasses

    for fld in dataclasses.fields(ConventionConfig):
        if fld.name == "weights":
            continue
        got, want = getattr(cfg, fld.name), getattr(ref, fld.name)
        if isinstance(want, tuple):
            got = tuple(float(x) for x in got)
            want = tuple(float(x) for x in want)
        assert got == want, fld.name


@pytest.mark.skipif(not SHIPPED.is_file(), reason="models/system.txt absent")
def test_shipped_file_alerts_cover_the_conventions():
    spec = parse_system_file(SHIPPED)
    # every rewarded convention kind has an alert and the artificial marker
    for calls, call in [
            (["1C", P, "1D", P, "1H", P], "2C"),     # xyz relay
            (["1C", P, "1D", P, "1H", P], "2D"),     # xyz GF
            (["1N", P, "2C", P, "2D", P], "3H"),     # smolen
            (["1S", P, "2S", P], "4N"),              # rkcb ask
            (["1S", P, "2S", P, "4N", P], "5C"),     # rkcb response
            (["1C", P, "1H", P, "2D", P], "2S"),     # fourth suit
            (["1S", P], "4D"),                       # splinter
            (["1H", "1S"], "2S"),                    # cuebid support
    ]:
        info = classify(calls, call)
        text = spec.alert_for(info)
        assert text, f"no alert for {calls} -> {call}"
        assert "{" not in text, f"unfilled placeholder: {text}"


# ---------------------------------------------------------------------------
# section_for resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("calls,call,section", [
    # openings
    ([], "1N", "open_1nt_range"),
    ([], "2N", "open_2nt_range"),
    ([], "2C", "open_2c_strong"),
    ([], "2H", "open_weak2"),
    ([], "1S", "open_major_5card"),
    ([], "1D", "open_minor_length"),
    ([], "3S", None),                       # preempt: undefined
    ([P, P, P], "1C", "open_minor_length"),
    # strong 2C structure
    (["2C", P], "2D", "resp_2c_waiting"),
    (["2C", P], "2H", "resp_2c_positive"),
    (["2C", P], "2N", "resp_2c_positive"),
    # responses / raises
    (["1H", P], "1N", "resp_1nt"),
    (["1H", P], "2H", "resp_raise"),
    (["1H", P], "3H", "resp_jump_raise"),
    (["1H", P], "2N", "resp_2nt"),
    (["1H", P, "2C", P, "2D", P], "3N", None),  # responder already called
    (["1C", P], "3N", "resp_3nt"),
    (["1H", P], "2C", "two_one_gf"),
    (["1S", P], "2C", "two_one_gf"),
    # opener rebids
    (["1C", P, "1H", P], "1N", "rebid_1nt"),
    (["1C", P, "1H", P], "2N", "rebid_2nt"),
    (["1C", P, "1H", P], "3D", "strong_jump_rebid"),
    (["1C", P, "1H", P], "2C", None),        # own rebid: natural
    # XYZ
    (["1C", P, "1D", P, "1H", P], "2C", "xyz_2c"),
    (["1C", P, "1D", P, "1H", P], "2D", "xyz_2d"),
    (["1C", P, "1D", P, "1H", P, "2C", P], "2D", "xyz_relay"),
    (["1C", P, "1D", P, "1H", P, "2C", P, "2D", P], P, "xyz_pass"),
    (["1C", P, "1D", "1S", "1H", P], "2C", None),   # contested: no XYZ
    # smolen
    (["1N", P, "2C", P, "2D", P], "3H", "smolen"),
    (["2N", P, "3C", P, "3D", P], "3S", "smolen"),
    # RKCB
    (["1S", P, "2S", P], "4N", "rkcb_ask"),
    (["1S", P, "2S", P, "4N", P], "5C", "rkcb_response"),
    (["1S", P, "2S", P, "4N", P], "5D", "rkcb_response"),
    # splinter / jump shift / fourth suit / cuebid
    (["1S", P], "4D", "splinter"),
    (["1C", "1D"], "2H", "inv_jump_shift"),
    (["1C", P, "1H", P, "2D", P], "2S", "fourth_suit"),  # 1C-1H-2D: no XYZ
    (["1C", P, "1H", P, "1S", P], "2D", "xyz_2d"),  # 1x-1y-1z: XYZ wins
    (["1H", "1S"], "2S", "cuebid_support"),
    # competition
    (["1H"], "1S", "overcall"),
    (["1H"], "1N", "overcall_1nt"),
    (["1H", P, P], "1N", "overcall_1nt_bal"),
    (["1H", P, P], "1S", "overcall"),
    (["1H"], X, "takeout_double"),
    (["1H", P, P], X, "balancing_takeout"),
    (["1C", "1H"], X, "negative_double"),
    (["1H", "1S", "2C", P, "2D"], X, "penalty_double"),
    (["1N"], X, "penalty_double"),           # nt_penalty folds into penalty
    ([], P, None),
    ([], "XX", None),
])
def test_section_for(calls, call, section):
    info = classify(calls, call)
    assert section_for(info) == section, f"{calls} -> {call}"


# ---------------------------------------------------------------------------
# Alert rendering
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SHIPPED.is_file(), reason="models/system.txt absent")
def test_render_placeholders():
    spec = parse_system_file(SHIPPED)
    # weak two: {call} and {suit}
    text = spec.alert_for(classify([], "2S"))
    assert text.startswith("2♠") and "weak two" in text
    # splinter: {suit} + {partner_suit}
    text = spec.alert_for(classify(["1S", P], "4D"))
    assert "singleton/void ♦" in text and "partner's ♠" in text
    # smolen: {suit} + {other}
    text = spec.alert_for(classify(["1N", P, "2C", P, "2D", P], "3H"))
    assert "4 ♥" in text and "5+ ♠" in text
    # rkcb ask/response: {trump} and {shows} from the steps line
    text = spec.alert_for(classify(["1S", P, "2S", P], "4N"))
    assert "RKCB for ♠" in text
    text = spec.alert_for(classify(["1S", P, "2S", P, "4N", P], "5D"))
    assert "1 or 4 keycards" in text and "for ♠" in text
    # doubles render Double
    text = spec.alert_for(classify(["1H"], "X"))
    assert text.startswith("Double")
    # natural, uncovered calls have no alert
    assert spec.alert_for(classify(["1C", P, "1H", P], "2C")) is None


def test_convention_alert_spec_and_fallback():
    spec = parse_system_file(SHIPPED) if SHIPPED.is_file() else None
    info = classify(["1C", P, "1D", P, "1H", P], "2D")
    fallback = convention_alert(info)
    assert fallback and "artificial game force" in fallback
    if spec is not None:
        text = convention_alert(info, spec)
        assert text and "XYZ" in text
    # a plain bid alerts in neither mode
    plain = classify([], "1S")
    assert convention_alert(plain) is None
    if spec is not None:
        assert "5+" in convention_alert(plain, spec)


def test_render_unknown_placeholders_stay():
    info = classify([], "1S")
    assert render("{call} {nope}", info) == "1♠ {nope}"


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
