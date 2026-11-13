"""Tests for the system-file pattern language (conventions/patterns.py):
parsing, sequence matching with generic suits and clauses, the replay state
behind the conditions, the training-side enforcement (branches, F/GF, TO/
PENALTY discipline, layer replacement) and the alert rendering.

Seat arithmetic: dealer N, the caller after k fed calls is seat k % 4.
Hands are 'S.H.D.C' with real ranks; K&R values quoted below were checked
with conventions.kr_points.
"""

import pathlib
import tempfile

import pytest

from bidding_dt.conventions import (AuctionReader, ConventionConfig,
                                    HandInfo, SystemEnforcer, match_first,
                                    parse_entry, parse_system_file,
                                    replay_state)
from bidding_dt.conventions.patterns import (call_id_of, hand_violations,
                                             render_desc)
from bidding_dt.data.vocab import call_to_id

B = call_to_id
P, X, XX = "P", "X", "XX"


def ids(calls):
    return [B(c) for c in calls]


def feed_all(calls, dealer=0):
    r = AuctionReader(dealer, 0)
    for c in calls:
        r.feed(B(c))
    return r


def spec_of(*lines):
    f = pathlib.Path(tempfile.mkdtemp()) / "system.txt"
    f.write_text("\n".join(lines))
    return parse_system_file(f)


def make_enforcer(lines, overrides=None):
    spec = spec_of(*lines)
    cfg = ConventionConfig.from_overrides(overrides or {})
    return SystemEnforcer(spec, cfg), spec


def codes(vs):
    return sorted(v.code for v in vs)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_entry_shapes():
    e = parse_entry("1X-1Y-2Z; X<Z = 5+X 4+Z p>16 F @2.0")
    assert e.pat == "1X-1Y-2Z" and not e.wildcard
    assert [(s.them, s.kind, s.level, s.denom) for s in e.steps] == [
        (False, "bid", 1, "X"), (False, "bid", 1, "Y"), (False, "bid", 2, "Z")]
    assert e.weight == 2.0 and not e.artificial
    assert e.flags == frozenset({"F"})
    assert len(e.branches) == 1
    assert [t.kind for t in e.branches[0]] == ["len", "len", "pts", "F"]
    assert e.checkable

    e = parse_entry("1X-(1Y)-2Z; Z<Y = 4+Z p>11.5 GF")
    assert [s.them for s in e.steps] == [False, True, False]
    assert e.flags == frozenset({"GF"}) and e.weight == 1.0

    e = parse_entry("1NT-2C = 4+H OR 4+S stayman ask @!")
    assert len(e.branches) == 2
    assert [t.kind for t in e.branches[0]] == ["len"]
    assert [t.kind for t in e.branches[1]] == ["len", "text", "text"]
    assert e.artificial and not e.flags

    # a ">=" inside a clause must not be mistaken for the description "="
    e = parse_entry("...-X; we_xx OR dbls_by_us>=2 OR gf OR their_bid>=4S"
                    " = PENALTY")
    assert e.wildcard and e.flags == frozenset({"PENALTY"})
    assert [g[0].kind for g in e.atoms] == ["we_xx", "cnt", "gf", "bidcmp"]
    assert e.atoms[3][0].call == B("4S")
    assert e.atoms[1][0].n == 2

    # no description at all (suppresses the alert for matching calls)
    e = parse_entry("1H-2NT-3S")
    assert e.desc == "" and e.pat == "1H-2NT-3S"

    # DBL/RDBL spellings, colon clause separator, exact/range lengths
    e = parse_entry("...-DBL: balancing = TO takeout")
    assert e.steps[-1].kind == "dbl"
    assert e.atoms[0][0].kind == "balancing"
    e = parse_entry("2X = 6-7X p4-12 weak two @! @0.1")
    tok_len, tok_pts = e.branches[0][0], e.branches[0][1]
    assert (tok_len.lo, tok_len.hi) == (6, 7)
    assert tok_pts.kind == "pts" and tok_pts.op == "range"
    assert e.weight == 0.1 and e.artificial

    with pytest.raises(ValueError):
        parse_entry("1Q-2X = nope")            # bad denom
    with pytest.raises(ValueError):
        parse_entry("1X = text @notanumber")   # bad attribute
    with pytest.raises(ValueError):
        parse_entry("1X; nonsense_atom = text")
    with pytest.raises(ValueError):
        parse_entry("...-X = TO and PENALTY")  # both meanings


def test_call_id_of():
    assert call_id_of("4S") == B("4S")
    assert call_id_of("1NT") == B("1N")
    assert call_id_of("P") == B("P")
    assert call_id_of("DBL") == B("X")
    assert call_id_of("RDBL") == B("XX")
    with pytest.raises(ValueError):
        call_id_of("1X")   # variables are not literal calls


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

REV = parse_entry("1X-1Y-2Z; X<Z = 5+X 4+Z p>16 F @2.0")


def test_match_reverse_and_bindings():
    e, b = match_first([REV], 0, ids(["1C", P, "1H", P]), B("2D"))
    assert e is REV and b == {"X": 0, "Y": 2, "Z": 1}      # C < D
    # not a reverse: the new suit ranks below the opening
    assert match_first([REV], 0, ids(["1H", P, "1S", P]), B("2D"))[0] is None
    # opener's own-suit rebid is not it either
    assert match_first([REV], 0, ids(["1C", P, "1H", P]), B("2C"))[0] is None
    # leading passes are skipped (3rd-seat opening)
    e, b = match_first([REV], 0, ids([P, P, "1C", P, "1H", P]), B("2D"))
    assert e is REV and b["X"] == 0
    # opponents opened -> the anchored pattern must not fire for our calls
    assert match_first([REV], 0, ids(["1H", "1S"]), B("2C"))[0] is None
    # dealer E: 1D(E)-P-1S(W)-P-2H(E) is a reverse for E-W (D < H)
    e, b = match_first([REV], 1, ids(["1D", P, "1S", P]), B("2H"))
    assert e is REV and b == {"X": 1, "Y": 3, "Z": 2}


def test_match_distinctness_and_literals():
    # the variable may not bind to a literal suit of the pattern
    e = parse_entry("1H-2X = artificial")
    assert match_first([e], 0, ids(["1H", P]), B("2D"))[0] is e
    assert match_first([e], 0, ids(["1H", P]), B("2H"))[0] is None
    # same letter = same suit
    e = parse_entry("1X-2X = raise")
    assert match_first([e], 0, ids(["1S", P]), B("2S"))[0] is e
    assert match_first([e], 0, ids(["1S", P]), B("2D"))[0] is None
    # fully literal sequences
    e = parse_entry("1H-2N-3S")
    assert match_first([e], 0, ids(["1H", P, "2N", P]), B("3S"))[0] is e
    assert match_first([e], 0, ids(["1H", P, "2N", P]), B("3D"))[0] is None


def test_match_own_pass_not_skipped():
    # 1C(N) P P(S!) 1D(W) 1S(N): partner passed between our bids, so the
    # implicit opponent-pass skip must NOT turn this into 1X-1Y
    e = parse_entry("1X-1Y = two responses")
    assert match_first([e], 0, ids(["1C", P, P, "1D"]), B("1S"))[0] is None
    # but opponent passes between our calls are skipped
    assert match_first([e], 0, ids(["1C", P]), B("1S"))[0] is e


def test_match_wildcard_and_parens():
    tail = parse_entry("...-2N = anywhere")
    assert match_first([tail], 0,
                       ids(["1C", P, "1D", P, "1H", P]), B("2N"))[0] is tail
    # an anchored 2N only matches the opening
    open2n = parse_entry("2N = opening")
    assert match_first([open2n], 0,
                       ids(["1C", P, "1D", P, "1H", P]), B("2N"))[0] is None
    assert match_first([open2n], 0, ids([P, P]), B("2N"))[0] is open2n
    # parenthesized steps match opponent calls only
    e = parse_entry("1X-(1Y)-2Z; Z<Y = GF")
    assert match_first([e], 0, ids(["1C", "1S"]), B("2D"))[0] is e   # D<S
    assert match_first([e], 0, ids(["1C", P]), B("2D"))[0] is None   # no overcall
    assert match_first([e], 0, ids(["1C", "1D"]), B("2H"))[0] is None  # H>D


def test_match_first_wins_in_file_order():
    a = parse_entry("...-X = first")
    b = parse_entry("...-X = second")
    assert match_first([a, b], 0, ids(["1H"]), B("X"))[0] is a
    assert match_first([b, a], 0, ids(["1H"]), B("X"))[0] is b


# ---------------------------------------------------------------------------
# Conditions (clause atoms over the replay state)
# ---------------------------------------------------------------------------

PEN = parse_entry("...-X; we_xx OR dbls_by_us>=2 OR gf OR their_bid>=4S"
                  " = PENALTY")
TO = parse_entry("...-X = TO")
DBLS = [PEN, TO]


def dbl_kind(calls, call="X", dealer=0):
    e, _ = match_first(DBLS, dealer, ids(calls), B(call))
    return sorted(e.flags)[0] if e else None


def test_double_conditions():
    assert dbl_kind(["1H"]) == "TO"
    assert dbl_kind(["1H", P, P]) == "TO"                  # balancing: still TO
    assert dbl_kind([P, "4S"]) == "PENALTY"                # their 4S
    assert dbl_kind([P, "4N"]) == "PENALTY"                # 4N ranks above 4S
    assert dbl_kind([P, "3S"]) == "TO"                     # below 4S
    # our side redoubled earlier: 1H(N) X(E) XX(S) 2D(W) P P, S doubles again
    assert dbl_kind(["1H", X, XX, "2D", P, P]) == "PENALTY"
    # our side already doubled twice: 1C 1H X(S#1) 2D X(N#2) 2H X(S -> penalty)
    assert dbl_kind(["1C", "1H", X, "2D", X, "2H"]) == "PENALTY"
    # ... one prior double is not enough (candidate is E's double here)
    assert dbl_kind(["1C", "1H", X, "2D", P]) == "TO"
    # file-defined GF: 1C-2D marked GF, a later double is penalty
    gf_entry = parse_entry("1X-2Y; Y>X = GF")
    seq = ["1C", P, "2D", "2H", "3C", "3H"]
    e, _ = match_first([gf_entry, PEN, TO], 0, ids(seq), B("X"))
    assert e is PEN


def test_state_conditions():
    st = replay_state(DBLS, 0, ids(["1H", P, P]))
    assert st.passes_since_bid == 2 and st.activity[1] == 0
    e = parse_entry("...-1S; balancing = balance seat")
    assert match_first([e], 0, ids(["1H", P, P]), B("1S"))[0] is e
    e = parse_entry("...-1S; uncontested = quiet")
    assert match_first([e], 0, ids(["1H", P, P]), B("1S"))[0] is None
    assert match_first([e], 0, ids([P, P]), B("1S"))[0] is e
    # passed hand: E passed first, then overcalls 1S when it comes around
    e = parse_entry("...-1S; passed = we passed")
    assert match_first([e], 0, ids([P, P, "1C", P, P]), B("1S"))[0] is e
    assert match_first([e], 0, ids(["1C", P]), B("1S"))[0] is None
    # our side has bid at 1C+: S doubles/bids after N opened
    e = parse_entry("...-2D; our_bid>=1C = we opened")
    assert match_first([e], 0, ids(["1C", "1H"]), B("2D"))[0] is e
    assert match_first([e], 0, ids(["1H"]), B("2D"))[0] is None


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------

def test_reverse_hand_enforcement():
    enf, _ = make_enforcer([REV.raw])
    r = feed_all(["1C", P, "1H", P])
    # 5+C 4+D, K&R 23.4: clean reverse
    vs = enf.check(r, 0, HandInfo("A92.5.AKJ4.AKJ32"), B("2D"))
    assert codes(vs) == []
    # only 4 clubs -> violates 5+X
    vs = enf.check(r, 0, HandInfo("AK92.5.AKJ4.AKJ3"), B("2D"))
    assert codes(vs) == ["pat:1X-1Y-2Z"]
    # 5+C 4+D but K&R 9.2 -> violates p>16
    vs = enf.check(r, 0, HandInfo("982.5.AK42.98543"), B("2D"))
    assert codes(vs) == ["pat:1X-1Y-2Z"]


def test_or_branches_stayman():
    enf, _ = make_enforcer(["1N-2C = 4+H OR 4+S stayman ask @1.0"])
    r = feed_all(["1N", P])
    # 4 hearts, no spade: branch 1 satisfied
    assert enf.check(r, 2, HandInfo("982.AK42.543.982"), B("2C")) == []
    # 4 spades, no heart: branch 2 satisfied (free text is not checkable)
    assert enf.check(r, 2, HandInfo("AK42.982.543.982"), B("2C")) == []
    # neither major: no branch fits
    vs = enf.check(r, 2, HandInfo("982.543.AK42.KJ4"), B("2C"))
    assert codes(vs) == ["pat:1N-2C"]


def test_wildcard_branch_is_lenient():
    enf, _ = make_enforcer(["2H = weak OR p15-17 @1.0"])
    r = feed_all([])
    # branch 1 has no checkable token -> any hand passes
    assert enf.check(r, 0, HandInfo("AKQ.AKQ.AKQ.AKQJ"), B("2H")) == []


def test_entry_replaces_rule_checks():
    # the file redefines 1NT as 10-12: a 10.9 K&R balanced hand is CLEAN by
    # the file and must not also eat the layer-1 open_1nt_range (15-17)
    enf, _ = make_enforcer(["1NT = p10-12 balanced @1.0"])
    r = feed_all([])
    vs = enf.check(r, 0, HandInfo("KJ4.Q82.AJ62.Q82"), B("1N"))
    assert codes(vs) == []
    # a 14.0 K&R hand is outside the file's range -> pat violation only
    vs = enf.check(r, 0, HandInfo("AJ74.KQ6.AJ7.654"), B("1N"))
    assert codes(vs) == ["pat:1NT"]


def test_alert_only_entry_keeps_rule_checks():
    enf, _ = make_enforcer(["1NT = p10-12 balanced @0"])
    r = feed_all([])
    # weight 0: the file entry never enforces, layer 1 (15-17) still does
    vs = enf.check(r, 0, HandInfo("KJ4.Q82.AJ62.Q82"), B("1N"))
    assert codes(vs) == ["open_1nt_range"]


def test_f_pass_discipline():
    enf, _ = make_enforcer([REV.raw])
    junk = HandInfo("982.543.652.9872")
    # N's reverse (F) -> S passing it is a violation
    r = feed_all(["1C", P, "1H", P, "2D", P])
    vs = enf.check(r, 2, junk, B("P"))
    assert "f_pass" in codes(vs)
    # an opponent bid in between releases the force
    r = feed_all(["1C", P, "1H", P, "2D", "2H"])
    vs = enf.check(r, 2, junk, B("P"))
    assert "f_pass" not in codes(vs)
    # after the partner (S) bid 2H the force is consumed; N may pass
    r = feed_all(["1C", P, "1H", P, "2D", P, "2H", P])
    vs = enf.check(r, 0, HandInfo("A92.5.AKJ4.AKJ32"), B("P"))
    assert "f_pass" not in codes(vs)


def test_file_gf_pass_and_no_duplicate():
    enf, _ = make_enforcer(["1N-2C = GF artificial checkback @1.0"])
    # N passes below game with the file-defined GF on -> one gf_pass
    r = feed_all(["1N", P, "2C", P])
    vs = enf.check(r, 0, HandInfo("982.543.652.9872"), B("P"))
    assert codes(vs).count("gf_pass") == 1
    # structural GF (2/1) -> exactly one gf_pass (layer 1; not doubled up)
    enf2, _ = make_enforcer(["[general]"])
    r = feed_all(["1C", P, "2D", P])
    vs = enf2.check(r, 0, HandInfo("982.543.652.9872"), B("P"))
    assert codes(vs).count("gf_pass") == 1


def test_to_pass_discipline():
    enf, _ = make_enforcer([PEN.raw, TO.raw])
    # E doubled our 1H (TO); W (partner) passing without a trump stack
    r = feed_all(["1H", X, P])
    vs = enf.check(r, 3, HandInfo("982.543.AK42.KJ4"), B("P"))
    assert "to_pass" in codes(vs)
    # with 4+ hearts the conversion is fine
    vs = enf.check(r, 3, HandInfo("982.KJ42.A42.982"), B("P"))
    assert "to_pass" not in codes(vs)
    # the doubler's side opponent passing is not a conversion of OUR double
    r = feed_all(["1H", X, P, P])
    vs = enf.check(r, 0, HandInfo("982.543.AK42.KJ4"), B("P"))
    assert "to_pass" not in codes(vs)


def test_penalty_pull_discipline():
    enf, _ = make_enforcer([PEN.raw, TO.raw], overrides={"overcall": 0.0})
    # their 4S -> our X is PENALTY; partner pulling with 3 spades is wrong
    r = feed_all([P, "4S", X, P])
    vs = enf.check(r, 0, HandInfo("KQ2.5.AK982.9842"), B("5C"))
    assert "penalty_pull" in codes(vs)
    # with a singleton spade the pull is fine
    vs = enf.check(r, 0, HandInfo("5.4.AK9842.AK982"), B("5C"))
    assert "penalty_pull" not in codes(vs)


def test_doubler_hand_not_checked_under_to():
    # a TO entry with weight>0 replaces the layer-1 takeout_double check:
    junk = HandInfo("92.543.6542.AKQJ")     # 2 spades, K&R 10.0
    enf, _ = make_enforcer([PEN.raw, TO.raw])
    r = feed_all(["1H"])
    vs = enf.check(r, 1, junk, B("X"))
    assert "takeout_double" not in codes(vs)
    # without the file entry the rule fires
    enf2, _ = make_enforcer(["[general]"])
    vs = enf2.check(r, 1, junk, B("X"))
    assert "takeout_double" in codes(vs)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_desc():
    e, b = match_first([REV], 0, ids(["1C", P, "1H", P]), B("2D"))
    assert render_desc(e, b) == "5+ ♣ 4+ ♦ 16+ pts forcing"
    e = parse_entry("1NT-2C = 4+H OR 4+S stayman ask @! @1.0")
    _, b = match_first([e], 0, ids(["1N", P]), B("2C"))
    assert render_desc(e, b) == "4+ ♥ or 4+ ♠ stayman ask"
    e = parse_entry("...-X = TO takeout")
    assert render_desc(e, {}) == "takeout takeout"
    e = parse_entry("2X = 6-7X p4-12 weak two")
    _, b = match_first([e], 0, [], B("2S"))
    assert render_desc(e, b) == "6-7 ♠ 4-12 pts weak two"


def test_alert_for_call_priority_and_suppression():
    spec = spec_of(
        "[open_1nt_range]",
        "alert = {call}: 15-17 balanced",
        "",
        "1NT = p10-12 balanced @1.0",
    )
    r = feed_all([])
    # the pattern entry wins over the section
    assert spec.alert_for_call(r, B("1N")) == "10-12 pts balanced"
    spec2 = spec_of(
        "[open_1nt_range]",
        "alert = {call}: 15-17 balanced",
        "",
        "1NT = @0",                       # matched but empty: no alert at all
    )
    assert spec2.alert_for_call(r, B("1N")) is None
    spec3 = spec_of("[open_1nt_range]", "alert = {call}: 15-17 balanced")
    assert spec3.alert_for_call(r, B("1N")) == "1NT: 15-17 balanced"


def test_spec_params_and_general():
    spec = spec_of("[general]", "kr_tolerance = 0.75",
                   "to_pass_min_len = 5", "f_pass_weight = 1.0",
                   "", "1X = opening")
    cfg = spec.config()
    assert cfg.kr_tolerance == 0.75
    assert not hasattr(cfg, "to_pass_min_len")       # filtered from the cfg
    p = spec.spec_params
    assert p["to_pass_min_len"] == 5 and p["f_pass_weight"] == 1.0
    assert p["penalty_pull_max_len"] == 2            # default kept
    enf = SystemEnforcer(spec, cfg)
    assert enf.to_len == 5 and enf.f_w == 1.0
    assert len(spec.entries) == 1


def test_hand_violations_direct():
    e = parse_entry("1X = 5+X p>10")
    binds = {"X": 3}                                   # spades
    hi = HandInfo("AK982.5.42.AK982")                  # 5 spades, K&R ~17
    assert hand_violations(e, binds, hi, 1.0) == []
    hi = HandInfo("AK9.5.AKJ42.AK92")                 # 3 spades
    vs = hand_violations(e, binds, hi, 1.0)
    assert len(vs) == 1 and vs[0].code == "pat:1X"
