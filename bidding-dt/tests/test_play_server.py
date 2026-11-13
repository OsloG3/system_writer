import numpy as np
import pytest

from bidding_dt.play_server import Service, ServiceError, parse_ckpt_specs

HAS_ENDPLAY = True
try:
    import endplay  # noqa: F401
except ImportError:  # pragma: no cover
    HAS_ENDPLAY = False

DEAL1 = ["AT7.KT943.A42.K8", "J6543.75.K8.AQJT", "K982.AQ6.T73.952", "Q.J82.QJ965.7643"]


@pytest.fixture(scope="module")
def svc(tmp_path_factory):
    # no checkpoints: /deal, /legal and /score work without a model
    return Service([], dd_cache=tmp_path_factory.mktemp("cache") / "dd.sqlite")


def test_parse_ckpt_specs():
    assert parse_ckpt_specs(["runs/tiny/best.pt", "rl=runs/rl_tinyt/best.pt"]) == [
        ("tiny", "runs/tiny/best.pt"),
        ("rl", "runs/rl_tinyt/best.pt"),
    ]


def test_health(svc):
    h = svc.health()
    assert h["ok"] is True and h["models"] == {} and h["default_model"] is None


def test_deal_random(svc):
    d = svc.deal({})
    assert len(d["hands"]) == 4 and 0 <= d["dealer"] <= 3 and 0 <= d["vuln"] <= 3
    for h in d["hands"]:
        assert len(h.split(".")) == 4
        assert sum(len(s) for s in h.split(".")) == 13
    # a full 52-card deck, no duplicates
    cards = [c for h in d["hands"] for suit, cs in zip("SHDC", h.split(".")) for c in cs]
    assert len(cards) == 52


def test_deal_pinned(svc):
    d = svc.deal({"dealer": 2, "vuln": 3})
    assert d["dealer"] == 2 and d["vuln"] == 3
    with pytest.raises(ServiceError):
        svc.deal({"dealer": 4})


def test_legal_opening(svc):
    r = svc.legal({"dealer": 1, "vuln": 0, "calls": []})
    assert r["seat"] == 1 and r["over"] is False
    legal = set(r["legal"])
    assert "P" in legal and "1C" in legal and "7N" in legal
    assert "X" not in legal and "XX" not in legal
    assert len(legal) == 36  # P + 35 bids


def test_legal_over_a_bid(svc):
    r = svc.legal({"dealer": 0, "vuln": 0, "calls": ["1H"]})
    assert r["seat"] == 1  # E to act
    legal = set(r["legal"])
    assert "X" in legal and "1S" in legal and "2C" in legal
    assert "1H" not in legal and "1D" not in legal and "XX" not in legal


def test_legal_redouble(svc):
    # N opens 1H, E doubles: S may redouble
    r = svc.legal({"dealer": 0, "vuln": 0, "calls": ["1H", "X"]})
    assert r["seat"] == 2
    assert "XX" in r["legal"] and "X" not in r["legal"]


def test_legal_over(svc):
    r = svc.legal({"dealer": 0, "vuln": 0, "calls": ["P", "P", "P", "P"]})
    assert r["over"] is True and r["legal"] == []


def test_auction_validation(svc):
    with pytest.raises(ServiceError):
        svc.legal({"dealer": 0, "vuln": 0, "calls": ["1Q"]})
    with pytest.raises(ServiceError):
        svc.legal({"dealer": 0, "vuln": 0, "calls": ["PAD"]})
    with pytest.raises(ServiceError):
        svc.legal({"dealer": 0, "vuln": 7, "calls": []})


def test_wire_calls_cap(svc):
    from bidding_dt.data.parse import MAX_CALLS
    from bidding_dt.play_server import WIRE_MAX_CALLS
    # the site allows longer auctions than the model can read: the non-/bid
    # endpoints accept up to WIRE_MAX_CALLS tokens (Go auto-passes the bots
    # past MAX_CALLS instead of asking /bid)
    long_calls = ["P", "1C"] * ((MAX_CALLS + 2) // 2)   # > MAX_CALLS, not over
    assert len(long_calls) > MAX_CALLS
    r = svc.legal({"dealer": 0, "vuln": 0, "calls": long_calls})
    assert r["over"] is False and "P" in r["legal"]
    with pytest.raises(ServiceError):
        svc.legal({"dealer": 0, "vuln": 0, "calls": ["P"] * (WIRE_MAX_CALLS + 1)})


def test_bid_without_models(svc):
    with pytest.raises(ServiceError) as e:
        svc.bid({"dealer": 0, "vuln": 0, "calls": [], "hands": DEAL1})
    assert e.value.code == 503


def test_hand_validation(svc):
    with pytest.raises(ServiceError):
        svc.score({"dealer": 0, "vuln": 0, "calls": ["P", "P", "P", "P"],
                   "hands": ["AT7.KT943.A42.K8"] * 3 + ["short.hand.xxx.yy"]})


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_score_known_deal(svc, tmp_path):
    body = {"dealer": 0, "vuln": 2, "calls": "1H P 2H P P P".split(), "hands": DEAL1}
    out = svc.score(body)
    assert out["contract"] == {"level": 2, "denom": "H", "declarer": 0, "penalty": 0}
    assert out["tricks"] == 10
    assert out["score_ns"] == 170 and out["par_ns"] == 430
    assert out["imps"] < 0

    passout = svc.score({**body, "calls": ["P", "P", "P", "P"]})
    assert passout["contract"] is None and passout["tricks"] is None
    assert passout["score_ns"] == 0 and passout["par_ns"] == 430
    assert passout["imps"] == pytest.approx(-10.0)  # IMPs(0 - 430)


def _legal_play_out(hands, opener, trump_int):
    """Play all 52 cards in true order (following suit) -> [[seat, 'AS'], ...]."""
    from bidding_dt.data.hands import str_to_cards
    from bidding_dt.dd import fastsolve
    ranks = "AKQJT98765432"
    rem = [set(str_to_cards(h).tolist()) for h in hands]
    cs = lambda cid: ranks[cid // 4] + "SHDC"[cid % 4]  # noqa: E731
    plays, leader = [], opener
    for _ in range(13):
        trick = []
        for k in range(4):
            s = (leader + k) % 4
            if k == 0:
                card = min(rem[s])
            else:
                ls = trick[0][1] % 4
                follow = [c for c in rem[s] if c % 4 == ls]
                card = min(follow) if follow else min(rem[s])
            rem[s].discard(card)
            trick.append((s, card))
            plays.append([s, cs(card)])
        leader = trick[fastsolve.trick_winner([c for _, c in trick], trump_int)][0]
    return plays


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_analyze_full_hand(svc):
    from bidding_dt.data.vocab import call_to_id
    from bidding_dt.dd.reward import contract_from_auction
    from bidding_dt.dd.solver import _PROJ_DENOM_TO_ENDPLAY

    calls = "1H P 2H P P P".split()
    contract = contract_from_auction(0, [call_to_id(c) for c in calls])
    trump_int = _PROJ_DENOM_TO_ENDPLAY[contract.denom]
    opener = (contract.declarer + 1) % 4
    plays = _legal_play_out(DEAL1, opener, trump_int)

    out = svc.analyze({"dealer": 0, "vuln": 2, "calls": calls, "hands": DEAL1,
                       "plays": plays})
    assert out["contract"] == {"level": 2, "denom": "H", "declarer": 0, "penalty": 0}
    assert out["target"] == 8 and out["dd_tricks"] == 10
    assert out["n"] == 52 and out["actual_tricks"] is not None
    assert len(out["plays"]) == 52
    # opening-lead invariant: DD-best declarer total == the deal's DD tricks
    assert out["plays"][0]["best_decl"] == out["dd_tricks"]
    for p in out["plays"]:
        assert p["cost"] >= 0 and isinstance(p["mistake"], bool)
        assert p["mistake"] == (p["cost"] > 0)
        if p["decl_side"]:
            assert p["best_decl"] >= p["played_decl"]
        else:
            assert p["best_decl"] <= p["played_decl"]
    for side in ("ns", "ew"):
        assert set(out["optimal"][side]) == {
            "best_tricks", "game", "small_slam", "grand_slam"}
    assert len(out["table"]) == 4 and all(len(r) == 5 for r in out["table"])


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_analyze_rejects_bad_input(svc):
    # passed out -> nothing to analyze
    with pytest.raises(ServiceError):
        svc.analyze({"dealer": 0, "vuln": 0, "calls": ["P"] * 4,
                     "hands": DEAL1, "plays": []})
    # a card the acting seat does not hold (K of spades sits with South)
    with pytest.raises(ServiceError):
        svc.analyze({"dealer": 0, "vuln": 2, "calls": "1H P 2H P P P".split(),
                     "hands": DEAL1, "plays": [[0, "KS"]]})
    # an incomplete deal
    with pytest.raises(ServiceError):
        svc.analyze({"dealer": 0, "vuln": 0, "calls": "1H P 2H P P P".split(),
                     "hands": [DEAL1[0]] * 4, "plays": []})
