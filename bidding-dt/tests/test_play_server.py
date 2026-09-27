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
