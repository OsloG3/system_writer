import numpy as np
import pytest
import torch

from bidding_dt.data.hands import encode_hand, str_to_cards
from bidding_dt.data.vocab import call_to_id
from bidding_dt.play_bot import (PlayConfig, PlaySession, card_above, card_to_str,
                                 is_honour, str_to_card, suit_of)
from bidding_dt.play_server import Service, ServiceError
from bidding_dt.hand.config import HandVQConfig
from bidding_dt.hand.model import build_hand_model

HAS_ENDPLAY = True
try:
    import endplay  # noqa: F401
    from endplay.utils.play import trick_winner
    import endplay.types as T
except ImportError:  # pragma: no cover
    HAS_ENDPLAY = False

# N-S in 4H by North. N,E,S,W.
DEAL = ["AT7.KT943.A42.K8", "J6543.75.K8.AQJT", "K982.AQ6.T73.952", "Q.J82.QJ965.7643"]
HANDS = [set(str_to_cards(h).tolist()) for h in DEAL]
CALLS = [call_to_id(c) for c in "1H P 4H P P P".split()]
DECLARER, DENOM_H = 0, 2            # North, hearts
RANKS = "AKQJT98765432"


def cid(rank_ch, suit_ch):
    return RANKS.index(rank_ch) * 4 + "SHDC".index(suit_ch)


def tiny_model():
    torch.manual_seed(0)
    cfg = HandVQConfig(d_model=64, d_ff=160, n_heads=4, n_enc_layers=2,
                       n_slot_layers=1, n_dec_layers=2, n_codes=4,
                       codebook_size=32, code_dim=16, revive_every=0)
    m = build_hand_model(cfg).eval()
    # seed the codebook so sampling is not degenerate
    from bidding_dt.hand.data import rotated_tokens
    import numpy as _np
    tok = torch.from_numpy(rotated_tokens(0, CALLS, 1)[None])
    with torch.no_grad():
        m.init_codebook(tok, torch.zeros(1, dtype=torch.long))
    return m


def decl_session(model=None, seed=0, cfg=None):
    dummy = (DECLARER + 2) % 4
    return PlaySession("d", 0, 0, CALLS, DECLARER, DENOM_H,
                       {DECLARER: DEAL[DECLARER], dummy: DEAL[dummy]},
                       model=model, cfg=cfg or PlayConfig(decl_pool=8, min_floor=2),
                       rng=np.random.default_rng(seed))


def def_session(seat, model=None, seed=0, cfg=None):
    dummy = (DECLARER + 2) % 4
    return PlaySession(f"x{seat}", 0, 0, CALLS, DECLARER, DENOM_H,
                       {seat: DEAL[seat], dummy: DEAL[dummy]},
                       model=model, cfg=cfg or PlayConfig(def_pool=8, min_floor=2),
                       rng=np.random.default_rng(seed))


# ---- card helpers ----------------------------------------------------------

def test_card_string_roundtrip():
    for c in range(52):
        assert str_to_card(card_to_str(c)) == c
    assert card_to_str(cid("A", "S")) == "AS"
    assert card_to_str(cid("9", "H")) == "9H"
    # suit+rank (endplay style) is accepted too
    assert str_to_card("SA") == cid("A", "S")


def test_card_above_and_honour():
    assert card_above(cid("2", "S")) == cid("3", "S")
    assert card_above(cid("K", "C")) == cid("A", "C")
    assert card_above(cid("A", "D")) is None
    assert is_honour(cid("T", "S")) and is_honour(cid("A", "S"))
    assert not is_honour(cid("9", "S")) and not is_honour(cid("2", "H"))


# ---- session basics --------------------------------------------------------

@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_session_seats_and_opener():
    s = decl_session()
    assert s.known_seats == [0, 2]
    assert s.hidden_seats == [1, 3]
    assert s.dummy == 2 and s.opener == 1        # declarer N -> dummy S, opener E


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_opening_lead_is_legal_defender_card():
    s = def_session(1, seed=3)
    card = s.choose([], 1)                        # East opens
    assert card in HANDS[1]


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_candidates_partition_the_unseen_cards():
    s = decl_session(seed=5)
    s._process_new_plays([])
    info = s._analyze([])
    s._topup(6, info)
    unseen = set(range(52)) - set(HANDS[0]) - set(HANDS[2])
    for cand in s.pool:
        e, w = cand.hands[1], cand.hands[3]
        assert len(e) == 13 and len(w) == 13
        assert not (e & w)
        assert e | w == unseen


# ---- consistency filtering (reuse across tricks) ---------------------------

@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_pool_filters_to_hands_that_hold_the_played_card():
    s = decl_session(seed=7)
    s._topup(24, s._analyze([]))                  # a pool at trick one
    before = len(s.pool)
    e_lead = max(c for c in HANDS[1] if suit_of(c) == 0)   # East's top spade
    s._process_new_plays([(1, e_lead)])
    assert 0 < len(s.pool) < before               # only hands that held it survive
    for cand in s.pool:
        assert e_lead not in cand.hands[1]        # played, so removed
        assert len(cand.hands[1]) == 12           # East is down to 12 cards


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_showout_drops_hands_that_could_have_followed():
    from bidding_dt.play_bot import _Candidate
    s = decl_session(seed=11)
    spade = cid("5", "S")
    heart = cid("5", "H")
    # one candidate where West still holds a spade, one where West is spade-void
    with_spade = _Candidate(hands={1: {cid("J", "S")}, 3: {spade}})
    void_spade = _Candidate(hands={1: {cid("J", "S")}, 3: {heart}})
    s.pool = [with_spade, void_spade]
    s.processed = 0
    # East leads a spade, South plays one, West discards a heart (a show-out)
    s._process_new_plays([(1, cid("J", "S")), (2, cid("2", "S")), (3, heart)])
    assert len(s.pool) == 1                        # the spade-holding hand is gone
    assert s.pool[0].hands[3] == set()             # West's heart was removed
    assert 0 in s.showouts[3]                       # West recorded void in spades


# ---- defender lead rule + inference ----------------------------------------

@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_lead_rule_keeps_top_of_touching_honours():
    s = def_session(1, seed=1)
    rem = {cid("K", "S"), cid("Q", "S"), cid("2", "S")}
    allowed = s._apply_lead_rule(set(rem), rem, is_defender=True, is_lead=True)
    assert cid("Q", "S") not in allowed          # holds the K above -> lead the K
    assert cid("K", "S") in allowed and cid("2", "S") in allowed
    # not applied when following, or for declarer
    assert s._apply_lead_rule(set(rem), rem, is_defender=True, is_lead=False) == rem
    assert s._apply_lead_rule(set(rem), rem, is_defender=False, is_lead=True) == rem


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_honour_lead_denies_the_card_above():
    s = decl_session(seed=13)
    q_clubs = cid("Q", "C")
    s._apply_play(1, q_clubs, None, is_lead=True)     # East leads the Q of clubs
    assert card_above(q_clubs) in s.lead_denials[1]    # the K of clubs is denied
    assert card_above(q_clubs) == cid("K", "C")
    # a low lead denies nothing
    s2 = decl_session(seed=13)
    s2._apply_play(1, cid("5", "S"), None, is_lead=True)
    assert s2.lead_denials[1] == set()


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_generated_hands_respect_lead_denial():
    s = decl_session(seed=17, cfg=PlayConfig(decl_pool=10, min_floor=2, topup_tries=8))
    q_clubs = cid("Q", "C")
    s._apply_play(1, q_clubs, None, is_lead=True)
    info = s._analyze([(1, q_clubs)])
    s.processed = 1
    s._topup(10, info)
    assert s.pool
    for cand in s.pool:
        assert cid("K", "C") not in cand.hands[1]      # East never holds the K


# ---- full hand plays out legally -------------------------------------------

def _play_out(model, cfg, seed):
    dummy = (DECLARER + 2) % 4
    trump = T.Denom((3, 2, 1, 0, 4)[DENOM_H])
    ds = PlaySession("d", 0, 0, CALLS, DECLARER, DENOM_H,
                     {DECLARER: DEAL[DECLARER], dummy: DEAL[dummy]}, model=model,
                     cfg=cfg, rng=np.random.default_rng(seed))
    xs = {s: PlaySession(f"x{s}", 0, 0, CALLS, DECLARER, DENOM_H,
                         {s: DEAL[s], dummy: DEAL[dummy]}, model=model, cfg=cfg,
                         rng=np.random.default_rng(seed + s))
          for s in (1, 3)}
    rem = [set(h) for h in HANDS]
    plays, to_act, decl_tricks = [], (DECLARER + 1) % 4, 0
    for i in range(52):
        sess = ds if to_act in (DECLARER, dummy) else xs[to_act]
        card = sess.choose(plays, to_act)
        assert card in rem[to_act], f"seat {to_act} played a card it does not hold"
        if i % 4:
            lead = suit_of(plays[(i // 4) * 4][1])
            if any(suit_of(c) == lead for c in rem[to_act]):
                assert suit_of(card) == lead, f"seat {to_act} revoked"
        rem[to_act].discard(card)
        plays.append((to_act, card))
        if len(plays) % 4 == 0:
            trick = plays[-4:]
            cards = [T.Card(suit=T.Denom(c % 4), rank=T.Rank.find(RANKS[c // 4]))
                     for _, c in trick]
            w = int(trick_winner(cards, T.Player(trick[0][0]), trump))
            decl_tricks += w in (DECLARER, dummy)
            to_act = w
        else:
            to_act = (to_act + 1) % 4
    return decl_tricks


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_full_hand_fallback():
    cfg = PlayConfig(decl_pool=8, def_pool=6, min_floor=2)
    tricks = _play_out(None, cfg, seed=2)
    assert 0 <= tricks <= 13


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_full_hand_model():
    cfg = PlayConfig(decl_pool=8, def_pool=6, min_floor=2)
    tricks = _play_out(tiny_model(), cfg, seed=2)
    assert 0 <= tricks <= 13


# ---- sidecar endpoints -----------------------------------------------------

@pytest.fixture(scope="module")
def svc(tmp_path_factory):
    return Service([], dd_cache=tmp_path_factory.mktemp("cache") / "dd.sqlite")


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_play_choose_endpoint(svc):
    body = {"session": "t:def1", "dealer": 0, "vuln": 0,
            "calls": "1H P 4H P P P".split(),
            "known": {"1": DEAL[1], "2": DEAL[2]}, "plays": [], "to_act": 1}
    out = svc.play_choose(body)
    assert out["seat"] == 1 and str_to_card(out["card"]) in HANDS[1]
    # to_act must be a seat the bot can see
    with pytest.raises(ServiceError):
        svc.play_choose({**body, "to_act": 0})
    # a passed-out auction has no play
    with pytest.raises(ServiceError):
        svc.play_choose({**body, "calls": ["P", "P", "P", "P"]})


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_play_result_endpoint(svc):
    body = {"dealer": 0, "vuln": 2, "calls": "1H P 4H P P P".split(),
            "hands": DEAL, "tricks": 10}
    out = svc.play_result(body)
    assert out["contract"]["declarer"] == 0 and out["contract"]["denom"] == "H"
    assert out["tricks"] == 10 and out["dd_tricks"] == 10
    assert out["score_ns"] == 420 and out["par_ns"] == 430
    down = svc.play_result({**body, "tricks": 8})
    assert down["score_ns"] == -100 and down["imps"] < 0
    with pytest.raises(ServiceError):
        svc.play_result({**body, "tricks": 14})
