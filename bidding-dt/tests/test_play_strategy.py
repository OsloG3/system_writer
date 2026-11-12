import numpy as np
import pytest

from bidding_dt.data.hands import str_to_cards
from bidding_dt.data.vocab import call_to_id
from bidding_dt.play_bot import _Candidate, PlayConfig, PlaySession
from bidding_dt.play_strategy import (_Node, Layout, StrategySearch, macro_key,
                                      merge_layouts, suit_relevance)

HAS_ENDPLAY = True
try:
    import endplay.types as T
    from endplay.dds.solve import SolveMode, solve_board
except ImportError:  # pragma: no cover
    HAS_ENDPLAY = False

# N-S in 4H by North (same fixture deal as test_play_bot). N,E,S,W.
DEAL = ["AT7.KT943.A42.K8", "J6543.75.K8.AQJT", "K982.AQ6.T73.952", "Q.J82.QJ965.7643"]
HANDS = [set(str_to_cards(h).tolist()) for h in DEAL]
CALLS = [call_to_id(c) for c in "1H P 4H P P P".split()]
DECLARER, DENOM_H = 0, 2            # North, hearts (target = 10 tricks)
RANKS = "AKQJT98765432"


def cid(rank_ch, suit_ch):
    return RANKS.index(rank_ch) * 4 + "SHDC".index(suit_ch)


def decl_session(cfg=None, seed=0, model=None):
    dummy = (DECLARER + 2) % 4
    return PlaySession("d", 0, 0, CALLS, DECLARER, DENOM_H,
                       {DECLARER: DEAL[DECLARER], dummy: DEAL[dummy]},
                       model=model,
                       cfg=cfg or PlayConfig(decl_pool=4, min_floor=0),
                       rng=np.random.default_rng(seed))


def swapped_diamond_candidate():
    """A second hidden-hand layout: E's CA and W's H8 are exchanged (plus the
    diamond spots keep the hands 13-cards). It stays consistent with the
    spade plays below, is NOT macro-equivalent to the truth (club ace and
    heart 8 are relevant ranks), and shifts the double-dummy values, so
    weighted vs unweighted evaluation of the pool genuinely differ."""
    e = (set(HANDS[1]) - {cid("A", "C"), cid("8", "D")}
         | {cid("8", "H"), cid("9", "D")})
    w = (set(HANDS[3]) - {cid("8", "H"), cid("9", "D")}
         | {cid("A", "C"), cid("8", "D")})
    return _Candidate(hands={1: e, 3: w})


# ---- low-card canonicalization ---------------------------------------------

def test_spots_below_every_known_card_are_ignorable():
    known = [cid("A", "S"), cid("K", "S")]
    ign = suit_relevance(known, [])
    # Q..2 of spades are all below both visible cards: interchangeable spots
    assert ign[0] == set(range(2, 13))
    # a suit the visible hands no longer hold has no tenace to protect: all spots
    assert ign[1] == set(range(13))


def test_sandwiched_honour_stays_relevant():
    # A,Q visible: the hidden K sits inside the tenace, its POSITION matters
    ign = suit_relevance([cid("A", "S"), cid("Q", "S")], [])
    assert 1 not in ign[0]                     # K relevant
    assert set(range(3, 13)) <= ign[0]         # J..2 are spots below Q
    # with A,9 visible, K,Q,J,T stay relevant (they can beat the 9 or be
    # finessed around it), but every hidden card BELOW the 9 is a spot: it
    # can never beat a visible card, and among spots the trick always goes
    # to the same side whatever their exact identity
    ign2 = suit_relevance([cid("A", "S"), cid("9", "S")], [])
    for r in ("K", "Q", "J", "T"):
        assert RANKS.index(r) not in ign2[0]
    assert set(range(6, 13)) <= ign2[0]        # 8..2 are spots


def test_played_cards_do_not_block_relevance():
    # K,Q gone: the visible A is the only live known card, everything below
    # it that the hidden hands hold is a spot
    ign = suit_relevance([cid("A", "S")], [cid("K", "S"), cid("Q", "S")])
    assert ign[0] == set(range(3, 13))


def test_merge_sums_weights_of_spot_swaps():
    ign = suit_relevance([cid("A", "S"), cid("K", "S")], [])
    l1 = Layout(rem={1: {cid("Q", "S"), cid("5", "S")}, 3: {cid("4", "S")}}, weight=0.5)
    l2 = Layout(rem={1: {cid("J", "S"), cid("4", "S")}, 3: {cid("5", "S")}}, weight=0.25)
    l3 = Layout(rem={1: {cid("Q", "S")}, 3: {cid("J", "S"), cid("4", "S")}}, weight=0.25)
    merged = merge_layouts([l1, l2, l3], [1, 3], ign)
    assert len(merged) == 2                    # l1 == l2 (spot swap), l3 differs (length)
    by_weight = sorted(merged, key=lambda L: -L.weight)
    assert by_weight[0].weight == pytest.approx(0.75)
    assert by_weight[1].weight == pytest.approx(0.25)


def test_relevant_honour_position_is_not_merged():
    ign = suit_relevance([cid("A", "S"), cid("Q", "S")], [])
    kw = Layout(rem={1: {cid("5", "S")}, 3: {cid("K", "S")}}, weight=1.0)
    ke = Layout(rem={1: {cid("K", "S")}, 3: {cid("5", "S")}}, weight=1.0)
    assert macro_key(kw, [1, 3], ign) != macro_key(ke, [1, 3], ign)
    assert len(merge_layouts([kw, ke], [1, 3], ign)) == 2


# ---- objectives ---------------------------------------------------------------

@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_objective_aggregates_ev_and_make():
    s = StrategySearch(T.Denom.nt, 0, 2, [0, 2], [1, 3], target=10,
                       objective="both", make_bonus=1.0)
    totals = [10, 9]           # declarer tricks per layout
    weights = [0.75, 0.25]
    assert s._objective(totals, weights) == pytest.approx(9.75 + 0.75)
    s.objective = "ev"
    assert s._objective(totals, weights) == pytest.approx(9.75)
    s.objective = "make"
    assert s._objective(totals, weights) == pytest.approx(0.75)


# ---- session-level weighting ----------------------------------------------------

@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_layouts_merge_and_weights_normalize():
    s = decl_session(cfg=PlayConfig(decl_pool=4, min_floor=0, strategy=False,
                                    play_temp=0.0))
    true_c = _Candidate(hands={1: set(HANDS[1]), 3: set(HANDS[3])})
    # the true layout three times + the diamond swap once -> weights 3:1
    s.pool = [true_c, true_c, true_c, swapped_diamond_candidate()]
    s.processed = 0
    plays = [(1, cid("6", "S")), (2, cid("2", "S")), (3, cid("Q", "S"))]
    s._process_new_plays(plays)
    assert len(s.pool) == 4
    info = s._analyze(plays)
    known_rem, legal = s._legal_for(info, DECLARER)
    layouts = s._weighted_layouts(info, known_rem)
    assert len(layouts) == 2
    assert sum(L.weight for L in layouts) == pytest.approx(1.0)
    assert sorted(round(L.weight, 6) for L in layouts) == [0.25, 0.75]


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_pool_rank_is_the_weighted_expectation():
    s = decl_session(cfg=PlayConfig(decl_pool=4, min_floor=0, strategy=False,
                                    play_temp=0.0))
    true_c = _Candidate(hands={1: set(HANDS[1]), 3: set(HANDS[3])})
    s.pool = [true_c, true_c, true_c, swapped_diamond_candidate()]
    s.processed = 0
    # East leads a spade, South and West follow: declarer (North) acts fourth
    plays = [(1, cid("6", "S")), (2, cid("2", "S")), (3, cid("Q", "S"))]
    s._process_new_plays(plays)
    info = s._analyze(plays)
    known_rem, legal = s._legal_for(info, DECLARER)
    layouts = s._weighted_layouts(info, known_rem)
    heavy, light = layouts[0], layouts[1]        # sorted by weight desc
    assert heavy.weight == pytest.approx(0.75)

    # independent per-layout double-dummy values for each legal card
    def tricks_for(layout):
        remaining = dict(known_rem)
        remaining.update(layout.rem)
        deal = s._build_deal(remaining, info["leader"], info["cur"])
        out = {}
        for card, tricks in solve_board(deal, SolveMode.Default):
            out[RANKS.index(card.rank.abbr) * 4 + int(card.suit)] = int(tricks)
        return out

    ta, tb = tricks_for(heavy), tricks_for(light)
    avg = s._pool_rank(info, known_rem, legal, layouts)
    assert legal == {cid("A", "S"), cid("T", "S"), cid("7", "S")}
    for c in legal:
        assert avg[c] == pytest.approx(0.75 * ta[c] + 0.25 * tb[c])


# ---- strategy search -------------------------------------------------------------

STRAT_FAST = PlayConfig(decl_pool=1, min_floor=0, strategy=True,
                        strat_layouts=4, strat_beam=13, strat_depth=1,
                        strat_def_depth=1, strat_def_branch=4,
                        strat_solves=2000, strat_time=60.0,
                        strat_objective="ev", weight_temp=0.0,
                        play_temp=0.0)


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_single_layout_search_matches_double_dummy():
    """With one layout (and every card expanded one trick deep) the plan's
    first card must be a double-dummy best card on the real deal."""
    s = decl_session(cfg=STRAT_FAST)
    s.pool = [_Candidate(hands={1: set(HANDS[1]), 3: set(HANDS[3])})]
    s.processed = 0
    plays = [(1, cid("6", "S")), (2, cid("2", "S")), (3, cid("Q", "S"))]
    card = s.choose(plays, DECLARER)
    assert s.last_search is not None and not s.last_search["exhausted"]
    # independent: solve the true deal at the same mid-trick position
    info = s._analyze(plays)
    known_rem, legal = s._legal_for(info, DECLARER)
    remaining = {0: known_rem[0], 2: known_rem[2],
                 1: set(HANDS[1]) - {cid("6", "S")}, 3: set(HANDS[3]) - {cid("Q", "S")}}
    deal = s._build_deal(remaining, info["leader"], info["cur"])
    res = {}
    for c, t in solve_board(deal, SolveMode.Default):
        res[RANKS.index(c.rank.abbr) * 4 + int(c.suit)] = int(t)
    best = max(res[c] for c in legal)
    assert card in legal and res[card] == best


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_search_prefers_the_weighted_majority_layout():
    """Two layouts weighted 3:1; at depth 1 the search value of a card is
    exactly its weighted double-dummy expectation, so the plan must anchor
    on a card that is optimal for the WEIGHTED set (recomputed here with
    independent per-layout solves), not for the unweighted pool."""
    s = decl_session(cfg=STRAT_FAST)
    true_c = _Candidate(hands={1: set(HANDS[1]), 3: set(HANDS[3])})
    s.pool = [true_c, true_c, true_c, swapped_diamond_candidate()]
    s.processed = 0
    plays = [(1, cid("6", "S")), (2, cid("2", "S")), (3, cid("Q", "S"))]
    s._process_new_plays(plays)
    info = s._analyze(plays)
    known_rem, legal = s._legal_for(info, DECLARER)
    layouts = s._weighted_layouts(info, known_rem)
    search = StrategySearch(
        s.trump, DECLARER, s.dummy, s.known_seats, s.hidden_seats,
        max_layouts=4, beam=13, depth=1, def_depth=1,
        solve_budget=2000, time_budget=60.0, objective="ev", target=s.target)
    fresh = [Layout(rem=dict(L.rem), weight=L.weight) for L in layouts]
    card, diag = search.solve(known_rem, fresh, info["leader"], info["cur"],
                              info["n_done"], info["tricks_decl"], DECLARER,
                              legal, frozenset(info["played_all"]))
    assert diag["layouts"] == 2 and not diag["exhausted"]

    def tricks_for(layout):
        remaining = dict(known_rem)
        remaining.update(layout.rem)
        deal = s._build_deal(remaining, info["leader"], info["cur"])
        out = {}
        for c, t in solve_board(deal, SolveMode.Default):
            out[RANKS.index(c.rank.abbr) * 4 + int(c.suit)] = int(t)
        return out

    heavy, light = layouts[0], layouts[1]
    ta, tb = tricks_for(heavy), tricks_for(light)
    scored = {c: 0.75 * ta[c] + 0.25 * tb[c] for c in legal}
    top = max(scored.values())
    assert card in legal
    # the search's chosen card attains the best weighted expectation
    assert scored[card] == pytest.approx(top, abs=1e-9)
    # and the plan's own value equals that expectation (depth 1 == DD leaf)
    assert diag["value"] == pytest.approx(top, abs=1e-9)
    # sanity: the two layouts really disagree somewhere, so weights matter
    assert any(ta[c] != tb[c] for c in legal)


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_fast_onestep_matches_per_card_path():
    """The one-solve-per-layout one-step must return exactly the legacy
    per-(layout, card) objective values while sending a |cards|x smaller
    batch to the solver."""
    s = decl_session(cfg=STRAT_FAST)
    s.pool = [_Candidate(hands={1: set(HANDS[1]), 3: set(HANDS[3])})]
    s.processed = 0
    plays = [(1, cid("6", "S")), (2, cid("2", "S")), (3, cid("Q", "S"))]
    s._process_new_plays(plays)
    info = s._analyze(plays)
    known_rem, legal = s._legal_for(info, DECLARER)
    layouts = s._weighted_layouts(info, known_rem)

    def onestep(fast):
        search = StrategySearch(
            s.trump, DECLARER, s.dummy, s.known_seats, s.hidden_seats,
            max_layouts=4, beam=13, depth=1, def_depth=1,
            solve_budget=2000, time_budget=60.0, objective="ev",
            target=s.target, fast_onestep=fast)
        # solve() stamps the trick-so-far onto every layout before building
        # the root; _position_for needs it to exclude the tabled cards
        root = search._node(
            known_rem,
            [Layout(rem=dict(L.rem), weight=L.weight, cur=list(info["cur"]))
             for L in layouts],
            info["leader"], info["n_done"], info["tricks_decl"],
            DECLARER, frozenset(info["played_all"]))
        return search, root, search._onestep_decl_cards(root, sorted(legal))

    sf, rf, fast_scores = onestep(True)
    sl, rl, slow_scores = onestep(False)
    assert set(fast_scores) == set(slow_scores) == set(legal)
    for c in legal:
        assert fast_scores[c] == pytest.approx(slow_scores[c])
    assert sf.solves == len(rf.layouts)                 # one board per layout
    assert sl.solves == len(rl.layouts) * len(legal)    # legacy: per card too


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_ceiling_skip_short_circuits_runout():
    """Two tricks left, declarer holds the top spades: the one-step already
    proves every remaining trick, so solve() must return at the ceiling
    without expanding a single search node."""
    a_s, k_s = cid("A", "S"), cid("K", "S")
    known_rem = {0: {a_s, k_s}, 2: {cid("3", "H"), cid("2", "H")}}
    rem = {1: {cid("6", "S"), cid("5", "S")}, 3: {cid("8", "S"), cid("7", "S")}}
    search = StrategySearch(T.Denom.nt, 0, 2, [0, 2], [1, 3],
                            beam=4, depth=4, def_depth=2,
                            solve_budget=500, time_budget=30.0,
                            objective="ev", rng=None)
    lay = Layout(rem={s: set(rem[s]) for s in rem}, weight=1.0)
    card, info = search.solve(known_rem, [lay], 0, [], 11, 9, 0,
                              {a_s, k_s}, frozenset())
    assert card in {a_s, k_s}
    assert info["value"] == pytest.approx(11.0)   # 9 banked + both remaining
    assert info["nodes"] == 0                     # no beam expansion happened
    assert search.solves == 1                     # just the one-step board


def _def_node(to_act, cur):
    """A minimal defender node/layout pair for _smart_pick tests (NT)."""
    search = StrategySearch(T.Denom.nt, 0, 2, [0, 2], [1, 3])
    lay = Layout(rem={1: set(), 3: set()}, weight=1.0, cur=list(cur))
    node = _Node({}, [lay], 0, 0, 0, to_act, frozenset(), {})
    return search, node, lay


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_smart_pick_follows_table_rules():
    # (a) last to play, declarer side winning cheap -> take it (cheapest winner)
    search, node, lay = _def_node(3, [(0, cid("5", "S")), (1, cid("2", "S")),
                                      (2, cid("3", "S"))])
    allowed = {cid("K", "S"), cid("4", "S")}
    assert search._smart_pick(node, lay, allowed) == cid("K", "S")
    # (b) partner is winning -> never overtake, play low
    search, node, lay = _def_node(3, [(1, cid("A", "S")), (2, cid("3", "S")),
                                      (0, cid("5", "S"))])
    assert search._smart_pick(node, lay, allowed) == cid("4", "S")
    # (c) covering an honour just played by the declaring side (partner has
    #     not played yet) -> cover with the cheapest winner
    search, node, lay = _def_node(3, [(2, cid("Q", "S"))])
    assert search._smart_pick(node, lay, allowed) == cid("K", "S")
    # (c2) an ace never covers -- it wins whenever it is played, so it is held;
    #      but last to play, the lone ace does take the trick
    ace_only = {cid("A", "S"), cid("4", "S")}
    search, node, lay = _def_node(3, [(2, cid("Q", "S"))])
    assert search._smart_pick(node, lay, ace_only) == cid("4", "S")
    search, node, lay = _def_node(3, [(0, cid("Q", "S")), (1, cid("2", "S")),
                                      (2, cid("3", "S"))])
    assert search._smart_pick(node, lay, ace_only) == cid("A", "S")
    # (d) second hand, LOW card from declarer side -> second hand low
    search, node, lay = _def_node(3, [(2, cid("5", "S"))])
    assert search._smart_pick(node, lay, allowed) == cid("4", "S")
    # (e) cannot beat the trick -> cheapest card, no honour wasted
    search, node, lay = _def_node(3, [(0, cid("A", "S"))])
    assert search._smart_pick(node, lay, allowed) == cid("4", "S")


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_smart_pick_ruffs_cheapest_trump():
    # hearts trump; defender is out of the led suit and ruffs with the lowest
    search = StrategySearch(T.Denom.hearts, 0, 2, [0, 2], [1, 3])
    lay = Layout(rem={1: set(), 3: set()}, weight=1.0,
                 cur=[(0, cid("5", "S")), (1, cid("2", "S")),
                      (2, cid("3", "S"))])
    node = _Node({}, [lay], 0, 0, 0, 3, frozenset(), {})
    allowed = {cid("2", "H"), cid("K", "H")}
    assert search._smart_pick(node, lay, allowed) == cid("2", "H")


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_search_respects_the_solve_budget():
    s = decl_session(cfg=STRAT_FAST)
    s.pool = [_Candidate(hands={1: set(HANDS[1]), 3: set(HANDS[3])})]
    s.processed = 0
    plays = [(1, cid("6", "S")), (2, cid("2", "S")), (3, cid("Q", "S"))]
    s._process_new_plays(plays)
    info = s._analyze(plays)
    known_rem, legal = s._legal_for(info, DECLARER)
    layouts = s._weighted_layouts(info, known_rem)
    search = StrategySearch(
        s.trump, DECLARER, s.dummy, s.known_seats, s.hidden_seats,
        max_layouts=4, beam=13, depth=4, def_depth=2,
        solve_budget=1, time_budget=60.0, objective="ev", target=s.target)
    fresh = [Layout(rem=dict(L.rem), weight=L.weight) for L in layouts]
    card, diag = search.solve(known_rem, fresh, info["leader"], info["cur"],
                              info["n_done"], info["tricks_decl"], DECLARER,
                              legal, frozenset(info["played_all"]))
    assert card in legal
    assert diag["exhausted"]
    assert search.solves <= 1 + len(legal) * len(layouts)   # one soft overshoot


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_search_refuses_a_defender_root():
    s = decl_session(cfg=STRAT_FAST)
    search = StrategySearch(s.trump, DECLARER, s.dummy, s.known_seats,
                            s.hidden_seats)
    layout = Layout(rem={1: set(HANDS[1]), 3: set(HANDS[3])}, weight=1.0)
    with pytest.raises(ValueError):
        search.solve({DECLARER: HANDS[DECLARER], 2: HANDS[2]}, [layout],
                     1, [], 0, 0, 1, {cid("6", "S")}, frozenset())


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_search_randomizes_tied_best_cards():
    """Two equal honours with equal plan values must not always pick the
    higher one: the tie is broken at random (per-decision rng)."""
    a_s, k_s = cid("A", "S"), cid("K", "S")
    known_rem = {0: {a_s, k_s}, 2: {cid("3", "S"), cid("2", "S")}}
    rem = {1: {cid("6", "S"), cid("5", "S")}, 3: {cid("8", "S"), cid("7", "S")}}
    picks = set()
    for seed in range(12):
        search = StrategySearch(T.Denom.nt, 0, 2, [0, 2], [1, 3],
                                beam=4, depth=2, def_depth=2,
                                solve_budget=500, time_budget=30.0,
                                objective="ev", rng=np.random.default_rng(seed))
        lay = Layout(rem={s: set(rem[s]) for s in rem}, weight=1.0)
        card, info = search.solve(known_rem, [lay], 0, [], 11, 9, 0,
                                  {a_s, k_s}, frozenset())
        assert card in {a_s, k_s}
        assert info["value"] == pytest.approx(11.0)   # both cards take the rest
        picks.add(card)
    assert picks == {a_s, k_s}


# ---- model-based weighting --------------------------------------------------------

@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_model_weights_are_positive_and_normalized():
    import torch
    from bidding_dt.hand.config import HandVQConfig
    from bidding_dt.hand.model import build_hand_model
    from bidding_dt.hand.data import rotated_tokens

    torch.manual_seed(0)
    cfgm = HandVQConfig(d_model=64, d_ff=160, n_heads=4, n_enc_layers=2,
                        n_slot_layers=1, n_dec_layers=2, n_codes=4,
                        codebook_size=32, code_dim=16, revive_every=0)
    model = build_hand_model(cfgm).eval()
    tok = torch.from_numpy(rotated_tokens(0, CALLS, 1)[None])
    with torch.no_grad():
        model.init_codebook(tok, torch.zeros(1, dtype=torch.long))

    s = decl_session(cfg=PlayConfig(decl_pool=8, min_floor=2, strategy=False,
                                    weight_temp=1.0), seed=5, model=model)
    s._topup(8, s._analyze([]))
    info = s._analyze([])
    known_rem = {DECLARER: set(HANDS[DECLARER]), 2: set(HANDS[2])}
    w = s._model_weights(info)
    assert w.shape == (len(s.pool),) and np.all(w > 0)
    layouts = s._weighted_layouts(info, known_rem)
    assert layouts
    assert sum(L.weight for L in layouts) == pytest.approx(1.0)
    # weighting is not degenerate: at least two distinct weights in the pool
    assert len({round(float(x), 6) for x in w}) >= 2
