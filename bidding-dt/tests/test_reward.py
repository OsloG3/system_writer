import pytest

from bidding_dt.data.vocab import call_to_id
from bidding_dt.dd.reward import contract_from_auction, deal_reward

HAS_ENDPLAY = True
try:
    import endplay  # noqa: F401
except ImportError:  # pragma: no cover
    HAS_ENDPLAY = False

DEAL1 = ("AT7.KT943.A42.K8", "J6543.75.K8.AQJT", "K982.AQ6.T73.952", "Q.J82.QJ965.7643")


def ids(auction: str) -> list[int]:
    return [call_to_id(c) for c in auction.split()]


@pytest.mark.parametrize("dealer,auction,expected", [
    (0, "1H P 4H P P P", (4, 2, 0, 0)),          # 4H by N undoubled
    (0, "1N P 2H P P P", (2, 2, 2, 0)),          # transfer: 2H by S
    (0, "1H X P P P", (1, 2, 0, 1)),             # 1H by N doubled by E
    (0, "1H X XX P P P", (1, 2, 0, 2)),          # redoubled
    (0, "4H P P X P P P", (4, 2, 0, 1)),         # X after intervening passes
    (0, "1H 2H P P P", (2, 2, 1, 0)),            # E wins hearts N named first
    (0, "1H 2H X P P P", (2, 2, 1, 1)),          # 2H by E doubled by S
    (1, "P 1C X P P XX P P P", (1, 0, 2, 2)),    # 1C by S XX (W doubled)
    (0, "P P P P", None),                        # passout
    (2, "P P 1S P 2N P P P", (2, 4, 2, 0)),      # 2N by S
])
def test_contract_from_auction(dealer, auction, expected):
    got = contract_from_auction(dealer, ids(auction))
    if expected is None:
        assert got is None
    else:
        assert (got.level, got.denom, got.declarer, got.penalty) == expected


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_deal_reward_known_deal():
    from bidding_dt.dd.scoring import points_to_imps
    # deal 1, dealer N, vuln E-W: human auction 1H P 2H -> 2H by N with
    # 10 DD tricks (+2 over), N-S non-vul: 60 + 50 + 2*30 = 170.
    calls = ids("1H P 2H P P P")
    r = deal_reward(DEAL1, dealer=0, vuln=2, calls=calls)
    assert r.contract is not None
    assert (r.contract.level, r.contract.denom, r.contract.declarer) == (2, 2, 0)
    assert r.tricks == 10
    assert r.score_ns == 170
    assert r.par_ns == 430  # 3NT+1 non-vul by N beats EW sacrifices here
    assert r.reward_imps == points_to_imps(r.score_ns - r.par_ns)
    assert r.reward_imps < 0  # stopping in 2H loses versus par

    # passout scores 0 vs par on the same deal
    r2 = deal_reward(DEAL1, dealer=0, vuln=2, calls=ids("P P P P"))
    assert r2.contract is None and r2.score_ns == 0 and r2.tricks is None
    assert r2.par_ns == r.par_ns
    assert r2.reward_imps == points_to_imps(-r2.par_ns)

    # making the par-ish contract 3NT by N (10 DD NT tricks) scores 430
    r3 = deal_reward(DEAL1, dealer=0, vuln=2, calls=ids("1H P 2H P 3N P P P"))
    assert r3.score_ns == 430


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_deal_reward_cache(tmp_path):
    from bidding_dt.dd.solver import TableCache
    cache = TableCache(tmp_path / "dd.sqlite")
    calls = ids("1H P 4H P P P")
    r1 = deal_reward(DEAL1, dealer=0, vuln=3, calls=calls, cache=cache)
    assert cache.misses == 1 and cache.hits == 0
    r2 = deal_reward(DEAL1, dealer=0, vuln=3, calls=calls, cache=cache)
    assert cache.hits == 1
    assert r1 == r2
    cache.close()


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_warm_cache(tmp_path):
    """Background presolve helper: dedupes, fills the cache, idempotent."""
    from bidding_dt.dd.reward import warm_cache
    from bidding_dt.dd.solver import TableCache
    cache = TableCache(tmp_path / "dd.sqlite")
    n = warm_cache([DEAL1, DEAL1], cache)      # duplicate hands solved once
    assert n == 1
    assert warm_cache([DEAL1], cache) == 0     # already warm -> no solves
    r = deal_reward(DEAL1, dealer=0, vuln=2, calls=ids("1H P 2H P P P"),
                    cache=cache)
    assert r.par_ns == 430                     # scoring reads the warm table
    assert cache.misses == 1 and cache.hits == 2   # first warm missed, re-warm + reward hit
    assert warm_cache([DEAL1], None) == 0      # no cache -> no-op
    cache.close()


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_deal_rewards_dedupes_identical_deals(tmp_path):
    """Two-table batches repeat each deal: solve each distinct deal once."""
    from bidding_dt.dd.reward import RewardRequest, deal_rewards
    from bidding_dt.dd.solver import TableCache
    cache = TableCache(tmp_path / "dd.sqlite")
    req = RewardRequest(DEAL1, 0, 2, ids("1H P 2H P P P"))
    req_pass = RewardRequest(DEAL1, 0, 2, ids("P P P P"))
    rs = deal_rewards([req, req_pass, req], cache=cache)
    assert cache.misses == 1 and cache.hits == 0  # one group, one get, one solve
    assert rs[0] == rs[2]
    assert rs[0].par_ns == rs[1].par_ns and rs[1].score_ns == 0
    deal_rewards([req], cache=cache)
    assert cache.hits == 1
    cache.close()
