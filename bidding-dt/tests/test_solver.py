import numpy as np
import pytest

pytest.importorskip("endplay")

from bidding_dt.dd.solver import (TableCache, dd_table, dd_tables_batch,  # noqa: E402
                                  par_score)

DEAL1 = ("AT7.KT943.A42.K8", "J6543.75.K8.AQJT", "K982.AQ6.T73.952", "Q.J82.QJ965.7643")

# Known DD table for DEAL1, rows N,E,S,W x cols C,D,H,S,NT
EXPECTED_TABLE = np.array([
    [5, 5, 10, 9, 10],
    [8, 7, 3, 4, 3],
    [4, 5, 10, 8, 9],
    [8, 7, 3, 4, 3],
], dtype=np.int8)


def test_dd_table_known_deal():
    assert np.array_equal(dd_table(DEAL1), EXPECTED_TABLE)


def test_par_score_known_deal():
    # values verified against endplay's par() on the same table
    assert par_score(EXPECTED_TABLE, 1, 0) == 500   # N-S vuln, dealer N
    assert par_score(EXPECTED_TABLE, 0, 0) == 430   # none vuln, dealer N


def test_batch_matches_serial():
    deals = [DEAL1, (DEAL1[1], DEAL1[2], DEAL1[3], DEAL1[0])]
    serial = np.stack([dd_table(d) for d in deals])
    batched = dd_tables_batch(deals)
    assert np.array_equal(serial, batched)


def test_table_cache_roundtrip(tmp_path):
    cache = TableCache(tmp_path / "dd.sqlite")
    key = " ".join(DEAL1).encode()
    assert cache.get(key) is None
    cache.put(key, EXPECTED_TABLE)
    got = cache.get(key)
    assert got is not None
    assert np.array_equal(got, EXPECTED_TABLE)
    assert got.dtype == np.int8
    assert len(cache) == 1
    # second process-simulating reopen
    cache2 = TableCache(tmp_path / "dd.sqlite")
    assert np.array_equal(cache2.get(key), EXPECTED_TABLE)
    assert cache2.hits == 1 and cache2.misses == 0
    cache.close()
    cache2.close()
