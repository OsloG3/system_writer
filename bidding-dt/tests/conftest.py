import numpy as np
import pytest

from bidding_dt.data.parse import build_cache

MINI_TRAINING = """\
AT7.KT943.A42.K8 J6543.75.K8.AQJT K982.AQ6.T73.952 Q.J82.QJ965.7643
N E-W 1H P 2H P P P
T953.AT973.KJT.4 2.Q.A842.AKQJT92 AJ7.K654.Q963.83 KQ864.J82.75.765
E N-S 1C P 1S P 2D P 2H P 3C P P P
QT9.AT94.A96.A65 J8.K.QT74.KQJT98 6543.J82.K53.732 AK72.Q7653.J82.4
E None 1C P 1H P 2C P 2N P P P
T987.64.KQ42.972 62.AKQT2.A5.A643 AQ3.7.JT863.QJ85 KJ54.J9853.97.KT
E Both 1H P 4H P P P
"""


@pytest.fixture(scope="session")
def mini_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "mini.txt"
    p.write_text(MINI_TRAINING)
    return p


@pytest.fixture(scope="session")
def mini_cache(mini_file, tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    build_cache(mini_file, cache, split=(0.5, 0.25, 0.25), seed=7)
    return cache
