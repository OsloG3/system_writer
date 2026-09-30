import pytest

from bidding_dt.dd.scoring import (contract_score, declarer_is_vul,
                                   points_to_imps)

HAS_ENDPLAY = True
try:
    import endplay.types as T
except ImportError:  # pragma: no cover
    HAS_ENDPLAY = False

# project denom order CDHSN
C, D, H, S, N = 0, 1, 2, 3, 4


@pytest.mark.parametrize("args,expected", [
    # making, undoubled
    ((4, H, 0, 10, True), 620),
    ((4, H, 0, 10, False), 420),
    ((4, H, 0, 11, False), 450),
    ((3, N, 0, 9, False), 400),
    ((3, N, 0, 9, True), 600),
    ((3, N, 0, 10, False), 430),
    ((2, C, 0, 8, False), 90),
    ((2, C, 0, 9, False), 110),
    ((1, N, 0, 7, False), 90),
    ((1, S, 0, 8, True), 110),
    ((6, S, 0, 12, False), 980),
    ((6, S, 0, 12, True), 1430),
    ((7, N, 0, 13, False), 1520),
    ((7, N, 0, 13, True), 2220),
    ((5, D, 0, 11, True), 600),
    # doubled making
    ((4, S, 1, 10, False), 590),
    ((4, S, 2, 10, True), 1080),
    ((2, H, 1, 8, False), 470),
    ((2, H, 1, 9, False), 570),
    ((2, H, 1, 9, True), 870),
    ((1, C, 1, 7, False), 140),
    ((1, C, 2, 8, True), 630),
    # down undoubled
    ((4, H, 0, 9, False), -50),
    ((4, H, 0, 9, True), -100),
    ((4, H, 0, 8, True), -200),
    # down doubled
    ((4, S, 1, 9, False), -100),
    ((4, S, 1, 8, False), -300),
    ((4, S, 1, 7, False), -500),
    ((4, S, 1, 6, False), -800),
    ((4, S, 1, 9, True), -200),
    ((4, S, 1, 8, True), -500),
    ((4, S, 1, 7, True), -800),
    ((4, S, 1, 6, True), -1100),
    # down redoubled
    ((4, S, 2, 9, False), -200),
    ((4, S, 2, 8, False), -600),
    ((4, S, 2, 7, False), -1000),
    ((4, S, 2, 6, False), -1600),
    ((4, S, 2, 9, True), -400),
    ((4, S, 2, 8, True), -1000),
    ((4, S, 2, 7, True), -1600),
])
def test_contract_score(args, expected):
    assert contract_score(*args) == expected


@pytest.mark.parametrize("points,expected", [
    (0, 0), (10, 0), (19, 0), (20, 1), (49, 1), (50, 2), (120, 3),
    (430, 10), (500, 11), (1750, 18), (4000, 24), (9999, 24),
    (-19, 0), (-20, -1), (-45, -1), (-4001, -24),
])
def test_points_to_imps(points, expected):
    assert points_to_imps(points) == expected


@pytest.mark.parametrize("seat,vuln,expected", [
    (0, 0, False), (0, 1, True), (0, 2, False), (0, 3, True),
    (2, 1, True), (2, 2, False),
    (1, 1, False), (1, 2, True), (3, 3, True), (3, 0, False),
])
def test_declarer_is_vul(seat, vuln, expected):
    assert declarer_is_vul(seat, vuln) is expected


@pytest.mark.skipif(not HAS_ENDPLAY, reason="endplay not installed")
def test_cross_check_endplay_exhaustive():
    """Cross-check against endplay's Contract.score over every contract."""
    denom_map = [T.Denom.clubs, T.Denom.diamonds, T.Denom.hearts,
                 T.Denom.spades, T.Denom.nt]
    pen_map = [T.Penalty.passed, T.Penalty.doubled, T.Penalty.redoubled]
    players = [T.Player.north, T.Player.east, T.Player.south, T.Player.west]
    vuln_map = [T.Vul.none, T.Vul.ns, T.Vul.ew, T.Vul.both]
    n = 0
    for level in range(1, 8):
        for denom in range(5):
            for pen in range(3):
                for tricks in range(0, 14):
                    for decl in (0, 1):  # one NS + one EW declarer
                        for vuln in range(4):
                            is_vul = declarer_is_vul(decl, vuln)
                            mine = contract_score(level, denom, pen, tricks, is_vul)
                            c = T.Contract(level=level, denom=denom_map[denom],
                                           declarer=players[decl],
                                           penalty=pen_map[pen],
                                           result=tricks - (level + 6))
                            assert c.score(vuln_map[vuln]) == mine, (
                                level, denom, pen, tricks, decl, vuln)
                            n += 1
    assert n == 7 * 5 * 3 * 14 * 2 * 4
