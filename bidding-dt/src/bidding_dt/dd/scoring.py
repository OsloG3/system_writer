"""Duplicate bridge scoring and IMP conversion (pure python, no deps).

Scores are from the declarer's side perspective unless stated otherwise.
Denominations use the project order CDHSN (see data/vocab.py): 0=C 1=D 2=H
3=S 4=N. Penalty: 0=undoubled, 1=doubled, 2=redoubled.
"""

PENALTY_NONE = 0
PENALTY_DOUBLE = 1
PENALTY_REDOUBLE = 2

_MINOR = (0, 1)  # C, D in project denom order


def declarer_is_vul(declarer_seat: int, vuln: int) -> bool:
    """vuln: project index 0=None 1=N-S 2=E-W 3=Both; seats N=0 E=1 S=2 W=3."""
    ns = declarer_seat in (0, 2)
    return vuln == 3 or (vuln == 1 and ns) or (vuln == 2 and not ns)


def contract_score(level: int, denom: int, penalty: int, tricks: int,
                   decl_vul: bool) -> int:
    """Duplicate score for a contract taking `tricks` (0..13) total tricks."""
    if not 1 <= level <= 7:
        raise ValueError(f"bad level {level}")
    result = tricks - (level + 6)
    if result < 0:
        u = -result
        if penalty == PENALTY_NONE:
            return -(100 if decl_vul else 50) * u
        if penalty == PENALTY_DOUBLE:
            if decl_vul:
                return -200 * u
            per = [100, 200, 200] + [300] * 10
            return -sum(per[:u])
        if decl_vul:
            return -400 * u
        per = [200, 400, 400] + [600] * 10
        return -sum(per[:u])

    trick_score = 20 * level if denom in _MINOR else 30 * level
    if denom == 4:
        trick_score += 10  # NT: 40 for the first trick, 30 each after
    mult = (1, 2, 4)[penalty]
    total = trick_score * mult
    if total >= 100:
        total += 500 if decl_vul else 300
    else:
        total += 50
    if level == 6:
        total += 750 if decl_vul else 500
    elif level == 7:
        total += 1500 if decl_vul else 1000
    if penalty == PENALTY_DOUBLE:
        total += 50
    elif penalty == PENALTY_REDOUBLE:
        total += 100
    if penalty == PENALTY_NONE:
        total += (20 if denom in _MINOR else 30) * result
    elif penalty == PENALTY_DOUBLE:
        total += (200 if decl_vul else 100) * result
    else:
        total += (400 if decl_vul else 200) * result
    return total


# Standard IMP table: (point threshold, imps); the last entry covers all above.
_IMP_STEPS = (
    (20, 0), (50, 1), (90, 2), (130, 3), (170, 4), (220, 5), (270, 6),
    (320, 7), (370, 8), (430, 9), (500, 10), (600, 11), (750, 12),
    (900, 13), (1100, 14), (1300, 15), (1500, 16), (1750, 17), (2000, 18),
    (2250, 19), (2500, 20), (3000, 21), (3500, 22), (4000, 23),
)
MAX_IMPS = 24


def points_to_imps(points: int) -> float:
    """Signed IMP value of a point difference."""
    sign = -1 if points < 0 else 1
    p = abs(points)
    imps = MAX_IMPS
    for threshold, value in _IMP_STEPS:
        if p < threshold:
            imps = value
            break
    return sign * imps
