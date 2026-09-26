"""Double-dummy solver wrapper around endplay (bundles libdds).

endplay is imported lazily so the rest of the package (pure-python scoring)
works without the `rl` extra installed.

Canonical table layout used across the project: int8 array (4, 5),
rows = seats N,E,S,W (project seat order), cols = denoms C,D,H,S,N
(project vocab order). Values are DD tricks for declarer=row in strain=col.

A sqlite cache keyed by the raw deal bytes (4 hand strings) avoids re-solving
deals across runs, which matters because a full table costs ~ms of CPU each.
"""

import sqlite3
import threading
from pathlib import Path

import numpy as np

# endplay enum ints (probed): Player N=0 E=1 S=2 W=3 matches seat order.
# Denom: spades=0 hearts=1 diamonds=2 clubs=3 nt=4 -> project CDHSN order:
_PROJ_DENOM_TO_ENDPLAY = (3, 2, 1, 0, 4)
_PROJ_VULN_TO_ENDPLAY = ("none", "ns", "ew", "both")

_TABLE_DIM = (4, 5)
_INIT_LOCK = threading.Lock()
_INITED = False
_ep = None  # (types, dds, _dds) module tuple, filled by _endplay()


def _endplay():
    global _ep, _INITED
    if _ep is None:
        try:
            import endplay._dds as _dds
            import endplay.dds as dds
            import endplay.types as T
            from endplay.dds.ddtable import DDTable
        except ImportError as e:
            raise ImportError(
                "endplay is required for DD solving: uv sync --extra rl"
            ) from e
        _ep = (T, dds, _dds, DDTable)
        with _INIT_LOCK:
            if not _INITED:
                _dds.SetMaxThreads(0)  # auto; required on Linux per DDS docs
                _INITED = True
    return _ep


def deal_from_hands(hands4) -> object:
    """hands4: sequence of 4 'S.H.D.C' hand strings in N,E,S,W order."""
    T = _endplay()[0]
    return T.Deal.from_pbn("N:" + " ".join(hands4))


def deal_key(hands4) -> bytes:
    return " ".join(hands4).encode("ascii")


def _np_from_endplay_table(table) -> np.ndarray:
    T = _endplay()[0]
    arr = np.zeros(_TABLE_DIM, dtype=np.int8)
    for proj_den, ep_den in enumerate(_PROJ_DENOM_TO_ENDPLAY):
        den = T.Denom(ep_den)
        for seat, pl in enumerate((T.Player.north, T.Player.east,
                                   T.Player.south, T.Player.west)):
            arr[seat, proj_den] = table[den, pl]
    return arr


def _endplay_from_np(arr: np.ndarray):
    _, _, _dds, DDTable = _endplay()
    data = _dds.ddTableResults()
    for proj_den, ep_den in enumerate(_PROJ_DENOM_TO_ENDPLAY):
        for seat in range(4):
            data.resTable[ep_den][seat] = int(arr[seat, proj_den])
    return DDTable(data)


def dd_table(hands4) -> np.ndarray:
    """Solve one deal -> (4,5) int8 table [seat NESW][denom CDHSN]."""
    dds = _endplay()[1]
    return _np_from_endplay_table(dds.calc_dd_table(deal_from_hands(hands4)))


def dd_tables_batch(hands_list, chunk: int = 40) -> np.ndarray:
    """hands_list: list of 4-hand tuples -> (B,4,5) int8, multithreaded."""
    dds = _endplay()[1]
    out = np.zeros((len(hands_list),) + _TABLE_DIM, dtype=np.int8)
    for i in range(0, len(hands_list), chunk):
        batch = hands_list[i:i + chunk]
        deals = [deal_from_hands(h) for h in batch]
        tables = dds.calc_all_tables(deals)
        for j in range(len(batch)):
            out[i + j] = _np_from_endplay_table(tables[j])
    return out


def par_score(table: np.ndarray, vuln: int, dealer: int) -> int:
    """Par score from the N-S perspective (may be negative), given a
    (4,5) DD table, project vuln index (0..3) and dealer seat (0..3)."""
    T, dds, _, _ = _endplay()
    ep_vul = getattr(T.Vul, _PROJ_VULN_TO_ENDPLAY[vuln])
    players = (T.Player.north, T.Player.east, T.Player.south, T.Player.west)
    return dds.par(_endplay_from_np(table), ep_vul, players[dealer]).score


class TableCache:
    """sqlite cache of (4,5) int8 DD tables keyed by raw deal bytes."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS dd (key BLOB PRIMARY KEY, tbl BLOB)")
        self._db.commit()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: bytes) -> np.ndarray | None:
        with self._lock:
            row = self._db.execute("SELECT tbl FROM dd WHERE key=?",
                                   (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return np.frombuffer(row[0], dtype=np.int8).reshape(_TABLE_DIM).copy()

    def put(self, key: bytes, table: np.ndarray):
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO dd VALUES (?,?)",
                (key, np.ascontiguousarray(table, dtype=np.int8).tobytes()))
            self._db.commit()

    def put_many(self, items):
        with self._lock:
            self._db.executemany(
                "INSERT OR IGNORE INTO dd VALUES (?,?)",
                [(k, np.ascontiguousarray(t, dtype=np.int8).tobytes())
                 for k, t in items])
            self._db.commit()

    def close(self):
        self._db.close()

    def __len__(self):
        return self._db.execute("SELECT COUNT(*) FROM dd").fetchone()[0]
