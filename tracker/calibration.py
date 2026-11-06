from __future__ import annotations
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np

EDGES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
MAX_VOTE_LLR = 4.6
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "vote_calibration.json"


class VoteCalibration:
    def __init__(self, table: dict[str, dict[str, list[float]]]):
        self.table = table

    def llr(self, node_type: str, drone_vote: bool, conf: float) -> float | None:
        t = self.table.get(node_type)
        if t is None:
            return None
        b = int(np.searchsorted(EDGES, conf, side="right")) - 1
        b = min(max(b, 0), len(EDGES) - 2)
        return t["drone" if drone_vote else "bird"][b]


def fit(votes, alpha: float = 1.0) -> VoteCalibration:
    """votes: iterable of (node_type, drone_vote, confidence, truth_is_drone)."""
    nb = len(EDGES) - 1
    counts: dict[str, dict[bool, dict[tuple[str, int], int]]] = {}
    for node_type, drone_vote, conf, truth in votes:
        b = min(max(int(np.searchsorted(EDGES, conf, side="right")) - 1, 0), nb - 1)
        cell = ("drone" if drone_vote else "bird", b)
        c = counts.setdefault(node_type, {True: {}, False: {}})[bool(truth)]
        c[cell] = c.get(cell, 0) + 1
    table = {}
    for node_type, c in counts.items():
        k = 2 * nb
        tot = {t: sum(c[t].values()) for t in (True, False)}
        table[node_type] = {"drone": [], "bird": []}
        for d in ("drone", "bird"):
            for b in range(nb):
                p1 = (c[True].get((d, b), 0) + alpha) / (tot[True] + alpha * k)
                p0 = (c[False].get((d, b), 0) + alpha) / (tot[False] + alpha * k)
                table[node_type][d].append(float(np.clip(math.log(p1 / p0), -MAX_VOTE_LLR, MAX_VOTE_LLR)))
    return VoteCalibration(table)


def save(cal: VoteCalibration, path: Path = DEFAULT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"edges": EDGES, "table": cal.table}, indent=2))


@lru_cache(maxsize=1)
def default_calibration() -> VoteCalibration | None:
    if DEFAULT_PATH.exists():
        return VoteCalibration(json.loads(DEFAULT_PATH.read_text())["table"])
    return None
