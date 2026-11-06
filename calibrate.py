from __future__ import annotations
import numpy as np

from schema import ObjectClass
from simulator.targets import SwarmScenario
from simulator.sensors import SensorSuite
from tracker.calibration import VoteCalibration, fit, save
from tracker.track_manager import DRONE_CLASSES

DT = 0.5
DURATION_S = 120.0
CAL_SEEDS = range(1001, 1031)


def harvest(seeds=CAL_SEEDS, uninformative: bool = False, n_drones: int = 5, n_decoys: int = 3):
    votes = []
    for seed in seeds:
        timeline = SwarmScenario(n_drones=n_drones, n_decoys=n_decoys, dt=DT, duration_s=DURATION_S, seed=seed).generate()
        suite = SensorSuite(np.random.default_rng(seed))
        suite.configure(uninformative_conf=uninformative)
        for k, truths in enumerate(timeline):
            for dets in suite.observe(truths, k * DT, DT).values():
                for d in dets:
                    if not d.classification:
                        continue
                    c = d.classification[0]
                    if c.object_class in DRONE_CLASSES:
                        drone_vote = True
                    elif c.object_class == ObjectClass.BIRD:
                        drone_vote = False
                    else:
                        continue
                    best = min(truths, key=lambda g: np.hypot(g.east_m - d.location.east_m, g.north_m - d.location.north_m))
                    dist = np.hypot(best.east_m - d.location.east_m, best.north_m - d.location.north_m)
                    if dist > 5 * (d.location.east_error_m or 10.0):
                        continue
                    votes.append((d.node_type.value, drone_vote, c.confidence, best.is_threat))
    return votes


def fit_for_world(uninformative: bool = False) -> VoteCalibration:
    return fit(harvest(uninformative=uninformative))


if __name__ == "__main__":
    cal = fit_for_world(False)
    save(cal)
    print(cal.table)
