"""
simulator/targets.py — Generates ground-truth trajectories for drone swarms
(and decoy birds/clutter) that sensors will then "observe" with noise.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class TargetTruth:
    """Ground truth state for one moving object at one timestep."""
    target_id: str
    east_m: float
    north_m: float
    up_m: float
    east_rate_mps: float
    north_rate_mps: float
    up_rate_mps: float
    is_threat: bool  # True = drone, False = bird/clutter (used only for scoring, not given to tracker)


class SwarmScenario:
    """
    Produces ground-truth trajectories for a swarm of drones approaching a
    protected point, plus a configurable number of bird/clutter decoys.

    Coordinate frame: ENU metres, origin at the protected asset (0, 0, 0).
    Drones start at the edge of a circle of `approach_radius_m` and fly
    roughly toward the origin with some per-drone heading jitter and
    altitude variation, at randomised speeds.
    """

    def __init__(
        self,
        n_drones: int = 5,
        n_decoys: int = 3,
        approach_radius_m: float = 2000.0,
        drone_speed_range_mps: tuple[float, float] = (8.0, 20.0),
        decoy_speed_range_mps: tuple[float, float] = (3.0, 12.0),
        dt: float = 0.5,
        duration_s: float = 180.0,
        seed: int | None = 42,
    ):
        self.dt = dt
        self.n_steps = int(duration_s / dt)
        self.rng = np.random.default_rng(seed)

        self.targets: list[dict] = []

        for i in range(n_drones):
            angle = self.rng.uniform(0, 2 * np.pi)
            speed = self.rng.uniform(*drone_speed_range_mps)
            start_e = approach_radius_m * np.cos(angle)
            start_n = approach_radius_m * np.sin(angle)
            # heading roughly toward origin, with jitter
            heading_to_origin = np.arctan2(-start_n, -start_e)
            heading = heading_to_origin + self.rng.normal(0, 0.15)
            alt = self.rng.uniform(40, 150)
            self.targets.append({
                "target_id": f"drone_{i}",
                "is_threat": True,
                "pos": np.array([start_e, start_n, alt]),
                "vel": np.array([speed * np.cos(heading), speed * np.sin(heading), 0.0]),
                "heading_jitter_std": 0.02,
            })

        for i in range(n_decoys):
            angle = self.rng.uniform(0, 2 * np.pi)
            speed = self.rng.uniform(*decoy_speed_range_mps)
            radius = self.rng.uniform(200, approach_radius_m)
            start_e = radius * np.cos(angle)
            start_n = radius * np.sin(angle)
            heading = self.rng.uniform(0, 2 * np.pi)  # birds wander randomly
            alt = self.rng.uniform(10, 80)
            self.targets.append({
                "target_id": f"decoy_{i}",
                "is_threat": False,
                "pos": np.array([start_e, start_n, alt]),
                "vel": np.array([speed * np.cos(heading), speed * np.sin(heading), 0.0]),
                "heading_jitter_std": 0.25,  # birds wander much more erratically
            })

    def generate(self) -> list[list[TargetTruth]]:
        """Returns a list (length n_steps) of lists of TargetTruth, one per target."""
        timeline: list[list[TargetTruth]] = []
        for _step in range(self.n_steps):
            frame = []
            for t in self.targets:
                # random walk on heading to keep motion plausible but not perfectly linear
                jitter = self.rng.normal(0, t["heading_jitter_std"])
                speed = np.linalg.norm(t["vel"][:2])
                if speed > 1e-6:
                    heading = np.arctan2(t["vel"][1], t["vel"][0]) + jitter
                    t["vel"][0] = speed * np.cos(heading)
                    t["vel"][1] = speed * np.sin(heading)

                t["pos"] = t["pos"] + t["vel"] * self.dt
                t["pos"][2] = max(t["pos"][2], 0.0)  # don't go underground

                frame.append(TargetTruth(
                    target_id=t["target_id"],
                    east_m=t["pos"][0], north_m=t["pos"][1], up_m=t["pos"][2],
                    east_rate_mps=t["vel"][0], north_rate_mps=t["vel"][1], up_rate_mps=t["vel"][2],
                    is_threat=t["is_threat"],
                ))
            timeline.append(frame)
        return timeline
