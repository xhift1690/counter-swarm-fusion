"""
tracker/kalman.py — Constant-velocity Kalman filter for a single track,
operating in ENU Cartesian coordinates.

State vector: [east, north, east_rate, north_rate]
This is the standard textbook CV (constant velocity) model. It's the right
starting point for this domain: drones maneuver, but over a single radar
revisit interval (~0.5-2s) a CV assumption is a reasonable local
approximation, and any deviation shows up as filter residual that a more
advanced model (e.g. constant-acceleration, or an IMM with multiple motion
models) could pick up later.
"""

from __future__ import annotations
import numpy as np


class KalmanTrack:
    def __init__(self, east_m: float, north_m: float, east_rate_mps: float, north_rate_mps: float,
                 process_noise_std: float = 1.0, initial_pos_std: float = 10.0, initial_vel_std: float = 5.0):
        # State: [e, n, e_dot, n_dot]
        self.x = np.array([east_m, north_m, east_rate_mps, north_rate_mps], dtype=float)
        self.P = np.diag([initial_pos_std**2, initial_pos_std**2, initial_vel_std**2, initial_vel_std**2])
        self.q = process_noise_std  # process noise tuning knob

        self.age = 0
        self.hits = 1
        self.misses = 0
        self.last_update_t: float | None = None

    def F(self, dt: float) -> np.ndarray:
        """State transition matrix for constant-velocity model."""
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    def Q(self, dt: float) -> np.ndarray:
        """Process noise covariance — discretised white-noise acceleration model."""
        q = self.q
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        # Standard discrete white noise acceleration (DWNA) model, per-axis, then combined for 2D
        block = np.array([
            [dt4 / 4, dt3 / 2],
            [dt3 / 2, dt2],
        ]) * q**2
        Q = np.zeros((4, 4))
        Q[0, 0], Q[0, 2] = block[0, 0], block[0, 1]
        Q[2, 0], Q[2, 2] = block[1, 0], block[1, 1]
        Q[1, 1], Q[1, 3] = block[0, 0], block[0, 1]
        Q[3, 1], Q[3, 3] = block[1, 0], block[1, 1]
        return Q

    def predict(self, dt: float):
        F = self.F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q(dt)
        self.age += 1

    def predicted_position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    def innovation(self, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (innovation y, innovation covariance S) for gating, without committing the update."""
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        return y, S

    def mahalanobis_distance_sq(self, z: np.ndarray, R: np.ndarray) -> float:
        y, S = self.innovation(z, R)
        return float(y.T @ np.linalg.inv(S) @ y)

    def update(self, z: np.ndarray, R: np.ndarray):
        """z = [east, north] measurement, R = measurement covariance (2x2)."""
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        y, S = self.innovation(z, R)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.hits += 1
        self.misses = 0

    def mark_missed(self):
        self.misses += 1
