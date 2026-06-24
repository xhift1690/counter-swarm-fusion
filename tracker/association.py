"""
tracker/association.py — Data association between predicted tracks and
incoming detections.

Implements gated nearest-neighbor (GNN) assignment via the Hungarian
algorithm, using Mahalanobis distance for gating. This is "JPDA-lite":
true JPDA would maintain probabilistic weights over multiple hypotheses;
GNN instead commits to the single best global assignment each timestep,
which is simpler to reason about and a perfectly reasonable starting point
before adding probabilistic data association.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.kalman import KalmanTrack

# Chi-square 99% threshold for 2 degrees of freedom (gating measurements vs. track predictions).
# Widened from the textbook 95% (5.99) because fused cross-sensor detections carry
# more positional noise than a single sensor's raw detections — too tight a gate
# here was the main cause of track fragmentation (see track_manager.py).
GATE_THRESHOLD_SQ = 9.21


def associate(tracks: list[KalmanTrack], detections_xy: list[np.ndarray], R: np.ndarray
              ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Args:
        tracks: list of KalmanTrack (already predicted to current timestep)
        detections_xy: list of [east, north] measurement vectors
        R: measurement covariance (2x2), assumed shared across detections in this call
           (in practice this varies by sensor; pass per-detection R in a future iteration)

    Returns:
        matches: list of (track_index, detection_index)
        unmatched_tracks: list of track_index with no assigned detection
        unmatched_detections: list of detection_index with no assigned track
    """
    n_tracks = len(tracks)
    n_dets = len(detections_xy)

    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))

    cost = np.full((n_tracks, n_dets), 1e6)
    for i, trk in enumerate(tracks):
        for j, z in enumerate(detections_xy):
            d2 = trk.mahalanobis_distance_sq(z, R)
            if d2 < GATE_THRESHOLD_SQ:
                cost[i, j] = d2

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_tracks = set()
    matched_dets = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 1e6:  # within gate
            matches.append((r, c))
            matched_tracks.add(r)
            matched_dets.add(c)

    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_detections = [j for j in range(n_dets) if j not in matched_dets]
    return matches, unmatched_tracks, unmatched_detections
