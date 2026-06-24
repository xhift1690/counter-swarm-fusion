"""
tracker/track_manager.py — Orchestrates the full fusion + tracking pipeline
for one timestep:

  1. Cross-sensor fusion: detections from different sensors at the same
     instant that are spatially close get merged into a single fused
     detection (mirrors SAPIENT's associated_detection/derived_detection
     relationship — see schema.py).
  2. Association: fused detections are matched to existing tracks via
     gated nearest-neighbor (tracker/association.py).
  3. Track lifecycle: matched tracks are Kalman-updated; unmatched
     detections spawn new TENTATIVE tracks; tracks with no recent updates
     are eventually dropped.
  4. Classification fusion + threat scoring: per-track classification
     confidence is combined across whichever sensors reported it.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from schema import DetectionReport, ObjectClass, new_id
from tracker.kalman import KalmanTrack


class TrackStatus(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    DROPPED = "DROPPED"


@dataclass
class FusedDetection:
    east_m: float
    north_m: float
    east_rate_mps: float
    north_rate_mps: float
    timestamp: float
    contributing_report_ids: list[str] = field(default_factory=list)
    contributing_node_ids: list[str] = field(default_factory=list)
    classification_votes: list[tuple[ObjectClass, float]] = field(default_factory=list)
    mean_detection_confidence: float = 1.0


@dataclass
class Track:
    track_id: str = field(default_factory=new_id)
    kf: KalmanTrack | None = None
    status: TrackStatus = TrackStatus.TENTATIVE
    history: list[tuple[float, float, float]] = field(default_factory=list)  # (t, e, n)
    class_score: dict[ObjectClass, float] = field(default_factory=dict)  # running classification belief
    last_seen: float = 0.0

    def threat_score(self) -> float:
        """Simple threat score: P(drone) from fused classification belief."""
        drone_mass = (
            self.class_score.get(ObjectClass.DRONE_MULTIROTOR, 0.0)
            + self.class_score.get(ObjectClass.DRONE_FIXED_WING, 0.0)
        )
        total = sum(self.class_score.values()) or 1.0
        return drone_mass / total


# --- Cross-sensor fusion (step 1) -------------------------------------------------

FUSION_DISTANCE_THRESHOLD_M = 25.0  # detections within this distance, same timestep, get merged


def fuse_detections(detections: list[DetectionReport]) -> list[FusedDetection]:
    """Greedy spatial clustering of same-instant, cross-sensor detections."""
    unclustered = list(detections)
    fused: list[FusedDetection] = []

    while unclustered:
        seed = unclustered.pop(0)
        cluster = [seed]
        remaining = []
        for d in unclustered:
            dist = np.hypot(d.location.east_m - seed.location.east_m,
                             d.location.north_m - seed.location.north_m)
            if dist < FUSION_DISTANCE_THRESHOLD_M:
                cluster.append(d)
            else:
                remaining.append(d)
        unclustered = remaining

        # Inverse-variance weighted mean position across the cluster
        weights = np.array([1.0 / (d.location.east_error_m or 10.0) ** 2 for d in cluster])
        weights = weights / weights.sum()
        east = float(np.sum(weights * [d.location.east_m for d in cluster]))
        north = float(np.sum(weights * [d.location.north_m for d in cluster]))
        e_rate = float(np.mean([d.velocity.east_rate_mps for d in cluster if d.velocity]))
        n_rate = float(np.mean([d.velocity.north_rate_mps for d in cluster if d.velocity]))

        class_votes = []
        for d in cluster:
            for c in d.classification:
                class_votes.append((c.object_class, c.confidence))

        fused.append(FusedDetection(
            east_m=east, north_m=north,
            east_rate_mps=e_rate, north_rate_mps=n_rate,
            timestamp=cluster[0].timestamp,
            contributing_report_ids=[d.report_id for d in cluster],
            contributing_node_ids=[d.node_id for d in cluster],
            classification_votes=class_votes,
            mean_detection_confidence=float(np.mean([d.detection_confidence for d in cluster])),
        ))

    return fused


# --- Track manager (steps 2-4) ----------------------------------------------------

class TrackManager:
    def __init__(
        self,
        measurement_std_m: float = 12.0,
        confirm_hits: int = 3,
        drop_after_misses: int = 5,
        duplicate_suppression_distance_m: float = 60.0,
    ):
        self.tracks: list[Track] = []
        self.R = np.diag([measurement_std_m**2, measurement_std_m**2])
        self.confirm_hits = confirm_hits
        self.drop_after_misses = drop_after_misses
        # If an "unmatched" detection is actually still close to an existing track
        # (just outside the statistical gate), treat it as a soft update on that
        # track rather than spawning a duplicate. Without this, sensor-to-sensor
        # fusion noise and process-noise underestimation fragment a single real
        # target into several overlapping tracks (see README "known limitations").
        self.dup_dist = duplicate_suppression_distance_m

    def step(self, raw_detections: list[DetectionReport], timestamp: float, dt: float) -> list[Track]:
        fused = fuse_detections(raw_detections)

        # Predict all existing tracks forward
        for trk in self.tracks:
            if trk.kf is not None:
                trk.kf.predict(dt)

        from tracker.association import associate
        kfs = [t.kf for t in self.tracks]
        det_xy = [np.array([f.east_m, f.north_m]) for f in fused]
        matches, unmatched_tracks, unmatched_dets = associate(kfs, det_xy, self.R)

        # Update matched tracks
        for ti, di in matches:
            trk = self.tracks[ti]
            f = fused[di]
            trk.kf.update(np.array([f.east_m, f.north_m]), self.R)
            trk.last_seen = timestamp
            trk.history.append((timestamp, *trk.kf.predicted_position()))
            self._update_classification(trk, f)
            if trk.status == TrackStatus.TENTATIVE and trk.kf.hits >= self.confirm_hits:
                trk.status = TrackStatus.CONFIRMED

        # Penalize unmatched tracks
        for ti in unmatched_tracks:
            trk = self.tracks[ti]
            trk.kf.mark_missed()
            if trk.kf.misses >= self.drop_after_misses:
                trk.status = TrackStatus.DROPPED

        # Spawn new tentative tracks from unmatched detections — unless one is
        # actually still close to an existing track (duplicate suppression).
        live_tracks = [t for t in self.tracks if t.status != TrackStatus.DROPPED]
        for di in unmatched_dets:
            f = fused[di]
            nearest_dist = min(
                (np.hypot(f.east_m - t.kf.x[0], f.north_m - t.kf.x[1]) for t in live_tracks),
                default=float("inf"),
            )
            if nearest_dist < self.dup_dist:
                # Soft-update the nearest existing track instead of spawning a duplicate.
                nearest_trk = min(live_tracks, key=lambda t: np.hypot(f.east_m - t.kf.x[0], f.north_m - t.kf.x[1]))
                nearest_trk.kf.update(np.array([f.east_m, f.north_m]), self.R)
                nearest_trk.last_seen = timestamp
                nearest_trk.history.append((timestamp, *nearest_trk.kf.predicted_position()))
                self._update_classification(nearest_trk, f)
                continue

            kf = KalmanTrack(f.east_m, f.north_m, f.east_rate_mps, f.north_rate_mps)
            trk = Track(kf=kf, status=TrackStatus.TENTATIVE, last_seen=timestamp)
            trk.history.append((timestamp, f.east_m, f.north_m))
            self._update_classification(trk, f)
            self.tracks.append(trk)
            live_tracks.append(trk)

        self.tracks = [t for t in self.tracks if t.status != TrackStatus.DROPPED]
        return self.tracks

    @staticmethod
    def _update_classification(trk: Track, f: FusedDetection):
        """Naive Bayesian-style running update: multiply in each vote, renormalize."""
        if not f.classification_votes:
            return
        for cls in ObjectClass:
            trk.class_score.setdefault(cls, 1.0)  # uniform prior
        for cls, conf in f.classification_votes:
            for c in ObjectClass:
                trk.class_score[c] *= conf if c == cls else (1 - conf) / max(len(ObjectClass) - 1, 1)
        total = sum(trk.class_score.values()) or 1.0
        for c in trk.class_score:
            trk.class_score[c] /= total
