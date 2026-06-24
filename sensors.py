"""
simulator/sensors.py — Synthetic sensor models.

Each sensor "observes" the ground-truth targets each timestep and emits
DetectionReport objects (schema.py) with:
  - Gaussian location noise (sensor-specific std dev)
  - missed detections (per sensor detection probability P(d))
  - false alarms (clutter/birds appearing as spurious detections)
  - partial classification ability (some sensors can classify, some can't)

This deliberately mirrors real sensor tradeoffs: radar has long range but
coarse classification; cameras classify well but have a narrow field of
view and degrade at range/at night; RF detects emissions but not silent
drones.
"""

from __future__ import annotations
import numpy as np
from schema import (
    DetectionReport, Location, Velocity, ClassificationResult,
    ObjectClass, NodeType, NodeRegistration, TrackingType, new_id,
)
from simulator.targets import TargetTruth


class SyntheticSensor:
    """Base class for a synthetic sensor node."""

    def __init__(
        self,
        node_type: NodeType,
        name: str,
        location_error_std_m: float,
        detection_probability: float,
        false_alarm_rate_hz: float,
        max_range_m: float,
        can_classify: bool,
        classification_accuracy: float,
        rng: np.random.Generator,
    ):
        self.node_id = new_id()
        self.node_type = node_type
        self.name = name
        self.location_error_std_m = location_error_std_m
        self.detection_probability = detection_probability
        self.false_alarm_rate_hz = false_alarm_rate_hz
        self.max_range_m = max_range_m
        self.can_classify = can_classify
        self.classification_accuracy = classification_accuracy
        self.rng = rng

    def registration(self) -> NodeRegistration:
        return NodeRegistration(
            node_id=self.node_id,
            node_type=self.node_type,
            name=self.name,
            tracking_type=TrackingType.NONE,  # raw edge sensors: no tracking, fusion engine does it
            location_error_std_m=self.location_error_std_m,
            detection_probability=self.detection_probability,
            false_alarm_rate_hz=self.false_alarm_rate_hz,
            can_classify=self.can_classify,
        )

    def _maybe_classify(self, truth: TargetTruth) -> list[ClassificationResult]:
        if not self.can_classify:
            return []
        true_class = ObjectClass.DRONE_MULTIROTOR if truth.is_threat else ObjectClass.BIRD
        if self.rng.random() < self.classification_accuracy:
            conf = self.rng.uniform(0.7, 0.97)
            return [ClassificationResult(object_class=true_class, confidence=conf)]
        else:
            # misclassify into the other bucket
            wrong_class = ObjectClass.BIRD if truth.is_threat else ObjectClass.DRONE_MULTIROTOR
            conf = self.rng.uniform(0.5, 0.8)
            return [ClassificationResult(object_class=wrong_class, confidence=conf)]

    def observe(self, truths: list[TargetTruth], timestamp: float, dt: float) -> list[DetectionReport]:
        detections: list[DetectionReport] = []

        for truth in truths:
            r = np.hypot(truth.east_m, truth.north_m)
            if r > self.max_range_m:
                continue
            if self.rng.random() > self.detection_probability:
                continue  # missed detection

            noise_e = self.rng.normal(0, self.location_error_std_m)
            noise_n = self.rng.normal(0, self.location_error_std_m)
            loc = Location(
                east_m=truth.east_m + noise_e,
                north_m=truth.north_m + noise_n,
                up_m=truth.up_m,
                east_error_m=self.location_error_std_m,
                north_error_m=self.location_error_std_m,
            )
            vel = Velocity(
                east_rate_mps=truth.east_rate_mps + self.rng.normal(0, 0.5),
                north_rate_mps=truth.north_rate_mps + self.rng.normal(0, 0.5),
            )
            det = DetectionReport(
                node_id=self.node_id,
                node_type=self.node_type,
                timestamp=timestamp,
                location=loc,
                velocity=vel,
                detection_confidence=float(np.clip(self.rng.normal(0.9, 0.05), 0.4, 0.99)),
                classification=self._maybe_classify(truth),
            )
            detections.append(det)

        # False alarms: Poisson-distributed spurious detections within range
        n_false = self.rng.poisson(self.false_alarm_rate_hz * dt)
        for _ in range(n_false):
            angle = self.rng.uniform(0, 2 * np.pi)
            r = self.rng.uniform(50, self.max_range_m)
            loc = Location(east_m=r * np.cos(angle), north_m=r * np.sin(angle), up_m=self.rng.uniform(0, 100))
            det = DetectionReport(
                node_id=self.node_id,
                node_type=self.node_type,
                timestamp=timestamp,
                location=loc,
                velocity=Velocity(east_rate_mps=self.rng.normal(0, 2), north_rate_mps=self.rng.normal(0, 2)),
                detection_confidence=float(self.rng.uniform(0.3, 0.6)),
                classification=[ClassificationResult(object_class=ObjectClass.CLUTTER, confidence=0.4)]
                    if self.can_classify else [],
            )
            detections.append(det)

        return detections


def build_default_sensor_suite(rng: np.random.Generator) -> list[SyntheticSensor]:
    """A representative 3-sensor suite: radar, RF, EO camera."""
    radar = SyntheticSensor(
        node_type=NodeType.RADAR, name="Radar-1",
        location_error_std_m=8.0, detection_probability=0.92, false_alarm_rate_hz=0.15,
        max_range_m=2500.0, can_classify=False, classification_accuracy=0.0, rng=rng,
    )
    rf = SyntheticSensor(
        node_type=NodeType.PASSIVE_RF, name="RF-1",
        location_error_std_m=15.0, detection_probability=0.80, false_alarm_rate_hz=0.05,
        max_range_m=1500.0, can_classify=True, classification_accuracy=0.7, rng=rng,
    )
    eo = SyntheticSensor(
        node_type=NodeType.CAMERA, name="EO-1",
        location_error_std_m=4.0, detection_probability=0.85, false_alarm_rate_hz=0.08,
        max_range_m=800.0, can_classify=True, classification_accuracy=0.88, rng=rng,
    )
    return [radar, rf, eo]
