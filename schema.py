"""
schema.py — Simplified SAPIENT-aligned data model.

This mirrors the structure of the real SAPIENT Interface Control Document
(DSTL/PUB145591) closely enough to demonstrate standards-aware design,
without implementing the full Protobuf wire format. Field names and
relationships (detection vs. track vs. fused detection) intentionally
follow the ICD's actual semantics:

- A Detection is a single sensor's report of an object at one instant.
- A Track is a time-series of detections sharing the same object_id.
- A fused detection is just a DetectionReport from a fusion node, where
  `associated_detections` lists the raw sensor detections it was built from.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import uuid
import time


class NodeType(str, Enum):
    RADAR = "RADAR"
    PASSIVE_RF = "PASSIVE_RF"
    CAMERA = "CAMERA"  # EO/IR
    ACOUSTIC = "ACOUSTIC"
    FUSION = "FUSION"


class TrackingType(str, Enum):
    NONE = "NONE"          # sensor cannot associate detections over time
    TRACKLET = "TRACKLET"  # sensor maintains object_id short-term
    TRACK = "TRACK"        # sensor has its own tracker


class ObjectClass(str, Enum):
    DRONE_MULTIROTOR = "DRONE_MULTIROTOR"
    DRONE_FIXED_WING = "DRONE_FIXED_WING"
    BIRD = "BIRD"
    CLUTTER = "CLUTTER"
    UNKNOWN = "UNKNOWN"


def new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Location:
    """Cartesian ENU (East, North, Up), metres, relative to a fixed origin."""
    east_m: float
    north_m: float
    up_m: float = 0.0
    east_error_m: float | None = None
    north_error_m: float | None = None


@dataclass
class Velocity:
    """ENU velocity, metres/second."""
    east_rate_mps: float
    north_rate_mps: float
    up_rate_mps: float = 0.0


@dataclass
class ClassificationResult:
    object_class: ObjectClass
    confidence: float  # 0-1


@dataclass
class DetectionReport:
    """
    A single sensor (or fusion node) report of an object at one instant.
    Mirrors SAPIENT's DetectionReport message (ICD section 6.2), simplified.
    """
    report_id: str = field(default_factory=new_id)
    object_id: str = field(default_factory=new_id)  # stable only if tracking_type != NONE
    node_id: str = ""
    node_type: NodeType = NodeType.RADAR
    timestamp: float = field(default_factory=time.time)

    location: Location | None = None
    velocity: Velocity | None = None
    detection_confidence: float = 1.0
    classification: list[ClassificationResult] = field(default_factory=list)

    # Fusion traceability, mirroring ICD 6.2.18 / 6.2.20
    associated_detections: list[str] = field(default_factory=list)  # report_ids fused into this one
    derived_detections: list[str] = field(default_factory=list)     # report_ids of fused results derived from this one


@dataclass
class NodeRegistration:
    """Mirrors SAPIENT's Registration message (ICD section 5.1), simplified."""
    node_id: str = field(default_factory=new_id)
    node_type: NodeType = NodeType.RADAR
    name: str = ""
    tracking_type: TrackingType = TrackingType.NONE
    # Typical per-sensor performance characterisation (ICD 5.5.1 GeometricError)
    location_error_std_m: float = 5.0
    detection_probability: float = 0.95   # P(d) — probability of detecting a true target
    false_alarm_rate_hz: float = 0.02     # false detections per second
    can_classify: bool = False
