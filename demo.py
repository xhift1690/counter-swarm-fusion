"""
demo.py — Run the full pipeline: synthetic swarm scenario -> multi-sensor
observation -> fusion -> Kalman tracking -> visualization + metrics.

Usage:
    python demo.py
Outputs:
    outputs/tracking_demo.gif   — animated plot of ground truth vs. fused tracks
    outputs/metrics.json        — summary tracking/classification metrics
"""

from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from simulator.targets import SwarmScenario
from simulator.sensors import build_default_sensor_suite
from tracker.track_manager import TrackManager, TrackStatus
from schema import ObjectClass


def run_simulation(seed: int = 42):
    rng = np.random.default_rng(seed)
    scenario = SwarmScenario(n_drones=5, n_decoys=3, dt=0.5, duration_s=120.0, seed=seed)
    sensors = build_default_sensor_suite(rng)
    manager = TrackManager()

    timeline = scenario.generate()

    history = []  # per-frame: (truths, tracks_snapshot)
    for step_idx, truths in enumerate(timeline):
        t = step_idx * scenario.dt
        all_detections = []
        for sensor in sensors:
            all_detections.extend(sensor.observe(truths, timestamp=t, dt=scenario.dt))

        tracks = manager.step(all_detections, timestamp=t, dt=scenario.dt)
        snapshot = [
            {
                "track_id": trk.track_id,
                "status": trk.status.value,
                "pos": trk.kf.predicted_position(),
                "threat_score": trk.threat_score(),
            }
            for trk in tracks if trk.status != TrackStatus.DROPPED
        ]
        history.append((truths, snapshot))

    return history, scenario, manager


def compute_metrics(history) -> dict:
    """
    Simple, honest metrics:
      - confirmed-track count vs. true threat count over time (track formation lag)
      - mean threat_score for tracks actually overlapping a real drone vs. a real decoy
        (nearest-truth labelling, for evaluation only — not given to the tracker)
    """
    final_truths, final_tracks = history[-1]
    n_true_threats = sum(1 for t in final_truths if t.is_threat)
    n_confirmed = sum(1 for tr in final_tracks if tr["status"] == "CONFIRMED")

    # Label each final track by nearest ground-truth target (eval-only)
    drone_scores, decoy_scores = [], []
    for tr in final_tracks:
        best_truth, best_dist = None, float("inf")
        for truth in final_truths:
            d = np.hypot(tr["pos"][0] - truth.east_m, tr["pos"][1] - truth.north_m)
            if d < best_dist:
                best_dist, best_truth = d, truth
        if best_truth is None or best_dist > 50:
            continue
        if best_truth.is_threat:
            drone_scores.append(tr["threat_score"])
        else:
            decoy_scores.append(tr["threat_score"])

    return {
        "n_true_threats_final_frame": n_true_threats,
        "n_confirmed_tracks_final_frame": n_confirmed,
        "n_total_tracks_final_frame": len(final_tracks),
        "mean_threat_score_on_real_drones": float(np.mean(drone_scores)) if drone_scores else None,
        "mean_threat_score_on_real_decoys": float(np.mean(decoy_scores)) if decoy_scores else None,
        "n_drone_tracks_evaluated": len(drone_scores),
        "n_decoy_tracks_evaluated": len(decoy_scores),
    }


def render_animation(history, out_path: str, frame_stride: int = 2):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-2200, 2200)
    ax.set_ylim(-2200, 2200)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Synthetic Multi-Sensor Fusion + Kalman Tracking")
    ax.scatter([0], [0], marker="^", s=120, c="black", label="Protected asset")

    drone_scatter = ax.scatter([], [], c="red", marker="x", s=40, label="True drone")
    decoy_scatter = ax.scatter([], [], c="gray", marker=".", s=20, label="True bird/clutter")
    track_scatter = ax.scatter([], [], facecolors="none", edgecolors="blue", s=150, label="Fused track")
    track_texts = []
    ax.legend(loc="upper right", fontsize=8)

    frames = history[::frame_stride]

    def update(frame_idx):
        truths, tracks = frames[frame_idx]
        drone_xy = np.array([[t.east_m, t.north_m] for t in truths if t.is_threat])
        decoy_xy = np.array([[t.east_m, t.north_m] for t in truths if not t.is_threat])
        track_xy = np.array([[tr["pos"][0], tr["pos"][1]] for tr in tracks]) if tracks else np.empty((0, 2))

        drone_scatter.set_offsets(drone_xy if len(drone_xy) else np.empty((0, 2)))
        decoy_scatter.set_offsets(decoy_xy if len(decoy_xy) else np.empty((0, 2)))
        track_scatter.set_offsets(track_xy)

        for txt in track_texts:
            txt.remove()
        track_texts.clear()
        for tr in tracks:
            label = f"{tr['threat_score']:.2f}"
            txt = ax.annotate(label, (tr["pos"][0], tr["pos"][1]), fontsize=7, color="blue",
                               xytext=(4, 4), textcoords="offset points")
            track_texts.append(txt)

        ax.set_title(f"Synthetic Multi-Sensor Fusion + Kalman Tracking  (t={frame_idx*frame_stride*0.5:.1f}s)")
        return drone_scatter, decoy_scatter, track_scatter

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
    anim.save(out_path, writer="pillow", fps=12)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    history, scenario, manager = run_simulation()
    metrics = compute_metrics(history)

    print(json.dumps(metrics, indent=2))
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    render_animation(history, "outputs/tracking_demo.gif")
    print("Saved outputs/tracking_demo.gif and outputs/metrics.json")
