"""
api.py — Minimal FastAPI service exposing the fusion engine's track state.

This mirrors how a real fusion engine would sit behind an API consumed by
a downstream command-and-control (C2) system or effector integrator: poll
or stream current tracks, each with position, velocity, status, and threat
score (see Stage 6 of the architecture in the project handoff — effector
integrators consume exactly this kind of output, they don't need to know
how fusion happened internally).

Run:
    uvicorn api:app --reload --port 8000

Then open:
    http://localhost:8000/demo/        -> live browser demo (static/index.html)

API endpoints:
    GET  /tracks          -> current snapshot of all live tracks
    GET  /health           -> liveness check
    POST /simulate/step     -> advance the simulation by one timestep and
                                 return the updated track snapshot (this is
                                 what the live demo polls every ~200ms)
    POST /simulate/reset     -> restart the simulation from t=0
"""

from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np

from simulator.targets import SwarmScenario
from simulator.sensors import build_default_sensor_suite
from tracker.track_manager import TrackManager, TrackStatus

app = FastAPI(title="Counter-Swarm Fusion Engine — Prototype API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/demo", StaticFiles(directory="static", html=True), name="demo")

_rng = np.random.default_rng(7)
_scenario = SwarmScenario(n_drones=5, n_decoys=3, dt=0.5, duration_s=600.0, seed=7)
_sensors = build_default_sensor_suite(_rng)
_manager = TrackManager()
_timeline = _scenario.generate()
_step_idx = 0


def _current_snapshot():
    live = [t for t in _manager.tracks if t.status != TrackStatus.DROPPED]
    return [
        {
            "track_id": t.track_id,
            "status": t.status.value,
            "position": {"east_m": t.kf.x[0], "north_m": t.kf.x[1]},
            "velocity": {"east_mps": t.kf.x[2], "north_mps": t.kf.x[3]},
            "threat_score": t.threat_score(),
            "hits": t.kf.hits,
        }
        for t in live
    ]


@app.get("/health")
def health():
    return {"status": "ok", "step": _step_idx}


@app.get("/tracks")
def get_tracks():
    return {"step": _step_idx, "tracks": _current_snapshot()}


@app.post("/simulate/step")
def simulate_step():
    global _step_idx
    if _step_idx >= len(_timeline):
        return {"done": True, "step": _step_idx, "tracks": _current_snapshot()}

    truths = _timeline[_step_idx]
    t = _step_idx * _scenario.dt
    detections = []
    for sensor in _sensors:
        detections.extend(sensor.observe(truths, timestamp=t, dt=_scenario.dt))
    _manager.step(detections, timestamp=t, dt=_scenario.dt)
    _step_idx += 1
    return {"done": False, "step": _step_idx, "timestamp_s": t, "tracks": _current_snapshot()}


@app.post("/simulate/reset")
def simulate_reset():
    global _manager, _step_idx
    _manager = TrackManager()
    _step_idx = 0
    return {"status": "reset"}
