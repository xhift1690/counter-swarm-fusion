from __future__ import annotations
import json
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from simulator.targets import SwarmScenario
from simulator.sensors import build_default_sensor_suite
from tracker.track_manager import TrackManager, TrackStatus

BASE = Path(__file__).parent
DT = 0.5
DURATION_S = 300.0
MAX_SESSIONS = 40

app = FastAPI(title="Counter-Swarm Fusion Engine — Prototype API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Session:
    def __init__(self, n_drones: int = 5, n_decoys: int = 3, seed: int = 7):
        self.n_drones = n_drones
        self.n_decoys = n_decoys
        self.seed = seed
        scenario = SwarmScenario(n_drones=n_drones, n_decoys=n_decoys, dt=DT, duration_s=DURATION_S, seed=seed)
        self.timeline = scenario.generate()
        self.sensors = build_default_sensor_suite(np.random.default_rng(seed))
        self.manager = TrackManager()
        self.step_idx = 0
        self.touched = time.time()

    def snapshot(self):
        live = [t for t in self.manager.tracks if t.status != TrackStatus.DROPPED]
        tracks = [
            {
                "track_id": t.track_id,
                "status": t.status.value,
                "position": {"east_m": float(t.kf.x[0]), "north_m": float(t.kf.x[1])},
                "velocity": {"east_mps": float(t.kf.x[2]), "north_mps": float(t.kf.x[3])},
                "threat_score": float(t.threat_score()),
                "hits": int(t.kf.hits),
            }
            for t in live
        ]
        idx = min(max(self.step_idx - 1, 0), len(self.timeline) - 1)
        truth = [
            {"id": g.target_id, "east_m": float(g.east_m), "north_m": float(g.north_m), "is_threat": bool(g.is_threat)}
            for g in self.timeline[idx]
        ] if self.step_idx > 0 else []
        return tracks, truth

    def step(self):
        if self.step_idx >= len(self.timeline):
            tracks, truth = self.snapshot()
            return {"done": True, "step": self.step_idx, "tracks": tracks, "truth": truth}
        t = self.step_idx * DT
        detections = []
        for sensor in self.sensors:
            detections.extend(sensor.observe(self.timeline[self.step_idx], timestamp=t, dt=DT))
        self.manager.step(detections, timestamp=t, dt=DT)
        self.step_idx += 1
        tracks, truth = self.snapshot()
        return {"done": False, "step": self.step_idx, "timestamp_s": t, "tracks": tracks, "truth": truth}


_sessions: dict[str, Session] = {}


def _get(sid: str) -> Session:
    s = _sessions.get(sid)
    if s is None:
        if len(_sessions) >= MAX_SESSIONS:
            oldest = min(_sessions, key=lambda k: _sessions[k].touched)
            del _sessions[oldest]
        s = _sessions[sid] = Session()
    s.touched = time.time()
    return s


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(_sessions)}


@app.get("/metrics")
def metrics():
    return json.loads((BASE / "outputs" / "metrics.json").read_text())


@app.get("/tracks")
def get_tracks(sid: str = "default"):
    s = _get(sid)
    tracks, truth = s.snapshot()
    return {"step": s.step_idx, "tracks": tracks, "truth": truth}


@app.post("/simulate/step")
def simulate_step(sid: str = "default"):
    return _get(sid).step()


@app.post("/simulate/reset")
def simulate_reset(sid: str = "default", n_drones: int = 5, n_decoys: int = 3, seed: int = 7):
    n_drones = max(1, min(n_drones, 12))
    n_decoys = max(0, min(n_decoys, 10))
    _sessions[sid] = Session(n_drones, n_decoys, seed)
    return {"status": "reset", "n_drones": n_drones, "n_decoys": n_decoys, "seed": seed}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE / "site" / "index.html")


app.mount("/outputs", StaticFiles(directory=BASE / "outputs"), name="outputs")
app.mount("/demo", StaticFiles(directory=BASE / "static", html=True), name="demo")
