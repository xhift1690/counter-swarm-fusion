# Counter-Swarm Fusion Engine — Synthetic Multi-Sensor Simulator + Kalman Tracker

A self-contained prototype of a SAPIENT-aligned multi-sensor fusion and tracking
engine for counter-drone (C-UAS) systems. Generates a synthetic drone-swarm
scenario, simulates noisy multi-sensor observation (radar / passive RF / EO
camera), fuses cross-sensor detections, tracks objects over time with a
Kalman filter, and produces a threat score per track — then exposes all of
it through a small live API.

This is deliberately scoped as a **fusion software prototype**, not a
hardware or weapons system: it contains no sensor-design, jamming, or
interceptor logic. The goal is to demonstrate the same architecture a
real C-UAS fusion/command-and-control layer uses (see "Design notes" below).

## What it does

1. **`simulator/targets.py`** — generates ground-truth trajectories for a
   swarm of drones approaching a protected point, plus bird/clutter decoys
   with more erratic motion.
2. **`simulator/sensors.py`** — three synthetic sensors (radar, passive RF,
   EO camera) observe the ground truth each timestep, each with its own
   noise, detection probability, false-alarm rate, range limit, and
   classification capability — modeling realistic sensor tradeoffs (e.g.
   radar: long range, no classification; camera: short range, good
   classification).
3. **`tracker/track_manager.py`** — fuses same-instant, cross-sensor
   detections that are spatially close (mirroring SAPIENT's
   `associated_detection` / `derived_detection` relationship), associates
   fused detections to existing tracks, updates a per-track Kalman filter,
   and maintains track lifecycle (TENTATIVE → CONFIRMED → DROPPED).
4. **`tracker/kalman.py`** — constant-velocity Kalman filter per track.
5. **`tracker/association.py`** — gated nearest-neighbor data association
   (Hungarian algorithm + Mahalanobis-distance gating) — described as
   "JPDA-lite" since it commits to a single best global assignment rather
   than maintaining full probabilistic hypotheses.
6. **`demo.py`** — runs the full pipeline, renders an animated visualization,
   and computes tracking/classification metrics.
7. **`api.py`** — a minimal FastAPI service exposing live track state,
   mirroring how a real fusion engine would sit behind an API consumed by a
   downstream C2 system or effector integrator.

## Running it

```bash
pip install -r requirements.txt

# Run the simulation, produce outputs/tracking_demo.gif and outputs/metrics.json
python demo.py

# Run the live API
uvicorn api:app --reload --port 8000
# then: curl -X POST http://localhost:8000/simulate/step
#       curl http://localhost:8000/tracks
```

## Results (current run, 5 drones + 3 decoys, 120s scenario)

```
n_true_threats_final_frame: 5
n_confirmed_tracks_final_frame: 8
n_total_tracks_final_frame: 9
mean_threat_score_on_real_drones: 1.0
mean_threat_score_on_real_decoys: ~0.0
```

The threat-scoring separation between real drones and decoys is clean in
this run — which is partly a genuine effect of fusing two independent
classifying sensors (RF + EO), and partly an artifact of the naive
multiplicative Bayesian update used for classification fusion (see
"Known limitations" below). 9 tracks for 8 true targets is close to ideal;
some residual fragmentation from false alarms is expected and is the
correct thing to tune next (see below), not something to be hidden.

## Design notes — how this maps to the SAPIENT standard

This project intentionally mirrors real SAPIENT ICD (DSTL/PUB145591)
semantics rather than inventing its own ad hoc data model:

- **Detection vs. fusion vs. track are three distinct relationships.**
  A `DetectionReport` is one sensor's instantaneous report. Fusion combines
  same-instant, cross-sensor detections (`associated_detections` /
  `derived_detections` in `schema.py`). A track is a time-series of fused
  detections sharing an `object_id` / `track_id`. Conflating these is the
  most common architectural mistake in fusion systems — this codebase keeps
  them separate on purpose.
- **Per-sensor capability declaration.** Each `SyntheticSensor` declares its
  own detection probability, false-alarm rate, location error, and whether
  it can classify — mirroring SAPIENT's Registration message, where every
  node only reports the fields its specific capability profile allows.
- **Track lifecycle states** (TENTATIVE / CONFIRMED / DROPPED) mirror how a
  real DMM (Decision Making Module) manages track confidence before
  recommending an engagement response.

## Known limitations (honest, by design)

- **Naive classification fusion** uses a simple multiplicative Bayesian
  update that overconfidently drives scores to 0 or 1 given just a few
  votes. A real system would use calibrated likelihood ratios or a proper
  Bayesian fusion model (e.g. a Beta-Binomial or Dirichlet model) — this is
  the natural next iteration, and a good one given prior Bayesian/NumPyro
  experience.
- **Constant-velocity motion model only.** Real drones maneuver; an
  Interacting Multiple Model (IMM) filter blending constant-velocity and
  constant-acceleration (or coordinated-turn) models would track
  maneuvering targets more accurately.
- **Greedy nearest-neighbor association**, not full JPDA. Under dense
  swarms with overlapping gates, a probabilistic data association approach
  would handle ambiguous assignments more gracefully than committing to a
  single global least-cost assignment each timestep.
- **Single shared measurement covariance** is used in association for
  simplicity; a production system would carry per-detection covariance
  through fusion and into the Kalman update.
- **Track fragmentation/duplication** was observed and partially fixed via
  a duplicate-suppression heuristic (see `track_manager.py`); a more
  principled fix would tune process noise and gate thresholds per-sensor
  rather than globally, or merge tracks via track-to-track association.

## Next steps (toward a deployable fusion engine)

1. Replace the naive classification fusion with a proper probabilistic
   model.
2. Add an IMM filter for maneuvering targets.
3. Move from gated nearest-neighbor to full JPDA or a multi-hypothesis
   tracker (MHT) for dense-swarm scenarios.
4. Replace the synthetic sensors with adapters that speak the real
   SAPIENT Protobuf wire format (request `.proto` files from
   SAPIENT@dstl.gov.uk) so this can ingest live sensor data.
5. Add a small web front-end polling `/tracks` for a genuinely live demo
   (rather than the pre-rendered GIF).
