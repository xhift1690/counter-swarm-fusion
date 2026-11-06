from __future__ import annotations
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

from simulator.targets import SwarmScenario
from simulator.sensors import SensorSuite
from calibrate import fit_for_world
from tracker.calibration import VoteCalibration
from tracker.track_manager import TrackManager, TrackStatus

DT = 0.5
DURATION_S = 120.0
GATE_M = 50.0

CONDITIONS = [
    ("baseline", "baseline", "5 drones, 3 decoys, all sensors", dict(n_drones=5)),
    ("size_3", "swarm size", "3 drones", dict(n_drones=3)),
    ("size_8", "swarm size", "8 drones", dict(n_drones=8)),
    ("size_12", "swarm size", "12 drones", dict(n_drones=12)),
    ("no_radar", "sensor ablation", "No radar", dict(radar=False)),
    ("no_rf", "sensor ablation", "No RF", dict(rf=False)),
    ("no_eo", "sensor ablation", "No camera", dict(eo=False)),
    ("radar_only", "sensor ablation", "Radar only", dict(rf=False, eo=False)),
    ("noise_2x", "stress", "2x sensor noise", dict(noise=2.0)),
    ("noise_3x", "stress", "3x sensor noise", dict(noise=3.0)),
    ("pd_70", "stress", "70% detection probability", dict(pd=0.7)),
    ("pd_50", "stress", "50% detection probability", dict(pd=0.5)),
    ("conf_uninformative", "stress", "Uninformative classifier confidence", dict(uninformative_conf=True)),
]


def run_episode(seed: int, params: dict, calibration: dict | None = None) -> dict:
    params = dict(params)
    n_drones = params.pop("n_drones", 5)
    n_decoys = params.pop("n_decoys", 3)
    timeline = SwarmScenario(n_drones=n_drones, n_decoys=n_decoys, dt=DT, duration_s=DURATION_S, seed=seed).generate()
    suite = SensorSuite(np.random.default_rng(seed))
    suite.configure(**params)
    manager = TrackManager(calibration=VoteCalibration(calibration) if calibration else None)

    last_track, was_tracked, first_hit = {}, {}, {}
    track_ids, last_score, is_threat = {}, {}, {}
    last_votes, cal, voted_steps = {}, {}, 0
    sq_err, hits, total, idsw, frag, false_tracks = [], 0, 0, 0, 0, []

    for k, truths in enumerate(timeline):
        t = k * DT
        dets = [d for ds in suite.observe(truths, t, DT).values() for d in ds]
        tracks = manager.step(dets, t, DT)
        conf = [tr for tr in tracks if tr.status == TrackStatus.CONFIRMED]

        match = {}
        if conf:
            cost = np.array([[np.hypot(tr.kf.x[0] - g.east_m, tr.kf.x[1] - g.north_m) for tr in conf] for g in truths])
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] <= GATE_M:
                    match[i] = (j, cost[i, j])
        false_tracks.append(len(conf) - len(match))

        for i, g in enumerate(truths):
            tid = g.target_id
            is_threat[tid] = g.is_threat
            total += 1
            if i in match:
                j, dist = match[i]
                tr = conf[j]
                hits += 1
                sq_err.append(dist ** 2)
                if tid in last_track and last_track[tid] != tr.track_id:
                    idsw += 1
                last_track[tid] = tr.track_id
                track_ids.setdefault(tid, set()).add(tr.track_id)
                first_hit.setdefault(tid, t)
                last_score[tid] = tr.threat_score()
                last_votes[tid] = tr.n_votes
                if tr.n_votes > 0:
                    voted_steps += 1
                    cal.setdefault(tid, []).append((t, tr.threat_score(), tr.threat_score_naive()))
                was_tracked[tid] = True
            else:
                if was_tracked.get(tid):
                    frag += 1
                was_tracked[tid] = False

    n_targets = n_drones + n_decoys
    acquired = list(first_hit)
    p_new, p_naive, ys, lat_new, lat_naive = [], [], [], [], []
    for tid, seq in cal.items():
        y = is_threat[tid]
        for _, a, b in seq:
            p_new.append(a)
            p_naive.append(b)
            ys.append(y)
        lat_new.append(_settle([(t, a if y else 1 - a) for t, a, _ in seq], first_hit[tid]))
        lat_naive.append(_settle([(t, b if y else 1 - b) for t, _, b in seq], first_hit[tid]))
    return {
        "recall": hits / total,
        "rmse_m": float(np.sqrt(np.mean(sq_err))) if sq_err else None,
        "id_switches_per_target": idsw / n_targets,
        "fragmentations_per_target": frag / n_targets,
        "tracks_per_target": float(np.mean([len(track_ids[t]) for t in acquired])) if acquired else None,
        "false_tracks_per_step": float(np.mean(false_tracks)),
        "acquisition_latency_s": float(np.mean([first_hit[t] for t in acquired])) if acquired else None,
        "acquired_fraction": len(acquired) / n_targets,
        "unclassified_fraction": 1 - voted_steps / hits if hits else None,
        "scores": [(float(last_score[t]), bool(is_threat[t])) for t in acquired if last_votes.get(t, 0) > 0],
        "cal_new": p_new, "cal_naive": p_naive, "cal_y": ys,
        "lat_new": lat_new, "lat_naive": lat_naive,
    }


def _settle(seq, t0, level: float = 0.9):
    start = None
    for t, c in reversed(seq):
        if c < level:
            break
        start = t
    return None if start is None else start - t0


def calib_metrics(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    bins = np.minimum((p * 10).astype(int), 9)
    rel, ece = [], 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            rel.append([float(p[m].mean()), float(y[m].mean()), int(m.sum())])
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    confident = np.maximum(p, 1 - p) >= 0.9
    wrong = confident & ((p >= 0.5) != (y == 1))
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "ece": float(ece),
        "confident_wrong_rate": float(wrong.sum() / confident.sum()) if confident.any() else None,
        "n_samples": int(len(p)),
    }, rel


def auc(scores, labels) -> float | None:
    pos = np.array([s for s, l in zip(scores, labels) if l])
    neg = np.array([s for s, l in zip(scores, labels) if not l])
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def summarize(episodes: list[dict]) -> dict:
    out = {}
    skip = {"scores", "cal_new", "cal_naive", "cal_y", "lat_new", "lat_naive"}
    for key in episodes[0]:
        if key in skip:
            continue
        vals = np.array([e[key] for e in episodes if e[key] is not None], dtype=float)
        out[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "n": int(len(vals))} if len(vals) else None
    pooled = [s for e in episodes for s in e["scores"]]
    scores = [s for s, _ in pooled]
    labels = [l for _, l in pooled]
    out["threat_auc"] = auc(scores, labels)
    out["threat_accuracy_at_0.5"] = float(np.mean([(s >= 0.5) == l for s, l in pooled])) if pooled else None
    out["n_scored_targets"] = len(pooled)

    ys = [y for e in episodes for y in e["cal_y"]]
    out["classification"], out["reliability"] = {}, {}
    for name, key, lat in (("calibrated", "cal_new", "lat_new"), ("naive", "cal_naive", "lat_naive")):
        if not ys:
            continue
        m, rel = calib_metrics([p for e in episodes for p in e[key]], ys)
        lats = [v for e in episodes for v in e[lat]]
        reached = [v for v in lats if v is not None]
        m["time_to_confident_s"] = float(np.mean(reached)) if reached else None
        m["confident_fraction"] = len(reached) / len(lats) if lats else None
        out["classification"][name] = m
        out["reliability"][name] = rel
    return out


def _job(args):
    return run_episode(*args)


def _cal_for(params, cals):
    return cals["uninformative"] if params.get("uninformative_conf") else cals["informative"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="outputs/benchmark.json")
    args = ap.parse_args()

    t0 = time.time()
    cals = {w: fit_for_world(w == "uninformative").table for w in ("informative", "uninformative")}
    partial = Path(args.out + ".partial")
    results = json.loads(partial.read_text()) if partial.exists() else []
    done = {r["id"] for r in results}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for cid, group, label, params in CONDITIONS:
            if cid in done:
                continue
            eps = list(pool.map(_job, [(s, params, _cal_for(params, cals)) for s in range(1, args.seeds + 1)]))
            results.append({"id": cid, "group": group, "label": label, "params": params, "metrics": summarize(eps)})
            partial.write_text(json.dumps(results))
            m = results[-1]["metrics"]
            print(f"{cid:12s} recall={m['recall']['mean']:.3f} rmse={m['rmse_m']['mean']:.1f} auc={m['threat_auc']}", flush=True)

    payload = {
        "meta": {"n_seeds": args.seeds, "duration_s": DURATION_S, "dt_s": DT, "match_gate_m": GATE_M,
                 "n_decoys": 3, "score_clip": 1e-4, "vote_calibration": cals, "runtime_s": round(time.time() - t0, 1)},
        "conditions": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    partial.unlink(missing_ok=True)
    print(f"saved {args.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
