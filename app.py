# ============================================================
# ORBITX — Flask Backend (Vercel + Local compatible)
# Local:  python app.py  →  http://localhost:5000
# Vercel: deployed via api/index.py
# ============================================================

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from math import degrees, pi, sqrt
from pathlib import Path

import numpy as np
import requests as http_requests
from flask import Flask, jsonify, request, send_from_directory
from sgp4.api import Satrec, jday

from satiliteposi import (
    get_epoch_dt,
    ecef_to_geodetic,
    propagate_satellite,
    predict_passes,
    SATELLITES,
    CELESTRAK_URL,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ── Static folder works both locally and on Vercel ─────────
BASE_DIR    = Path(__file__).parent
STATIC_DIR  = BASE_DIR / "data" / "webapp"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

# ── In-memory stores (per-instance, fine for serverless) ───
_tle_cache:      dict = {}
_trained_models: dict = {}
_train_lock = threading.Lock()

# ============================================================
# TLE  — fetch live, no file writes needed on Vercel
# ============================================================

def load_tle() -> dict:
    global _tle_cache
    if _tle_cache:
        return _tle_cache

    # Try local cache file first (works locally)
    cache_file = BASE_DIR / "data" / "tle_cache.json"
    if cache_file.exists():
        age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if age < 3600:
            with open(cache_file) as f:
                _tle_cache = json.load(f)
            return _tle_cache

    # Fetch live from CelesTrak
    tle_data = {}
    for catnr in SATELLITES:
        try:
            r = http_requests.get(f"{CELESTRAK_URL}?CATNR={catnr}&FORMAT=JSON", timeout=10)
            data = r.json()
            entry = data[0] if data else {}

            r2 = http_requests.get(f"{CELESTRAK_URL}?CATNR={catnr}&FORMAT=TLE", timeout=10)
            lines = [l.strip() for l in r2.text.splitlines() if l.strip()]
            if len(lines) >= 2:
                if lines[0].startswith("1 "):
                    entry["TLE_LINE1"], entry["TLE_LINE2"] = lines[0], lines[1]
                else:
                    entry["OBJECT_NAME"] = lines[0]
                    entry["TLE_LINE1"],   entry["TLE_LINE2"] = lines[1], lines[2]

            if "TLE_LINE1" in entry:
                tle_data[str(catnr)] = entry
        except Exception:
            pass

    _tle_cache = tle_data
    return _tle_cache


def get_current_position(entry: dict):
    sat = Satrec.twoline2rv(entry["TLE_LINE1"], entry["TLE_LINE2"])
    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
    err, pos, vel = sat.sgp4(jd, fr)
    if err != 0:
        return None
    x, y, z    = pos
    vx, vy, vz = vel
    lat, lon, alt = ecef_to_geodetic(x, y, z)
    speed = sqrt(vx**2 + vy**2 + vz**2)
    return {"lat": round(lat,4), "lon": round(lon,4), "alt_km": round(alt,2),
            "speed_km_s": round(speed,4), "x": round(x,2), "y": round(y,2), "z": round(z,2)}


# ============================================================
# ML TRAINING  — synchronous, fast (1000 samples, RF only)
# ============================================================

def train_model_for_satellite(catnr_str: str):
    from satiliteposi import generate_dataset

    tle   = load_tle()
    entry = tle.get(catnr_str)
    if not entry:
        return None, "Satellite not found"

    df = generate_dataset(entry, num_samples=1500)

    features = ["time_offset","inclination","eccentricity","raan",
                "arg_perigee","mean_motion","mean_anomaly","bstar"]
    targets  = ["x","y","z"]

    df    = df.sort_values("time_offset").reset_index(drop=True)
    split = int(len(df) * 0.8)
    X_tr, y_tr = df[features].iloc[:split], df[targets].iloc[:split]
    X_te, y_te = df[features].iloc[split:],  df[targets].iloc[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=-1),
    }

    results = {}
    best_model, best_r2 = None, -np.inf

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds   = model.predict(X_te)
        r2      = r2_score(y_te, preds)
        mse     = mean_squared_error(y_te, preds)
        avg_err = np.sqrt(
            (y_te["x"].values - preds[:,0])**2 +
            (y_te["y"].values - preds[:,1])**2 +
            (y_te["z"].values - preds[:,2])**2
        ).mean()
        results[name] = {"r2": round(r2,4), "mse": round(mse,2), "avg_error_km": round(avg_err,2)}
        if r2 > best_r2:
            best_r2, best_model = r2, model

    sat = Satrec.twoline2rv(entry["TLE_LINE1"], entry["TLE_LINE2"])
    orbital_params = {
        "inclination":  round(degrees(sat.inclo), 4),
        "eccentricity": round(sat.ecco, 6),
        "raan":         round(degrees(sat.nodeo), 4),
        "arg_perigee":  round(degrees(sat.argpo), 4),
        "mean_motion":  round(sat.no_kozai * 1440 / (2*pi), 6),
        "mean_anomaly": round(degrees(sat.mo), 4),
        "bstar":        sat.bstar,
    }

    return {"model": best_model, "metrics": results,
            "orbital_params": orbital_params, "features": features}, None


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/satellites")
def api_satellites():
    tle    = load_tle()
    result = []
    for catnr_str, entry in tle.items():
        pos = get_current_position(entry)
        result.append({"catnr": catnr_str,
                        "name": entry.get("OBJECT_NAME", f"SAT-{catnr_str}"),
                        "position": pos})
    return jsonify(result)


@app.route("/api/track/<catnr>")
def api_track(catnr):
    tle   = load_tle()
    entry = tle.get(catnr)
    if not entry:
        return jsonify({"error": "Not found"}), 404

    hours    = float(request.args.get("hours", 2))
    sat      = Satrec.twoline2rv(entry["TLE_LINE1"], entry["TLE_LINE2"])
    epoch_dt = get_epoch_dt(sat)
    records  = propagate_satellite(sat, epoch_dt, int(hours*3600), step_seconds=60)

    track = [{"lat": round(r["lat"],4), "lon": round(r["lon"],4),
              "alt_km": round(r["alt_km"],2), "speed_km_s": round(r["speed_km_s"],4),
              "x": round(r["x"],2), "y": round(r["y"],2), "z": round(r["z"],2),
              "time_offset": r["time_offset"]} for r in records]

    return jsonify({"catnr": catnr, "name": entry.get("OBJECT_NAME"), "track": track})


@app.route("/api/passes/<catnr>")
def api_passes(catnr):
    tle   = load_tle()
    entry = tle.get(catnr)
    if not entry:
        return jsonify({"error": "Not found"}), 404

    lat    = float(request.args.get("lat",    40.7128))
    lon    = float(request.args.get("lon",   -74.006))
    name   = request.args.get("name",  "My Location")
    hours  = int(request.args.get("hours",   24))
    min_el = float(request.args.get("min_el", 10.0))

    gs     = {"name": name, "lat": lat, "lon": lon, "alt_km": 0.01}
    passes = predict_passes(entry, gs=gs, duration_hours=hours, min_elevation=min_el)

    result = [{"aos": p["aos"].isoformat(),
               "los": p["los"].isoformat() if "los" in p else None,
               "duration_min":  round(p.get("duration_min", 0), 1),
               "max_elevation": round(p["max_el"], 1),
               "range_km":      round(p["range_km"], 0)} for p in passes]

    return jsonify({"satellite": entry.get("OBJECT_NAME"),
                    "ground_station": gs, "passes": result})


@app.route("/api/train/<catnr>", methods=["POST"])
def api_train(catnr):
    """
    Synchronous training — runs inline.
    Vercel Hobby: 10s limit  (1500 samples fits)
    Vercel Pro:   60s limit
    Local:        no limit
    """
    with _train_lock:
        if _trained_models.get(catnr, {}).get("status") == "training":
            return jsonify({"status": "already_training"})
        _trained_models[catnr] = {"status": "training"}

    result, err = train_model_for_satellite(catnr)

    with _train_lock:
        if err:
            _trained_models[catnr] = {"status": "error", "error": err}
        else:
            _trained_models[catnr] = {
                "status":         "ready",
                "metrics":        result["metrics"],
                "orbital_params": result["orbital_params"],
                "_model":         result["model"],
                "_features":      result["features"]
            }

    safe = {k: v for k, v in _trained_models[catnr].items() if not k.startswith("_")}
    return jsonify(safe)


@app.route("/api/model/<catnr>")
def api_model_status(catnr):
    info = _trained_models.get(catnr, {"status": "not_trained"})
    return jsonify({k: v for k, v in info.items() if not k.startswith("_")})


@app.route("/api/predict/<catnr>", methods=["POST"])
def api_predict(catnr):
    info = _trained_models.get(catnr)
    if not info or info.get("status") != "ready":
        return jsonify({"error": "Model not trained yet."}), 400

    body        = request.get_json(force=True)
    time_offset = float(body.get("time_offset", 0))

    op  = info["orbital_params"]
    row = np.array([[time_offset, op["inclination"], op["eccentricity"],
                     op["raan"],  op["arg_perigee"], op["mean_motion"],
                     op["mean_anomaly"], op["bstar"]]])

    pred = info["_model"].predict(row)[0]
    x, y, z       = pred
    lat, lon, alt = ecef_to_geodetic(x, y, z)

    tle   = load_tle()
    entry = tle.get(catnr)
    true_pos = None
    if entry:
        sat      = Satrec.twoline2rv(entry["TLE_LINE1"], entry["TLE_LINE2"])
        epoch_dt = get_epoch_dt(sat)
        dt       = epoch_dt + timedelta(seconds=time_offset)
        jd, fr   = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        err2, pos, _ = sat.sgp4(jd, fr)
        if err2 == 0:
            tlat, tlon, talt = ecef_to_geodetic(*pos)
            error_km = sqrt((x-pos[0])**2 + (y-pos[1])**2 + (z-pos[2])**2)
            true_pos = {"x": round(pos[0],2), "y": round(pos[1],2), "z": round(pos[2],2),
                        "lat": round(tlat,4), "lon": round(tlon,4), "alt_km": round(talt,2),
                        "error_km": round(error_km,2)}

    return jsonify({
        "predicted":     {"x": round(x,2), "y": round(y,2), "z": round(z,2),
                          "lat": round(lat,4), "lon": round(lon,4), "alt_km": round(alt,2)},
        "true":          true_pos,
        "time_offset_s": time_offset
    })


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    global _tle_cache
    _tle_cache = {}          # clear cache — next request re-fetches live
    load_tle()
    return jsonify({"status": "ok", "count": len(_tle_cache)})


# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  ORBITX Web App  →  http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=5000)
