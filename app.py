# ============================================================
# ORBITX — Self-contained Flask backend
# No satiliteposi imports — safe for Vercel cold starts
# ============================================================

import json
import threading
from datetime import datetime, timedelta, timezone
from math import degrees, radians, asin, atan2, cos, sin, sqrt, pi
from pathlib import Path

import numpy as np
import requests as http_requests
from flask import Flask, jsonify, request, send_from_directory
from sgp4.api import Satrec, jday
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ── paths ───────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
STATIC_DIR = BASE_DIR / "data" / "webapp"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

_tle_cache:      dict = {}
_trained_models: dict = {}
_train_lock = threading.Lock()

EARTH_RADIUS_KM = 6371.0
CELESTRAK_URL   = "https://celestrak.org/NORAD/elements/gp.php"

SATELLITES = {
    25544: "ISS (ZARYA)",  20580: "HUBBLE",
    27424: "AQUA",         25994: "TERRA",
    33591: "NOAA-19",      28654: "NOAA-18",
    43013: "NOAA-20",      37849: "SUOMI NPP",
    39084: "LANDSAT 8",    49260: "LANDSAT 9",
}

# ============================================================
# CORE UTILITIES
# ============================================================

def ecef_to_geodetic(x, y, z):
    r   = sqrt(x**2 + y**2 + z**2)
    lat = degrees(asin(z / r))
    lon = degrees(atan2(y, x))
    alt = r - EARTH_RADIUS_KM
    return lat, lon, alt


def get_epoch_dt(sat):
    year = sat.epochyr
    year += 2000 if year < 57 else 1900
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=sat.epochdays - 1)


def propagate(sat, epoch_dt, duration_s, step_s=60):
    records, t = [], 0
    while t <= duration_s:
        dt = epoch_dt + timedelta(seconds=t)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        err, pos, vel = sat.sgp4(jd, fr)
        if err == 0:
            x, y, z    = pos
            vx, vy, vz = vel
            lat, lon, alt = ecef_to_geodetic(x, y, z)
            records.append({
                "time_offset": t, "x": x, "y": y, "z": z,
                "lat": lat, "lon": lon, "alt_km": alt,
                "speed_km_s": sqrt(vx**2 + vy**2 + vz**2)
            })
        t += step_s
    return records


def elevation_angle(sx, sy, sz, gs_lat, gs_lon, gs_alt=0.01):
    lat_r, lon_r = radians(gs_lat), radians(gs_lon)
    r  = EARTH_RADIUS_KM + gs_alt
    gx = r * cos(lat_r) * cos(lon_r)
    gy = r * cos(lat_r) * sin(lon_r)
    gz = r * sin(lat_r)
    dx, dy, dz = sx - gx, sy - gy, sz - gz
    rng = sqrt(dx**2 + dy**2 + dz**2)
    ux  = cos(lat_r) * cos(lon_r)
    uy  = cos(lat_r) * sin(lon_r)
    uz  = sin(lat_r)
    el  = degrees(asin((dx*ux + dy*uy + dz*uz) / rng))
    return el, rng


def predict_passes(entry, gs_lat, gs_lon, gs_name, duration_h=24, min_el=10.0):
    sat      = Satrec.twoline2rv(entry["TLE_LINE1"], entry["TLE_LINE2"])
    epoch_dt = get_epoch_dt(sat)
    passes, current = [], None

    for t in range(0, duration_h * 3600, 30):
        dt = epoch_dt + timedelta(seconds=t)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        err, pos, _ = sat.sgp4(jd, fr)
        if err != 0:
            continue
        el, rng = elevation_angle(pos[0], pos[1], pos[2], gs_lat, gs_lon)
        if el >= min_el:
            if current is None:
                current = {"aos": dt, "max_el": el, "range_km": rng}
            elif el > current["max_el"]:
                current["max_el"] = el
                current["range_km"] = rng
        else:
            if current:
                current["los"]          = dt
                current["duration_min"] = (dt - current["aos"]).seconds / 60
                passes.append(current)
                current = None
    return passes


def generate_dataset(entry, n=1500):
    sat      = Satrec.twoline2rv(entry["TLE_LINE1"], entry["TLE_LINE2"])
    epoch_dt = get_epoch_dt(sat)
    inc  = degrees(sat.inclo);  ecc  = sat.ecco
    raan = degrees(sat.nodeo);  argp = degrees(sat.argpo)
    mm   = sat.no_kozai * 1440 / (2*pi)
    ma   = degrees(sat.mo);     bstar = sat.bstar

    rows = []
    rng  = np.random.default_rng(42)
    for _ in range(n):
        dt_s = float(rng.uniform(0, 86400 * 2))
        dt   = epoch_dt + timedelta(seconds=dt_s)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        err, pos, vel = sat.sgp4(jd, fr)
        if err != 0:
            continue
        rows.append([dt_s, inc, ecc, raan, argp, mm, ma, bstar,
                     pos[0], pos[1], pos[2]])

    import pandas as pd
    return pd.DataFrame(rows, columns=[
        "time_offset","inclination","eccentricity","raan",
        "arg_perigee","mean_motion","mean_anomaly","bstar",
        "x","y","z"
    ])


# ============================================================
# TLE LOADING
# ============================================================

def load_tle() -> dict:
    global _tle_cache
    if _tle_cache:
        return _tle_cache

    # local cache file (works when running locally)
    cache_file = BASE_DIR / "data" / "tle_cache.json"
    if cache_file.exists():
        age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if age < 3600:
            with open(cache_file) as f:
                _tle_cache = json.load(f)
            return _tle_cache

    # fetch live
    tle_data = {}
    for catnr, name in SATELLITES.items():
        try:
            r = http_requests.get(
                f"{CELESTRAK_URL}?CATNR={catnr}&FORMAT=TLE", timeout=8)
            lines = [l.strip() for l in r.text.splitlines() if l.strip()]
            if len(lines) >= 3:
                entry = {"OBJECT_NAME": lines[0],
                         "TLE_LINE1":   lines[1],
                         "TLE_LINE2":   lines[2]}
            elif len(lines) == 2 and lines[0].startswith("1 "):
                entry = {"OBJECT_NAME": name,
                         "TLE_LINE1":   lines[0],
                         "TLE_LINE2":   lines[1]}
            else:
                continue

            # enrich with JSON metadata
            try:
                rj = http_requests.get(
                    f"{CELESTRAK_URL}?CATNR={catnr}&FORMAT=JSON", timeout=8)
                data = rj.json()
                if data:
                    entry.update(data[0])
            except Exception:
                pass

            tle_data[str(catnr)] = entry
        except Exception:
            pass

    if tle_data:
        _tle_cache = tle_data
    return _tle_cache


def get_current_position(entry):
    sat = Satrec.twoline2rv(entry["TLE_LINE1"], entry["TLE_LINE2"])
    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
    err, pos, vel = sat.sgp4(jd, fr)
    if err != 0:
        return None
    x, y, z    = pos
    vx, vy, vz = vel
    lat, lon, alt = ecef_to_geodetic(x, y, z)
    return {"lat": round(lat,4), "lon": round(lon,4), "alt_km": round(alt,2),
            "speed_km_s": round(sqrt(vx**2+vy**2+vz**2),4),
            "x": round(x,2), "y": round(y,2), "z": round(z,2)}


# ============================================================
# ML TRAINING
# ============================================================

def train_satellite(catnr_str):
    tle   = load_tle()
    entry = tle.get(catnr_str)
    if not entry:
        return None, "Satellite not found"

    df = generate_dataset(entry, n=1500)
    features = ["time_offset","inclination","eccentricity","raan",
                "arg_perigee","mean_motion","mean_anomaly","bstar"]
    targets  = ["x","y","z"]

    df    = df.sort_values("time_offset").reset_index(drop=True)
    split = int(len(df) * 0.8)
    Xtr, ytr = df[features].iloc[:split], df[targets].iloc[:split]
    Xte, yte = df[features].iloc[split:],  df[targets].iloc[split:]

    mdls = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=-1),
    }
    results, best_model, best_r2 = {}, None, -np.inf

    for name, m in mdls.items():
        m.fit(Xtr, ytr)
        p   = m.predict(Xte)
        r2  = r2_score(yte, p)
        mse = mean_squared_error(yte, p)
        err = np.sqrt((yte["x"].values-p[:,0])**2 +
                      (yte["y"].values-p[:,1])**2 +
                      (yte["z"].values-p[:,2])**2).mean()
        results[name] = {"r2": round(r2,4), "mse": round(mse,2), "avg_error_km": round(err,2)}
        if r2 > best_r2:
            best_r2, best_model = r2, m

    sat = Satrec.twoline2rv(entry["TLE_LINE1"], entry["TLE_LINE2"])
    op  = {
        "inclination":  round(degrees(sat.inclo), 4),
        "eccentricity": round(sat.ecco, 6),
        "raan":         round(degrees(sat.nodeo), 4),
        "arg_perigee":  round(degrees(sat.argpo), 4),
        "mean_motion":  round(sat.no_kozai * 1440 / (2*pi), 6),
        "mean_anomaly": round(degrees(sat.mo), 4),
        "bstar":        sat.bstar,
    }
    return {"model": best_model, "metrics": results,
            "orbital_params": op, "features": features}, None


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/satellites")
def api_satellites():
    tle = load_tle()
    return jsonify([{
        "catnr":    k,
        "name":     v.get("OBJECT_NAME", f"SAT-{k}"),
        "position": get_current_position(v)
    } for k, v in tle.items()])


@app.route("/api/track/<catnr>")
def api_track(catnr):
    tle   = load_tle()
    entry = tle.get(catnr)
    if not entry:
        return jsonify({"error": "Not found"}), 404

    hours    = float(request.args.get("hours", 2))
    sat      = Satrec.twoline2rv(entry["TLE_LINE1"], entry["TLE_LINE2"])
    epoch_dt = get_epoch_dt(sat)
    records  = propagate(sat, epoch_dt, int(hours*3600), step_s=60)

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

    passes = predict_passes(entry, lat, lon, name, hours, min_el)
    result = [{"aos": p["aos"].isoformat(),
               "los": p["los"].isoformat() if "los" in p else None,
               "duration_min":  round(p.get("duration_min",0), 1),
               "max_elevation": round(p["max_el"], 1),
               "range_km":      round(p["range_km"], 0)} for p in passes]

    return jsonify({"satellite": entry.get("OBJECT_NAME"),
                    "ground_station": {"name":name,"lat":lat,"lon":lon},
                    "passes": result})


@app.route("/api/train/<catnr>", methods=["POST"])
def api_train(catnr):
    result, err = train_satellite(catnr)
    if err:
        return jsonify({"status": "error", "error": err}), 500

    _trained_models[catnr] = {
        "status":         "ready",
        "metrics":        result["metrics"],
        "orbital_params": result["orbital_params"],
        "_model":         result["model"],
        "_features":      result["features"]
    }
    return jsonify({"status": "ready",
                    "metrics": result["metrics"],
                    "orbital_params": result["orbital_params"]})


@app.route("/api/model/<catnr>")
def api_model_status(catnr):
    info = _trained_models.get(catnr, {"status": "not_trained"})
    return jsonify({k: v for k, v in info.items() if not k.startswith("_")})


@app.route("/api/predict/<catnr>", methods=["POST"])
def api_predict(catnr):
    info = _trained_models.get(catnr)
    if not info or info.get("status") != "ready":
        return jsonify({"error": "Model not trained yet."}), 400

    time_offset = float(request.get_json(force=True).get("time_offset", 0))
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
            true_pos = {
                "x": round(pos[0],2), "y": round(pos[1],2), "z": round(pos[2],2),
                "lat": round(tlat,4), "lon": round(tlon,4), "alt_km": round(talt,2),
                "error_km": round(sqrt((x-pos[0])**2+(y-pos[1])**2+(z-pos[2])**2), 2)
            }

    return jsonify({
        "predicted":    {"x":round(x,2),"y":round(y,2),"z":round(z,2),
                         "lat":round(lat,4),"lon":round(lon,4),"alt_km":round(alt,2)},
        "true":          true_pos,
        "time_offset_s": time_offset
    })


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    global _tle_cache
    _tle_cache = {}
    load_tle()
    return jsonify({"status": "ok", "count": len(_tle_cache)})


# ============================================================

if __name__ == "__main__":
    print("  ORBITX  →  http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
