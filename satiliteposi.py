# ============================================================
# ORBITX — ADVANCED SATELLITE ANALYTICS PLATFORM
# Portfolio Project by James
# ============================================================
# Features:
#   - Multi-satellite TLE fetching & caching
#   - SGP4 orbital propagation dataset generation
#   - Ground track generation (lat/lon)
#   - Pass prediction for a ground station
#   - Conjunction (close approach) analysis
#   - Altitude & velocity analytics
#   - ML model training: Linear, Random Forest, Gradient Boost, MLP
#   - Model comparison dashboard
#   - Interactive Plotly HTML dashboard
#   - Animated 3D orbit visualization
#   - CLI interface
# ============================================================

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from math import degrees, radians, asin, atan2, cos, sin, sqrt, pi

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D
from sgp4.api import Satrec, jday

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

TLE_CACHE_FILE = DATA_DIR / "tle_cache.json"

# Expanded satellite catalog
SATELLITES = {
    25544: "ISS (ZARYA)",
    20580: "HUBBLE",
    27424: "AQUA",
    25994: "TERRA",
    33591: "NOAA-19",
    28654: "NOAA-18",
    43013: "NOAA-20",
    37849: "SUOMI NPP",
    39084: "LANDSAT 8",
    49260: "LANDSAT 9",
}

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"

# Default ground station (New York)
GROUND_STATION = {
    "name": "New York",
    "lat": 40.7128,
    "lon": -74.0060,
    "alt_km": 0.01
}

EARTH_RADIUS_KM = 6371.0

# ============================================================
# TLE FETCHING
# ============================================================

def fetch_tle_from_celestrak(catnr):
    import requests
    result = None

    try:
        resp = requests.get(f"{CELESTRAK_URL}?CATNR={catnr}&FORMAT=JSON", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data:
            result = data[0]
    except Exception as e:
        print(f"  JSON fetch failed for {catnr}: {e}")

    try:
        resp = requests.get(f"{CELESTRAK_URL}?CATNR={catnr}&FORMAT=TLE", timeout=15)
        resp.raise_for_status()
        lines = [l.strip() for l in resp.text.splitlines() if l.strip()]
        if len(lines) >= 2:
            if lines[0].startswith("1 "):
                sat_name, tle1, tle2 = f"SAT-{catnr}", lines[0], lines[1]
            else:
                sat_name, tle1, tle2 = lines[0], lines[1], lines[2]
            if result is None:
                result = {"OBJECT_NAME": sat_name}
            result["TLE_LINE1"] = tle1
            result["TLE_LINE2"] = tle2
    except Exception as e:
        print(f"  TLE fetch failed for {catnr}: {e}")

    if result and "TLE_LINE1" not in result:
        return None
    return result


def fetch_all_tle(catalog_numbers, force_refresh=False):
    print("=" * 60)
    print("FETCHING TLE DATA")
    print("=" * 60)

    if not force_refresh and TLE_CACHE_FILE.exists():
        cache_age = datetime.now().timestamp() - TLE_CACHE_FILE.stat().st_mtime
        if cache_age < 3600:
            print("Using cached TLE data (< 1 hour old)")
            with open(TLE_CACHE_FILE) as f:
                return json.load(f)

    tle_data = {}
    for catnr in catalog_numbers:
        name = SATELLITES.get(catnr, f"SAT-{catnr}")
        print(f"Fetching {name} ({catnr})...")
        result = fetch_tle_from_celestrak(catnr)
        if result:
            tle_data[str(catnr)] = result
            print(f"  OK: {result.get('OBJECT_NAME', name)}")
        else:
            print(f"  FAILED")

    if tle_data:
        with open(TLE_CACHE_FILE, "w") as f:
            json.dump(tle_data, f, indent=2)
        print(f"\nCached {len(tle_data)} satellites to {TLE_CACHE_FILE}")
    elif TLE_CACHE_FILE.exists():
        print("Network failed — loading cache fallback...")
        with open(TLE_CACHE_FILE) as f:
            tle_data = json.load(f)

    if not tle_data:
        print("No TLE data available. Exiting.")
        sys.exit(1)

    return tle_data

# ============================================================
# COORDINATE UTILITIES
# ============================================================

def ecef_to_geodetic(x, y, z):
    """Convert ECI-approximate XYZ (km) to lat/lon/alt using simple spherical model."""
    r = sqrt(x**2 + y**2 + z**2)
    lat = degrees(asin(z / r))
    lon = degrees(atan2(y, x))
    alt = r - EARTH_RADIUS_KM
    return lat, lon, alt


def propagate_satellite(satellite, epoch_dt, duration_seconds, step_seconds=60):
    """Propagate a satellite over a time range. Returns list of dicts."""
    records = []
    t = 0
    while t <= duration_seconds:
        dt = epoch_dt + timedelta(seconds=t)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
        err, pos, vel = satellite.sgp4(jd, fr)
        if err == 0:
            x, y, z = pos
            vx, vy, vz = vel
            lat, lon, alt = ecef_to_geodetic(x, y, z)
            speed = sqrt(vx**2 + vy**2 + vz**2)
            records.append({
                "time_offset": t,
                "datetime": dt.isoformat(),
                "x": x, "y": y, "z": z,
                "vx": vx, "vy": vy, "vz": vz,
                "lat": lat, "lon": lon, "alt_km": alt,
                "speed_km_s": speed
            })
        t += step_seconds
    return records


def get_epoch_dt(satellite):
    year = satellite.epochyr
    year += 2000 if year < 57 else 1900
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=satellite.epochdays - 1)

# ============================================================
# GROUND TRACK
# ============================================================

def generate_ground_track(tle_entry, duration_hours=2, step_seconds=60):
    sat = Satrec.twoline2rv(tle_entry["TLE_LINE1"], tle_entry["TLE_LINE2"])
    epoch_dt = get_epoch_dt(sat)
    records = propagate_satellite(sat, epoch_dt, duration_hours * 3600, step_seconds)
    return pd.DataFrame(records)


def plot_ground_track_folium(tle_data, duration_hours=2):
    print("\nGenerating interactive ground track map...")
    m = folium.Map(location=[0, 0], zoom_start=2, tiles="CartoDB dark_matter")

    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow", "white", "lime"]

    for idx, (catnr, entry) in enumerate(tle_data.items()):
        name = entry.get("OBJECT_NAME", f"SAT-{catnr}")
        df = generate_ground_track(entry, duration_hours=duration_hours)
        if df.empty:
            continue

        color = colors[idx % len(colors)]
        points = list(zip(df["lat"], df["lon"]))

        # Draw track line (split at dateline crossings)
        segment = []
        for i, (lat, lon) in enumerate(points):
            if i > 0 and abs(lon - points[i-1][1]) > 180:
                if len(segment) > 1:
                    folium.PolyLine(segment, color=color, weight=2, opacity=0.8).add_to(m)
                segment = []
            segment.append([lat, lon])
        if len(segment) > 1:
            folium.PolyLine(segment, color=color, weight=2, opacity=0.8).add_to(m)

        # Current position marker
        row = df.iloc[-1]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            popup=folium.Popup(
                f"<b>{name}</b><br>Alt: {row['alt_km']:.1f} km<br>"
                f"Speed: {row['speed_km_s']:.2f} km/s<br>"
                f"Lat: {row['lat']:.2f}° Lon: {row['lon']:.2f}°",
                max_width=200
            ),
            tooltip=name
        ).add_to(m)

    # Ground station
    gs = GROUND_STATION
    folium.Marker(
        location=[gs["lat"], gs["lon"]],
        popup=f"Ground Station: {gs['name']}",
        tooltip=gs["name"],
        icon=folium.Icon(color="red", icon="star")
    ).add_to(m)

    out_path = PLOTS_DIR / "ground_track_map.html"
    m.save(str(out_path))
    print(f"  Saved: {out_path}")
    return out_path

# ============================================================
# PASS PREDICTION
# ============================================================

def elevation_angle(sat_x, sat_y, sat_z, gs_lat, gs_lon, gs_alt_km):
    """Calculate elevation angle of satellite from ground station (degrees)."""
    lat_r = radians(gs_lat)
    lon_r = radians(gs_lon)
    r = EARTH_RADIUS_KM + gs_alt_km

    gx = r * cos(lat_r) * cos(lon_r)
    gy = r * cos(lat_r) * sin(lon_r)
    gz = r * sin(lat_r)

    dx, dy, dz = sat_x - gx, sat_y - gy, sat_z - gz
    range_vec = sqrt(dx**2 + dy**2 + dz**2)

    up_x = cos(lat_r) * cos(lon_r)
    up_y = cos(lat_r) * sin(lon_r)
    up_z = sin(lat_r)

    dot = dx * up_x + dy * up_y + dz * up_z
    el = degrees(asin(dot / range_vec))
    return el, range_vec


def predict_passes(tle_entry, gs=None, duration_hours=24, min_elevation=10.0):
    if gs is None:
        gs = GROUND_STATION

    sat = Satrec.twoline2rv(tle_entry["TLE_LINE1"], tle_entry["TLE_LINE2"])
    epoch_dt = get_epoch_dt(sat)

    passes = []
    current_pass = None
    step = 30  # seconds

    for t in range(0, duration_hours * 3600, step):
        dt = epoch_dt + timedelta(seconds=t)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        err, pos, vel = sat.sgp4(jd, fr)
        if err != 0:
            continue

        el, rng = elevation_angle(pos[0], pos[1], pos[2], gs["lat"], gs["lon"], gs["alt_km"])

        if el >= min_elevation:
            if current_pass is None:
                current_pass = {"aos": dt, "max_el": el, "max_el_time": dt, "range_km": rng}
            else:
                if el > current_pass["max_el"]:
                    current_pass["max_el"] = el
                    current_pass["max_el_time"] = dt
                    current_pass["range_km"] = rng
        else:
            if current_pass is not None:
                current_pass["los"] = dt
                current_pass["duration_min"] = (dt - current_pass["aos"]).seconds / 60
                passes.append(current_pass)
                current_pass = None

    return passes


def print_pass_predictions(tle_data, duration_hours=24):
    print("\n" + "=" * 60)
    print(f"PASS PREDICTIONS — Ground Station: {GROUND_STATION['name']}")
    print("=" * 60)

    all_passes = []
    for catnr, entry in tle_data.items():
        name = entry.get("OBJECT_NAME", f"SAT-{catnr}")
        passes = predict_passes(entry, duration_hours=duration_hours)
        for p in passes:
            p["satellite"] = name
            all_passes.append(p)
        print(f"{name}: {len(passes)} passes in {duration_hours}h")

    if all_passes:
        df = pd.DataFrame(all_passes)
        df = df.sort_values("aos")
        df_out = df[["satellite", "aos", "los", "duration_min", "max_el", "range_km"]].copy()
        df_out.columns = ["Satellite", "AOS", "LOS", "Duration (min)", "Max Elevation (°)", "Range (km)"]
        df_out["Max Elevation (°)"] = df_out["Max Elevation (°)"].round(1)
        df_out["Duration (min)"] = df_out["Duration (min)"].round(1)
        df_out["Range (km)"] = df_out["Range (km)"].round(0)
        print("\nUpcoming Passes:")
        print(df_out.to_string(index=False))
        df_out.to_csv(DATA_DIR / "pass_predictions.csv", index=False)
        print(f"\nSaved to {DATA_DIR / 'pass_predictions.csv'}")
    return all_passes

# ============================================================
# CONJUNCTION ANALYSIS
# ============================================================

def conjunction_analysis(tle_data, duration_hours=6, threshold_km=100):
    print("\n" + "=" * 60)
    print("CONJUNCTION ANALYSIS (Close Approach Detection)")
    print("=" * 60)

    sat_list = []
    for catnr, entry in tle_data.items():
        name = entry.get("OBJECT_NAME", f"SAT-{catnr}")
        sat = Satrec.twoline2rv(entry["TLE_LINE1"], entry["TLE_LINE2"])
        epoch_dt = get_epoch_dt(sat)
        sat_list.append({"name": name, "sat": sat, "epoch_dt": epoch_dt})

    conjunctions = []
    step = 60

    print(f"Analyzing {len(sat_list)} satellites over {duration_hours}h...")

    for t in tqdm(range(0, duration_hours * 3600, step), desc="Scanning"):
        positions = []
        for s in sat_list:
            dt = s["epoch_dt"] + timedelta(seconds=t)
            jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
            err, pos, _ = s["sat"].sgp4(jd, fr)
            if err == 0:
                positions.append({"name": s["name"], "pos": np.array(pos), "dt": dt})

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i]["pos"] - positions[j]["pos"])
                if dist < threshold_km:
                    conjunctions.append({
                        "sat_a": positions[i]["name"],
                        "sat_b": positions[j]["name"],
                        "distance_km": round(dist, 2),
                        "time": positions[i]["dt"].isoformat()
                    })

    if conjunctions:
        df = pd.DataFrame(conjunctions).drop_duplicates(subset=["sat_a", "sat_b"])
        df = df.sort_values("distance_km")
        print(f"\nFound {len(df)} conjunction events (< {threshold_km} km):")
        print(df.to_string(index=False))
        df.to_csv(DATA_DIR / "conjunctions.csv", index=False)
    else:
        print(f"No conjunctions found within {threshold_km} km threshold.")
        df = pd.DataFrame()

    return df

# ============================================================
# ALTITUDE & VELOCITY ANALYTICS
# ============================================================

def altitude_velocity_analytics(tle_data, duration_hours=3):
    print("\n" + "=" * 60)
    print("ALTITUDE & VELOCITY ANALYTICS")
    print("=" * 60)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Altitude Over Time", "Speed Over Time",
                        "Altitude Distribution", "Speed vs Altitude"],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )

    colors = px.colors.qualitative.Plotly

    for idx, (catnr, entry) in enumerate(tle_data.items()):
        name = entry.get("OBJECT_NAME", f"SAT-{catnr}")
        df = generate_ground_track(entry, duration_hours=duration_hours, step_seconds=120)
        if df.empty:
            continue

        color = colors[idx % len(colors)]
        show_legend = True

        fig.add_trace(go.Scatter(x=df["time_offset"] / 3600, y=df["alt_km"],
                                 name=name, line=dict(color=color),
                                 legendgroup=name, showlegend=show_legend), row=1, col=1)

        fig.add_trace(go.Scatter(x=df["time_offset"] / 3600, y=df["speed_km_s"],
                                 name=name, line=dict(color=color),
                                 legendgroup=name, showlegend=False), row=1, col=2)

        fig.add_trace(go.Histogram(x=df["alt_km"], name=name, opacity=0.6,
                                   marker_color=color,
                                   legendgroup=name, showlegend=False), row=2, col=1)

        fig.add_trace(go.Scatter(x=df["speed_km_s"], y=df["alt_km"], mode="markers",
                                 name=name, marker=dict(color=color, size=4, opacity=0.5),
                                 legendgroup=name, showlegend=False), row=2, col=2)

        print(f"  {name}: alt range {df['alt_km'].min():.0f}–{df['alt_km'].max():.0f} km | "
              f"avg speed {df['speed_km_s'].mean():.2f} km/s")

    fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
    fig.update_xaxes(title_text="Altitude (km)", row=2, col=1)
    fig.update_xaxes(title_text="Speed (km/s)", row=2, col=2)
    fig.update_yaxes(title_text="Altitude (km)", row=1, col=1)
    fig.update_yaxes(title_text="Speed (km/s)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Altitude (km)", row=2, col=2)

    fig.update_layout(
        title="ORBITX — Altitude & Velocity Analytics",
        template="plotly_dark",
        height=700
    )

    out_path = PLOTS_DIR / "altitude_velocity.html"
    fig.write_html(str(out_path))
    print(f"\nSaved interactive chart: {out_path}")
    return out_path

# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(tle_entry, num_samples=10000):
    print("\n" + "=" * 60)
    print("GENERATING TRAINING DATASET")
    print("=" * 60)

    sat = Satrec.twoline2rv(tle_entry["TLE_LINE1"], tle_entry["TLE_LINE2"])
    epoch_dt = get_epoch_dt(sat)

    inclination   = degrees(sat.inclo)
    eccentricity  = sat.ecco
    raan          = degrees(sat.nodeo)
    arg_perigee   = degrees(sat.argpo)
    mean_motion   = sat.no_kozai * 1440 / (2 * pi)
    mean_anomaly  = degrees(sat.mo)
    bstar         = sat.bstar

    rows = []
    for i in tqdm(range(num_samples), desc="Propagating"):
        delta_seconds = np.random.uniform(0, 86400 * 3)
        current_dt = epoch_dt + timedelta(seconds=delta_seconds)
        jd, fr = jday(current_dt.year, current_dt.month, current_dt.day,
                      current_dt.hour, current_dt.minute, current_dt.second)
        err, pos, vel = sat.sgp4(jd, fr)
        if err != 0:
            continue

        x, y, z = pos
        vx, vy, vz = vel
        lat, lon, alt = ecef_to_geodetic(x, y, z)
        speed = sqrt(vx**2 + vy**2 + vz**2)

        rows.append({
            "time_offset": delta_seconds,
            "inclination": inclination,
            "eccentricity": eccentricity,
            "raan": raan,
            "arg_perigee": arg_perigee,
            "mean_motion": mean_motion,
            "mean_anomaly": mean_anomaly,
            "bstar": bstar,
            "x": x, "y": y, "z": z,
            "vx": vx, "vy": vy, "vz": vz,
            "lat": lat, "lon": lon,
            "alt_km": alt,
            "speed_km_s": speed
        })

    df = pd.DataFrame(rows)
    print(f"\nDataset shape: {df.shape}")
    return df

# ============================================================
# ML MODEL TRAINING
# ============================================================

def train_models(df):
    print("\n" + "=" * 60)
    print("TRAINING ML MODELS")
    print("=" * 60)

    features = ["time_offset", "inclination", "eccentricity", "raan",
                "arg_perigee", "mean_motion", "mean_anomaly", "bstar"]
    targets  = ["x", "y", "z"]

    df = df.sort_values("time_offset").reset_index(drop=True)
    split = int(len(df) * 0.8)
    X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
    y_train, y_test = df[targets].iloc[:split],  df[targets].iloc[split:]
    # keep time_offset in y_test so 3D plot can sort by it
    y_test = y_test.copy()
    y_test["time_offset"] = df["time_offset"].iloc[split:].values

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": MultiOutputRegressor(
                                GradientBoostingRegressor(n_estimators=100, random_state=42), n_jobs=-1),
        "Neural Network":    Pipeline([
                                ("scaler", StandardScaler()),
                                ("mlp",    MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                                                        max_iter=500, random_state=42))
                             ])
    }

    results = {}
    best_model, best_r2, best_preds = None, -np.inf, None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2  = r2_score(y_test, preds)
        errors = np.sqrt(
            (y_test["x"].values - preds[:, 0])**2 +
            (y_test["y"].values - preds[:, 1])**2 +
            (y_test["z"].values - preds[:, 2])**2
        )
        avg_err = errors.mean()
        print(f"  MSE: {mse:.2f} | R²: {r2:.4f} | Avg Position Error: {avg_err:.2f} km")

        results[name] = {"mse": mse, "r2": r2, "avg_error_km": avg_err}

        if r2 > best_r2:
            best_r2, best_model, best_preds = r2, model, preds

    # Save metrics
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(DATA_DIR / "model_metrics.csv")
    print(f"\nBest model: {max(results, key=lambda k: results[k]['r2'])}")

    return models, results, y_test, best_preds, features

# ============================================================
# MODEL COMPARISON DASHBOARD
# ============================================================

def model_comparison_dashboard(results):
    print("\nGenerating model comparison dashboard...")

    names  = list(results.keys())
    r2s    = [results[n]["r2"] for n in names]
    errors = [results[n]["avg_error_km"] for n in names]
    mses   = [results[n]["mse"] for n in names]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["R² Score (Higher = Better)",
                                        "Avg Position Error km (Lower = Better)",
                                        "Mean Squared Error (Lower = Better)"])

    colors = ["#00d4ff" if v == max(r2s) else "#ff6b6b" for v in r2s]
    fig.add_trace(go.Bar(x=names, y=r2s, marker_color=colors, name="R²"), row=1, col=1)

    colors2 = ["#00d4ff" if v == min(errors) else "#ff6b6b" for v in errors]
    fig.add_trace(go.Bar(x=names, y=errors, marker_color=colors2, name="Avg Error"), row=1, col=2)

    colors3 = ["#00d4ff" if v == min(mses) else "#ff6b6b" for v in mses]
    fig.add_trace(go.Bar(x=names, y=mses, marker_color=colors3, name="MSE"), row=1, col=3)

    fig.update_layout(title="ORBITX — ML Model Comparison", template="plotly_dark",
                      height=450, showlegend=False)

    out_path = PLOTS_DIR / "model_comparison.html"
    fig.write_html(str(out_path))
    print(f"  Saved: {out_path}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(models, features):
    print("\nGenerating feature importance plots...")
    rf_model = models.get("Random Forest")
    if rf_model is None:
        return

    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=feat_df["Importance"], y=feat_df["Feature"],
        orientation="h", marker_color="#00d4ff"
    ))
    fig.update_layout(title="ORBITX — Feature Importance (Random Forest)",
                      template="plotly_dark", height=400,
                      xaxis_title="Importance", yaxis_title="Feature")

    out_path = PLOTS_DIR / "feature_importance.html"
    fig.write_html(str(out_path))
    print(f"  Saved: {out_path}")

# ============================================================
# 3D ORBIT VISUALIZATION (Interactive)
# ============================================================

def plot_orbits_3d_interactive(y_test, preds, tle_data):
    print("\nGenerating interactive 3D orbit visualization...")

    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    ex = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    ey = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    ez = EARTH_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()

    # Earth sphere
    fig.add_trace(go.Surface(x=ex, y=ey, z=ez,
                             colorscale=[[0, "#1a3a5c"], [1, "#2e7d32"]],
                             opacity=0.6, showscale=False, name="Earth"))

    # Sort by time so orbit traces a smooth continuous path
    sort_idx = y_test["time_offset"].argsort() if "time_offset" in y_test.columns else np.arange(len(y_test))
    y_sorted = y_test.iloc[sort_idx]
    p_sorted = preds[sort_idx]

    # True orbit — use SGP4 sequential propagation for clean smooth orbit
    iss_entry = list(tle_data.values())[0]
    orbit_df  = generate_ground_track(iss_entry, duration_hours=3, step_seconds=30)

    fig.add_trace(go.Scatter3d(
        x=orbit_df["x"].values, y=orbit_df["y"].values, z=orbit_df["z"].values,
        mode="lines", line=dict(color="#00d4ff", width=3),
        name="True Orbit (SGP4)"
    ))

    # Predicted orbit — sort by time so line is continuous
    fig.add_trace(go.Scatter3d(
        x=p_sorted[:, 0], y=p_sorted[:, 1], z=p_sorted[:, 2],
        mode="lines", line=dict(color="#ff6b6b", width=2, dash="dot"),
        name="ML Predicted"
    ))

    # Additional satellite tracks
    colors = ["#ffeb3b", "#76ff03", "#ff4081", "#40c4ff"]
    for idx, (catnr, entry) in enumerate(list(tle_data.items())[1:5]):
        name = entry.get("OBJECT_NAME", f"SAT-{catnr}")
        df = generate_ground_track(entry, duration_hours=1, step_seconds=60)
        if not df.empty:
            fig.add_trace(go.Scatter3d(
                x=df["x"], y=df["y"], z=df["z"],
                mode="lines", line=dict(color=colors[idx % len(colors)], width=2),
                name=name
            ))

    fig.update_layout(
        title="ORBITX — 3D Multi-Satellite Orbit Visualization",
        template="plotly_dark",
        scene=dict(
            xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)",
            bgcolor="#0a0a1a"
        ),
        height=700
    )

    out_path = PLOTS_DIR / "orbit_3d_interactive.html"
    fig.write_html(str(out_path))
    print(f"  Saved: {out_path}")

# ============================================================
# MAIN DASHBOARD (Master HTML)
# ============================================================

def generate_master_dashboard(results):
    print("\nGenerating master dashboard...")

    best_model_name = max(results, key=lambda k: results[k]["r2"])
    best_r2         = results[best_model_name]["r2"]
    best_err        = results[best_model_name]["avg_error_km"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ORBITX — Satellite Analytics Platform</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0a0a1a; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }}
  header {{ background: linear-gradient(135deg, #0d47a1, #1a237e);
            padding: 40px 20px; text-align: center; border-bottom: 2px solid #00d4ff; }}
  header h1 {{ font-size: 3em; color: #00d4ff; letter-spacing: 6px; }}
  header p  {{ color: #90caf9; margin-top: 10px; font-size: 1.1em; }}
  .badge {{ display: inline-block; background: #00d4ff22; border: 1px solid #00d4ff;
            color: #00d4ff; padding: 4px 12px; border-radius: 20px; margin: 4px; font-size: 0.85em; }}
  .stats {{ display: flex; justify-content: center; flex-wrap: wrap; gap: 20px; padding: 30px 20px; }}
  .stat-card {{ background: #111130; border: 1px solid #00d4ff33; border-radius: 12px;
                padding: 20px 30px; text-align: center; min-width: 180px; }}
  .stat-card .value {{ font-size: 2em; color: #00d4ff; font-weight: bold; }}
  .stat-card .label {{ font-size: 0.85em; color: #888; margin-top: 4px; }}
  .section {{ padding: 20px; max-width: 1200px; margin: 0 auto; }}
  .section h2 {{ color: #00d4ff; margin-bottom: 16px; font-size: 1.4em; letter-spacing: 2px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
  .card {{ background: #111130; border: 1px solid #1e1e4a; border-radius: 10px; overflow: hidden;
           transition: transform 0.2s, border-color 0.2s; }}
  .card:hover {{ transform: translateY(-4px); border-color: #00d4ff; }}
  .card a {{ display: block; padding: 24px; text-decoration: none; color: inherit; }}
  .card .icon {{ font-size: 2.5em; margin-bottom: 10px; }}
  .card h3 {{ color: #00d4ff; font-size: 1.1em; margin-bottom: 6px; }}
  .card p  {{ color: #888; font-size: 0.88em; line-height: 1.5; }}
  .model-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
  .model-table th {{ background: #00d4ff22; color: #00d4ff; padding: 10px; text-align: left; border-bottom: 1px solid #00d4ff44; }}
  .model-table td {{ padding: 10px; border-bottom: 1px solid #1e1e4a; font-size: 0.9em; }}
  .best {{ color: #76ff03; font-weight: bold; }}
  footer {{ text-align: center; padding: 30px; color: #444; border-top: 1px solid #1e1e4a; margin-top: 40px; }}
  footer a {{ color: #00d4ff; text-decoration: none; }}
</style>
</head>
<body>
<header>
  <h1>&#x1F6F0; ORBITX</h1>
  <p>Advanced Satellite Analytics &amp; ML Prediction Platform</p>
  <div style="margin-top:16px">
    <span class="badge">Python</span>
    <span class="badge">SGP4</span>
    <span class="badge">Machine Learning</span>
    <span class="badge">Plotly</span>
    <span class="badge">Folium</span>
    <span class="badge">scikit-learn</span>
    <span class="badge">Real-Time TLE</span>
    <span class="badge">10 Satellites</span>
  </div>
</header>

<div class="stats">
  <div class="stat-card"><div class="value">10</div><div class="label">Satellites Tracked</div></div>
  <div class="stat-card"><div class="value">4</div><div class="label">ML Models Compared</div></div>
  <div class="stat-card"><div class="value">{best_r2:.3f}</div><div class="label">Best R² Score</div></div>
  <div class="stat-card"><div class="value">{best_err:.1f} km</div><div class="label">Best Position Error</div></div>
  <div class="stat-card"><div class="value">10K+</div><div class="label">Training Samples</div></div>
</div>

<div class="section">
  <h2>&#x1F4CA; INTERACTIVE VISUALIZATIONS</h2>
  <div class="grid">
    <div class="card"><a href="plots/orbit_3d_interactive.html">
      <div class="icon">&#x1F30D;</div>
      <h3>3D Orbit Visualization</h3>
      <p>Interactive 3D plot of true vs predicted satellite orbits with Earth globe and multi-satellite tracks.</p>
    </a></div>
    <div class="card"><a href="plots/ground_track_map.html">
      <div class="icon">&#x1F5FA;</div>
      <h3>Ground Track Map</h3>
      <p>Live ground track of all satellites on an interactive world map with position markers.</p>
    </a></div>
    <div class="card"><a href="plots/altitude_velocity.html">
      <div class="icon">&#x1F4C8;</div>
      <h3>Altitude &amp; Velocity Analytics</h3>
      <p>Altitude decay, orbital speed, distribution analysis and speed-altitude correlation for all satellites.</p>
    </a></div>
    <div class="card"><a href="plots/model_comparison.html">
      <div class="icon">&#x1F916;</div>
      <h3>ML Model Comparison</h3>
      <p>R², MSE, and average position error compared across Linear Regression, Random Forest, Gradient Boosting, and Neural Network.</p>
    </a></div>
    <div class="card"><a href="plots/feature_importance.html">
      <div class="icon">&#x1F50D;</div>
      <h3>Feature Importance</h3>
      <p>Which orbital parameters matter most for position prediction — ranked by Random Forest importance scores.</p>
    </a></div>
  </div>
</div>

<div class="section" style="margin-top:30px">
  <h2>&#x1F9E0; MODEL PERFORMANCE</h2>
  <table class="model-table">
    <thead><tr><th>Model</th><th>R² Score</th><th>Avg Position Error (km)</th><th>MSE</th></tr></thead>
    <tbody>
"""
    for name, m in results.items():
        is_best = name == best_model_name
        cls = ' class="best"' if is_best else ''
        html += f'      <tr{cls}><td>{"★ " if is_best else ""}{name}</td><td>{m["r2"]:.4f}</td><td>{m["avg_error_km"]:.2f}</td><td>{m["mse"]:.2f}</td></tr>\n'

    html += f"""    </tbody>
  </table>
</div>

<div class="section" style="margin-top:30px">
  <h2>&#x1F4CB; DATA FILES</h2>
  <div class="grid">
    <div class="card"><a href="orbit_dataset.csv">
      <div class="icon">&#x1F4BE;</div><h3>Orbit Dataset (CSV)</h3>
      <p>10,000 propagated position samples with orbital elements, XYZ coordinates, lat/lon, altitude and speed.</p>
    </a></div>
    <div class="card"><a href="model_metrics.csv">
      <div class="icon">&#x1F4CA;</div><h3>Model Metrics (CSV)</h3>
      <p>Exported R², MSE and average error for all trained models.</p>
    </a></div>
    <div class="card"><a href="pass_predictions.csv">
      <div class="icon">&#x1F4E1;</div><h3>Pass Predictions (CSV)</h3>
      <p>Upcoming satellite passes over ground station: AOS, LOS, duration, max elevation and range.</p>
    </a></div>
  </div>
</div>

<footer>
  <p>Built with Python &bull; SGP4 &bull; scikit-learn &bull; Plotly &bull; Folium</p>
  <p style="margin-top:8px">Data source: <a href="https://celestrak.org" target="_blank">CelesTrak</a></p>
</footer>
</body>
</html>"""

    out_path = DATA_DIR / "index.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved: {out_path}")
    return out_path

# ============================================================
# CLI
# ============================================================

def build_argparser():
    p = argparse.ArgumentParser(
        prog="orbitx",
        description="ORBITX — Advanced Satellite Analytics Platform"
    )
    p.add_argument("--refresh",      action="store_true", help="Force refresh TLE data from CelesTrak")
    p.add_argument("--samples",      type=int, default=10000, help="Training dataset size (default: 10000)")
    p.add_argument("--passes",       action="store_true", help="Run pass prediction only")
    p.add_argument("--conjunction",  action="store_true", help="Run conjunction analysis only")
    p.add_argument("--groundtrack",  action="store_true", help="Generate ground track map only")
    p.add_argument("--no-ml",        action="store_true", help="Skip ML training")
    p.add_argument("--gs-lat",       type=float, default=GROUND_STATION["lat"], help="Ground station latitude")
    p.add_argument("--gs-lon",       type=float, default=GROUND_STATION["lon"], help="Ground station longitude")
    p.add_argument("--gs-name",      type=str,   default=GROUND_STATION["name"], help="Ground station name")
    return p

# ============================================================
# MAIN
# ============================================================

def main():
    parser = build_argparser()
    args   = parser.parse_args()

    GROUND_STATION["lat"]  = args.gs_lat
    GROUND_STATION["lon"]  = args.gs_lon
    GROUND_STATION["name"] = args.gs_name

    print("=" * 60)
    print("  ORBITX — SATELLITE ANALYTICS PLATFORM")
    print("=" * 60)
    print(f"  Ground Station : {GROUND_STATION['name']} ({GROUND_STATION['lat']}, {GROUND_STATION['lon']})")
    print(f"  Satellites     : {len(SATELLITES)}")
    print(f"  Output Dir     : {DATA_DIR.resolve()}")
    print("=" * 60)

    # --- TLE ---
    tle_data = fetch_all_tle(list(SATELLITES.keys()), force_refresh=args.refresh)

    # --- Single-module modes ---
    if args.passes:
        print_pass_predictions(tle_data)
        return

    if args.conjunction:
        conjunction_analysis(tle_data)
        return

    if args.groundtrack:
        plot_ground_track_folium(tle_data)
        return

    # --- Full pipeline ---
    print_pass_predictions(tle_data)
    conjunction_analysis(tle_data)
    plot_ground_track_folium(tle_data)
    altitude_velocity_analytics(tle_data)

    results = {}
    y_test  = None
    preds   = None
    models  = {}
    features = []

    if not args.no_ml:
        first_entry = list(tle_data.values())[0]
        df = generate_dataset(first_entry, num_samples=args.samples)

        dataset_path = DATA_DIR / "orbit_dataset.csv"
        df.to_csv(dataset_path, index=False)
        print(f"\nDataset saved: {dataset_path}")

        models, results, y_test, preds, features = train_models(df)
        model_comparison_dashboard(results)
        plot_feature_importance(models, features)

    if y_test is not None and preds is not None:
        plot_orbits_3d_interactive(y_test, preds, tle_data)

    if results:
        master = generate_master_dashboard(results)
        print(f"\nMaster dashboard: {master.resolve()}")

    print("\n" + "=" * 60)
    print("  ORBITX COMPLETED SUCCESSFULLY")
    print(f"  Open {DATA_DIR / 'index.html'} in your browser")
    print("=" * 60)

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
