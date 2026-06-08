# 🛰️ ORBITX — Advanced Satellite Analytics Platform

> A real-time satellite tracking, orbital prediction, and machine learning platform built with Python.

---

## 🚀 Features

| Feature | Description |
|---|---|
| **Multi-Satellite Tracking** | Tracks 10 real satellites (ISS, Hubble, NOAA, Landsat, Terra, Aqua, Suomi NPP) |
| **SGP4 Orbital Propagation** | High-fidelity position propagation using the SGP4/SDP4 model |
| **Ground Track Map** | Interactive world map with live satellite ground tracks (Folium) |
| **Pass Prediction** | Predicts upcoming overhead passes for any ground station (AOS, LOS, max elevation, range) |
| **Conjunction Analysis** | Detects close approaches between satellites below a configurable distance threshold |
| **Altitude & Velocity Analytics** | Interactive charts of altitude decay, orbital speed, and speed-altitude correlation |
| **ML Position Prediction** | Trains 4 models to predict X/Y/Z position from orbital elements |
| **Model Comparison Dashboard** | Side-by-side R², MSE, and position error for all models |
| **Feature Importance** | Ranked orbital parameter importance from Random Forest |
| **Interactive 3D Orbit** | 3D true vs predicted orbit with Earth sphere and multi-satellite tracks (Plotly) |
| **Master HTML Dashboard** | Single `index.html` linking all charts, maps, and data files |
| **CLI Interface** | Run individual modules or full pipeline via command line |

---

## 📊 ML Models

- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor (Multi-output)
- Neural Network — MLP with StandardScaler pipeline

**Targets:** X, Y, Z position in km (ECI frame)  
**Features:** time offset, inclination, eccentricity, RAAN, arg. of perigee, mean motion, mean anomaly, BSTAR

---

## 🛠️ Tech Stack

- **Python 3.10+**
- [sgp4](https://pypi.org/project/sgp4/) — orbital mechanics
- [scikit-learn](https://scikit-learn.org/) — machine learning
- [Plotly](https://plotly.com/python/) — interactive charts
- [Folium](https://python-visualization.github.io/folium/) — interactive maps
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data processing
- [tqdm](https://tqdm.github.io/) — progress bars
- [CelesTrak API](https://celestrak.org/) — live TLE data

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/orbitx.git
cd orbitx
pip install -r requirements.txt
```

---

## 🖥️ Usage

### Full pipeline (recommended)
```bash
python satiliteposi.py
```

### Custom ground station
```bash
python satiliteposi.py --gs-lat 51.5074 --gs-lon -0.1278 --gs-name "London"
```

### Larger training dataset
```bash
python satiliteposi.py --samples 50000
```

### Individual modules
```bash
python satiliteposi.py --passes          # Pass predictions only
python satiliteposi.py --conjunction     # Conjunction analysis only
python satiliteposi.py --groundtrack     # Ground track map only
python satiliteposi.py --no-ml           # Skip ML training
python satiliteposi.py --refresh         # Force refresh TLE from CelesTrak
```

---

## 📁 Output Files

```
data/
├── index.html                    ← Master portfolio dashboard
├── tle_cache.json                ← Cached TLE data
├── orbit_dataset.csv             ← 10K propagated samples
├── model_metrics.csv             ← ML model comparison metrics
├── pass_predictions.csv          ← Upcoming satellite passes
├── conjunctions.csv              ← Close approach events
└── plots/
    ├── orbit_3d_interactive.html ← 3D orbit visualization
    ├── ground_track_map.html     ← World map with ground tracks
    ├── altitude_velocity.html    ← Altitude & velocity charts
    ├── model_comparison.html     ← ML model comparison dashboard
    └── feature_importance.html   ← Feature importance chart
```

---

## 📡 Tracked Satellites

| NORAD ID | Name |
|---|---|
| 25544 | ISS (ZARYA) |
| 20580 | Hubble Space Telescope |
| 27424 | Aqua |
| 25994 | Terra |
| 33591 | NOAA-19 |
| 28654 | NOAA-18 |
| 43013 | NOAA-20 |
| 37849 | Suomi NPP |
| 39084 | Landsat 8 |
| 49260 | Landsat 9 |

---

## 📸 Screenshots

> Run the project to generate all interactive HTML dashboards in `data/`

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

Built by James — [GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-profile)
