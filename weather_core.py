"""Core logic: data loading, feature engineering, model training, forecasting, climate stats."""
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REQUIRED = ["date", "temp_max", "temp_min", "humidity", "rainfall", "wind_speed", "pressure"]
BASE_COLS = ["temp_avg", "temp_max", "temp_min", "humidity", "rainfall", "wind_speed", "pressure"]
HORIZONS = list(range(1, 8))      # forecast 1..7 days ahead
RAIN_MM = 1.0                     # a "rain day" has >= 1 mm
FEATURES = BASE_COLS + ["temp_lag1", "temp_lag2", "temp_lag3", "temp_lag7", "temp_roll3", "temp_roll7",
                        "rain_roll3", "rain_roll7", "hum_roll3", "pressure_diff1", "pressure_diff3",
                        "doy_sin", "doy_cos"]

# Limits for the manual-observation form: key -> (min, max, label)
FIELDS = {"temp_max": (-10, 55, "Max temperature (°C)"), "temp_min": (-20, 45, "Min temperature (°C)"),
          "humidity": (0, 100, "Humidity (%)"), "rainfall": (0, 500, "Rainfall today (mm)"),
          "wind_speed": (0, 200, "Wind speed (km/h)"), "pressure": (900, 1100, "Pressure (hPa)")}


# ----------------------------------------------------------------- data
def clean_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df = df[REQUIRED].drop_duplicates("date").sort_values("date").set_index("date").asfreq("D")
    df = df.apply(pd.to_numeric, errors="coerce")
    df["rainfall"] = df["rainfall"].fillna(0.0)
    other = [c for c in df.columns if c != "rainfall"]
    df[other] = df[other].interpolate(limit=5, limit_direction="both")
    if df.isna().any().any():
        raise ValueError("Data has long gaps of missing values (more than 5 days in a row).")
    if len(df) < 400:
        raise ValueError("Need at least ~400 days of data to train.")
    df["temp_avg"] = ((df["temp_max"] + df["temp_min"]) / 2).round(1)
    return df[BASE_COLS]


def load_data(path):
    return clean_data(pd.read_csv(path))


# ------------------------------------------------------------- features
def build_features(df):
    f = df[BASE_COLS].copy()
    t = df["temp_avg"]
    for k in (1, 2, 3, 7):
        f[f"temp_lag{k}"] = t.shift(k)
    f["temp_roll3"], f["temp_roll7"] = t.rolling(3).mean(), t.rolling(7).mean()
    f["rain_roll3"], f["rain_roll7"] = df["rainfall"].rolling(3).sum(), df["rainfall"].rolling(7).sum()
    f["hum_roll3"] = df["humidity"].rolling(3).mean()
    f["pressure_diff1"], f["pressure_diff3"] = df["pressure"].diff(1), df["pressure"].diff(3)
    doy = df.index.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return f[FEATURES]


def _split(df, h, split_date):
    """Features at day t -> targets at day t+h. Train rows never peek into the test period."""
    X = build_features(df)
    nxt_rain = df["rainfall"].shift(-h)
    data = X.assign(y_temp=df["temp_avg"].shift(-h), y_rain=(nxt_rain >= RAIN_MM).astype(float).where(nxt_rain.notna()))
    data = data.dropna()
    tr = data[data.index + pd.Timedelta(days=h) < split_date]
    te = data[data.index >= split_date]
    return (tr[FEATURES], tr["y_temp"], tr["y_rain"].astype(int),
            te[FEATURES], te["y_temp"], te["y_rain"].astype(int))


def climatology(temp):
    m = temp.groupby(temp.index.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    ext = pd.concat([m.iloc[-15:], m, m.iloc[:15]])
    sm = ext.rolling(15, center=True).mean().iloc[15:-15]
    sm.index = range(1, 367)
    return sm


# ------------------------------------------------------------- training
def _reg_metrics(y, p):
    return {"mae": round(float(mean_absolute_error(y, p)), 3),
            "rmse": round(float(mean_squared_error(y, p) ** 0.5), 3),
            "r2": round(float(r2_score(y, p)), 3)}


def rain_proba(clf, X):
    if 1 not in list(clf.classes_):
        return np.zeros(len(X))
    return clf.predict_proba(X)[:, list(clf.classes_).index(1)]


def train_all(df, fast=False, test_fraction=0.2):
    n_rf, n_gb = (30, 40) if fast else (100, 200)
    zoo = {
        "Linear Regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=n_rf, max_depth=14, min_samples_leaf=3,
                                                       n_jobs=-1, random_state=42),
        "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=n_gb, max_depth=3, learning_rate=0.05,
                                                               subsample=0.8, random_state=42),
    }
    split_date = df.index[int(len(df) * (1 - test_fraction))]
    Xtr, ytr, _, Xte, yte, _ = _split(df, 1, split_date)
    n_train, n_test = len(Xtr), len(Xte)

    clim = climatology(df.loc[df.index < split_date, "temp_avg"])
    comparison = {
        "Baseline: Persistence (tomorrow = today)": _reg_metrics(yte, Xte["temp_avg"]),
        "Baseline: Climatology (seasonal average)": _reg_metrics(
            yte, clim.loc[(Xte.index + pd.Timedelta(days=1)).dayofyear].values),
    }
    fitted = {}
    for name, make in zoo.items():
        fitted[name] = make().fit(Xtr, ytr)
        comparison[name] = _reg_metrics(yte, fitted[name].predict(Xte))
    best = min(zoo, key=lambda k: comparison[k]["rmse"])

    reg, clf, hm = {}, {}, {}
    for h in HORIZONS:
        Xtr, ytr, rtr, Xte, yte, rte = _split(df, h, split_date)
        reg[h] = zoo[best]().fit(Xtr, ytr)
        clf[h] = RandomForestClassifier(n_estimators=n_rf, max_depth=12, min_samples_leaf=3,
                                        n_jobs=-1, random_state=42).fit(Xtr, rtr)
        proba = rain_proba(clf[h], Xte)
        pred = (proba >= 0.5).astype(int)
        hm[h] = {"reg": _reg_metrics(yte, reg[h].predict(Xte)),
                 "rain": {"accuracy": round(float(accuracy_score(rte, pred)), 3),
                          "precision": round(float(precision_score(rte, pred, zero_division=0)), 3),
                          "recall": round(float(recall_score(rte, pred, zero_division=0)), 3),
                          "f1": round(float(f1_score(rte, pred, zero_division=0)), 3),
                          "auc": round(float(roc_auc_score(rte, proba)), 3) if rte.nunique() > 1 else None}}
        if h == 1:
            persist = (Xte["rainfall"] >= RAIN_MM).astype(int)
            rain_base = {"persistence_accuracy": round(float(accuracy_score(rte, persist)), 3),
                         "always_dry_accuracy": round(float((rte == 0).mean()), 3),
                         "wet_day_share": round(float(rte.mean()), 3)}
            Xte1, yte1 = Xte, yte
    test_pred = pd.DataFrame({"actual": yte1.values, "predicted": reg[1].predict(Xte1)},
                             index=Xte1.index + pd.Timedelta(days=1))
    return {"sklearn_version": sklearn.__version__, "split_date": str(split_date.date()), "best_model": best,
            "reg": reg, "clf": clf, "comparison": comparison, "horizon_metrics": hm, "rain_baselines": rain_base,
            "importance": dict(zip(FEATURES, fitted["Random Forest"].feature_importances_.round(4))),
            "test_pred": test_pred, "last_date": str(df.index[-1].date()), "n_rows": len(df),
            "n_train": n_train, "n_test": n_test}


def get_artifacts(df, path, fast=False):
    """Load saved models if valid for this data + sklearn version, otherwise (re)train and save."""
    path = Path(path)
    if path.exists():
        try:
            art = joblib.load(path)
            if (art["sklearn_version"] == sklearn.__version__ and art["n_rows"] == len(df)
                    and art["last_date"] == str(df.index[-1].date())):
                return art
        except Exception:
            pass
    art = train_all(df, fast=fast)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, path, compress=3)
    return art


# ------------------------------------------------------------ forecasting
def describe(temp, rain_prob):
    if rain_prob >= 0.6:
        return "Rainy"
    if rain_prob >= 0.35:
        return "Chance of showers"
    if temp >= 38:
        return "Very hot"
    if temp >= 32:
        return "Hot"
    if temp >= 24:
        return "Warm"
    if temp >= 16:
        return "Pleasant"
    if temp >= 10:
        return "Cool"
    return "Cold"


def forecast(art, X, start):
    """7-day forecast from a one-row feature frame X observed on date `start`."""
    out = []
    for h in HORIZONS:
        t = float(art["reg"][h].predict(X)[0])
        p = float(rain_proba(art["clf"][h], X)[0])
        r = art["horizon_metrics"][h]["reg"]["rmse"]
        d = pd.Timestamp(start) + timedelta(days=h)
        out.append({"day": h, "date": d.strftime("%Y-%m-%d"), "weekday": d.strftime("%a"),
                    "temp": round(t, 1), "low": round(t - 1.28 * r, 1), "high": round(t + 1.28 * r, 1),
                    "rain_prob": int(round(p * 100)), "condition": describe(t, p)})
    return out


def validate_observation(form):
    """Parse the manual-observation form. Returns (obs_dict, date, errors)."""
    obs, errors = {}, []
    for key, (lo, hi, label) in FIELDS.items():
        raw = str(form.get(key, "")).strip()
        try:
            val = float(raw)
            if not np.isfinite(val):
                raise ValueError
        except ValueError:
            errors.append(f"{label}: enter a number.")
            continue
        if not lo <= val <= hi:
            errors.append(f"{label}: must be between {lo} and {hi}.")
        obs[key] = val
    if not errors and obs["temp_max"] < obs["temp_min"]:
        errors.append("Max temperature cannot be lower than min temperature.")
    try:
        date = pd.Timestamp(str(form.get("date", "")).strip())
        if pd.isna(date) or not 1900 <= date.year <= 2100:
            raise ValueError
    except Exception:
        date = None
        errors.append("Date: enter a valid date (YYYY-MM-DD).")
    return obs, date, errors


def features_from_observation(obs, date):
    """One-row feature frame from a single day's readings (history assumed steady: simplification)."""
    t = round((obs["temp_max"] + obs["temp_min"]) / 2, 1)
    doy = date.dayofyear
    row = {"temp_avg": t, "temp_max": obs["temp_max"], "temp_min": obs["temp_min"], "humidity": obs["humidity"],
           "rainfall": obs["rainfall"], "wind_speed": obs["wind_speed"], "pressure": obs["pressure"],
           "temp_lag1": t, "temp_lag2": t, "temp_lag3": t, "temp_lag7": t, "temp_roll3": t, "temp_roll7": t,
           "rain_roll3": obs["rainfall"], "rain_roll7": obs["rainfall"], "hum_roll3": obs["humidity"],
           "pressure_diff1": 0.0, "pressure_diff3": 0.0,
           "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)), "doy_cos": float(np.cos(2 * np.pi * doy / 365.25))}
    return pd.DataFrame([row])[FEATURES]


# ------------------------------------------------------- climate analysis
def annual_summary(df):
    g = df.groupby(df.index.year)
    out = pd.DataFrame({"days": g.size(), "temp": g["temp_avg"].mean(), "rain": g["rainfall"].sum(),
                        "rain_days": g["rainfall"].apply(lambda s: int((s >= RAIN_MM).sum())),
                        "hottest": g["temp_max"].max()})
    return out[out["days"] >= 360]


def temperature_trend(annual):
    if len(annual) < 3:
        return None
    r = stats.linregress(annual.index.values, annual["temp"].values)
    return {"slope_per_decade": round(float(r.slope) * 10, 3), "slope": float(r.slope),
            "intercept": float(r.intercept), "p_value": round(float(r.pvalue), 4), "r2": round(float(r.rvalue ** 2), 3)}


def monthly_climatology(df):
    temp = df.groupby(df.index.month)["temp_avg"].mean()
    tot = df["rainfall"].groupby([df.index.year, df.index.month]).sum().groupby(level=1).mean()
    return pd.DataFrame({"temp": temp, "rain": tot})
