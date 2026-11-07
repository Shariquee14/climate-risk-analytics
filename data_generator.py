"""Generates a realistic synthetic daily climate dataset (central-India style:
hot summers, monsoon Jun-Sep, mild winters, slow warming trend).

Run:  python data_generator.py
Replace data/weather_data.csv with real data (same columns) to use real data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Monthly climatology (value at mid-month), interpolated smoothly through the year
MID = [15.5, 45, 74.5, 105, 135.5, 166, 196.5, 227.5, 258, 288.5, 319, 349.5]
TEMP = [16.5, 19.5, 25, 30.5, 34, 32, 27, 26, 26.5, 24.5, 20, 16.5]
HUM = [55, 45, 35, 28, 28, 50, 80, 85, 78, 58, 52, 55]
RAINP = [.05, .05, .04, .04, .06, .35, .68, .72, .5, .1, .05, .04]
RAINSCALE = [5, 5, 4, 4, 6, 12, 14, 14, 12, 8, 5, 5]
RANGE = [16, 17, 17, 17, 14, 10, 6, 6, 8, 13, 16, 16]
WIND = [6, 7, 8, 9, 10, 13, 14, 12, 9, 6, 5, 5]
PRES = [1016, 1014, 1011, 1007, 1003, 1000, 1000, 1002, 1005, 1010, 1014, 1016]


def _clim(table, doy):
    xs = [MID[-1] - 365] + MID + [MID[0] + 365]
    ys = [table[-1]] + table + [table[0]]
    return float(np.interp(doy, xs, ys))


def generate(start="2010-01-01", end="2024-12-31", seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    z = e = 0.0          # z: hidden "weather system" state, e: temperature noise
    wet_prev = 0
    rows = []
    for d in dates:
        doy = d.dayofyear
        z = 0.8 * z + rng.normal(0, 0.6)
        e = 0.6 * e + rng.normal(0, 1.1)
        p = min(0.95, _clim(RAINP, doy) * (0.75 + 0.5 * wet_prev) * np.exp(0.6 * z))
        wet = int(rng.random() < p)
        rain = round(max(0.2, rng.gamma(0.9, _clim(RAINSCALE, doy) / 0.9)), 1) if wet else 0.0
        trend = 0.03 * (d - dates[0]).days / 365.25          # ~0.3 C per decade warming
        t = _clim(TEMP, doy) + trend + e - 0.6 * z - 1.2 * wet
        spread = max(3.0, _clim(RANGE, doy) - 3 * wet + rng.normal(0, 1.0))
        hum = float(np.clip(_clim(HUM, doy) + 6 * z + 12 * wet + rng.normal(0, 4), 10, 100))
        wind = max(0.5, _clim(WIND, doy) + 1.5 * abs(z) + rng.normal(0, 1.5))
        pres = _clim(PRES, doy) - 2.2 * z + rng.normal(0, 0.8)
        rows.append({"date": d.strftime("%Y-%m-%d"),
                     "temp_max": round(t + spread / 2, 1), "temp_min": round(t - spread / 2, 1),
                     "humidity": round(hum, 1), "rainfall": rain,
                     "wind_speed": round(wind, 1), "pressure": round(pres, 1)})
        wet_prev = wet
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "weather_data.csv"
    out.parent.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
