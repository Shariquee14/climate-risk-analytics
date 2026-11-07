"""Matplotlib charts returned as base64 PNG strings (thread-safe: no pyplot)."""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib.figure import Figure

BLUE, RED, GREEN, GREY, ORANGE = "#2563eb", "#dc2626", "#16a34a", "#64748b", "#f59e0b"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _new(w=9, h=3.6):
    fig = Figure(figsize=(w, h))
    ax = fig.subplots()
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def _b64(fig):
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    return base64.b64encode(buf.getvalue()).decode()


def history_chart(df, days=180):
    d = df.iloc[-days:]
    fig, ax = _new()
    ax.fill_between(d.index, d["temp_min"], d["temp_max"], color=BLUE, alpha=0.15, label="Min–Max")
    ax.plot(d.index, d["temp_avg"], color=BLUE, lw=1.5, label="Average temp")
    ax.set_ylabel("Temperature (°C)")
    ax2 = ax.twinx()
    ax2.bar(d.index, d["rainfall"], color=GREEN, alpha=0.6, width=1.0, label="Rainfall")
    ax2.set_ylabel("Rainfall (mm)")
    ax2.set_ylim(0, max(60, d["rainfall"].max() * 1.5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False)
    return _b64(fig)


def forecast_chart(df, start, fc):
    import pandas as pd
    start = pd.Timestamp(start)
    fig, ax = _new(9, 3.4)
    hist = df.loc[start - pd.Timedelta(days=14):start, "temp_avg"] if start in df.index else None
    if hist is not None and len(hist):
        ax.plot(hist.index, hist.values, color=GREY, marker="o", ms=3, label="Recent observed")
    dates = pd.to_datetime([f["date"] for f in fc])
    ax.plot(dates, [f["temp"] for f in fc], color=RED, ls="--", marker="o", ms=4, label="Forecast")
    ax.fill_between(dates, [f["low"] for f in fc], [f["high"] for f in fc], color=RED, alpha=0.15,
                    label="Likely range (80%)")
    act = df["temp_avg"].reindex(dates).dropna()
    if len(act):
        ax.scatter(act.index, act.values, color=GREEN, zorder=5, s=28, label="Actual (backtest)")
    ax.set_ylabel("Avg temperature (°C)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    return _b64(fig)


def annual_trend_chart(annual, trend):
    fig, ax = _new(9, 3.4)
    ax.plot(annual.index, annual["temp"], marker="o", color=ORANGE, label="Annual mean temp")
    if trend:
        ax.plot(annual.index, trend["intercept"] + trend["slope"] * annual.index, color=RED, ls="--",
                label=f"Trend: {trend['slope_per_decade']:+.2f} °C / decade")
    ax.set_ylabel("°C")
    ax.legend(fontsize=8, frameon=False)
    return _b64(fig)


def climatograph(monthly):
    fig, ax = _new(9, 3.4)
    ax.bar(MONTHS, monthly["rain"].values, color=GREEN, alpha=0.6)
    ax.set_ylabel("Avg monthly rainfall (mm)")
    ax2 = ax.twinx()
    ax2.plot(MONTHS, monthly["temp"].values, color=RED, marker="o")
    ax2.set_ylabel("Avg temperature (°C)")
    return _b64(fig)


def annual_rain_chart(annual):
    fig, ax = _new(9, 3.2)
    ax.bar(annual.index.astype(str), annual["rain"], color=BLUE, alpha=0.7)
    ax.axhline(annual["rain"].mean(), color=RED, ls="--", lw=1, label=f"Mean {annual['rain'].mean():.0f} mm")
    ax.set_ylabel("Total rainfall (mm)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, frameon=False)
    return _b64(fig)


def model_compare_chart(comparison):
    fig, ax = _new(9, 3.2)
    names = list(comparison)
    vals = [comparison[n]["rmse"] for n in names]
    cols = [GREY if n.startswith("Baseline") else BLUE for n in names]
    ax.barh([n.replace("Baseline: ", "") for n in names], vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("RMSE on unseen test data (°C) — lower is better")
    ax.invert_yaxis()
    return _b64(fig)


def pred_actual_chart(test_pred, n=120):
    d = test_pred.iloc[:n]
    fig, ax = _new(9, 3.2)
    ax.plot(d.index, d["actual"], color=GREY, label="Actual")
    ax.plot(d.index, d["predicted"], color=RED, ls="--", label="Predicted (1 day ahead)")
    ax.set_ylabel("°C")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax.legend(fontsize=8, frameon=False)
    return _b64(fig)


def importance_chart(imp, top=10):
    items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:top]
    fig, ax = _new(9, 3.2)
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color=BLUE)
    ax.set_xlabel("Importance (Random Forest)")
    return _b64(fig)


def horizon_chart(hm):
    hs = sorted(hm)
    fig, ax = _new(9, 3.0)
    ax.plot(hs, [hm[h]["reg"]["rmse"] for h in hs], marker="o", color=RED, label="RMSE")
    ax.plot(hs, [hm[h]["reg"]["mae"] for h in hs], marker="o", color=BLUE, label="MAE")
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("Error (°C)")
    ax.legend(fontsize=8, frameon=False)
    return _b64(fig)
