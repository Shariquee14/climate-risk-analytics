"""Flask web app: climate & weather forecasting dashboard.   Run: python app.py"""
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

import charts
from weather_core import (FIELDS, annual_summary, build_features, features_from_observation, forecast,
                          get_artifacts, load_data, monthly_climatology, temperature_trend, validate_observation)

BASE = Path(__file__).parent


def create_app(data_path=None, model_path=None, fast=False):
    app = Flask(__name__)
    df = load_data(data_path or BASE / "data" / "weather_data.csv")
    art = get_artifacts(df, model_path or BASE / "models" / "weather_models.joblib", fast=fast)
    feats = build_features(df).dropna()
    annual = annual_summary(df)
    trend = temperature_trend(annual)
    monthly = monthly_climatology(df)
    split_date = pd.Timestamp(art["split_date"])
    last = df.index[-1]
    cache = {}

    def cached(key, fn):
        if key not in cache:
            cache[key] = fn()
        return cache[key]

    def parse_date(raw):
        try:
            d = pd.Timestamp(str(raw).strip()).normalize()
        except Exception:
            return None
        return d if d in feats.index else None

    def run_forecast(date):
        fc = forecast(art, feats.loc[[date]], date)
        for f in fc:
            d = pd.Timestamp(f["date"])
            if d in df.index:
                f["actual_temp"] = float(df.loc[d, "temp_avg"])
                f["actual_rain"] = float(df.loc[d, "rainfall"])
                f["error"] = round(f["temp"] - f["actual_temp"], 1)
        return fc

    date_range_msg = f"Pick a date between {feats.index[0].date()} and {last.date()}."

    @app.route("/")
    def index():
        return render_template("index.html", fc=run_forecast(last), today=df.iloc[-1], last=last.date(),
                               chart=cached("history", lambda: charts.history_chart(df)),
                               n_days=len(df), art=art, start=df.index[0].date(), page="home")

    @app.route("/forecast")
    def forecast_page():
        raw, error, date = request.args.get("date"), None, last
        if raw:
            parsed = parse_date(raw)
            if parsed is None:
                error = f"'{raw}' is not a usable date. {date_range_msg}"
            else:
                date = parsed
        fc = run_forecast(date)
        return render_template("forecast.html", fc=fc, date=date.date(), error=error, page="forecast",
                               chart=charts.forecast_chart(df, date, fc), unseen=date >= split_date,
                               split=art["split_date"], dmin=feats.index[0].date(), dmax=last.date(),
                               obs=df.loc[date]), (400 if error else 200)

    @app.route("/custom", methods=["GET", "POST"])
    def custom():
        errors, fc, chart, form = [], None, None, {k: round(float(df.iloc[-1][k]), 1) for k in FIELDS}
        form["date"] = str(last.date())
        if request.method == "POST":
            form = request.form.to_dict()
            obs, date, errors = validate_observation(request.form)
            if not errors:
                fc = forecast(art, features_from_observation(obs, date), date)
                chart = charts.forecast_chart(df, pd.Timestamp("1900-01-01"), fc)
        return render_template("custom.html", fc=fc, chart=chart, errors=errors, form=form, fields=FIELDS,
                               page="custom"), (400 if errors else 200)

    @app.route("/climate")
    def climate():
        return render_template("climate.html", trend=trend, annual=annual, page="climate",
                               c_trend=cached("c_trend", lambda: charts.annual_trend_chart(annual, trend)),
                               c_clim=cached("c_clim", lambda: charts.climatograph(monthly)),
                               c_rain=cached("c_rain", lambda: charts.annual_rain_chart(annual)))

    @app.route("/models")
    def models():
        return render_template("models.html", art=art, page="models",
                               c_cmp=cached("m1", lambda: charts.model_compare_chart(art["comparison"])),
                               c_pa=cached("m2", lambda: charts.pred_actual_chart(art["test_pred"])),
                               c_imp=cached("m3", lambda: charts.importance_chart(art["importance"])),
                               c_hor=cached("m4", lambda: charts.horizon_chart(art["horizon_metrics"])))

    @app.route("/about")
    def about():
        return render_template("about.html", page="about", art=art, n_days=len(df), start=df.index[0].date(),
                               end=last.date())

    @app.route("/api/forecast")
    def api_forecast():
        raw = request.args.get("date")
        date = last if not raw else parse_date(raw)
        if date is None:
            return jsonify({"error": f"Invalid date. {date_range_msg}"}), 400
        return jsonify({"origin_date": str(date.date()), "model": art["best_model"], "forecast": run_forecast(date)})

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "best_model": art["best_model"], "rows": len(df)})

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", message="Page not found.", page=""), 404

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
