"""Run:  python -m unittest discover -s tests -v   (from the project folder)"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_generator
import weather_core as wc
from app import create_app

DATA = Path(__file__).resolve().parent.parent / "data" / "weather_data.csv"


class DataTests(unittest.TestCase):
    def test_generator_is_deterministic_and_complete(self):
        a = data_generator.generate("2020-01-01", "2020-12-31")
        b = data_generator.generate("2020-01-01", "2020-12-31")
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 366)
        self.assertFalse(a.isna().any().any())

    def test_bundled_csv_is_valid_and_physical(self):
        df = wc.load_data(DATA)
        self.assertGreater(len(df), 5000)
        self.assertTrue((df["temp_max"] >= df["temp_min"]).all())
        self.assertTrue(df["humidity"].between(0, 100).all())
        self.assertTrue((df["rainfall"] >= 0).all())
        self.assertTrue(df.index.is_monotonic_increasing)

    def test_clean_data_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            wc.clean_data(pd.DataFrame({"date": ["2020-01-01"], "temp_max": [30]}))   # missing columns
        short = data_generator.generate("2020-01-01", "2020-03-01")
        with self.assertRaises(ValueError):
            wc.clean_data(short)                                                       # too short

    def test_clean_data_fills_short_gaps(self):
        raw = data_generator.generate("2020-01-01", "2021-12-31")
        raw.loc[10:12, "temp_max"] = np.nan
        raw.loc[20, "rainfall"] = np.nan
        out = wc.clean_data(raw)
        self.assertFalse(out.isna().any().any())


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = wc.load_data(DATA)

    def test_features_have_no_lookahead(self):
        X = wc.build_features(self.df)
        d = pd.Timestamp("2018-07-15")
        i = self.df.index.get_loc(d)
        self.assertAlmostEqual(X.loc[d, "temp_lag1"], self.df["temp_avg"].iloc[i - 1])
        self.assertAlmostEqual(X.loc[d, "temp_lag7"], self.df["temp_avg"].iloc[i - 7])
        self.assertAlmostEqual(X.loc[d, "rain_roll3"], self.df["rainfall"].iloc[i - 2:i + 1].sum())
        self.assertAlmostEqual(X.loc[d, "temp_roll7"], self.df["temp_avg"].iloc[i - 6:i + 1].mean())

    def test_split_targets_and_no_train_test_overlap(self):
        split = pd.Timestamp("2022-01-01")
        for h in (1, 7):
            Xtr, ytr, rtr, Xte, yte, rte = wc._split(self.df, h, split)
            d = Xte.index[5]
            self.assertAlmostEqual(yte.loc[d], self.df["temp_avg"].loc[d + pd.Timedelta(days=h)])
            self.assertEqual(rte.loc[d], int(self.df["rainfall"].loc[d + pd.Timedelta(days=h)] >= wc.RAIN_MM))
            self.assertLess(Xtr.index.max() + pd.Timedelta(days=h), split)
            self.assertGreaterEqual(Xte.index.min(), split)
            self.assertFalse(Xtr.isna().any().any() or Xte.isna().any().any())

    def test_climate_stats(self):
        annual = wc.annual_summary(self.df)
        self.assertEqual(len(annual), 15)
        trend = wc.temperature_trend(annual)
        self.assertGreater(trend["slope_per_decade"], 0)           # built-in warming trend is detected
        monthly = wc.monthly_climatology(self.df)
        self.assertEqual(len(monthly), 12)
        self.assertGreater(monthly.loc[7, "rain"], 10 * monthly.loc[1, "rain"])   # monsoon vs winter
        self.assertGreater(monthly.loc[5, "temp"], monthly.loc[1, "temp"])


class ValidationTests(unittest.TestCase):
    good = {"date": "2024-05-01", "temp_max": "40", "temp_min": "27", "humidity": "30",
            "rainfall": "0", "wind_speed": "10", "pressure": "1004"}

    def test_valid(self):
        obs, date, errs = wc.validate_observation(self.good)
        self.assertEqual(errs, [])
        self.assertEqual(date, pd.Timestamp("2024-05-01"))

    def test_invalid_cases(self):
        for change in ({"humidity": "150"}, {"temp_max": "abc"}, {"temp_max": "10", "temp_min": "20"},
                       {"date": "not-a-date"}, {"pressure": "nan"}, {"rainfall": "-1"}, {"wind_speed": ""}):
            _, _, errs = wc.validate_observation({**self.good, **change})
            self.assertTrue(errs, f"expected an error for {change}")


class ModelAndAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.tmp.name) / "m.joblib"
        cls.app = create_app(model_path=cls.model_path, fast=True)
        cls.client = cls.app.test_client()
        cls.df = wc.load_data(DATA)
        cls.art = wc.get_artifacts(cls.df, cls.model_path, fast=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_models_beat_baselines(self):
        c = self.art["comparison"]
        best = c[self.art["best_model"]]["rmse"]
        self.assertLess(best, c["Baseline: Climatology (seasonal average)"]["rmse"])
        self.assertLess(best, c["Baseline: Persistence (tomorrow = today)"]["rmse"])
        self.assertGreater(c[self.art["best_model"]]["r2"], 0.9)
        for h in wc.HORIZONS:                                      # every horizon trained, error is sane
            self.assertLess(self.art["horizon_metrics"][h]["reg"]["rmse"], 3.0)
            self.assertGreater(self.art["horizon_metrics"][h]["rain"]["auc"], 0.7)

    def test_artifacts_are_cached_and_reloaded(self):
        self.assertTrue(self.model_path.exists())
        again = wc.get_artifacts(self.df, self.model_path, fast=True)
        self.assertEqual(again["best_model"], self.art["best_model"])

    def test_forecast_structure(self):
        X = wc.build_features(self.df).dropna().loc[[self.df.index[-1]]]
        fc = wc.forecast(self.art, X, self.df.index[-1])
        self.assertEqual(len(fc), 7)
        for f in fc:
            self.assertTrue(f["low"] < f["temp"] < f["high"])
            self.assertTrue(0 <= f["rain_prob"] <= 100)
            self.assertTrue(-5 < f["temp"] < 50)
        self.assertEqual(fc[0]["date"], "2025-01-01")

    def test_seasonal_sanity(self):
        def run(date, **kw):
            obs = {"temp_max": 40, "temp_min": 27, "humidity": 30, "rainfall": 0, "wind_speed": 10, "pressure": 1004}
            obs.update(kw)
            d = pd.Timestamp(date)
            return wc.forecast(self.art, wc.features_from_observation(obs, d), d)
        summer = run("2025-05-10")
        winter = run("2025-01-10", temp_max=25, temp_min=9, humidity=55, pressure=1016)
        monsoon = run("2025-07-20", temp_max=29, temp_min=24, humidity=88, rainfall=25, pressure=998)
        self.assertGreater(np.mean([f["temp"] for f in summer]), np.mean([f["temp"] for f in winter]) + 10)
        self.assertGreater(np.mean([f["rain_prob"] for f in monsoon]), np.mean([f["rain_prob"] for f in summer]) + 15)

    def test_all_pages_load(self):
        for url in ("/", "/forecast", "/forecast?date=2024-08-15", "/custom", "/climate", "/models", "/about"):
            r = self.client.get(url)
            self.assertEqual(r.status_code, 200, url)
            self.assertIn(b"<html", r.data)
        self.assertIn(b"data:image/png;base64,", self.client.get("/models").data)

    def test_forecast_page_backtest_columns_and_bad_dates(self):
        r = self.client.get("/forecast?date=2024-08-15")
        self.assertIn(b"Actual temp", r.data)
        self.assertIn(b"unseen test period", r.data)
        for bad in ("garbage", "1999-01-01", "2099-01-01", "2010-01-02"):
            self.assertEqual(self.client.get(f"/forecast?date={bad}").status_code, 400, bad)

    def test_custom_form(self):
        ok = self.client.post("/custom", data=TestForm.good)
        self.assertEqual(ok.status_code, 200)
        self.assertIn(b"Result", ok.data)
        bad = self.client.post("/custom", data={**TestForm.good, "humidity": "500"})
        self.assertEqual(bad.status_code, 400)
        self.assertIn(b"Humidity", bad.data)
        self.assertEqual(self.client.post("/custom", data={}).status_code, 400)

    def test_api(self):
        j = self.client.get("/api/forecast").get_json()
        self.assertEqual(len(j["forecast"]), 7)
        self.assertEqual(j["origin_date"], "2024-12-31")
        self.assertEqual(self.client.get("/api/forecast?date=bad").status_code, 400)
        self.assertEqual(self.client.get("/api/health").get_json()["status"], "ok")
        self.assertEqual(self.client.get("/nope").status_code, 404)


class TestForm:
    good = {"date": "2025-03-01", "temp_max": "33", "temp_min": "18", "humidity": "35",
            "rainfall": "0", "wind_speed": "8", "pressure": "1011"}


if __name__ == "__main__":
    unittest.main()
