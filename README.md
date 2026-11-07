# Climate & Weather Forecasting using Machine Learning (Minor Project)

A web app that forecasts **daily average temperature** and **chance of rain for the next 7 days**, and analyses
**long-term climate trends** (warming trend, seasonal pattern, rainfall).

## Quick start
```bash
pip install -r requirements.txt
python app.py            # open http://127.0.0.1:5000
```
First start trains the models (~30–60 s) if `models/weather_models.joblib` is missing or was made with a different
scikit-learn version. Later starts are instant.

## Pages
| Page | What it does |
|---|---|
| Dashboard | 7-day forecast from the latest data + last 6 months chart |
| Forecast | Pick any date → 7-day forecast, compared with what actually happened (backtest) |
| Your Readings | Type in one day's readings → 7-day forecast |
| Climate Trends | Annual warming trend (°C/decade, p-value), climatograph, rainfall totals |
| Model Performance | Model comparison vs baselines, per-horizon accuracy, rain metrics, feature importance |
| About | Method, limitations, data note |

JSON API: `/api/forecast?date=YYYY-MM-DD`, `/api/health`.

## Files
- `data_generator.py` – creates the synthetic dataset (`python data_generator.py`)
- `weather_core.py` – data cleaning, features, training, forecasting, climate statistics
- `charts.py` – matplotlib charts · `app.py` – Flask app · `train.py` – train/evaluate from the command line
- `tests/test_project.py` – 17 automated tests: `python -m unittest discover -s tests -v`

## Method (for viva)
1. **Features:** today's readings, temperature lags (1/2/3/7 days), rolling means/sums, pressure change, day-of-year sin/cos.
2. **Targets:** temperature and "rain ≥ 1 mm" for each of the next 1–7 days (one model per day ahead).
3. **Models:** Linear Regression, Random Forest, Gradient Boosting vs baselines *persistence* and *climatology*;
   best by RMSE forecasts temperature. Random Forest classifier gives rain probability.
4. **Evaluation:** time-ordered split (train on earlier 80 %, test on latest 20 %), no shuffling, no future leakage.
5. **Metrics:** MAE, RMSE, R² (temperature); accuracy, precision, recall, F1, ROC-AUC (rain).

## Using real data
The bundled data is **synthetic** (central-India-style climate). To use real data (Open-Meteo, NASA POWER, Kaggle…),
save a CSV with columns `date, temp_max, temp_min, humidity, rainfall, wind_speed, pressure` as
`data/weather_data.csv`, then run `python train.py` (or just start the app). Short gaps (≤5 days) are filled automatically.
