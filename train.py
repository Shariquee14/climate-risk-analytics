"""Train and evaluate the models from the command line.
   python train.py                 -> uses data/weather_data.csv
   python train.py --csv my.csv    -> use your own data
"""
import argparse
import json
from pathlib import Path

import joblib

from weather_core import load_data, train_all

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(Path(__file__).parent / "data" / "weather_data.csv"))
ap.add_argument("--out", default=str(Path(__file__).parent / "models" / "weather_models.joblib"))
args = ap.parse_args()
df = load_data(args.csv)
art = train_all(df)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
joblib.dump(art, args.out, compress=3)
print(f"Rows: {len(df)} | train: {art['n_train']} | test: {art['n_test']} | split date: {art['split_date']}")
print("Best model:", art["best_model"])
print(json.dumps(art["comparison"], indent=2))
for h, m in art["horizon_metrics"].items():
    print(f"+{h}d", m)
