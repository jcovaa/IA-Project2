"""
train.py - Train the model and save it to model.pkl
Run once before launching the app
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


DATA_PATH = Path(__file__).parent / "data" / "highway_accidents.csv"

FEATURES = [
   "hour", "weekday", "month", "holiday",
   "precipitation_mm", "visibility_km", "temperature_c", "wind_speed_kmh",
   "traffic_density",
   "num_lanes", "ilumination", "work_zone",
]
TARGET = "num_accidents"

def train_and_test():
   print("Loading data...")
   df = pd.read_csv(DATA_PATH)

   x = df[FEATURES]
   y = df[TARGET]

   X_train, X_test, y_train, y_test = train_test_split(
      x, y, test_size=0.2, random_state=42
   )

   print("Training RandomForestRegressor...")
   model = RandomForestRegressor()
   model.fit(X_train, y_train)

   y_pred = model.predict(X_test)
   mae = mean_absolute_error(y_test, y_pred)
   rmse = mean_squared_error(y_test, y_pred) ** 0.5
   r2 = r2_score(y_test, y_pred)

   print(f"\nModel performance on test set:")
   print(f"  MAE  : {mae:.3f}")
   print(f"  RMSE : {rmse:.3f}")
   print(f"  R^2  : {r2:.3f}")

   return model

MODEL_MAP = {
   "RandomForestRegressor": RandomForestRegressor(random_state=42, max_depth= 20, max_features='sqrt', min_samples_leaf=5, min_samples_split=20, n_estimators =200),
   "XGBoost": XGBRegressor(random_state=42, colsample_bytree=0.8, learning_rate=0.05, max_depth=3, min_child_weight=2, n_estimators=200,reg_alpha=0.1, reg_lambda=5.0, subsample=0.8),
   "LightGBM": LGBMRegressor(random_state=42, colsample_bytree=0.8, learning_rate=0.1, max_depth=5, min_child_samples=50, n_estimators=100, num_leaves=31, reg_alpha=1.0, reg_lambda=5.0, subsample=0.8),
}

def train_and_export_models():
   print("Loading data...")
   df = pd.read_csv(DATA_PATH)

   x = df[FEATURES]
   y = df[TARGET]

   model_dir = Path(__file__).parent / "models"

   for name, model in MODEL_MAP.items():
      print(f"Training {name}...")
      model.fit(x, y)
      filename = model_dir / f"{name}.pkl"
      joblib.dump(model, filename)
      print(f"Saved {filename}")

   print("All models exported.")


if __name__ == "__main__":
   train_and_export_models()