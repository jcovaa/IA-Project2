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

DATA_PATH = Path(__file__).parent / "data" / "highway_accidents.csv"
MODEL_PATH = Path(__file__).parent / "model.pkl"

FEATURES = [
   "hour", "weekday", "month", "holiday",
   "precipitation_mm", "visibility_km", "temperature_c", "wind_speed_kmh",
   "traffic_density",
   "num_lanes", "ilumination", "work_zone",
]
TARGET = "num_accidents"

def train_and_save():
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

   joblib.dump(model, MODEL_PATH)
   print(f"\nModel saved to {MODEL_PATH}")
   return model

if __name__ == "__main__":
   train_and_save()