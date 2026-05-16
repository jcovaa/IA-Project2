import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "highway_accidents.csv"

def load_data():
   if DATA_PATH.exists():
      return pd.read_csv(DATA_PATH)
   else:
      raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
   
# Max value for each feature
def max_values():
   df = load_data()
   return {
      "hour": df["hour"].max(),
      "weekday": df["weekday"].max(),
      "month": df["month"].max(),
      "holiday": df["holiday"].max(),
      "precipitation_mm": df["precipitation_mm"].max(),
      "visibility_km": df["visibility_km"].max(),
      "temperature_c": df["temperature_c"].max(),
      "wind_speed_kmh": df["wind_speed_kmh"].max(),
      "traffic_density": df["traffic_density"].max(),
      "num_lanes": df["num_lanes"].max(),
      "ilumination": df["ilumination"].max(),
      "work_zone": df["work_zone"].max(),
   }

# Min value for each feature
def min_values():
   df = load_data()
   return {
      "hour": df["hour"].min(),
      "weekday": df["weekday"].min(),
      "month": df["month"].min(),
      "holiday": df["holiday"].min(),
      "precipitation_mm": df["precipitation_mm"].min(),
      "visibility_km": df["visibility_km"].min(),
      "temperature_c": df["temperature_c"].min(),
      "wind_speed_kmh": df["wind_speed_kmh"].min(),
      "traffic_density": df["traffic_density"].min(),
      "num_lanes": df["num_lanes"].min(),
      "ilumination": df["ilumination"].min(),
      "work_zone": df["work_zone"].min(),
   }

def main():
   print("Max values for each feature:")
   for feature, value in max_values().items():
      print(f"  {feature}: {value}")
   
   print("\nMin values for each feature:")
   for feature, value in min_values().items():
      print(f"  {feature}: {value}")

if __name__ == "__main__":
   main()