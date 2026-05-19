import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os
from pathlib import Path

full_df = pd.read_csv(Path(__file__).parent / "data" / "highway_accidents.csv")

st.set_page_config(
   page_title="Highway Accident Predictor",
   page_icon="🚗",
   layout="wide",
)

MODEL_MAP = {
   "RandomForestRegressor": "models/randomforestregressor.pkl",
   "XGBoost": "models/xgboost.pkl",
   "LightGBM": "models/lightgbm.pkl",
   "Upload custom model": None,
}

def load_model(path):
   if Path(path).exists():
      return joblib.load(path)

# Input widgets

st.title("Highway Accident Predictor 🚗")

st.subheader("Predict number of accidents and vehicles needed")

model_index = st.selectbox("Select model", list(MODEL_MAP.keys()), index=0)

model = None
if model_index == "Upload custom model":
   uploaded_model = st.file_uploader("Upload model (.pkl)", type=["pkl"])
   if uploaded_model is not None:
      try:
         model = joblib.load(io.BytesIO(uploaded_model.read()))
         st.success("Custom model loaded.")
      except Exception as exc:
         st.error(f"Failed to load model: {exc}")
else:
   model = load_model(MODEL_MAP[model_index])

binary_map = {"No": 0, "Yes": 1}

hour = st.slider("Hour of the day", 0, 23, 8)
weekday = st.slider("Day of the week (0=Mon, 6=Sun)", 0, 6, 0)
month = st.slider("Month of the year", 1, 12, 1)
holiday = st.selectbox("Public holiday", ["No", "Yes"])
holiday = binary_map[holiday]

precipitation_mm = st.slider("Precipitation (mm)", 0.0, 60.0, 0.0)
visibility_km = st.slider("Visibility (km)", 0.3, 11.0, 8.0)
temperature_c = st.slider("Temperature (ºC)", 0.0, 35.0, 20.0)
wind_speed_kmh = st.slider("Wind speed (km/h)", 0.0, 100.0, 15.0)

traffic_density = st.slider("Traffic density", 1, 10, 5)

num_lanes = st.slider("Number of lanes", 1, 4, 2)
ilumination = st.selectbox("Road has ilumination", ["No", "Yes"])
ilumination = binary_map[ilumination]
work_zone = st.selectbox("Work zone present", ["No", "Yes"])
work_zone = binary_map[work_zone]

# Predict and show result
if st.button("Predict", type="primary", width='stretch'):
   if model is None:
      st.warning("Please load a model before predicting.")
   else:
      input_data = pd.DataFrame([{
         "hour": hour,
         "weekday": weekday,
         "month": month,
         "holiday": holiday,
         "precipitation_mm": precipitation_mm,
         "visibility_km": visibility_km,
         "temperature_c": temperature_c,
         "wind_speed_kmh": wind_speed_kmh,
         "traffic_density": traffic_density,
         "num_lanes": num_lanes,
         "ilumination": ilumination,
         "work_zone": work_zone,
      }])

      prediction = model.predict(input_data)[0]
      accidents = max(0, round(prediction))
      vehicles = accidents * 3


      st.subheader("Prediction Results")
      col1, col2 = st.columns(2)
      with col1:
         st.metric("Predicted Accidents", accidents)
      with col2:
         st.metric("Estimated Vehicles Needed", vehicles)

if full_df is not None:
   st.subheader("Dataset Sample")
   st.dataframe(full_df.head(), width='stretch')