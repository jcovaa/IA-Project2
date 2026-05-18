import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

st.set_page_config(
   page_title="Highway Accident Predictor",
   page_icon="🚗",
   layout="wide",
)

MODEL_MAP = {
   "RandomForestRegressor": "models/model.pkl",
}

def load_model(path):
   if Path(path).exists():
      return joblib.load(path)

# Input widgets

st.title("Highway Accident Predictor 🚗")

model_index = st.selectbox("Select model", ["RandomForestRegressor"], index=0)

model = load_model(MODEL_MAP[model_index])

hour = st.slider("Hour of the day", 0, 23, 8)
weekday = st.slider("Day of the week (0=Mon, 6=Sun)", 0, 6, 0)
month = st.slider("Month of the year", 1, 12, 1)
holiday = st.selectbox("Public holiday", [0, 1])

precipitation_mm = st.slider("Precipitation (mm)", 0.0, 60.0, 0.0)
visibility_km = st.slider("Visibility (km)", 0.3, 11.0, 8.0)
temperature_c = st.slider("Temperature (ºC)", 0.0, 35.0, 20.0)
wind_speed_kmh = st.slider("Wind speed (km/h)", 0.0, 100.0, 15.0)

traffic_density = st.slider("Traffic density", 1, 10, 5)

num_lanes = st.slider("Number of lanes", 1, 4, 2)
ilumination = st.selectbox("Road has ilumination", [0, 1])
work_zone = st.selectbox("Work zone present", [0, 1])

# Predict and show result
if st.button("Predict"):
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

   st.write("Predicted accidents:", accidents)
   st.write("Vehicles needed:", vehicles)