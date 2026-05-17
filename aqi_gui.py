import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Set page config
st.set_page_config(page_title="AQI Prediction System", layout="centered", page_icon="🌤️")

st.title("🌤️ Air Quality Index (AQI) Predictor")
st.write("Enter the pollutant concentrations to predict the Air Quality Index.")

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('./outputs/models/best_model.pkl')
        scaler = joblib.load('./outputs/models/scaler.pkl')
        le = joblib.load('./outputs/models/label_encoder.pkl')
        return model, scaler, le
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

model, scaler, le = load_models()

if model is None:
    st.error("Model files not found! Please wait for the ML pipeline to finish generating the `.pkl` files in `./outputs/models/`.")
    st.stop()

# Input layout
st.subheader("Base Pollutants")
col1, col2, col3 = st.columns(3)
with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=30.0, help="Fine particulate matter")
    so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, value=10.0, help="Sulfur dioxide")
with col2:
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=60.0, help="Respirable particulate matter")
    co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.0, help="Carbon monoxide")
with col3:
    no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, value=20.0, help="Nitrogen dioxide")
    ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=30.0, help="Ozone")

st.subheader("Additional Pollutants")
col4, col5, col6 = st.columns(3)
with col4:
    no = st.number_input("NO (µg/m³)", min_value=0.0, value=10.0, help="Nitric oxide")
    benzene = st.number_input("Benzene (µg/m³)", min_value=0.0, value=1.0)
with col5:
    nox = st.number_input("NOx (ppb)", min_value=0.0, value=20.0, help="Nitrogen oxides")
    toluene = st.number_input("Toluene (µg/m³)", min_value=0.0, value=5.0)
with col6:
    nh3 = st.number_input("NH3 (µg/m³)", min_value=0.0, value=10.0, help="Ammonia")

st.markdown("---")

if st.button("Predict AQI Category", type="primary", use_container_width=True):
    # Feature Engineering exactly as done in training
    pm_ratio = pm25 / (pm10 + 1)
    pm_product = pm25 * pm10
    nox_ratio = no2 / (co + 0.1)
    pollutant_sum = pm25 + pm10 + no2 + so2 + co
    pm25_squared = pm25 ** 2
    pm10_squared = pm10 ** 2
    no2_squared = no2 ** 2

    # Feature List matching the exact order used in aqi_forecasting_ml_project.py
    features = [
        pm25, pm10, no2, so2, co, ozone,
        no, nox, nh3, benzene, toluene,
        pm_ratio, pm_product, nox_ratio, pollutant_sum,
        pm25_squared, pm10_squared, no2_squared
    ]
    
    # Predict
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    prediction_encoded = model.predict(features_scaled)[0]
    prediction_label = le.inverse_transform([prediction_encoded])[0]
    
    # Display logic
    color_map = {
        "Good": "#2ecc71",
        "Moderate": "#f1c40f",
        "Poor": "#e67e22",
        "Very Poor": "#e74c3c",
        "Severe": "#8e44ad"
    }
    color = color_map.get(prediction_label, "#34495e")
    
    st.markdown(f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white; margin-top: 10px;">
        <h2 style="margin: 0; font-family: sans-serif;">Predicted AQI Category:</h2>
        <h1 style="margin: 10px 0 0 0; font-size: 3.5em; font-family: sans-serif; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">{prediction_label}</h1>
    </div>
    """, unsafe_allow_html=True)
