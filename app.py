import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/best_aqi_model.pkl")

# ---------------- AQI CLASSIFICATION ----------------
def classify_aqi(aqi_value):
    if 0 <= aqi_value <= 50:
        return "Good ✅"
    elif 51 <= aqi_value <= 100:
        return "Moderate 🙂"
    elif 101 <= aqi_value <= 200:
        return "Unhealthy 😷"
    elif 201 <= aqi_value <= 300:
        return "Very Unhealthy ⚠️"
    elif 301 <= aqi_value <= 500:
        return "Hazardous ☠️"
    else:
        return "Invalid AQI"

st.title("🌍 Air Quality Index Prediction System")

st.subheader("Enter Pollutant Details")

# ----------- SLIDERS (VARIABLE NAMES MATCH BELOW) -----------

co_val = st.slider("CO (ppm)", 0.0, 50.0, 2.5)
nox_val = st.slider("NOx (ppb)", 0.0, 2000.0, 150.0)
no2_val = st.slider("NO2 (ppb)", 0.0, 2000.0, 100.0)

c6h6_val = st.slider("C6H6 (µg/m³)", 0.0, 100.0, 10.0)
nmhc_val = st.slider("NMHC (µg/m³)", 0.0, 1000.0, 100.0)

pm10_val = st.slider("PM10 (µg/m³)", 0.0, 500.0, 120.0)
o3_val = st.slider("O3 Sensor Value", 0.0, 2000.0, 1000.0)
so2_val = st.slider("SO2 (ppb)", 0.0, 500.0, 20.0)

temp_val = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
rh_val = st.slider("Humidity (%)", 0.0, 100.0, 55.0)
ah_val = st.slider("Absolute Humidity", 0.0, 5.0, 0.75)

# ---------------- PREDICTION ----------------
if st.button("Predict AQI"):

    new_sample = {
        "CO(GT)": co_val,
        "NOx(GT)": nox_val,
        "NO2(GT)": no2_val,
        "C6H6(GT)": c6h6_val,
        "NMHC(GT)": nmhc_val,
        "PM10": pm10_val,
        "PT08.S5(O3)": o3_val,
        "SO2": so2_val,
        "T": temp_val,
        "RH": rh_val,
        "AH": ah_val
    }

    input_df = pd.DataFrame([new_sample])

    prediction = model.predict(input_df)
    prediction = float(prediction[0])

    category = classify_aqi(prediction)

    st.subheader("📊 Predicted AQI")
    st.success(f"{prediction:.2f}")

    st.subheader("Health Category")
    st.info(category)

    # Save results
    results_file = "aqi_predictions_log.csv"

    new_sample["Predicted_AQI"] = prediction
    new_sample["Category"] = category

    df_save = pd.DataFrame([new_sample])

    if os.path.exists(results_file):
        df_save.to_csv(results_file, mode="a", header=False, index=False)
    else:
        df_save.to_csv(results_file, index=False)

    st.write("Prediction saved to aqi_predictions_log.csv")
    st.write(model.named_steps["preprocessor"].feature_names_in_)