import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# ---------------------------------------
# Load Dataset
# ---------------------------------------
df = pd.read_excel("Crop_Yield_Numeric_Ridge_Dataset.xlsx")

df.drop_duplicates(inplace=True)

# ---------------------------------------
# Prepare Features
# ---------------------------------------
X = df.drop(["farm_id", "crop_yield_ton_per_ha"], axis=1)

# Train Scaler
scaler = StandardScaler()
scaler.fit(X)

# ---------------------------------------
# Load Model
# ---------------------------------------
with open("model_pickle", "rb") as f:
    model = pickle.load(f)

# ---------------------------------------
# Streamlit Page
# ---------------------------------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Crop Yield Prediction")
st.write("Predict Crop Yield (Ton/Ha) using Ridge Regression")

st.divider()

# ---------------------------------------
# User Inputs
# ---------------------------------------

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=5000.0,
    value=800.0
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=28.0
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=65.0
)

soil_ph = st.number_input(
    "Soil pH",
    min_value=3.0,
    max_value=10.0,
    value=6.5
)

nitrogen = st.number_input(
    "Nitrogen",
    min_value=0.0,
    value=90.0
)

phosphorus = st.number_input(
    "Phosphorus",
    min_value=0.0,
    value=45.0
)

potassium = st.number_input(
    "Potassium",
    min_value=0.0,
    value=40.0
)

# ---------------------------------------
# Prediction
# ---------------------------------------

if st.button("Predict Crop Yield"):

    input_df = pd.DataFrame({
        "rainfall_mm":[rainfall],
        "temperature_c":[temperature],
        "humidity_percent":[humidity],
        "soil_ph":[soil_ph],
        "nitrogen":[nitrogen],
        "phosphorus":[phosphorus],
        "potassium":[potassium]
    })

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    st.success(
        f"🌾 Predicted Crop Yield : {prediction[0]:.2f} Ton/Ha"
    )

    st.subheader("Input Details")
    st.dataframe(input_df)