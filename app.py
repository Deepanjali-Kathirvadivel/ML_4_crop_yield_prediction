import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
df = pd.read_excel("Crop_Yield_Numeric_Ridge_Dataset.xlsx")
df.drop_duplicates(inplace=True)
X = df.drop(["farm_id", "crop_yield_ton_per_ha"], axis=1)
scaler = StandardScaler()
scaler.fit(X)
with open("model_pickle", "rb") as f:
    model = pickle.load(f)
st.set_page_config(page_title="Crop Yield Prediction",page_icon="🌾",layout="centered")
st.title("🌾 Crop Yield Prediction")
st.write("Predict Crop Yield using Machine Learning")
st.divider()
rainfall = st.number_input("Rainfall (mm)", 0.0, 5000.0, 1200.0)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 28.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
soil_ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
nitrogen = st.number_input("Nitrogen (kg/ha)", 0.0, 500.0, 100.0)
phosphorus = st.number_input("Phosphorus (kg/ha)", 0.0, 500.0, 60.0)
potassium = st.number_input("Potassium (kg/ha)", 0.0, 500.0, 80.0)
fertilizer = st.number_input("Fertilizer (kg/ha)", 0.0, 1000.0, 200.0)
irrigation = st.selectbox("Irrigation",[0, 1],format_func=lambda x: "No" if x == 0 else "Yes")
farm_area = st.number_input("Farm Area (ha)", 0.1, 100.0, 2.5)
sunshine = st.number_input("Sunshine Hours", 0.0, 24.0, 8.0)
if st.button("Predict Crop Yield"):
    input_df = pd.DataFrame({
        "rainfall_mm": [rainfall],
        "temperature_c": [temperature],
        "humidity_percent": [humidity],
        "soil_ph": [soil_ph],
        "nitrogen_kg_ha": [nitrogen],
        "phosphorus_kg_ha": [phosphorus],
        "potassium_kg_ha": [potassium],
        "fertilizer_kg_ha": [fertilizer],
        "irrigation": [irrigation],
        "farm_area_ha": [farm_area],
        "sunshine_hours": [sunshine]
    })
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"🌾 Predicted Crop Yield: {prediction[0]:.2f} tons/hectare")
    st.subheader("Input Details")
    st.dataframe(input_df)
