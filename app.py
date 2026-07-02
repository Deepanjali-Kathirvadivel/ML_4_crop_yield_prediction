import streamlit as st
import pickle
import numpy as np

# ---------------- Page ----------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)

# ---------------- Load Model ----------------
with open("model_pickle", "rb") as f:
    model = pickle.load(f)

with open("scaler_pickle", "rb") as f:
    scaler = pickle.load(f)

# ---------------- CSS ----------------
st.markdown("""
<style>

.main{
    background-color:#f7f9fc;
}

h1{
    text-align:center;
}

.stButton>button{
    width:100%;
    height:50px;
    border-radius:10px;
    background:#2E8B57;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#256f47;
    color:white;
}

.result{
    background:#d4edda;
    padding:20px;
    border-radius:12px;
    text-align:center;
    color:#155724;
    font-size:28px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.title("🌾 Crop Yield Prediction")

st.caption("Predict crop yield using Machine Learning")

st.divider()

# ---------------- Inputs ----------------

col1, col2 = st.columns(2)

with col1:

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=1200.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        value=27.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        value=65.0
    )

    soil_ph = st.number_input(
        "Soil pH",
        value=6.8
    )

    nitrogen = st.number_input(
        "Nitrogen (kg/ha)",
        value=120.0
    )

    phosphorus = st.number_input(
        "Phosphorus (kg/ha)",
        value=60.0
    )

with col2:

    potassium = st.number_input(
        "Potassium (kg/ha)",
        value=80.0
    )

    fertilizer = st.number_input(
        "Fertilizer (kg/ha)",
        value=180.0
    )

    irrigation = st.selectbox(
        "Irrigation",
        ["No", "Yes"]
    )

    farm_area = st.number_input(
        "Farm Area (ha)",
        value=5.5
    )

    sunshine = st.number_input(
        "Sunshine Hours",
        value=8.0
    )

irrigation = 1 if irrigation == "Yes" else 0

# ---------------- Prediction ----------------

if st.button("Predict Crop Yield"):

    features = np.array([[
        rainfall,
        temperature,
        humidity,
        soil_ph,
        nitrogen,
        phosphorus,
        potassium,
        fertilizer,
        irrigation,
        farm_area,
        sunshine
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    st.markdown(
        f"""
        <div class="result">
        🌾 Estimated Crop Yield<br><br>
        {prediction[0]:.2f} ton/ha
        </div>
        """,
        unsafe_allow_html=True
    )