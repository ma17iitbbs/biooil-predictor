import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('svr_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title("Bio-oil Yield Prediction")

temp = st.slider("Temperature (°C)", min_value=200, max_value=600, step=10)
time = st.slider("Time (seconds)", min_value=5, max_value=30, step=1)

if st.button("Predict"):
    input_data = np.array([[temp, time]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    st.success(f"Predicted Bio-oil Yield: {prediction:.2f}%")
