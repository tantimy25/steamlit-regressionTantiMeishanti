import streamlit as st
import pandas as pd
import joblib
# If your model used a specific scaler or encoder, 
# ensure scikit-learn is imported even if not called directly.
import sklearn 

model = joblib.load("regression_model.sav")

st.title("Aplikasi Prediksi Regresi")

model = joblib.load("regression_model.sav")

st.write("Masukkan nilai fitur:")

f1 = st.number_input("Fitur 1", value=0.0)
f2 = st.number_input("Fitur 2", value=0.0)

if st.button("Prediksi"):
    data = np.array([[f1, f2]])
    hasil = model.predict(data)
    st.success(f"Hasil Prediksi: {hasil[0]}")







