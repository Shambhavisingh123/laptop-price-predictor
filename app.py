# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# ---- Load Model ----
model = joblib.load('outputs/laptop_model.pkl')
expected_features = model.feature_names_in_

# ---- Page Setup ----
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")
st.title("💻 Laptop Price Predictor")
st.markdown("Estimate laptop prices based on specifications")

# ---- Sidebar Inputs ----
st.sidebar.header("📋 Laptop Specifications")

# Basic inputs
ram = st.sidebar.selectbox("RAM (GB)", [4, 8, 16, 32])
weight = st.sidebar.slider("Weight (kg)", 0.5, 5.0, step=0.1)
inches = st.sidebar.slider("Screen Size (inches)", 10.0, 18.0, step=0.1)
screenw = st.sidebar.selectbox("Screen Width (px)", [1366, 1920, 2560, 2880, 3840])
screenh = st.sidebar.selectbox("Screen Height (px)", [768, 1080, 1440, 1600, 2160])
touchscreen = st.sidebar.radio("Touchscreen", ['Yes', 'No'])
ips = st.sidebar.radio("IPS Panel", ['Yes', 'No'])
retina = st.sidebar.radio("Retina Display", ['Yes', 'No'])
cpu_freq = st.sidebar.slider("CPU Frequency (GHz)", 1.0, 4.5, step=0.1)
cpu_company = st.sidebar.selectbox("CPU Brand", ['Intel', 'Samsung'])

# Storage inputs
primary_storage = st.sidebar.selectbox("Primary Storage (GB)", [128, 256, 512, 1024, 2048])
primary_type = st.sidebar.selectbox("Primary Storage Type", ['HDD', 'SSD', 'Hybrid'])
secondary_storage = st.sidebar.selectbox("Secondary Storage (GB)", [0, 128, 256, 512, 1024, 2048])
secondary_type = st.sidebar.selectbox("Secondary Storage Type", ['No', 'SSD', 'Hybrid'])

# Categorical One-hot encoded values
company = st.sidebar.selectbox("Company", ['Apple', 'Asus', 'Chuwi', 'Dell', 'Fujitsu', 'Google', 'HP', 'Huawei', 'LG', 'Lenovo', 'MSI', 'Mediacom', 'Microsoft', 'Razer', 'Samsung', 'Toshiba', 'Vero', 'Xiaomi'])
typename = st.sidebar.selectbox("Laptop Type", ['Notebook', 'Ultrabook', 'Gaming', 'Netbook', 'Workstation'])
os = st.sidebar.selectbox("Operating System", ['Windows 10', 'Windows 10 S', 'Windows 7', 'macOS', 'Mac OS X', 'Linux', 'Chrome OS', 'No OS'])
gpu_company = st.sidebar.selectbox("GPU Company", ['Intel', 'Nvidia', 'ARM'])

# ---- Build Full Input Dict with Default 0s ----
input_dict = {col: 0 for col in expected_features}

# Fill numeric/base values
input_dict['Ram'] = ram
input_dict['Weight'] = weight
input_dict['Inches'] = inches
input_dict['ScreenW'] = screenw
input_dict['ScreenH'] = screenh
input_dict['Touchscreen'] = 1 if touchscreen == 'Yes' else 0
input_dict['IPSpanel'] = 1 if ips == 'Yes' else 0
input_dict['RetinaDisplay'] = 1 if retina == 'Yes' else 0
input_dict['CPU_freq'] = cpu_freq
input_dict['PrimaryStorage'] = primary_storage
input_dict['SecondaryStorage'] = secondary_storage

# Fill one-hot encoded values
if f'Company_{company}' in input_dict:
    input_dict[f'Company_{company}'] = 1
if f'TypeName_{typename}' in input_dict:
    input_dict[f'TypeName_{typename}'] = 1
if f'OS_{os}' in input_dict:
    input_dict[f'OS_{os}'] = 1
if f'CPU_company_{cpu_company}' in input_dict:
    input_dict[f'CPU_company_{cpu_company}'] = 1
if f'PrimaryStorageType_{primary_type}' in input_dict:
    input_dict[f'PrimaryStorageType_{primary_type}'] = 1
if f'SecondaryStorageType_{secondary_type}' in input_dict:
    input_dict[f'SecondaryStorageType_{secondary_type}'] = 1
if f'GPU_company_{gpu_company}' in input_dict:
    input_dict[f'GPU_company_{gpu_company}'] = 1

# ---- Convert to DataFrame ----
input_df = pd.DataFrame([input_dict])

# ---- Main Display ----
st.markdown("---")
st.subheader("🧾 Preview of Input Data")
with st.expander("🔍 Click to view model input features"):
    st.dataframe(input_df)

# ---- Prediction ----
st.markdown("---")
st.subheader("💡 Predicted Price")
if st.button("🚀 Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Laptop Price: €{prediction:.2f}")

# ---- Footer ----
st.markdown("---")
st.markdown("Made with ❤️ by Shambhavi")
