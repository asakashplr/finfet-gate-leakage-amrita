import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# FINFET GATE LEAKAGE PREDICTION
# =========================================================

st.set_page_config(
    page_title="22nm FinFET Gate Leakage Predictor",
    page_icon="⚡",
    layout="wide"
)
st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:15px;">
    <h4>Developed by A. S. Akash & Dr Sundaresh Amrita Chaithanya</h4>
    <p>M.Tech VLSI Design</p>
    <p>Amrita School of Engineering, Amritapuri</p>
    <p>Semiconductor device Gate Leakage current Prediction using Artificial Intelligence</p>
    <small>Academic Project | 2026</small>
</div>
""", unsafe_allow_html=True)
st.title("⚡ 22nm FinFET Gate Leakage Predictor")
st.markdown("### Developed by A S AKASH and Dr Sundar , Amrita School of Engineering  Amritapuri| VLSI Design")

# =========================================================
# LOAD TRAINED MODELS
# =========================================================

@st.cache_resource
def load_models():
    with open("models.pkl", "rb") as file:
        return pickle.load(file)

if not os.path.exists("models.pkl"):
    st.error("models.pkl file not found.")
    st.stop()

models = load_models()

# =========================================================
# SIDEBAR - MODEL SELECTION
# =========================================================

st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)

selected_model = models[model_name]["model"]
model_r2 = models[model_name]["r2"]
model_rmse = models[model_name]["rmse"]

st.sidebar.metric("Holdout R²", f"{model_r2:.4f}")
st.sidebar.metric("Holdout RMSE", f"{model_rmse:.4f}")

# =========================================================
# DEVICE PARAMETERS
# =========================================================

st.header("FinFET Device Parameters")

col1, col2 = st.columns(2)

with col1:
    Tox = st.slider("Oxide Thickness (nm)", 0.05, 2.50, 0.85, 0.05)
    Vgs = st.slider("Gate Voltage Vgs (V)", 0.0, 1.2, 0.8, 0.01)
    Lgate = st.slider("Gate Length (nm)", 14.0, 40.0, 22.0, 1.0)
    Hfin = st.slider("Fin Height (nm)", 5.0, 35.0, 11.0, 1.0)

with col2:
    Wfin = st.slider("Fin Width (nm)", 5.0, 15.0, 11.0, 1.0)
    Na = st.selectbox(
        "Doping Concentration (cm⁻³)",
        [1e17, 5e17, 1e18]
    )
    Temp = st.slider("Temperature (K
