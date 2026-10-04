import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Page Configuration
st.set_page_config(
    page_title="Gate leakage prediction - Amrita Vishwa Vidyapeetham",
    page_icon="⚡",
    layout="wide"
)

# Cache loaded models so the app runs fast
@st.cache_resource
def load_all_models():
    model_path = "models.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

models_dict = load_all_models()

# App Header & Institutional Affiliation
st.title("22nm FinFET Gate Leakage Predictor")
st.subheader("Amrita Vishwa Vidyapeetham, Amritapuri Campus")
st.caption("Developed by: **A S Akash** & **Dr. Sundar** | Department of VLSI Design")
st.markdown("---")

if models_dict is None:
    st.error("Model file `models.pkl` not found! Please run `python train_models.py` in your terminal first.")
    st.stop()

# Sidebar: Model Selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose ML Algorithm",
    list(models_dict.keys())
)

# Show model performance metrics in sidebar
selected_model_info = models_dict[model_choice]
st.sidebar.markdown(f"**Test Set $R^2$ Score:** `{selected_model_info['r2']:.4f}`")
st.sidebar.markdown(f"**Test Set RMSE:** `{selected_model_info['rmse']:.4f}`")

# Sidebar: Project Branding
st.sidebar.markdown("---")
st.sidebar.markdown("**Project Details**")
st.sidebar.write("🏫 **Amrita Vishwa Vidyapeetham**")
st.sidebar.write("📍 **Amritapuri Campus**")
st.sidebar.write("🔬 **Dept. of VLSI Design**")
st.sidebar.write("👨‍💻 **A S Akash & Dr. Sundar**")
st.sidebar.markdown("---")

# Sidebar: Device Parameters
st.sidebar.header("Device & Operating Parameters")
tox = st.sidebar.slider("Oxide Thickness (Tox in nm)", 0.8, 2.5, 1.2, 0.05)
vgs = st.sidebar.slider("Gate-Source Voltage (Vgs in V)", 0.0, 1.2, 0.6, 0.05)
lgate = st.sidebar.slider("Gate Length (Lgate in nm)", 14.0, 40.0, 22.0, 1.0)
hfin = st.sidebar.slider("Fin Height (Hfin in nm)", 20.0, 50.0, 30.0, 1.0)
wfin = st.sidebar.slider("Fin Width (Wfin in nm)", 5.0, 15.0, 8.0, 0.5)
na = st.sidebar.selectbox("Channel Doping (Na in cm-3)", [1e17, 5e17, 1e18])
temp = st.sidebar.slider("Temperature (T in K)", 300.0, 400.0, 300.0, 5.0)
nfin = st.sidebar.slider("Number of Fins (Nfin)", 1, 4, 1, 1)

# Main Prediction Section
st.write("Select a machine learning model and adjust the 22nm FinFET device parameters to predict $\\log_{10}(I_G)$ and $I_G$.")

if st.button("Predict Gate Leakage", type="primary"):
    # Format input features matching training column order
    input_data = pd.DataFrame([[tox, vgs, lgate, hfin, wfin, na, temp, nfin]], 
                              columns=['Tox_nm', 'Vgs_V', 'Lgate_nm', 'Hfin_nm', 'Wfin_nm', 'Na_cm3', 'T_K', 'Nfin'])
    
    # Run inference using the selected model
    model = selected_model_info["model"]
    pred_log_ig = float(model.predict(input_data)[0])
    pred_ig = 10 ** pred_log_ig

    # Display Results Dashboard
    st.subheader(f"Prediction Results ({model_choice})")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Predicted $\\log_{10}(I_G)$", value=f"{pred_log_ig:.4f}")
    with col2:
        st.metric(label="Predicted Gate Leakage ($I_G$)", value=f"{pred_ig:.3e} A")
    with col3:
        st.metric(label="Model Validation $R^2$", value=f"{selected_model_info['r2']:.4f}")
        
    st.success(f"Inference successfully calculated using the trained {model_choice} model!")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>M.Tech VLSI Thesis Research Project • Amrita Vishwa Vidyapeetham, Amritapuri</p>", unsafe_allow_html=True)