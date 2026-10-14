import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="Gate leakage prediction - Amrita Vishwa Vidyapeetham",
    page_icon="⚡",
    layout="wide"
)

# 2. Session State for Comparison History Table
if "history" not in st.session_state:
    st.session_state.history = []

# 3. Cache loaded models
@st.cache_resource
def load_all_models():
    model_path = "models.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

models_dict = load_all_models()

# 4. App Header & Institutional Affiliation
st.title("22nm FinFET Gate Leakage Predictor")
st.subheader("Amrita Vishwa Vidyapeetham, Amritapuri Campus")
st.caption("Developed by: **A S Akash** & **Dr. Sundar** | Department of ECE -MTech VLSI Design 2025-27")
st.markdown("---")

if models_dict is None:
    st.error("Model file `models.pkl` not found! Please run `python train_models.py` first.")
    st.stop()

# 5. Sidebar: Model Selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose ML Algorithm", list(models_dict.keys()))

selected_model_info = models_dict[model_choice]
st.sidebar.markdown(f"**Holdout Test $R^2$ Score:** `{selected_model_info['r2']:.4f}`")
st.sidebar.markdown(f"**Holdout Test RMSE:** `{selected_model_info['rmse']:.4f}`")

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Details**")
st.sidebar.write("🏫 **Amrita Vishwa Vidyapeetham**")
st.sidebar.write("📍 **Amritapuri Campus**")
st.sidebar.write("🔬 **Dept. of VLSI Design**")
st.sidebar.write("👨‍💻 **A S Akash & Dr. Sundar**")
st.sidebar.markdown("---")

# 6. Sidebar: Device Parameters
st.sidebar.header("Device & Operating Parameters")
tox = st.sidebar.slider("Oxide Thickness (Tox in nm)",0,0.05,0,10,0.4,0.65, 0.8, 2.5, 0.85, 0.05)
vgs = st.sidebar.slider("Gate-Source Voltage (Vgs in V)", 0.0, 1.2, 0.8, 0.05)
lgate = st.sidebar.slider("Gate Length (Lgate in nm)", 14.0, 40.0, 22.0, 1.0)
hfin = st.sidebar.slider("Fin Height (Hfin in nm)", 20.0, 50.0, 22.0, 1.0)
wfin = st.sidebar.slider("Fin Width (Wfin in nm)", 5.0, 15.0, 12.0, 0.5)
na = st.sidebar.selectbox("Channel Doping (Na in cm-3)", [1e17, 5e17, 1e18])
temp = st.sidebar.slider("Temperature (T in K)", 300.0, 400.0, 300.0, 5.0)
nfin = st.sidebar.slider("Number of Fins (Nfin)", 1, 4, 4, 1)

# 7. Main Prediction Section
st.write("Select a machine learning model and adjust the 22nm FinFET device parameters to predict $\\log_{10}(I_G)$ and $I_G$.")

if st.button("Predict Gate Leakage", type="primary"):
    input_data = pd.DataFrame(
        [[tox, vgs, lgate, hfin, wfin, na, temp, nfin]], 
        columns=['Tox_nm', 'Vgs_V', 'Lgate_nm', 'Hfin_nm', 'Wfin_nm', 'Na_cm3', 'T_K', 'Nfin']
    )
    
    model = selected_model_info["model"]
    raw_pred_log_ig = float(model.predict(input_data)[0])
    
    # Rescale to physical 10^-12 to 10^-2 range
    pred_log_ig = raw_pred_log_ig + 40.0
    pred_ig = 10 ** pred_log_ig

    # Append to comparison history table
    st.session_state.history.append({
        "Model": model_choice,
        "Tox (nm)": tox,
        "Vgs (V)": vgs,
        "Lgate (nm)": lgate,
        "Hfin (nm)": hfin,
        "Wfin (nm)": wfin,
        "Na (cm⁻³)": f"{na:.1e}",
        "Temp (K)": temp,
        "Nfin": nfin,
        "log10(Ig)": f"{pred_log_ig:.4f}",
        "Ig (A)": f"{pred_ig:.3e}"
    })

    # Results Layout
    st.subheader(f"Prediction Results ({model_choice})")
    col_metrics, col_gauge = st.columns([1, 1.2])

    with col_metrics:
        st.metric(label="Predicted $\\log_{10}(I_G)$", value=f"{pred_log_ig:.4f}")
        st.metric(label="Predicted Gate Leakage ($I_G$)", value=f"{pred_ig:.3e} A")
        st.metric(label="Holdout Test $R^2$", value=f"{selected_model_info['r2']:.4f}")

        # Physical 2-tier threshold at -8.5 (~3.16 nA standby limit)
        if pred_log_ig < -8.5:
            st.success("🟢 **Leakage Status: LOW (Safe Standby)**")
        else:
            st.error("🔴 **Leakage Status: HIGH (Excessive Tunneling)**")

    with col_gauge:
        # Semicircular 2-zone gauge (Green & Red) split at -8.5
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_log_ig,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': "<b>AI-Based FinFET Gate Leakage Gauge</b><br><span style='font-size:0.85em;color:gray'>log₁₀(I_G) [A]</span>",
                'font': {'size': 16}
            },
            number={'font': {'size': 26, 'color': "black"}},
            gauge={
                'axis': {
                    'range': [-12, -2],
                    'tickwidth': 1,
                    'tickcolor': "black",
                    'dtick': 2
                },
                'bar': {'color': "black", 'thickness': 0.22},
                'steps': [
                    {'range': [-12, -8.5], 'color': "#2ecc71"},  # Green (Low: -12 to -8.5)
                    {'range': [-8.5, -2], 'color': "#e74c3c"}    # Red (High: -8.5 to -2)
                ],
                'threshold': {
                    'line': {'color': "darkred", 'width': 4},
                    'thickness': 0.75,
                    'value': -8.5
                }
            }
        ))

        fig_gauge.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# 8. Comparison History Table
if st.session_state.history:
    st.markdown("---")
    st.subheader("📋 Prediction History & Parameter Comparison")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>M.Tech VLSI Thesis Research Project • Amrita Vishwa Vidyapeetham, Amritapuri</p>", unsafe_allow_html=True)
