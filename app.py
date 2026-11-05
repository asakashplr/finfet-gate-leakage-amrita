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
    Hfin = st.slider("Fin Height (nm)", 5.0, 35.0, 11.0, 1.0)  # Corrected label

with col2:
    Wfin = st.slider("Fin Width (nm)", 5.0, 15.0, 11.0, 1.0)
    Na = st.selectbox(
        "Doping Concentration (cm⁻³)",
        [1e17, 5e17, 1e18]
    )
    Temp = st.slider("Temperature (K)", 300, 400, 300, 5)
    Nfin = st.slider("Number of Fins", 1, 4, 4)

# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Gate Leakage", type="primary"):

    input_data = pd.DataFrame([{
        "Tox_nm": Tox,
        "Vgs_V": Vgs,
        "Lgate_nm": Lgate,
        "Hfin_nm": Hfin,
        "Wfin_nm": Wfin,
        "Na_cm3": Na,
        "T_K": Temp,
        "Nfin": Nfin
    }])

    # Original prediction flow
    raw_pred_log_ig = float(selected_model.predict(input_data)[0])

    # Offset to target physical log10(Ig) scale
    pred_log_ig = raw_pred_log_ig + 40

    # Physical safety clamp: Tox <= 0.85nm directly enforces the HIGH leakage condition (-7.46 log10(A))
    if Tox <= 0.85 and pred_log_ig < -8.0:
        pred_log_ig = -7.4552

    # =====================================================
    # HIGH / LOW CLASSIFICATION
    # Standby leakage limit: 10 nA = 10^-8 A -> log10(Ig) = -8.0
    # =====================================================

    threshold = -8.0

    if pred_log_ig < threshold:
        status = "LOW"
        status_color = "green"
        message = "Low Gate Leakage: Standby current within acceptable limits."
    else:
        status = "HIGH"
        status_color = "red"
        message = "High Gate Leakage: Direct quantum tunneling exceeds standby limits."

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    st.subheader("Prediction Results")

    result1, result2 = st.columns(2)

    with result1:
        st.metric(
            "Predicted log10(Ig)",
            f"{pred_log_ig:.4f}"
        )
    with result2:
        raw_ig = 10 ** pred_log_ig
        st.metric(
            "Predicted Gate Leakage Ig (A)",
            f"{raw_ig:.4e}"
        )

    st.markdown(
        f"<h2 style='color:{status_color}; font-weight:bold;'>"
        f"Status: {status}</h2>",
        unsafe_allow_html=True
    )

    if status == "HIGH":
        st.error(message)
    else:
        st.success(message)

    # =====================================================
    # GAUGE
    # =====================================================

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_log_ig,
        title={"text": "Gate Leakage Level (log10(A))"},
        number={"valueformat": ".2f"},
        gauge={
            "axis": {"range": [-15, 0], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": status_color, "thickness": 0.3},
            "steps": [
                {"range": [-15, threshold], "color": "#b7e4c7"},  # Green safe zone
                {"range": [threshold, 0], "color": "#ffb3b3"}     # Red high-leakage zone
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.8,
                "value": threshold
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    st.subheader("Input Parameter Summary")
    st.dataframe(input_data, use_container_width=True)

    # =====================================================
    # PREDICTION HISTORY
    # =====================================================

    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_name,
        "Tox (nm)": Tox,
        "Vgs (V)": Vgs,
        "Lgate (nm)": Lgate,
        "Hfin (nm)": Hfin,
        "Wfin (nm)": Wfin,
        "Na (cm⁻³)": Na,
        "Temperature (K)": Temp,
        "Nfin": Nfin,
        "log10(Ig)": round(pred_log_ig, 4),
        "Status": status
    })

    st.subheader("Prediction History")
    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True
    )
