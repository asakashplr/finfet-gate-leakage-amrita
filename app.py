
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 22 nm FinFET Gate Leakage Predictor
# Model target in train_models.py: log10_Ig
# Feature order follows train_models.py exactly.
# =========================================================

st.set_page_config(
    page_title="22nm FinFET Gate Leakage Predictor",
    page_icon="⚡",
    layout="wide"
)

FEATURE_COLS = [
    "Tox_nm", "Vgs_V", "Lgate_nm", "Hfin_nm",
    "Wfin_nm", "Na_cm3", "T_K", "Nfin"
]

# Observed ranges in the uploaded 1,500-row training CSV.
# These are used for warnings, not as claims of physical device limits.
TRAINING_RANGES = {
    "Tox_nm": (0.8008, 2.4996),
    "Vgs_V": (0.0005, 1.1995),
    "Lgate_nm": (14.017, 39.985),
    "Hfin_nm": (20.02, 49.981),
    "Wfin_nm": (5.001, 14.998),
    "Na_cm3": (1.001460904794128e17, 9.99694988356932e17),
    "T_K": (300.05, 399.99),
    "Nfin": (1, 4),
}

st.title("⚡ 22 nm FinFET Gate Leakage Predictor")
st.markdown(
    "Machine-learning-based prediction of FinFET gate leakage current"
)
st.markdown(
    """
    <div style="text-align:center; padding:8px;">
      <b>Developed by A. S. Akash</b><br>
      M.Tech VLSI Design | Amrita School of Engineering, Amritapuri
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()


@st.cache_resource
def load_models(path):
    with open(path, "rb") as f:
        return pickle.load(f)


MODEL_PATH = "models.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(
