
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
        "`models.pkl` was not found. Run `python train_models.py` "
        "in the project folder to generate it."
    )
    st.stop()

try:
    models = load_models(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load models.pkl: {exc}")
    st.stop()

if not isinstance(models, dict) or not models:
    st.error("models.pkl does not contain the expected trained-model dictionary.")
    st.stop()

st.sidebar.header("Model Selection")
model_names = list(models.keys())
ann_name = next(
    (name for name in model_names if "artificial neural network" in name.lower() or "ann" in name.lower()),
    model_names[0]
)
model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    model_names,
    index=model_names.index(ann_name)
)

model_info = models[model_name]
selected_model = model_info["model"]

if "r2" in model_info:
    st.sidebar.metric("Holdout R²", f"{float(model_info['r2']):.4f}")
if "rmse" in model_info:
    st.sidebar.metric("Holdout RMSE", f"{float(model_info['rmse']):.4f}")

st.sidebar.caption(
    "The selected estimator is loaded from models.pkl. "
    "For the ANN, StandardScaler is already included in the saved Pipeline."
)

st.header("FinFET Device Parameters")
st.caption(
    "Enter the device conditions. Values outside the uploaded training-data "
    "range are allowed for exploration but will be flagged as extrapolation."
)

left, right = st.columns(2)

with left:
    tox = st.number_input(
        "Oxide Thickness, Tox (nm)",
        min_value=0.05, max_value=5.0, value=1.15, step=0.05, format="%.4f"
    )
    vgs = st.number_input(
        "Gate Voltage, Vgs (V)",
        min_value=0.0, max_value=2.0, value=0.80, step=0.01, format="%.4f"
    )
    lgate = st.number_input(
        "Gate Length, Lgate (nm)",
        min_value=1.0, max_value=100.0, value=22.0, step=1.0, format="%.4f"
    )
    hfin = st.number_input(
        "Fin Height, Hfin (nm)",
        min_value=1.0, max_value=100.0, value=35.0, step=1.0, format="%.4f"
    )

with right:
    wfin = st.number_input(
        "Fin Width, Wfin (nm)",
        min_value=1.0, max_value=50.0, value=10.0, step=1.0, format="%.4f"
    )
    na = st.number_input(
        "Channel Doping, Na (cm⁻³)",
        min_value=1.0e15, max_value=1.0e20,
        value=5.0e17, step=1.0e16, format="%.4e"
    )
    temp = st.number_input(
        "Temperature, T (K)",
        min_value=200.0, max_value=600.0, value=300.0, step=5.0, format="%.2f"
    )
    nfin = st.selectbox("Number of Fins, Nfin", [1, 2, 3, 4], index=3)

input_data = pd.DataFrame([{
    "Tox_nm": float(tox),
    "Vgs_V": float(vgs),
    "Lgate_nm": float(lgate),
    "Hfin_nm": float(hfin),
    "Wfin_nm": float(wfin),
    "Na_cm3": float(na),
    "T_K": float(temp),
    "Nfin": int(nfin),
}], columns=FEATURE_COLS)

# Show training-domain checks before prediction.
outside = []
for feature in FEATURE_COLS:
    low, high = TRAINING_RANGES[feature]
    value = float(input_data.iloc[0][feature])
    if value < low or value > high:
        outside.append((feature, value, low, high))

if outside:
    st.warning(
        "One or more inputs are outside the observed training-data range. "
        "The model can still return a number, but this is extrapolation and "
        "the prediction may be less reliable."
    )
    with st.expander("View out-of-range parameters"):
        for feature, value, low, high in outside:
            st.write(
                f"**{feature}:** {value:.6g} — training range "
                f"{low:.6g} to {high:.6g}"
            )
else:
    st.success("All entered parameters are within the observed training-data ranges.")

st.info(
    "The training script defines the target as `log10_Ig`. "
    "Therefore this app uses the model output directly; it does not add an "
    "extra +40 offset. The saved ANN Pipeline performs its own scaling."
)

if st.button("Predict Gate Leakage", type="primary", use_container_width=True):
    try:
        # Keep the exact feature names and order used during model training.
        raw_pred_log_ig = float(np.asarray(
            selected_model.predict(input_data[FEATURE_COLS])
        ).reshape(-1)[0])

        # train_models.py trains directly on y = df['log10_Ig'].
        # No additional offset is applied here.
        pred_log_ig = raw_pred_log_ig

        # Convert log10(Ig) in amperes to Ig in amperes.
        pred_ig = float(10.0 ** pred_log_ig)

        if not np.isfinite(pred_log_ig) or not np.isfinite(pred_ig):
            raise ValueError("The model returned a non-finite prediction.")

        st.subheader("Prediction Results")
        c1, c2 = st.columns(2)
        c1.metric("Predicted log₁₀(Ig)", f"{pred_log_ig:.4f}")
        c2.metric("Predicted Gate Leakage Current, Ig", f"{pred_ig:.4e} A")

        st.caption(
            "Ig is calculated as 10 raised to the predicted log₁₀(Ig). "
            "The current is displayed in amperes."
        )

        # Gauge limits are based on the target range in the uploaded CSV,
        # with a small margin for visual context. It is not a high/low threshold.
        target_min, target_max = -58.09871252036223, -46.51729766354848
        gauge_min = min(-60.0, target_min - 1.0)
        gauge_max = max(-45.0, target_max + 1.0)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_log_ig,
            number={"valueformat": ".4f"},
            title={"text": "Predicted log₁₀(Ig)"},
            gauge={
                "axis": {"range": [gauge_min, gauge_max]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [gauge_min, target_min], "color": "#e5e7eb"},
                    {"range": [target_min, target_max], "color": "#bbf7d0"},
                    {"range": [target_max, gauge_max], "color": "#e5e7eb"},
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The shaded central band represents the observed log₁₀(Ig) "
            "target range in the supplied training CSV. It is not a clinical "
            "or device pass/fail threshold."
        )

        st.subheader("Input Parameter Summary")
        st.dataframe(input_data, use_container_width=True)

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Model": model_name,
            **input_data.iloc[0].to_dict(),
            "log10(Ig)": pred_log_ig,
            "Ig (A)": pred_ig,
            "Extrapolation": "Yes" if outside else "No",
        })

        st.subheader("Prediction History")
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.exception(exc)

with st.expander("About the model and prediction limits"):
    st.markdown(
        """
        - The uploaded training script uses eight input features:
          `Tox_nm`, `Vgs_V`, `Lgate_nm`, `Hfin_nm`, `Wfin_nm`,
          `Na_cm3`, `T_K`, and `Nfin`.
        - Its target is `log10_Ig`.
        - The ANN is a scikit-learn Pipeline containing `StandardScaler`
          followed by `MLPRegressor`; do not apply a second scaler in this app.
        - Predictions outside the observed training-data ranges are
          extrapolations and require additional validation.
        - A numerical prediction is not, by itself, proof of physical accuracy.
        """
    )
