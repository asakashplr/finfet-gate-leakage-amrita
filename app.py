
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="22nm FinFET Gate Leakage Predictor",
    page_icon="⚡",
    layout="wide"
)

FEATURES = [
    "Tox_nm", "Vgs_V", "Lgate_nm", "Hfin_nm",
    "Wfin_nm", "Na_cm3", "T_K", "Nfin"
]

FEATURE_LABELS = {
    "Tox_nm": "Oxide Thickness (nm)",
    "Vgs_V": "Gate-Source Voltage (V)",
    "Lgate_nm": "Gate Length (nm)",
    "Hfin_nm": "Fin Height (nm)",
    "Wfin_nm": "Fin Width (nm)",
    "Na_cm3": "Channel Doping (cm⁻³)",
    "T_K": "Temperature (K)",
    "Nfin": "Number of Fins"
}

# Reference ranges from the previously shared synthetic-data generator.
# Replace these with actual training-data ranges if they differ.
REFERENCE_RANGES = {
    "Tox_nm": (0.8, 2.0),
    "Vgs_V": (0.0, 1.0),
    "Lgate_nm": (18.0, 26.0),
    "Hfin_nm": (20.0, 35.0),
    "Wfin_nm": (8.0, 15.0),
    "Na_cm3": (1e16, 5e17),
    "T_K": (300, 375),
    "Nfin": (1, 5)
}

# This is an illustrative display threshold, not a universal safety limit.
LEAKAGE_THRESHOLD_LOG10 = -8.5

# ============================================================
# 2. SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# 3. LOAD MODEL PACKAGE
# ============================================================

@st.cache_resource
def load_all_models():
    model_path = "models.pkl"

    if not os.path.exists(model_path):
        return None

    with open(model_path, "rb") as file:
        return pickle.load(file)


models_dict = load_all_models()

# ============================================================
# 4. MODEL PACKAGE HELPERS
# ============================================================

def get_model_info(model_name):
    """
    Supports:
      models_dict[model_name] = {
          'model': trained_model,
          'scaler': fitted_scaler,
          'r2': ...,
          'rmse': ...,
          'target_offset': 0
      }

    A scikit-learn Pipeline can also be stored directly as model.
    """

    item = models_dict[model_name]

    if not isinstance(item, dict):
        return {
            "model": item,
            "scaler": None,
            "r2": None,
            "rmse": None,
            "target_offset": 0.0,
            "feature_cols": FEATURES
        }

    return {
        "model": item.get("model"),
        "scaler": item.get("scaler", models_dict.get("scaler")),
        "r2": item.get("r2"),
        "rmse": item.get("rmse"),
        "target_offset": item.get(
            "target_offset",
            models_dict.get("target_offset", 0.0)
        ),
        "feature_cols": item.get(
            "feature_cols",
            models_dict.get("feature_cols", FEATURES)
        )
    }


def make_input_dataframe(values, feature_cols):
    """
    Converts the eight UI inputs into the feature order
    expected by the saved model.
    """

    aliases = {
        "Tox": "Tox_nm",
        "Vgs": "Vgs_V",
        "Lgate": "Lgate_nm",
        "Hfin": "Hfin_nm",
        "Wfin": "Wfin_nm",
        "Na": "Na_cm3",
        "Temp": "T_K",
        "Nfin": "Nfin"
    }

    normalized_values = {}

    for name, value in values.items():
        normalized_values[name] = value

    for old_name, new_name in aliases.items():
        if new_name in values:
            normalized_values[old_name] = values[new_name]

    try:
        ordered_values = [normalized_values[name] for name in feature_cols]
    except KeyError as error:
        raise ValueError(
            f"Model expects an unsupported feature: {error}. "
            "Check feature_cols in models.pkl."
        )

    return pd.DataFrame([ordered_values], columns=feature_cols)


def predict_one(model_name, values):
    """
    Performs prediction using the model's saved preprocessing.
    No arbitrary output shift is applied.
    """

    info = get_model_info(model_name)

    model = info["model"]
    scaler = info["scaler"]
    feature_cols = info["feature_cols"]
    target_offset = float(info["target_offset"])

    if model is None:
        raise ValueError(f"Model object missing for {model_name}")

    input_df = make_input_dataframe(values, feature_cols)

    # Pipeline objects should receive their original raw input.
    if hasattr(model, "named_steps"):
        raw_prediction = float(model.predict(input_df)[0])

    elif scaler is not None:
        scaled_input = scaler.transform(input_df)
        raw_prediction = float(model.predict(scaled_input)[0])

    else:
        # Only correct if the saved model was trained on raw features.
        raw_prediction = float(model.predict(input_df)[0])

    predicted_log_ig = raw_prediction + target_offset
    predicted_ig = 10.0 ** predicted_log_ig

    if not np.isfinite(predicted_ig):
        raise ValueError("Prediction produced a non-finite current.")

    return predicted_log_ig, predicted_ig, info


# ============================================================
# 5. INPUT VALIDATION
# ============================================================

def check_reference_ranges(values):
    outside = []

    for feature, value in values.items():
        lower, upper = REFERENCE_RANGES[feature]

        if value < lower or value > upper:
            outside.append(
                f"{FEATURE_LABELS[feature]}: {value:g} "
                f"(reference range {lower:g}–{upper:g})"
            )

    return outside


# ============================================================
# 6. GAUGE
# ============================================================

def create_leakage_gauge(log_value):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=log_value,
            title={
                "text": "Predicted log₁₀(Ig) [A]"
            },
            number={
                "valueformat": ".4f"
            },
            gauge={
                "axis": {
                    "range": [-12, -2],
                    "dtick": 2
                },
                "bar": {
                    "color": "black",
                    "thickness": 0.2
                },
                "steps": [
                    {
                        "range": [-12, LEAKAGE_THRESHOLD_LOG10],
                        "color": "#2ecc71"
                    },
                    {
                        "range": [LEAKAGE_THRESHOLD_LOG10, -2],
                        "color": "#e74c3c"
                    }
                ],
                "threshold": {
                    "line": {
                        "color": "darkred",
                        "width": 3
                    },
                    "value": LEAKAGE_THRESHOLD_LOG10
                }
            }
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# ============================================================
# 7. APPLICATION HEADER
# ============================================================

st.title("22nm FinFET Gate Leakage Predictor")

st.subheader(
    "Amrita Vishwa Vidyapeetham, Amritapuri Campus"
)

st.caption(
    "Developed by: A S Akash & Dr. Sundar | "
    "Department of ECE | M.Tech VLSI Design 2025–27"
)

st.markdown("---")

if models_dict is None:
    st.error(
        "`models.pkl` was not found. "
        "Please upload the trained model package."
    )
    st.stop()

if not isinstance(models_dict, dict):
    st.error("Invalid models.pkl format. Expected a dictionary.")
    st.stop()

# Keep only entries containing model objects.
available_models = []

for name in models_dict:
    try:
        info = get_model_info(name)
        if info["model"] is not None:
            available_models.append(name)
    except Exception:
        continue

if not available_models:
    st.error("No valid trained models were found.")
    st.stop()

# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose ML Algorithm",
    available_models
)

selected_info = get_model_info(model_choice)

if selected_info["r2"] is not None:
    st.sidebar.metric(
        "Holdout Test R²",
        f"{float(selected_info['r2']):.4f}"
    )

if selected_info["rmse"] is not None:
    st.sidebar.metric(
        "Holdout Test RMSE",
        f"{float(selected_info['rmse']):.4f}"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Project Details")

st.sidebar.write("🏫 Amrita Vishwa Vidyapeetham")
st.sidebar.write("📍 Amritapuri Campus")
st.sidebar.write("🔬 Department of VLSI Design")
st.sidebar.write("👨‍💻 A S Akash & Dr. Sundar")

st.sidebar.markdown("---")
st.sidebar.header("Device & Operating Parameters")

tox = st.sidebar.slider(
    "Oxide Thickness (Tox in nm)",
    0.8, 2.0, 1.2, 0.05
)

vgs = st.sidebar.slider(
    "Gate-Source Voltage (Vgs in V)",
    0.0, 1.0, 0.8, 0.05
)

lgate = st.sidebar.slider(
    "Gate Length (Lgate in nm)",
    18.0, 26.0, 22.0, 1.0
)

hfin = st.sidebar.slider(
    "Fin Height (Hfin in nm)",
    20.0, 35.0, 25.0, 1.0
)

wfin = st.sidebar.slider(
    "Fin Width (Wfin in nm)",
    8.0, 15.0, 12.0, 0.5
)

na = st.sidebar.selectbox(
    "Channel Doping (Na in cm⁻³)",
    [1e16, 1e17, 2e17, 3e17, 5e17],
    index=1,
    format_func=lambda x: f"{x:.1e}"
)

temp = st.sidebar.select_slider(
    "Temperature (T in K)",
    options=[300, 325, 350, 375],
    value=300
)

nfin = st.sidebar.slider(
    "Number of Fins (Nfin)",
    1, 5, 2
)

values = {
    "Tox_nm": tox,
    "Vgs_V": vgs,
    "Lgate_nm": lgate,
    "Hfin_nm": hfin,
    "Wfin_nm": wfin,
    "Na_cm3": na,
    "T_K": temp,
    "Nfin": nfin
}

# ============================================================
# 9. MAIN PREDICTION
# ============================================================

st.write(
    "Adjust the FinFET device parameters and select a trained "
    "machine learning model to predict gate leakage current."
)

outside_ranges = check_reference_ranges(values)

if outside_ranges:
    st.warning(
        "Some input parameters are outside the reference "
        "ranges of the previously shared synthetic dataset. "
        "The model may be extrapolating."
    )

    for item in outside_ranges:
        st.caption("• " + item)

if st.button("Predict Gate Leakage", type="primary"):

    try:
        pred_log_ig, pred_ig, info = predict_one(
            model_choice,
            values
        )

        st.session_state.last_prediction = {
            "model": model_choice,
            "log10_Ig": pred_log_ig,
            "Ig": pred_ig
        }

        st.session_state.history.append({
            "Model": model_choice,
            "Tox (nm)": tox,
            "Vgs (V)": vgs,
            "Lgate (nm)": lgate,
            "Hfin (nm)": hfin,
            "Wfin (nm)": wfin,
            "Na (cm⁻³)": f"{na:.2e}",
            "Temp (K)": temp,
            "Nfin": nfin,
            "log10(Ig)": pred_log_ig,
            "Ig (A)": pred_ig
        })

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# ============================================================
# 10. DISPLAY PREDICTION
# ============================================================

if "last_prediction" in st.session_state:

    result = st.session_state.last_prediction

    st.markdown("---")
    st.subheader(
        f"Prediction Results ({result['model']})"
    )

    col1, col2 = st.columns([1, 1.2])

    with col1:

        st.metric(
            "Predicted log₁₀(Ig)",
            f"{result['log10_Ig']:.4f}"
        )

        st.metric(
            "Predicted Gate Leakage (Ig)",
            f"{result['Ig']:.4e} A"
        )

        if selected_info["r2"] is not None:
            st.metric(
                "Holdout Test R²",
                f"{float(selected_info['r2']):.4f}"
            )

        if result["log10_Ig"] < LEAKAGE_THRESHOLD_LOG10:
            st.success(
                "Lower predicted leakage relative to "
                "the selected display threshold."
            )
        else:
            st.warning(
                "Higher predicted leakage relative to "
                "the selected display threshold."
            )

        st.caption(
            "The threshold is an illustrative comparison "
            "setting, not a universal device safety specification."
        )

    with col2:

        st.plotly_chart(
            create_leakage_gauge(result["log10_Ig"]),
            use_container_width=True
        )

# ============================================================
# 11. MODEL COMPARISON
# ============================================================

st.markdown("---")
st.subheader("Machine Learning Model Performance Comparison")

comparison_rows = []

for name in available_models:

    info = get_model_info(name)

    comparison_rows.append({
        "Model": name,
        "Holdout R²": info["r2"],
        "Holdout RMSE": info["rmse"]
    })

comparison_df = pd.DataFrame(comparison_rows)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# 12. PREDICTION HISTORY
# ============================================================

if st.session_state.history:

    st.markdown("---")
    st.subheader("Prediction History & Parameter Comparison")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prediction History (CSV)",
        data=csv_data,
        file_name="finfet_prediction_history.csv",
        mime="text/csv"
    )

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# ============================================================
# 13. REPRESENTATIVE PARAMETER CASES
# ============================================================

st.markdown("---")
st.subheader("Representative FinFET Parameter Cases")

corner_cases = {
    "TT (Typical)": [
        1.70, 0.80, 22, 42, 11, 5e17, 300, 2
    ],
    "FF (Fast)": [
        1.20, 0.88, 14, 30, 6, 1e17, 300, 1
    ],
    "SS (Slow)": [
        2.20, 0.72, 35, 56, 16, 1e18, 300, 4
    ],
    "FS (Fast N)": [
        1.40, 0.84, 18, 35, 8, 2e17, 300, 2
    ],
    "SF (Slow N)": [
        2.00, 0.76, 28, 50, 14, 8e17, 300, 3
    ]
}

if st.button("Run Representative Case Analysis"):

    corner_results = []

    for case_name, parameters in corner_cases.items():

        case_values = dict(zip(FEATURES, parameters))

        try:
            log_value, current_value, _ = predict_one(
                model_choice,
                case_values
            )

            corner_results.append({
                "Case": case_name,
                "log10(Ig)": log_value,
                "Ig (A)": current_value
            })

        except Exception as error:
            st.error(f"{case_name}: {error}")

    if corner_results:

        corner_df = pd.DataFrame(corner_results)

        st.dataframe(
            corner_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            corner_df.set_index("Case")["log10(Ig)"]
        )

        st.caption(
            "These are representative parameter combinations, "
            "not calibrated foundry process corners."
        )

# ============================================================
# 14. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;color:gray;'>
    M.Tech VLSI Thesis Research Project<br>
    Amrita Vishwa Vidyapeetham, Amritapuri
    </div>
    """,
    unsafe_allow_html=True
)
