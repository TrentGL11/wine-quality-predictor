"""
Wine Quality Predictor — The star feature
Predict wine quality with SHAP explanations and preset profiles.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🔮", layout="wide")


@st.cache_resource
def load_model():
    model = pickle.load(open("model/trained_model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    with open("model/feature_names.json", "r") as f:
        meta = json.load(f)
    return model, scaler, meta


@st.cache_data
def load_data():
    return pd.read_csv("data/winequality-red.csv", sep=";")


model, scaler, meta = load_model()
df = load_data()
feature_names = meta["feature_names"]
feature_stats = meta["feature_stats"]

st.title("🔮 Wine Quality Predictor")
st.markdown("Adjust the physicochemical properties below to predict wine quality. Use the **preset profiles** to quickly compare good, average, and poor wines.")
st.markdown("---")

# Preset wine profiles (derived from dataset quartiles)
PRESETS = {
    "🏆 Excellent Wine": {
        "fixed acidity": 8.0, "volatile acidity": 0.35, "citric acid": 0.45,
        "residual sugar": 2.2, "chlorides": 0.065, "free sulfur dioxide": 14.0,
        "total sulfur dioxide": 28.0, "density": 0.995, "pH": 3.25,
        "sulphates": 0.78, "alcohol": 12.5
    },
    "👍 Above Average Wine": {
        "fixed acidity": 7.9, "volatile acidity": 0.45, "citric acid": 0.35,
        "residual sugar": 2.3, "chlorides": 0.075, "free sulfur dioxide": 15.0,
        "total sulfur dioxide": 40.0, "density": 0.996, "pH": 3.3,
        "sulphates": 0.65, "alcohol": 11.0
    },
    "😐 Average Wine": {
        "fixed acidity": 7.2, "volatile acidity": 0.55, "citric acid": 0.25,
        "residual sugar": 2.5, "chlorides": 0.08, "free sulfur dioxide": 14.0,
        "total sulfur dioxide": 50.0, "density": 0.997, "pH": 3.32,
        "sulphates": 0.58, "alcohol": 10.0
    },
    "👎 Poor Wine": {
        "fixed acidity": 7.5, "volatile acidity": 0.85, "citric acid": 0.05,
        "residual sugar": 2.0, "chlorides": 0.095, "free sulfur dioxide": 8.0,
        "total sulfur dioxide": 60.0, "density": 0.998, "pH": 3.4,
        "sulphates": 0.52, "alcohol": 9.2
    }
}

# Preset buttons
st.subheader("Quick Presets")
preset_cols = st.columns(len(PRESETS))
selected_preset = None
for i, (name, values) in enumerate(PRESETS.items()):
    if preset_cols[i].button(name, use_container_width=True):
        selected_preset = values

# Sliders
st.subheader("Wine Properties")
input_values = {}
slider_cols = st.columns(3)

for i, feat in enumerate(feature_names):
    col = slider_cols[i % 3]
    stats_f = feature_stats[feat]
    default_val = selected_preset[feat] if selected_preset else stats_f["median"]
    # Ensure default is within range
    default_val = max(stats_f["min"], min(stats_f["max"], default_val))
    step = (stats_f["max"] - stats_f["min"]) / 100
    input_values[feat] = col.slider(
        feat,
        min_value=float(stats_f["min"]),
        max_value=float(stats_f["max"]),
        value=float(default_val),
        step=float(step),
        format="%.3f"
    )

st.markdown("---")

# Prediction
X_input = np.array([[input_values[f] for f in feature_names]])
pred_proba = model.predict_proba(X_input)[0]
pred_class = model.predict(X_input)[0]
quality_label = "Good (≥6)" if pred_class == 1 else "Bad (<6)"
confidence = max(pred_proba) * 100

# Display results
result_cols = st.columns([1, 2, 2])

with result_cols[0]:
    if pred_class == 1:
        st.success(f"### ✅ {quality_label}")
    else:
        st.error(f"### ❌ {quality_label}")
    st.metric("Confidence", f"{confidence:.1f}%")

with result_cols[1]:
    # Probability bar chart
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Bar(
        x=["Bad (<6)", "Good (≥6)"],
        y=pred_proba,
        marker_color=["#d32f2f", "#388e3c"],
        text=[f"{p:.1%}" for p in pred_proba],
        textposition="outside"
    ))
    fig_prob.update_layout(
        title="Prediction Probabilities",
        yaxis_title="Probability",
        yaxis_range=[0, 1.15],
        height=350,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with result_cols[2]:
    # SHAP waterfall
    st.markdown("**SHAP Explanation** — What drove this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    fig_shap, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_input[0],
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig_shap)
    plt.close()

# Feature comparison table
st.markdown("---")
st.subheader("Your Wine vs Dataset Statistics")
comp_data = []
for feat in feature_names:
    comp_data.append({
        "Feature": feat,
        "Your Value": input_values[feat],
        "Dataset Mean": feature_stats[feat]["mean"],
        "Dataset Min": feature_stats[feat]["min"],
        "Dataset Max": feature_stats[feat]["max"],
        "Percentile": f"{(df[feat] <= input_values[feat]).mean() * 100:.0f}th"
    })
st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
