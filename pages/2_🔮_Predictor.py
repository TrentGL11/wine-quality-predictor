"""
Wine Quality Predictor  -  Interactive prediction with coefficient explanations
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go

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

# Preset wine profiles - median feature values per quality tier from the dataset
# Excellent: quality 7-8 (n=217), Above Average: quality 6 (n=638),
# Average: quality 5 (n=681), Poor: quality 3-4 (n=63)
PRESETS = {
    "🏆 Excellent Wine": {
        "volatile acidity": 0.370, "citric acid": 0.400,
        "chlorides": 0.073, "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 27.0, "pH": 3.27,
        "sulphates": 0.740, "alcohol": 11.6
    },
    "👍 Above Average Wine": {
        "volatile acidity": 0.490, "citric acid": 0.260,
        "chlorides": 0.078, "free sulfur dioxide": 14.0,
        "total sulfur dioxide": 35.0, "pH": 3.32,
        "sulphates": 0.640, "alcohol": 10.5
    },
    "😐 Average Wine": {
        "volatile acidity": 0.580, "citric acid": 0.230,
        "chlorides": 0.081, "free sulfur dioxide": 15.0,
        "total sulfur dioxide": 47.0, "pH": 3.30,
        "sulphates": 0.580, "alcohol": 9.7
    },
    "👎 Poor Wine": {
        "volatile acidity": 0.680, "citric acid": 0.080,
        "chlorides": 0.080, "free sulfur dioxide": 9.0,
        "total sulfur dioxide": 26.0, "pH": 3.38,
        "sulphates": 0.560, "alcohol": 10.0
    }
}

# Preset buttons
st.subheader("Quick Presets")
st.caption("Each preset uses the median feature values for wines in that quality tier (Excellent: quality 7-8, Above Average: 6, Average: 5, Poor: 3-4).")
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
X_scaled = scaler.transform(X_input)
pred_proba = model.predict_proba(X_scaled)[0]
pred_class = model.predict(X_scaled)[0]
quality_label = "Good (>=6)" if pred_class == 1 else "Bad (<6)"
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
        x=["Bad (<6)", "Good (>=6)"],
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
    # Coefficient contribution table
    st.markdown("**Coefficient Contributions** - What drove this prediction?")
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    scaled_values = X_scaled[0]

    contributions = []
    for j, feat in enumerate(feature_names):
        contrib = scaled_values[j] * coefficients[j]
        contributions.append({
            "Feature": feat,
            "Standardised Value": round(scaled_values[j], 3),
            "Coefficient": round(coefficients[j], 3),
            "Contribution (log-odds)": round(contrib, 3)
        })

    contrib_df = pd.DataFrame(contributions)
    contrib_df = contrib_df.reindex(
        contrib_df["Contribution (log-odds)"].abs().sort_values(ascending=False).index
    )

    # Style: green for positive, red for negative contributions
    def color_contribution(val):
        if val > 0:
            return "color: #388e3c; font-weight: bold"
        elif val < 0:
            return "color: #d32f2f; font-weight: bold"
        return ""

    styled = contrib_df.style.map(
        color_contribution, subset=["Contribution (log-odds)"]
    ).format({
        "Standardised Value": "{:.3f}",
        "Coefficient": "{:.3f}",
        "Contribution (log-odds)": "{:+.3f}"
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    total_log_odds = intercept + sum(c["Contribution (log-odds)"] for c in contributions)
    st.caption(f"Intercept: {intercept:+.3f} | Total log-odds: {total_log_odds:+.3f}")

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
