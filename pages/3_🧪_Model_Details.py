"""
Model Details  -  Performance metrics, coefficient analysis, and methodology
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

st.set_page_config(page_title="Model Details", page_icon="🧪", layout="wide")


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

st.title("🧪 Logistic Regression - Champion Model")
st.markdown(
    "Logistic Regression binary classifier (Good >= 6 vs Bad < 6) with 8 features, "
    "trained on all 1,599 samples with group-aware stratified 5-fold CV x 10 repeats. "
    "Champion selected for interpretability, stability, and governance suitability."
)
st.markdown("---")

# Performance metrics
st.markdown("**Cross-Validation (5-Fold x 10 Repeats, Training Set)**")
col1, col2, col3 = st.columns(3)
col1.metric("CV Accuracy", f"{meta['cv_accuracy']:.1%}")
col2.metric("CV F1 Score", f"{meta['cv_f1']:.1%}")
col3.metric("CV ROC AUC", f"{meta['cv_auc']:.3f}")

st.markdown("**Held-Out Test Set (20% unseen data)**")
col4, col5, col6 = st.columns(3)
col4.metric("Test Accuracy", f"{meta.get('test_accuracy', 0):.1%}")
col5.metric("Test F1 Score", f"{meta.get('test_f1', 0):.1%}")
col6.metric("Test ROC AUC", f"{meta['test_auc']:.3f}")

st.markdown("---")

# Feature importance - Coefficients and Odds Ratios
st.subheader("Feature Analysis")
left, right = st.columns(2)

coefficients = pd.Series(model.coef_[0], index=feature_names)

with left:
    st.markdown("**Logistic Regression Coefficients (standardised)**")
    coef_sorted = coefficients.sort_values(ascending=True)
    colors = ["#d32f2f" if v < 0 else "#388e3c" for v in coef_sorted.values]
    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(
        x=coef_sorted.values,
        y=coef_sorted.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in coef_sorted.values],
        textposition="outside"
    ))
    fig_coef.update_layout(
        height=450,
        xaxis_title="Coefficient",
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_coef, use_container_width=True)

with right:
    st.markdown("**Odds Ratios** (exp of coefficients)")
    odds_ratios = np.exp(coefficients)
    or_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients.values,
        "Odds Ratio": odds_ratios.values,
        "Direction": ["Increases quality odds" if c > 0 else "Decreases quality odds" for c in coefficients.values]
    }).sort_values("Odds Ratio", ascending=False)

    st.dataframe(
        or_df.style.format({"Coefficient": "{:+.3f}", "Odds Ratio": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Odds Ratio > 1 means a one-standard-deviation increase in that feature "
        "raises the odds of 'Good' quality. Odds Ratio < 1 means it lowers the odds."
    )

st.markdown("---")

# Cross-validated confusion matrix and ROC
st.subheader("Cross-Validated Performance")
y = (df["quality"] >= 6).astype(int)
X = df[feature_names]
X_scaled = scaler.transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(model, X_scaled, y, cv=cv)
y_prob_cv = cross_val_predict(model, X_scaled, y, cv=cv, method="predict_proba")[:, 1]

left2, right2 = st.columns(2)

with left2:
    st.markdown("**Confusion Matrix (5-Fold CV)**")
    cm = confusion_matrix(y, y_pred_cv)
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Bad (<6)", "Good (>=6)"],
        y=["Bad (<6)", "Good (>=6)"],
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual"}
    )
    fig_cm.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

with right2:
    st.markdown("**ROC Curve (5-Fold CV)**")
    fpr, tpr, _ = roc_curve(y, y_prob_cv)
    roc_auc = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                  name=f"Logistic Regression (AUC = {roc_auc:.3f})",
                                  line=dict(color="#1a3a5c", width=2)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  name="Random", line=dict(dash="dash", color="gray")))
    fig_roc.update_layout(
        height=400, xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# Classification report
st.markdown("**Classification Report:**")
report = classification_report(y, y_pred_cv, target_names=["Bad (<6)", "Good (>=6)"], output_dict=True)
st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# Methodology
st.markdown("---")
st.subheader("Methodology")
st.markdown("""
**Model:** Logistic Regression (scikit-learn) with L2 regularisation

**Features:** 8 physicochemical properties selected through a rigorous variable selection process:
- Started with all 11 features
- Reduced to 9 after removing multicollinear features (VIF filtering)
- Reduced to 8 after removing sign-unstable features (coefficient sign stability check across CV folds)
- Variable selection journey: 11 -> 9 (VIF) -> 8 (sign stability)

**Validation:** Group-aware stratified 5-fold cross-validation repeated 10 times

**Target:** Binary classification - Good (quality >= 6) vs Bad (quality < 6)

**Champion Selection Rationale:** Logistic Regression was chosen over XGBoost as the champion model for:
- **Interpretability:** Direct coefficient interpretation and odds ratios
- **Stability:** Consistent performance across CV folds with low variance
- **Governance:** Transparent, auditable model suitable for regulated environments

**Interpretability:** Coefficient contributions show exactly how much each standardised feature adds to or subtracts from the log-odds of a "Good" prediction. Odds ratios provide an intuitive multiplicative interpretation.

**Reference:** Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547-553.
""")
