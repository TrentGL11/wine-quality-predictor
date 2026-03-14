"""
Model Details — Performance metrics, feature importance, and methodology
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
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Details", page_icon="🧪", layout="wide")


@st.cache_resource
def load_model():
    model = pickle.load(open("model/trained_model.pkl", "rb"))
    with open("model/feature_names.json", "r") as f:
        meta = json.load(f)
    return model, meta


@st.cache_data
def load_data():
    return pd.read_csv("data/winequality-red.csv", sep=";")


model, meta = load_model()
df = load_data()
feature_names = meta["feature_names"]

st.title("🧪 Model Details")
st.markdown("XGBoost binary classifier (Good ≥ 6 vs Bad < 6) trained on all 1,599 samples with 5-fold stratified cross-validation.")
st.markdown("---")

# Performance metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("CV Accuracy", f"{meta['cv_accuracy']:.1%}")
col2.metric("CV F1 Score", f"{meta['cv_f1']:.1%}")
col3.metric("CV ROC AUC", f"{meta['cv_auc']:.3f}")
col4.metric("Test ROC AUC", f"{meta['test_auc']:.3f}")

st.markdown("---")

# Feature importance
st.subheader("Feature Importance")
left, right = st.columns(2)

with left:
    st.markdown("**XGBoost Gain-Based Importance**")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig_imp = px.bar(
        x=importances.values, y=importances.index,
        orientation="h",
        labels={"x": "Importance (Gain)", "y": ""},
        color=importances.values,
        color_continuous_scale="Teal"
    )
    fig_imp.update_layout(height=450, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

with right:
    st.markdown("**SHAP Summary Plot** (global feature impact)")
    X = df[feature_names]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig_shap, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X, show=False, plot_size=(8, 6))
    plt.tight_layout()
    st.pyplot(fig_shap)
    plt.close()

st.markdown("---")

# Cross-validated confusion matrix and ROC
st.subheader("Cross-Validated Performance")
y = (df["quality"] >= 6).astype(int)
X = df[feature_names]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(model, X, y, cv=cv)
y_prob_cv = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

left2, right2 = st.columns(2)

with left2:
    st.markdown("**Confusion Matrix (5-Fold CV)**")
    cm = confusion_matrix(y, y_pred_cv)
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Bad (<6)", "Good (≥6)"],
        y=["Bad (<6)", "Good (≥6)"],
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
                                  name=f"XGBoost (AUC = {roc_auc:.3f})",
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
report = classification_report(y, y_pred_cv, target_names=["Bad (<6)", "Good (≥6)"], output_dict=True)
st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# Methodology
st.markdown("---")
st.subheader("Methodology")
st.markdown("""
**Model:** XGBoost Classifier (gradient boosted trees)

**Hyperparameters:**
- `n_estimators`: 300
- `max_depth`: 4
- `learning_rate`: 0.1
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `scale_pos_weight`: Adjusted for class imbalance

**Validation:** 5-fold stratified cross-validation

**Target:** Binary classification — Good (quality ≥ 6) vs Bad (quality < 6)

**Interpretability:** SHAP (SHapley Additive exPlanations) values provide individual prediction explanations, showing which features contribute most to each prediction.

**Reference:** Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547–553.
""")
