"""
Red Wine Quality Analysis  -  Dashboard
Senior Data Scientist Case Study | UNIFI Credit
Trent Lemkus, Ph.D.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Wine Quality Predictor | Trent Lemkus",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UNIFI branding (navy/teal)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a3a5c;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a90a4;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f0f5fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #1a3a5c;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv("data/winequality-red.csv", sep=";")


df = load_data()
feature_cols = [c for c in df.columns if c != "quality"]

# Header
st.markdown('<p class="main-header">🍷 Red Wine Quality Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Senior Data Scientist Case Study  -  Trent Lemkus, Ph.D. | UNIFI Credit</p>', unsafe_allow_html=True)
st.markdown("---")

# Key Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Samples", f"{len(df):,}")
col2.metric("Features", len(feature_cols))
col3.metric("Quality Range", f"{df['quality'].min()} - {df['quality'].max()}")
col4.metric("Median Quality", f"{df['quality'].median():.0f}")

st.markdown("---")

# Two columns: correlation heatmap + quality distribution
left, right = st.columns([3, 2])

with left:
    st.subheader("Correlation Heatmap")
    corr = df.corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.where(~mask)

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>"
    ))
    fig_corr.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)

with right:
    st.subheader("Quality Distribution")
    counts = df["quality"].value_counts().sort_index()
    fig_qual = px.bar(
        x=counts.index, y=counts.values,
        labels={"x": "Quality Score", "y": "Count"},
        color=counts.values,
        color_continuous_scale=["#d32f2f", "#ff9800", "#388e3c"],
        text=[f"{v}<br>({v/len(df)*100:.1f}%)" for v in counts.values]
    )
    fig_qual.update_layout(
        height=500, showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig_qual.update_traces(textposition="outside")
    st.plotly_chart(fig_qual, use_container_width=True)

# Summary statistics
st.subheader("Summary Statistics")
desc = df.describe().T
desc["skewness"] = df.skew()
desc["kurtosis"] = df.kurtosis()
st.dataframe(desc.style.format("{:.3f}"), use_container_width=True)

st.markdown("---")
st.caption("Data: Red Wine Quality Dataset (Cortez et al., 2009) | Built by Trent Lemkus, Ph.D.")
