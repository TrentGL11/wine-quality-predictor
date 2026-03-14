"""
Data Explorer — Interactive scatter plots and distributions
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("data/winequality-red.csv", sep=";")


df = load_data()
feature_cols = [c for c in df.columns if c != "quality"]

st.title("📊 Data Explorer")
st.markdown("Interactively explore relationships between wine characteristics and quality.")
st.markdown("---")

# Scatter plot
col1, col2, col3 = st.columns(3)
with col1:
    x_var = st.selectbox("X-axis", feature_cols, index=feature_cols.index("alcohol"))
with col2:
    y_var = st.selectbox("Y-axis", feature_cols, index=feature_cols.index("volatile acidity"))
with col3:
    color_by = st.selectbox("Color by", ["quality", "None"], index=0)

if color_by == "quality":
    fig_scatter = px.scatter(
        df, x=x_var, y=y_var, color="quality",
        color_continuous_scale="RdYlGn",
        opacity=0.6, hover_data=feature_cols,
        title=f"{x_var} vs {y_var} (colored by quality)"
    )
else:
    fig_scatter = px.scatter(
        df, x=x_var, y=y_var, opacity=0.5,
        title=f"{x_var} vs {y_var}"
    )
fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# Distribution explorer
st.subheader("Distribution Explorer")
dist_var = st.selectbox("Select variable", df.columns.tolist(), index=0)

left, right = st.columns(2)

with left:
    fig_hist = px.histogram(
        df, x=dist_var, nbins=50, marginal="box",
        color_discrete_sequence=["#1a3a5c"],
        title=f"Distribution of {dist_var}"
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

with right:
    fig_violin = px.violin(
        df, y=dist_var, x="quality", box=True, points="outliers",
        color="quality", color_discrete_sequence=px.colors.sequential.Teal,
        title=f"{dist_var} by Quality Level"
    )
    fig_violin.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_violin, use_container_width=True)

# Pair plot for top correlated features with quality
st.markdown("---")
st.subheader("Top Features vs Quality")
quality_corr = df[feature_cols].corrwith(df["quality"]).abs().sort_values(ascending=False)
top_features = quality_corr.head(4).index.tolist()

fig_pair = px.scatter_matrix(
    df, dimensions=top_features + ["quality"],
    color="quality", color_continuous_scale="RdYlGn",
    opacity=0.4, height=700,
    title="Scatter Matrix — Top 4 Features Most Correlated with Quality"
)
fig_pair.update_traces(diagonal_visible=False, marker=dict(size=3))
st.plotly_chart(fig_pair, use_container_width=True)
