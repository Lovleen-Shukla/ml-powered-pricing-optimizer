# -------------------------------
# Fix import path (ABSOLUTE & SAFE)
# -------------------------------
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# -------------------------------
# Imports
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from src.features import add_time_features
from src.optimizer import find_optimal_price

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Dynamic Pricing Decision System",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Load model and data
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(
        os.path.join(PROJECT_ROOT, "data", "processed", "demand_model.pkl")
    )

@st.cache_data
def load_data():
    df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "processed", "daily_demand.csv"),
        dtype={"StockCode": str},     # 👈 FIX
        low_memory=False
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


model = load_model()
df = load_data()

# ============================================================
# HEADER
# ============================================================
st.title("📈 Dynamic Pricing & Revenue Optimization System")
st.caption(
    "ML-powered decision support tool that recommends optimal prices "
    "to maximize revenue based on demand forecasting."
)

st.markdown("---")

# ============================================================
# SIDEBAR – BUSINESS CONTROLS
# ============================================================
st.sidebar.header("⚙️ Pricing Controls")

product_id = st.sidebar.selectbox(
    "Product ID",
    df["StockCode"].value_counts().index[:50],
    help="Select the product for which pricing optimization will be performed."
)

price_min = st.sidebar.slider(
    "Lower Price Bound (%)",
    50, 100, 70,
    help="Minimum price (as % of current price) the model is allowed to test."
) / 100

price_max = st.sidebar.slider(
    "Upper Price Bound (%)",
    100, 150, 130,
    help="Maximum price (as % of current price) the model is allowed to test."
) / 100

st.sidebar.markdown("---")
st.sidebar.caption(
    "These controls define the safe pricing range. "
    "The ML model optimizes revenue within these limits."
)

# ============================================================
# DATA PREPARATION
# ============================================================
product_df = df[df["StockCode"] == product_id].copy()
product_df = add_time_features(product_df)

FEATURES = [
    "Price",
    "day",
    "month",
    "day_of_week",
    "is_weekend"
]

current_price = product_df["Price"].median()

# ============================================================
# PRICING OPTIMIZATION
# ============================================================
results_df = find_optimal_price(
    model=model,
    base_row=product_df.iloc[-1],
    features=FEATURES,
    price_min_factor=price_min,
    price_max_factor=price_max,
    steps=25
)

optimal_row = results_df.loc[
    results_df["expected_revenue"].idxmax()
]

# ============================================================
# KPI SECTION
# ============================================================
st.subheader("💰 Pricing Recommendation")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "Current Price",
    f"{current_price:.2f}",
    help="Median historical price for the selected product."
)

kpi2.metric(
    "Recommended Price",
    f"{optimal_row['price']:.2f}",
    help="Price that maximizes expected revenue according to the ML model."
)

kpi3.metric(
    "Expected Demand",
    f"{optimal_row['predicted_demand']:.0f}",
    help="Predicted number of units sold at the recommended price."
)

kpi4.metric(
    "Expected Revenue",
    f"{optimal_row['expected_revenue']:.2f}",
    help="Projected revenue at the recommended price."
)

st.markdown("---")

# ============================================================
# VISUAL ANALYSIS
# ============================================================
left, right = st.columns([2, 1])

with left:
    st.subheader("📊 Revenue vs Price Analysis")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        results_df["price"],
        results_df["expected_revenue"],
        marker="o",
        linewidth=2
    )

    ax.axvline(
        current_price,
        linestyle=":",
        color="gray",
        label="Current Price"
    )

    ax.axvline(
        optimal_row["price"],
        linestyle="--",
        color="red",
        label="Recommended Price"
    )

    ax.set_xlabel("Price")
    ax.set_ylabel("Expected Revenue")
    ax.set_title("Revenue Optimization Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

with right:
    st.subheader("🧠 Business Interpretation")

    st.markdown(
        """
        **How to interpret this chart:**
        
        • Each point shows expected revenue at a specific price  
        • Revenue increases until demand drops faster than price increases  
        • **Dashed red line** → revenue-maximizing price  
        • **Dotted gray line** → current price  
        
        This helps decision-makers justify pricing changes using data.
        """
    )

# ============================================================
# DATA INSPECTION
# ============================================================
with st.expander("🔍 View Pricing Simulation Table"):
    st.dataframe(
        results_df.style.format({
            "price": "{:.2f}",
            "predicted_demand": "{:.0f}",
            "expected_revenue": "{:.2f}"
        }),
        use_container_width=True
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Decision-Focused Machine Learning Project • Demand Forecasting + Optimization • Streamlit Dashboard"
)
