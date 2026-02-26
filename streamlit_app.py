import streamlit as st
import joblib
import pandas as pd

st.title("🛒 Shopper Spectrum")

# -----------------------------
# Load Segmentation Models
# -----------------------------

try:
    kmeans = joblib.load("kmeans.pkl")
    scaler = joblib.load("scaler.pkl")
    st.success("Segmentation model loaded ✅")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# -----------------------------
# Lightweight Product List
# -----------------------------

products = [
    "Laptop",
    "Mobile",
    "Headphones",
    "Smart Watch",
    "Tablet",
    "Shoes",
    "Backpack"
]

# -----------------------------
# Product Recommendation (Demo Version)
# -----------------------------

st.header("📦 Product Recommendation")

selected_product = st.selectbox("Select Product", products)

if st.button("Get Recommendations"):
    
    # Simple demo logic (lightweight)
    recommendations = {
        "Laptop": ["Mouse", "Keyboard", "Laptop Bag", "USB Hub"],
        "Mobile": ["Phone Case", "Charger", "Earbuds", "Power Bank"],
        "Headphones": ["Bluetooth Adapter", "Music Subscription", "Carrying Case"],
        "Smart Watch": ["Watch Strap", "Screen Protector", "Fitness Band"],
        "Tablet": ["Stylus", "Tablet Cover", "Keyboard Case"],
        "Shoes": ["Socks", "Shoe Cleaner", "Sports Bag"],
        "Backpack": ["Water Bottle", "Notebook", "Laptop Sleeve"]
    }

    st.subheader("Recommended Products:")
    for item in recommendations.get(selected_product, []):
        st.write("👉", item)

# -----------------------------
# Customer Segmentation
# -----------------------------

st.header("👤 Customer Segmentation (RFM Based)")

r = st.number_input("Recency (days)", min_value=0)
f = st.number_input("Frequency", min_value=0)
m = st.number_input("Monetary", min_value=0.0)

if st.button("Predict Cluster"):
    data = scaler.transform([[r, f, m]])
    cluster = kmeans.predict(data)[0]

    st.success(f"Predicted Cluster: {cluster}")

    if cluster == 0:
        st.write("💎 High-Value Customer")
    elif cluster == 1:
        st.write("⚡ Regular Customer")
    else:
        st.write("🔄 At-Risk Customer")
