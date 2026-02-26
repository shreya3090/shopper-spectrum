import streamlit as st
import joblib

st.title("🛒 Shopper Spectrum")

# -----------------------------
# Load Segmentation Models
# -----------------------------

try:
    kmeans = joblib.load("kmeans.pkl")
    scaler = joblib.load("scaler.pkl")
    st.success("Models loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# -----------------------------
# Customer Segmentation Module
# -----------------------------

st.header("👤 Customer Segmentation (RFM Based)")

r = st.number_input("Recency (days)", min_value=0)
f = st.number_input("Frequency", min_value=0)
m = st.number_input("Monetary", min_value=0.0)

if st.button("Predict Cluster"):
    data = scaler.transform([[r, f, m]])
    cluster = kmeans.predict(data)[0]

    st.success(f"Predicted Cluster: {cluster}")

    # Optional: Add simple interpretation
    if cluster == 0:
        st.write("💎 High-Value Customer")
    elif cluster == 1:
        st.write("⚡ Regular Customer")
    else:
        st.write("🔄 At-Risk Customer")
