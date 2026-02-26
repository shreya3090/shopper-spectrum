import streamlit as st
import joblib

st.title("🛒 Shopper Spectrum")

# Try loading models safely
try:
    kmeans = joblib.load("models/kmeans.pkl")
    scaler = joblib.load("models/scaler.pkl")
    similarity_df = joblib.load("models/product_similarity.pkl")
    st.success("Models loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading models: {e}")

# -----------------------------
# Product Recommendation Module
# -----------------------------

st.header("📦 Product Recommendation")

product = st.text_input("Enter Product Name")

if st.button("Get Recommendations"):
    if product in similarity_df.columns:
        recs = similarity_df[product].sort_values(ascending=False)[1:6]
        for r in recs.index:
            st.write("👉", r)
    else:
        st.error("Product not found!")

# -----------------------------
# Customer Segmentation Module
# -----------------------------

st.header("👤 Customer Segmentation")

r = st.number_input("Recency (days)", min_value=0)
f = st.number_input("Frequency", min_value=0)
m = st.number_input("Monetary", min_value=0.0)

if st.button("Predict Cluster"):
    data = scaler.transform([[r, f, m]])
    cluster = kmeans.predict(data)[0]
    st.success(f"Predicted Cluster: {cluster}")
product = st.selectbox("Select Product", similarity_df.columns)
