import streamlit as st
import joblib
import os
import requests

st.title("🛒 Shopper Spectrum")

# -----------------------------
# Load Models Safely
# -----------------------------

try:
    # Load segmentation models
    kmeans = joblib.load("kmeans.pkl")
    scaler = joblib.load("scaler.pkl")

    # Download similarity file if not present
    if not os.path.exists("product_similarity.pkl"):
        file_id = "1gO83w912PxAJl7Ydze4p7eoDF0qkyiOW"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)

        with open("product_similarity.pkl", "wb") as f:
            f.write(response.content)

    # Load similarity dataframe
    similarity_df = joblib.load("product_similarity.pkl")

    st.success("Models loaded successfully ✅")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# -----------------------------
# Product Recommendation Module
# -----------------------------

st.header("📦 Product Recommendation")

product = st.selectbox("Select Product", similarity_df.columns)

if st.button("Get Recommendations"):
    if product in similarity_df.columns:
        recs = similarity_df[product].sort_values(ascending=False)[1:6]
        st.subheader("Recommended Products:")
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
