import streamlit as st
import joblib
import os
import requests

st.title("🛒 Shopper Spectrum")

# -----------------------------
# Google Drive Download Function (Handles Large Files)
# -----------------------------

def download_file_from_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)

    # Handle large file confirmation token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(
                URL,
                params={"id": file_id, "confirm": value},
                stream=True,
            )

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# -----------------------------
# Load Models Safely
# -----------------------------

try:
    # Load segmentation models
    kmeans = joblib.load("kmeans.pkl")
    scaler = joblib.load("scaler.pkl")

    # Download similarity model if not present
    if not os.path.exists("product_similarity.pkl"):
        st.info("Downloading recommendation model... ⏳")
        download_file_from_drive(
            "1gO83w912PxAJl7Ydze4p7eoDF0qkyiOW",
            "product_similarity.pkl",
        )

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
