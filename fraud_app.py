import streamlit as st
import pandas as pd
import numpy as np
import pickle
import logging
import json
import base64

# ---------------- Logging ----------------
logging.basicConfig(
    filename="model_performance.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🔍",
    layout="centered"
)

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    with open("final_xgb_credit_model.pkl", "rb") as f1:
        xgb_credit = pickle.load(f1)
    with open("final_xgb_paysim_model.pkl", "rb") as f2:
        xgb_paysim = pickle.load(f2)
    return xgb_credit, xgb_paysim

xgb_credit, xgb_paysim = load_models()

# ---------------- Sidebar ----------------
st.sidebar.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
st.sidebar.title("🧠 AI Fraud Detection")
st.sidebar.markdown("Upload a transaction file (.JSON or .CSV) to detect fraud using trained XGBoost models.")

# ---------------- Main Interface ----------------
st.title("🔍 Fraud Detection for Transactions")

uploaded_file = st.file_uploader("📤 Upload a transaction file", type=["json", "csv"])

if uploaded_file:
    try:
        # ---------------- Load Data ----------------
        if uploaded_file.name.endswith(".json"):
            input_data = json.load(uploaded_file)
            df = pd.DataFrame([input_data])
        else:
            df = pd.read_csv(uploaded_file)

        with st.expander("📊 Click to Preview Uploaded Data"):
            st.write(df)

        # ---------------- Select Model ----------------
        if "Time" in df.columns:
            model = xgb_credit
            dataset_type = "Credit Card"
        elif "step" in df.columns:
            model = xgb_paysim
            dataset_type = "PaySim Mobile"
        else:
            st.error("🚫 Unrecognized format: Include a 'Time' or 'step' column.")
            st.stop()

        # ---------------- Prediction ----------------
        prediction = model.predict(df)
        prediction_prob = model.predict_proba(df)[:, 1]
        pred_label = "Fraud" if prediction[0] == 1 else "Not Fraud"
        pred_color = "red" if prediction[0] == 1 else "green"

        # ---------------- Display Result ----------------
        st.markdown(f"""
            <div style="border:2px solid {pred_color}; padding:20px; border-radius:10px; background-color:#f9f9f9;">
                <h3>📌 Prediction: <span style="color:{pred_color};">{pred_label}</span></h3>
                <p><b>Probability of Fraud:</b> {round(float(prediction_prob[0])*100, 2)}%</p>
                <p><b>Model Used:</b> {dataset_type}</p>
            </div>
        """, unsafe_allow_html=True)

        # ---------------- Log Activity ----------------
        logging.info(f"Input: {df.to_dict()}, Prediction: {pred_label}, Probability: {prediction_prob[0]}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        logging.error(f"Prediction failed: {e}")

else:
    st.info("Please upload a .json or .csv transaction file to begin.")
