import streamlit as st
from PIL import Image
import numpy as np
from preprocess import get_dft_rgb_feat, get_prediction
import matplotlib.pyplot as plt
import cv2

classes = ['fake','real']
st.set_page_config(page_title="Deepfake Detection – DFT+ELA", page_icon="🕵️", layout="centered")

st.markdown("""
<div style='text-align: center;'>
    <h1>🕵️ Deepfake Detection</h1>
    <span style='font-size:1.2em;'>Robust detection using DFT & ELA features</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("**Upload your image for verification**", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image")
    st.markdown(":mag_right: _Feature extraction and prediction will start when you click the button below._")

button = st.button("🔍 Verify Image", type="primary")

if uploaded_file and button:
    with st.spinner("Extracting features and running the model..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rgb_feat, dft_feat = get_dft_rgb_feat(img)

        pred = get_prediction([rgb_feat, dft_feat])
        pred_prob = float(pred)
        y_val_pred = (pred_prob >= 0.509)
        pred_label = classes[y_val_pred]  
        color = "#27ae60" if pred_label == "real" else "#c0392b"

        certainty = pred_prob if y_val_pred else (1 - pred_prob)
        certainty_percent = certainty * 100

        st.markdown(f"""
        <div style='text-align:center;'>
            <span style='background:{color}; color:white; padding:8px 30px; border-radius:8px; font-size:1.5em;'>
                Prediction: <b>{pred_label.capitalize()}</b>
            </span>
            <br><br>
            <span style='font-size:1.1em;'>Probability: <b>{pred_prob:.3f}</b></span><br>
            <span style='font-size:1.1em;'>Certainty: <b>{certainty_percent:.1f}%</b></span>
        </div>""", unsafe_allow_html=True)

        st.success("Prediction complete! You can try another image.")
else:
    st.info("Upload a JPG or PNG image and click 'Verify Image' to begin.")
