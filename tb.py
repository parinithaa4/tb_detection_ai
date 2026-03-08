import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import cv2
from PIL import Image

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="TB Detection AI", layout="centered")

st.title("🩺 Multimodal Tuberculosis Detection System")
st.markdown("AI-based Screening Prototype (Image + Symptoms)")

# -------------------------
# Load Models (cached)
# -------------------------
@st.cache_resource
def load_models():
    model_img = tf.keras.models.load_model("tb_image_model.keras")
    model_nlp = tf.keras.models.load_model("tb_nlp_model.keras")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model_img, model_nlp, vectorizer

model_img, model_nlp, vectorizer = load_models()

# -------------------------
# Prediction Function
# -------------------------
def multimodal_predict(image, text):
    # Image processing
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img / 255.0, axis=0)

    img_pred = model_img.predict(img)[0][0]

    # Text processing
    vec = vectorizer.transform([text]).toarray()
    text_pred = model_nlp.predict(vec)[0][0]

    # Weighted fusion
    final_score = (0.7 * img_pred) + (0.3 * text_pred)
    confidence = final_score * 100

    if final_score > 0.6:
        result = "TB Detected"
    else:
        result = "Normal"

    return result, confidence


# -------------------------
# UI Inputs
# -------------------------
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])
symptoms = st.text_area("Enter Patient Symptoms")

# -------------------------
# Prediction Button
# -------------------------
if st.button("Analyze"):
    if uploaded_file is not None and symptoms.strip() != "":
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        result, confidence = multimodal_predict(image, symptoms)

        st.subheader("Result")

        if result == "TB Detected":
            st.error(f"⚠️ High Risk of Tuberculosis\n\nConfidence: {confidence:.2f}%")
        else:
            st.success(f"✅ No TB Detected\n\nConfidence: {confidence:.2f}%")

        st.markdown("---")
        st.caption("⚠️ This is an AI-based screening prototype and not a medical diagnosis.")

    else:
        st.warning("Please upload an image and enter symptoms.")