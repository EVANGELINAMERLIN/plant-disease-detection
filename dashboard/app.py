import streamlit as st
import numpy as np
from PIL import Image
import json
import tensorflow as tf

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

IMG_SIZE = 96

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/plant_disease_model.h5")
    with open("models/class_indices.json", "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_model()

st.title("🌿 Plant Disease Detection System")
st.write("Upload a leaf image and the model will predict whether the plant is healthy or diseased.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing leaf image..."):
        prediction = model.predict(img_array)
        pred_idx = np.argmax(prediction)
        pred_class = idx_to_class[pred_idx]
        confidence = np.max(prediction) * 100

    st.subheader("Prediction Result")

    if "healthy" in pred_class.lower():
        st.success(f"✅ {pred_class}")
    else:
        st.error(f"⚠️ {pred_class}")

    st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("Top 3 Predictions")
    top_indices = prediction[0].argsort()[-3:][::-1]
    for idx in top_indices:
        st.write(f"- {idx_to_class[idx]}: {prediction[0][idx]*100:.2f}%")

st.markdown("---")
st.caption("Plant Disease Detection System | CNN Model trained on PlantVillage Dataset")
