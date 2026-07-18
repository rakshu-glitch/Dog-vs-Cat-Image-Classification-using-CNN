import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐶",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

classes = ["Cat", "Dog"]

# ----------------------------
# Title
# ----------------------------
st.title("🐶🐱 Dog vs Cat Image Classification")

st.write("""
Upload an image and the AI model will predict whether it is a **Cat** or a **Dog**.
""")

# ----------------------------
# Upload Image
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize image
    img = image.resize((224,224))

    img_array = np.array(img)

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction))*100

    st.markdown("---")

    st.subheader("Prediction")

    st.success(f"**{classes[predicted_index]}**")

    st.metric("Confidence", f"{confidence:.2f}%")

    st.markdown("---")

    st.write("Raw Prediction")

    st.write(prediction)