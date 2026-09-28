import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect plant diseases")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_model.keras')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

uploaded = st.file_uploader("Choose a leaf image", type=['jpg', 'jpeg', 'png'])

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess — NO /255 (EfficientNet handles it internally!)
    img = image.resize((224, 224))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)

    with st.spinner('Analyzing...'):
        preds = model.predict(arr, verbose=0)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100
        label = class_names[str(idx)]

    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Top 3
    st.subheader("Top 3 Predictions")
    top3 = np.argsort(preds[0])[-3:][::-1]
    for i in top3:
        st.write(f"- {class_names[str(int(i))]}: {preds[0][i]*100:.2f}%")