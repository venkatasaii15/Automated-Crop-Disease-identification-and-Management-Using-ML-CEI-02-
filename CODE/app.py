import streamlit as st

from tensorflow.keras.models import load_model # type: ignore

from tensorflow.keras.preprocessing import image # type: ignore

import numpy as np

from PIL import Image

# Load model

model = load_model('rice_disease_model.h5')

class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

# Streamlit UI

st.title("🌾 Rice Crop Disease Detection")

st.write("Upload an image of a rice leaf to detect the disease.")

uploaded_file = st.file_uploader("Choose a rice leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

  img = Image.open(uploaded_file).convert("RGB")

  st.image(img, caption='Uploaded Leaf Image', use_column_width=True)

  # Preprocess

  img = img.resize((150, 150))

  img_array = image.img_to_array(img)

  img_array = np.expand_dims(img_array, axis=0)

  img_array = img_array / 255.0

  # Predict

  prediction = model.predict(img_array)

  class_index = np.argmax(prediction)

  confidence = prediction[0][class_index]

  st.markdown(f"### 🩺 Prediction: `{class_names[class_index]}`")

  st.markdown(f"### ✅ Confidence: `{confidence*100:.2f}%`")















