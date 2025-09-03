import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import io
from tensorflow.keras.preprocessing.image import img_to_array

# —— Setup UI —— #
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶🐱",
    layout="centered"
)

st.title("🐶🐱 Cat vs Dog Classifier")
st.markdown("""
Upload gambar kucing atau anjing, dan model MobileNetV2 (Fine-Tuned) akan memprediksinya.  
Model dimuat langsung dari GitHub.
""")
https://github.com/alifiaazzahra1515-debug/cat
# —— Load Model dari GitHub —— #
@st.cache_resource 
def load_model_from_github():
    model_url = "https://github.com/alifiaazzahra1515-debug/cat"  # ganti dengan URL GitHub kamu
    response = requests.get(model_url)
    response.raise_for_status()  # biar error jelas kalau gagal download
    model_bytes = io.BytesIO(response.content)
    model = tf.keras.models.load_model(model_bytes, compile=False)
    return model

model = load_model_from_github()
IMG_SIZE = 128

# —— Preprocessing Function —— #
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img

# —— Upload dan Prediksi —— #
uploaded = st.file_uploader("🌄 Upload Gambar (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    img_pil = Image.open(uploaded)
    arr, img_display = preprocess(img_pil)

    st.image(img_display, caption="Gambar", use_column_width=True)

    pred_prob = model.predict(arr)[0][0]
    label = "🐶 Dog" if pred_prob > 0.5 else "🐱 Cat"
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

    st.subheader("Hasil Prediksi")
    st.metric("Kelas", label)
    st.metric("Confidence", f"{confidence*100:.2f}%")
    st.progress(float(confidence))

    if label == "🐶 Dog":
        st.success("Model yakin ini **anjing**")
    else:
        st.info("Model yakin ini **kucing**")

# —— Fitur Tambahan —— #
st.markdown("---")
st.markdown("""
### ✨ Kenapa Ini Layak Dihargai?
- **UI bersih & intuitif**: upload → langsung tampil hasil.
- **Feedback visual**: gambar, label, confidence meter.
- **Optimasi cache**: model tidak dimuat ulang setiap kali.
- **Aksesibilitas**: bisa dijalankan langsung di Streamlit Cloud.
""")

