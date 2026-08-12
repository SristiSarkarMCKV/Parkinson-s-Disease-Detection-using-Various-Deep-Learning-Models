import os
import urllib.request
import io
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Page Configuration
st.set_page_config(
    page_title="Parkinson's Disease Diagnostic Platform",
    page_icon="🧠",
    layout="centered"
)

MODEL_PATH = "PD_CNN_Model.h5"

# Automatically download model weights from Hugging Face if not present locally
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights from Hugging Face..."):
        MODEL_URL = "https://huggingface.co/SRISTISARKAR/parkinsons-cnn-model/resolve/main/PD_CNN_Model.h5"
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)
model = get_model()
CLASS_NAMES = ["Healthy Control", "Parkinson's Disease Detected"]

# GitHub Raw URLs for the exact sample images
AFFECTED_URLS = [
    "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_183520.jpg",
    "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_185834.jpg",
    "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_185906.jpg",
    "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_185944.jpg"
]

UNAFFECTED_URLS = [
    "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_183555.jpg",
    "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_185816.jpg",
    "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_185849.jpg",
    "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_185926.jpg"
]

@st.cache_data
def load_image_from_url(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")
    except Exception as e:
        return Image.new('RGB', (224, 224), color=(50, 50, 50))

affected_imgs = [load_image_from_url(url) for url in AFFECTED_URLS]
unaffected_imgs = [load_image_from_url(url) for url in UNAFFECTED_URLS]
all_sample_imgs = unaffected_imgs + affected_imgs

def predict_parkinsons(img):
    if img is None:
        return None
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  
    prediction = model.predict(img_array)[0][0]
    return {
        CLASS_NAMES[0]: float(1 - prediction), 
        CLASS_NAMES[1]: float(prediction)
    }

# App Header Styling & Navigation Radio
st.markdown("<h1 style='text-align: center;'>🧠 🩺 Parkinson's Disease Diagnostic Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><em>Automated deep learning biometric and image classification engine.</em></p>", unsafe_allow_html=True)
st.markdown("---")

page = st.radio("Navigation", ["🏠 Home", "🔮 Prediction", "ℹ️ About"], horizontal=True, label_visibility="collapsed")

# --- HOME PAGE ---
if page == "🏠 Home":
    st.markdown("## 🧬 Automated Deep Learning Recognition Engine")
    st.write(
        "Welcome! This application utilizes state-of-the-art Deep Computer Vision to instantly analyze "
        "and evaluate biomarker visual cohorts. Built on top of a custom **Sequential Convolutional Neural Network (CNN)**, "
        "the framework processes structural intensity inputs to classify medical scans into distinct diagnostic categories:"
    )
    st.markdown("* 🟢 **Healthy Control** or 🔴 **Parkinson's Disease Detected**.")
    
    st.markdown("---")
    st.markdown("### 📌 Core Engineering Highlights")
    st.markdown("* **State-of-the-Art Accuracy:** Achieved **97.0% Validation & Test Accuracy** within 10 training epochs.")
    st.markdown("* **Clinical Safety First:** Optimized for a **98.84% Recall rate** on positive cases to minimize dangerous False Negatives.")
    st.markdown("* **Modern Technical Stack:** Developed natively in Python using the latest **TensorFlow** runtime engine.")

# --- PREDICTION PAGE ---
elif page == "🔮 Prediction":
    st.markdown("### 🔍 Interactive Scan Predictor")
    st.markdown("Select a source below to either upload your own scan image or pick from the pre-loaded clinical sample database.")
    
    source_choice = st.radio("Source", ["Upload Image", "Sample Images"], horizontal=True)
    
    target_img = None
    
    if source_choice == "Upload Image":
        uploaded_file = st.file_uploader("Upload Scan Image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            target_img = Image.open(uploaded_file).convert("RGB")
            st.image(target_img, caption="Uploaded Scan", use_column_width=True)
    else:
        st.markdown("#### Click or select a sample image below:")
        # Display sample images in columns/grid layout
        cols = st.columns(4)
        selected_idx = None
        for idx, img in enumerate(all_sample_imgs):
            with cols[idx % 4]:
                st.image(img, use_column_width=True, caption=f"Sample {idx+1}")
                if st.button("Select", key=f"btn_{idx}"):
                    st.session_state['selected_sample_idx'] = idx
        
        if 'selected_sample_idx' in st.session_state:
            idx = st.session_state['selected_sample_idx']
            target_img = all_sample_imgs[idx]
            st.success(f"Selected Sample #{idx+1}")
            st.image(target_img, caption=f"Selected Sample #{idx+1}", width=250)

    st.markdown("---")
    if st.button("Run Inference", type="primary"):
        if target_img is None:
            st.warning("Please provide or select a valid scan image.")
        else:
            with st.spinner("Analyzing scan through CNN pipeline..."):
                results = predict_parkinsons(target_img)
                st.markdown("### 📊 Prediction Probabilities")
                for class_name, prob in results.items():
                    st.metric(label=class_name, value=f"{prob*100:.2f}%")
                    st.progress(prob)

# --- ABOUT PAGE ---
elif page == "ℹ️ About":
    st.markdown("## ⚙️ Classification System Architecture & Workflow")
    st.markdown("* **Dataset & Augmentation:** Real-time data generators scale pixel intensities ($1/255$), apply structural transformations ($15^{\circ}$ rotation, $0.2$ shear, $0.2$ zoom, and horizontal flips) to protect against overfitting.")
    st.markdown("* **Model Design:** Cascaded 2D Convolution and Max-Pooling blocks leading into dense operational layers with dropout regulation ($0.5$).")
    st.markdown("### Performance Metrics")
    st.markdown("""
    * **Accuracy:** `97.00%`
    * **Precision:** `0.9659`
    * **Recall:** `98.84%`
    * **F1-Score:** `0.9770`
    """)
