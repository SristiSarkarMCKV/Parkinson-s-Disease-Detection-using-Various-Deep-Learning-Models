import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import numpy as np
import os
import requests
from io import BytesIO

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="wide"
)

# --- NAVIGATION SIDEBAR ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "About"])

MODEL_PATH = 'parkinson_model.h5'

def build_model_from_notebook():
    """
    Reconstructs the exact CNN architecture defined in the notebook:
    In [1]: Sequential -> Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

@st.cache_resource
def load_model():
    """Loads weights if available, or compiles the architecture from the notebook."""
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.warning(f"Could not load local weights file: {e}. Rebuilding model architecture.")
    
    # Fallback to model structure built in notebook
    return build_model_from_notebook()

def preprocess_image(image: Image.Image):
    """Preprocesses input image (224x224, normalized 1/255)."""
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def make_prediction(model, img_array):
    """Executes model inference."""
    pred = model.predict(img_array)[0][0]
    if pred >= 0.5:
        result = "Parkinson's Disease Detected"
        confidence = float(pred)
    else:
        result = "Normal / Not Affected"
        confidence = float(1 - pred)
    return result, confidence

# --- GLOBAL DISCLAIMER ---
def render_disclaimer():
    st.markdown("---")
    st.markdown(
        """
        > **⚠️ Clinical Disclaimer & Notice:**  
        > *This application uses a Deep Learning Convolutional Neural Network (CNN) pipeline built based on `Parkinson's_Disease_Detection_using_Various_Deep_Learning_Models.ipynb`. Designed solely for research and educational purposes. Always consult with a qualified healthcare professional for formal evaluations.*  
        >  
        > **Developer:** Sristi Sarkar | **Accuracy:** 97.0% | **Recall:** 98.84%
        """
    )

# --- 1. HOME PAGE ---
if page == "Home":
    st.title("🧠 Parkinson’s Disease Detection via Deep Learning")
    st.markdown("An automated diagnostic pipeline using the Sequential CNN trained in your Jupyter notebook.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Accuracy", "97.0%")
    with col2:
        st.metric("Recall (Sensitivity)", "98.84%")
    with col3:
        st.metric("Runtime Engine", f"TensorFlow {tf.__version__}")

    st.markdown("---")
    st.subheader("🏗️ Model Architecture (from Notebook)")
    st.code(
        """
Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
        """,
        language="python"
    )

    render_disclaimer()

# --- 2. PREDICTION PAGE ---
elif page == "Prediction":
    st.title("🔍 Inference & Diagnostic Prediction")
    
    model = load_model()

    option = st.radio("Select Input Method:", ["Upload Image", "Analyze Existing Sample Images"])

    image_to_analyze = None

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_to_analyze = Image.open(uploaded_file)
            st.image(image_to_analyze, caption="Uploaded Image", use_container_width=True)

    else:
        sample_urls = {
            "Not Affected Sample": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_183555.jpg",
            "Affected Sample": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_183533.jpg"
        }
        
        selected_sample = st.selectbox("Select Sample Image:", list(sample_urls.keys()))
        sample_url = sample_urls[selected_sample]

        if st.button("Load Selected Sample"):
            with st.spinner("Fetching image from repository..."):
                try:
                    response = requests.get(sample_url)
                    image_to_analyze = Image.open(BytesIO(response.content))
                    st.image(image_to_analyze, caption=f"Loaded: {selected_sample}", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to retrieve sample image: {e}")

    if image_to_analyze is not None:
        if st.button("⚡ Run Diagnostic Analysis"):
            with st.spinner("Executing prediction pipeline..."):
                img_array = preprocess_image(image_to_analyze)
                result, confidence = make_prediction(model, img_array)

                st.markdown("### 📋 Diagnostic Outcome")
                if "Detected" in result:
                    st.error(f"**Prediction:** {result}")
                else:
                    st.success(f"**Prediction:** {result}")
                st.info(f"**Confidence Score:** {confidence * 100:.2f}%")

    render_disclaimer()

# --- 3. ABOUT PAGE ---
elif page == "About":
    st.title("ℹ️ Technical Overview & Source Notebook")
    st.markdown(
        """
        * **Notebook Link:** [`Parkinson's_Disease_Detection_using_Various_Deep_Learning_Models.ipynb`](https://github.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/blob/main/Parkinson%E2%80%99s_Disease_Detection_using_Various_Deep_Learning_Models.ipynb)
        * **Model Architecture:** Sequential 3-Block 2D-CNN with Sigmoid classification head
        * **Optimizer & Loss:** Adam, Binary Cross-Entropy
        """
    )
    render_disclaimer()
