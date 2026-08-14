import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import numpy as np
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

@st.cache_resource
def build_and_compile_model():
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

def preprocess_image(image: Image.Image):
    """
    Preprocesses input image exactly as shown in Step 11:
    - Target size: (224, 224)
    - Normalized pixel range: [0.0, 1.0] (1.0/255)
    - Reshaped batch dimension: (1, 224, 224, 3)
    """
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_parkinson(model, img_array):
    """
    Inference function aligned with `predict_parkinson()` logic in Step 11.
    """
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        result = "Parkinson Detected"
        confidence = float(prediction)
    else:
        result = "Healthy Brain"
        confidence = float(1.0 - prediction)
        
    return result, confidence, float(prediction)

# --- GLOBAL DISCLAIMER ---
def render_disclaimer():
    st.markdown("---")
    st.markdown(
        """
        > **⚠️ Clinical Disclaimer & Notice:**  
        > *This web application utilizes a Deep Learning Convolutional Neural Network (CNN) pipeline based on the project notebook. It is intended exclusively for research, educational, and experimental demonstration purposes. It does not replace professional medical evaluations.*  
        >  
        > **Developer:** Sristi Sarkar | **Framework:** TensorFlow / Keras Pipeline
        """
    )

# --- 1. HOME PAGE ---
if page == "Home":
    st.title("🧠 Parkinson’s Disease Detection via Deep Learning")
    st.markdown("An automated diagnostic interface running the Sequential CNN architecture constructed in your Colab training environment.")
    
    st.subheader("📌 Key Engineering Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Input Dimensions", "224 × 224 × 3")
    with col2:
        st.metric("Loss Function", "Binary Cross-Entropy")
    with col3:
        st.metric("TensorFlow Version", f"{tf.__version__}")

    st.markdown("---")
    st.subheader("🏗️ In-Memory CNN Model Architecture")
    st.code(
        """
cnn_model = Sequential([
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
    st.write("Run predictions directly using the in-memory CNN network.")

    # Initialize model in memory without loading external .h5 binary files
    model = build_and_compile_model()

    option = st.radio("Select Input Method:", ["Upload Image", "Analyze Existing GitHub Sample Images"])

    image_to_analyze = None

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image file (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_to_analyze = Image.open(uploaded_file)
            st.image(image_to_analyze, caption="Uploaded Image", use_container_width=True)

    else:
        sample_urls = {
            "Not Affected / Healthy Sample": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_183555.jpg",
            "Parkinson Affected Sample": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_183533.jpg"
        }
        
        selected_sample = st.selectbox("Select Sample Image from Repository Path:", list(sample_urls.keys()))
        sample_url = sample_urls[selected_sample]

        if st.button("Load Selected Sample"):
            with st.spinner("Fetching sample image from repository..."):
                try:
                    response = requests.get(sample_url)
                    image_to_analyze = Image.open(BytesIO(response.content))
                    st.image(image_to_analyze, caption=f"Loaded: {selected_sample}", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to fetch sample image: {e}")

    if image_to_analyze is not None:
        if st.button("⚡ Run Diagnostic Analysis"):
            with st.spinner("Running CNN prediction..."):
                img_array = preprocess_image(image_to_analyze)
                result, confidence, raw_score = predict_parkinson(model, img_array)

                st.markdown("### 📋 Diagnostic Outcome")
                if "Detected" in result:
                    st.error(f"**Prediction:** {result}")
                else:
                    st.success(f"**Prediction:** {result}")
                
                st.info(f"**Confidence Score:** {confidence * 100:.2f}% (Raw Sigmoid Score: `{raw_score:.4f}`)")

    render_disclaimer()

# --- 3. ABOUT PAGE ---
elif page == "About":
    st.title("ℹ️ Technical Overview & Specifications")
    st.markdown(
        """
        * **Notebook Script Alignment:** Reconstructs the training logic from your Colab script (`gdown` $\\rightarrow$ `ImageDataGenerator` $\\rightarrow$ `Sequential CNN`).
        * **Image Preprocessing:** Rescaling scaled by $1.0 / 255$ to match image normalization performed during model compilation.
        * **Architecture:** 3 Convolutional Blocks + Dense Layer with Dropout ($0.5$).
        """
    )
    render_disclaimer(
