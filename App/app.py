import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image
import streamlit.components.v1 as components
import requests
from io import BytesIO
import os
import zipfile
import gdown

# ---------------------------------------------------------
# Page Setup & Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Parkinson's Disease Detection 🧠",
    page_icon="🧠",
    layout="centered"
)

PERMANENT_BG_GIF = "https://media2.giphy.com/media/v1.Y2lksetItemzZjMDliOTUyemhtNmVzYTdldnp1endmNXRheTBzcHIyc2h0cG5xcHoxdXFyOXFiOSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xRGuaM7FFZSZq/giphy.gif"


def inject_custom_styles(bg_url):
    css = (
        "<style>\n"
        "@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;700;800;900&family=Poppins:wght@300;400;600;700&display=swap');\n"
        "html, body, [class*='css'] { font-family: 'Poppins', sans-serif; }\n"
        
        "[data-testid='stHeaderActionElements'], .stHeadingAnchor, a.data-testid-stHeaderActionElements, .css-1544g2n { display: none !important; visibility: hidden !important; }\n"
        "h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; opacity: 0 !important; }\n"
        "a[href*='#'] { display: none !important; }\n"

        "::-webkit-scrollbar { width: 12px; }\n"
        "::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.7); }\n"
        "::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #FF781F, #FF9800, #F57C00); border-radius: 10px; border: 2px solid rgba(255, 255, 255, 0.25); }\n"
        "::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #E65100, #FF6D00, #FF9800); }\n"
        
        ".stApp {\n"
        "  background-image: linear-gradient(rgba(15, 23, 42, 0.75), rgba(15, 23, 42, 0.75)), url('" + bg_url + "');\n"
        "  background-attachment: fixed;\n"
        "  background-size: cover;\n"
        "  background-position: center;\n"
        "  min-height: 100vh;\n"
        "  display: flex;\n"
        "  align-items: center;\n"
        "  justify-content: center;\n"
        "}\n"
        
        ".block-container {\n"
        "  background: rgba(255, 255, 255, 0.95);\n"
        "  color: #1A202C;\n"
        "  border-radius: 28px;\n"
        "  padding: 24px 20px !important;\n"
        "  margin: auto !important;\n"
        "  max-width: 720px;\n"
        "  width: 100%;\n"
        "  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);\n"
        "  backdrop-filter: blur(14px);\n"
        "  border: 1px solid rgba(255, 255, 255, 0.4);\n"
        "}\n"

        ".content-section {\n"
        "  margin-top: 100px !important;\n"
        "  margin-bottom: 0px !important;\n"
        "}\n"

        "div.element-container {\n"
        "  margin-bottom: 0px !important;\n"
        "  margin-top: 10px !important;\n"
        "}\n"
        "div[data-testid='stVerticalBlock'] {\n"
        "  gap: 0.4rem !important;\n"
        "}\n"
        "h3 { margin-top: 10px !important; margin-bottom: 0.3rem !important; }\n"
        "h4 { margin-top: 10px !important; margin-bottom: 0.3rem !important; }\n"
        "p { margin-bottom: 0.3rem !important; margin-top: 0px !important; }\n"

        "div[data-testid='stAlert'] { color: #1A202C !important; font-weight: 500; border-radius: 12px; margin-bottom: 0.3rem !important; margin-top: 0.2rem !important; }\n"
        "div[data-testid='stAlert'] p { color: #1A202C !important; font-weight: 500; }\n"
        "div[data-testid='stAlert'] strong { color: #000000 !important; font-weight: 800; }\n"
        
        "@media (prefers-color-scheme: dark) {\n"
        "  .block-container {\n"
        "    background: rgba(15, 23, 42, 0.92) !important;\n"
        "    color: #F7FAFC !important;\n"
        "    border: 1px solid rgba(255, 255, 255, 0.15);\n"
        "  }\n"
        "  .sub-text { color: #E2E8F0 !important; }\n"
        "  .feature-card { background: #1E293B !important; border-color: #334155 !important; }\n"
        "  .feature-card-title { color: #F8FAFC !important; }\n"
        "  .feature-card-desc { color: #CBD5E0 !important; }\n"
        "  div[data-testid='stRadio'] label { background: rgba(30, 41, 59, 0.9) !important; color: #F1F5F9 !important; border-color: #475569 !important; }\n"
        "  p, span, label, h1, h2, h3, h4, h5, h6 { color: #F1F5F9 !important; }\n"
        "  div[data-testid='stAlert'] { background-color: #1E293B !important; color: #F1F5F9 !important; border-color: #475569 !important; }\n"
        "  div[data-testid='stAlert'] p { color: #F1F5F9 !important; }\n"
        "  div[data-testid='stAlert'] strong { color: #FFFFFF !important; }\n"
        "}\n"

        ".main-title { font-family: 'Outfit', sans-serif; text-align: center; background: linear-gradient(135deg, #FF6B6B, #FF8E53, #4ECDC4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2rem; font-weight: 900; margin-bottom: 0px; padding-bottom: 0px; letter-spacing: -0.5px; }\n"
        ".sub-text { font-family: 'Poppins', sans-serif; text-align: center; font-size: 0.9rem; color: #4A5568; font-weight: 500; line-height: 1.3; margin-bottom: 6px; }\n"
        ".highlight-text { color: #FF6B6B; font-weight: 700; }\n"
        
        "div[data-testid='stRadio'] > div { justify-content: center; gap: 8px; border: none !important; margin-bottom: 2px; margin-top: 2px; }\n"
        "div[data-testid='stRadio'] label { background: rgba(240, 244, 248, 0.85); border: 1px solid #CBD5E0; border-radius: 10px; padding: 3px 12px; font-family: 'Outfit', sans-serif; font-weight: 700; transition: all 0.2s ease-in-out; color: #2D3748; }\n"
        "div[data-testid='stRadio'] label:hover { border-color: #FF6B6B; background: #FFFFFF; }\n"
        
        ".feature-card { background: #F8FAFC; border-radius: 10px; padding: 10px; border-left: 4px solid #4ECDC4; height: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.04); }\n"
        ".feature-card-title { font-family: 'Outfit', sans-serif; font-weight: 800; color: #2D3748; font-size: 0.95rem; margin-bottom: 2px; }\n"
        ".feature-card-desc { color: #4A5568; font-size: 0.82rem; line-height: 1.35; }\n"
        
        ".result-card { border-radius: 16px; padding: 14px; text-align: center; color: white !important; font-family: 'Outfit', sans-serif; font-weight: 800; margin-bottom: 10px; box-shadow: 0 8px 20px rgba(0,0,0,0.15); }\n"
        ".result-card p { color: white !important; }\n"
        ".result-affected { background: linear-gradient(135deg, #FF6B6B, #FF8E53); }\n"
        ".result-not-affected { background: linear-gradient(135deg, #48BB78, #38A169); }\n"
        ".card-title { font-size: 1.25rem; margin: 0; letter-spacing: 0.5px; color: #FFFFFF !important; }\n"
        
        "div[data-testid='stFileUploader'] { border: 2px dashed #4ECDC4; border-radius: 14px; background: rgba(247, 250, 252, 0.4); padding: 8px; transition: all 0.3s ease; }\n"
        "div[data-testid='stFileUploader']:hover { border-color: #FF6B6B; transform: translateY(-1px); }\n"
        
        ".sample-img-container { width: 100%; height: 85px; overflow: hidden; border-radius: 8px; display: flex; align-items: center; justify-content: center; background: #000000; margin-bottom: 2px; }\n"
        ".sample-img-container img { width: 100%; height: 100%; object-fit: cover; }\n"

        ".stButton>button { background: linear-gradient(135deg, #FF6B6B, #FF8E53); color: white !important; font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 0.72rem; border-radius: 8px; border: none; padding: 4px 6px; width: 100%; min-height: 38px; line-height: 1.1; transition: all 0.3s ease; box-shadow: 0 4px 14px rgba(255, 107, 107, 0.35); }\n"
        ".stButton>button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(255, 107, 107, 0.5); }\n"
        
        "[data-testid='stMetricValue'] { font-family: 'Outfit', sans-serif; font-size: 1.6rem !important; color: #3182CE !important; font-weight: 800; }\n"
        "hr { margin: 6px 0 !important; border-color: #E2E8F0 !important; }\n"
        "ul { list-style-type: none !important; padding-left: 0 !important; }\n"
        "li { padding: 1px 0; }\n"
        "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)


def render_css_flowchart():
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Poppins', sans-serif; background: transparent; padding: 0px; overflow: hidden; }
        
        :root {
          --bg-section: #F8FAFC;
          --border-color: #E2E8F0;
          --title-color: #4A5568;
          --node-gray-bg: #FFFFFF;
          --node-gray-border: #CBD5E0;
          --node-gray-text: #2D3748;
          --arrow-color: #CBD5E0;
        }

        @media (prefers-color-scheme: dark) {
          :root {
            --bg-section: #1E293B;
            --border-color: #334155;
            --title-color: #CBD5E0;
            --node-gray-bg: #0F172A;
            --node-gray-border: #475569;
            --node-gray-text: #F1F5F9;
            --arrow-color: #64748B;
          }
        }

        .flow-wrapper { display: flex; flex-direction: column; gap: 6px; width: 100%; min-width: 280px; }
        .flow-section { background: var(--bg-section); border: 1px solid var(--border-color); border-radius: 10px; padding: 6px 8px; }
        .section-title { font-size: 0.75rem; font-weight: 800; color: var(--title-color); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
        .stage-box { display: flex; gap: 8px; align-items: center; }
        .step-grid { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; flex: 1; }
        .node { padding: 5px 6px; border-radius: 6px; font-size: 0.73rem; font-weight: 600; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 2px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); flex: 1 1 auto; min-width: 75px; text-align: center; }
        .node-gray { background: var(--node-gray-bg); border: 1.5px solid var(--node-gray-border); color: var(--node-gray-text); }
        .node-blue { background: #EBF8FF; border: 1.5px solid #90CDF4; color: #2B6CB0; }
        .node-orange { background: #FFFAF0; border: 1.5px solid #FBD38D; color: #C05621; }
        .node-green { background: #F0FFF4; border: 1.5px solid #9AE6B4; color: #276749; }
        .arrow { color: var(--arrow-color); font-weight: bold; font-size: 0.75rem; }
        .down-arrow { text-align: center; font-size: 0.8rem; color: var(--arrow-color); margin: -3px 0; }
      </style>
    </head>
    <body>
      <div id="content-body" class="flow-wrapper">
        <div class="flow-section">
          <div class="section-title">1️⃣ Input & Image Preprocessing</div>
          <div class="stage-box">
            <div class="step-grid">
              <div class="node node-gray">📥 Gallery</div>
              <div class="arrow">➔</div>
              <div class="node node-gray">📸 RGB</div>
              <div class="arrow">➔</div>
              <div class="node node-gray">📏 224x224</div>
              <div class="arrow">➔</div>
              <div class="node node-gray">⚖️ Normalize (1/255)</div>
            </div>
          </div>
        </div>
        <div class="down-arrow">⬇️</div>
        <div class="flow-section">
          <div class="section-title">2️⃣ Sequential CNN Feature Extraction</div>
          <div class="stage-box">
            <div class="step-grid">
              <div class="node node-blue">⚡ Conv (32)</div>
              <div class="arrow">➔</div>
              <div class="node node-blue">⚡ Conv (64)</div>
              <div class="arrow">➔</div>
              <div class="node node-blue">⚡ Conv (128)</div>
              <div class="arrow">➔</div>
              <div class="node node-blue">📉 Dense Head</div>
            </div>
          </div>
        </div>
        <div class="down-arrow">⬇️</div>
        <div class="flow-section">
          <div class="section-title">3️⃣ Categorization & Prediction Output</div>
          <div class="stage-box">
            <div class="step-grid" style="width: 100%;">
              <div class="node node-orange">❓ Sigmoid (0-1)</div>
              <div class="arrow">➔</div>
              <div class="node node-green"><span>⚠️ Affected (YES)</span></div>
              <div class="node node-green"><span>✅ Not Affected (NO)</span></div>
            </div>
          </div>
        </div>
      </div>

      <script>
        function sendHeight() {
          const bodyHeight = document.getElementById('content-body').scrollHeight + 10;
          window.parent.postMessage({ type: 'streamlit:setFrameHeight', height: bodyHeight }, '*');
        }
        window.addEventListener('load', sendHeight);
        window.addEventListener('resize', sendHeight);
        setTimeout(sendHeight, 100);
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=340, scrolling=False)


inject_custom_styles(PERMANENT_BG_GIF)


# ---------------------------------------------------------
# Sidebar Diagnostic Controls (Fixes Label Inversion)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛠️ Model Diagnostic Settings")
    invert_outputs = st.checkbox("🔄 Invert Output Labels (Flip Affected/Not Affected)", value=False, help="Enable this toggle if your model outputs are inverted due to alphabetical folder indexing.")


# ---------------------------------------------------------
# AI Model & Inline Training/Testing Engine
# ---------------------------------------------------------
@st.cache_resource
def get_trained_model():
    model_path = 'PD_CNN_Model.h5'
    
    if os.path.exists(model_path):
        st.toast("Loading pre-trained model from disk...", icon="🎯")
        return tf.keras.models.load_model(model_path)

    st.warning("Model file not found. Downloading dataset and training CNN model inline... Please wait.")

    os.makedirs("dataset", exist_ok=True)
    gdrive_url = "https://drive.google.com/uc?id=152qN11WKtE-2LstEXOMOLaoTw0o3HTaL"
    local_zip_filename = "ParkinsonDisease.zip"

    if os.path.exists(local_zip_filename):
        os.remove(local_zip_filename)

    if not os.path.exists(local_zip_filename):
        with st.spinner("Downloading dataset from Google Drive..."):
            gdown.download(gdrive_url, local_zip_filename, quiet=False)

    extract_path = 'dataset'
    if os.path.exists(local_zip_filename):
        with st.spinner("Extracting dataset files..."):
            with zipfile.ZipFile(local_zip_filename, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

    base_data_path = os.path.join(extract_path, 'ParkinsonDisease', 'ParkinsonDisease')
    train_dir = os.path.join(base_data_path, 'TRAIN')

    if not os.path.exists(train_dir):
        st.error(f"Could not locate the TRAIN directory at path: {train_dir}")
        return None

    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=15,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), activation='relu'),
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

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    with st.spinner("Training CNN model dynamically across epochs..."):
        model.fit(
            train_generator,
            epochs=5,
            validation_data=val_generator
        )

    model.save(model_path)
    st.success("Model successfully trained and saved locally!")
    return model

model = get_trained_model()

def classify_image(image, invert_flag):
    img_resized = image.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = float(model.predict(img_array)[0][0])
    
    # Apply toggle inversion if enabled in the sidebar
    is_affected = prediction >= 0.5
    if invert_flag:
        is_affected = not is_affected

    if is_affected:
        pred_status = "Affected"
        top1_label = "YES"
        top1_score = prediction * 100 if not invert_flag else (1 - prediction) * 100
    else:
        pred_status = "Not Affected"
        top1_label = "NO"
        top1_score = (1 - prediction) * 100 if not invert_flag else prediction * 100

    details = [
        ("Affected (YES)", prediction * 100),
        ("Not Affected (NO)", (1 - prediction) * 100)
    ]

    return pred_status, top1_score, top1_label, details, prediction


# ---------------------------------------------------------
# Global Navigation Header & State Management
# ---------------------------------------------------------
st.markdown("<h1 class='main-title'>🧠 Parkinson's Disease Detection 🧠</h1>", unsafe_allow_html=True)

if 'nav' not in st.session_state:
    st.session_state.nav = '🏠 Home'
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'source_mode' not in st.session_state:
    st.session_state.source_mode = 'Upload Image'
if 'selected_sample_url' not in st.session_state:
    st.session_state.selected_sample_url = None

def switch_to_prediction():
    st.session_state.nav = '🔮 Prediction'

nav_choice = st.radio(
    "",
    ["🏠 Home", "🔮 Prediction", "ℹ️ About"],
    horizontal=True,
    key='nav',
    label_visibility="collapsed"
)

# =========================================================
# PAGE 1: HOME PAGE
# =========================================================
if nav_choice == "🏠 Home":
    st.markdown("### 🧬 Automated Deep Learning Parkinson's Recognition Engine")
    st.markdown(
        "<p style='font-size: 0.9rem; line-height: 1.4;'>"
        "Welcome! This application utilizes a custom Sequential Convolutional Neural Network (CNN) "
        "built to evaluate biomarkers from visual cohorts. Leveraging intensity range normalizations "
        "and real-time data augmentations, the framework achieves a definitive clinical test accuracy of "
        "<b>97.0%</b>, classifying scans into clinical indications: ⚠️ <b>Affected</b> or ✅ <b>Not Affected</b>."
        "</p>", 
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Classification System Architecture & Workflow")
    st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="feature-card" style="border-left-color: #FF6B6B;"><div class="feature-card-title">1. Preprocessing</div><div class="feature-card-desc">Raw image frames are normalized (1/255), color-space corrected (RGB), and resized to 224x224.</div></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="feature-card" style="border-left-color: #4ECDC4;"><div class="feature-card-title">2. Sequential CNN</div><div class="feature-card-desc">Three cascaded Conv2D and MaxPooling2D layers extract robust features down to a dense decision head.</div></div>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="feature-card" style="border-left-color: #4299E1;"><div class="feature-card-title">3. Logic & Classification</div><div class="feature-card-desc">Sigmoid output probability maps predictions into clinical status groupings with high sensitivity.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("#### 📊 Visual Workflow Diagram")
    st.markdown('</div>', unsafe_allow_html=True)
    render_css_flowchart()

    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Core Capabilities Highlight")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**⚡ 97.0% Accuracy**\n\nValidated performance achieved within training epochs.")
    with col2:
        st.success("**🔬 Clinical Safety**\n\nOptimized for high recall rates to minimize False Negatives.")
    with col3:
        st.warning("**🛡️ Modern Stack**\n\nDeveloped natively using TensorFlow runtime engine.")

    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.button("🚀 Launch Detection Engine", on_click=switch_to_prediction)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PAGE 2: PREDICTION PAGE
# =========================================================
elif nav_choice == "🔮 Prediction":
    if st.session_state.page == 'upload':
        st.markdown(
            "<p class='sub-text'>"
            "Select a source and choose or upload an image to analyze. "
            "The model will classify whether it indicates <span class='highlight-text'>⚠️ Affected</span> "
            "or <span class='highlight-text'>✅ Not Affected</span>"
            "</p>", 
            unsafe_allow_html=True
        )

        st.markdown("**Source:**")
        st.session_state.source_mode = st.radio(
            "Source",
            ["Upload Image", "Sample Images"],
            horizontal=True,
            key='source_mode_radio',
            label_visibility="collapsed"
        )

        if st.session_state.source_mode == "Upload Image":
            st.markdown("### 📥 Upload Your Image")
            file = st.file_uploader(
                "Choose a sample image file (JPG, JPEG, PNG, WEBP)", 
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed"
            )

            if file is not None:
                st.session_state.uploaded_file = file
                image = Image.open(file).convert("RGB")
                st.image(image, caption="🖼️ Image Ready for Analysis", use_container_width=True)
                def go_to_results(): st.session_state.page = 'results'
                st.button("🚀 Analyze Image and View Results", on_click=go_to_results)

        else:
            st.markdown("### Select a Sample Image:")
            
            sample_images = {
                "DaTscan 1": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_183520.jpg",
                "DaTscan 2": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_185834.jpg",
                "DaTscan 3": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_185906.jpg",
                "DaTscan 4": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_185944.jpg",
                "DaTscan 5": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_183555.jpg",
                "DaTscan 6": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_185816.jpg",
                "DaTscan 7": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_185849.jpg",
                "DaTscan 8": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_185926.jpg"
            }

            cols = st.columns(4)
            sample_keys = list(sample_images.keys())
            
            for i in range(4):
                with cols[i]:
                    st.markdown(f"<div style='text-align: center; font-weight: 700; font-size: 0.72rem; margin-bottom: 2px;'>{sample_keys[i]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='sample-img-container'><img src='{sample_images[sample_keys[i]]}' /></div>", unsafe_allow_html=True)
                    if st.button(f"🚀 Analyze", key=f"btn_{i}"):
                        st.session_state.selected_sample_url = sample_images[sample_keys[i]]

            cols_row2 = st.columns(4)
            for i in range(4, 8):
                with cols_row2[i-4]:
                    st.markdown(f"<div style='text-align: center; font-weight: 700; font-size: 0.72rem; margin-bottom: 2px;'>{sample_keys[i]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='sample-img-container'><img src='{sample_images[sample_keys[i]]}' /></div>", unsafe_allow_html=True)
                    if st.button(f"🚀 Analyze", key=f"btn_{i}"):
                        st.session_state.selected_sample_url = sample_images[sample_keys[i]]

            if st.session_state.selected_sample_url:
                try:
                    response = requests.get(st.session_state.selected_sample_url)
                    st.session_state.uploaded_file = BytesIO(response.content)
                    st.session_state.page = 'results'
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading sample image: {e}")

    elif st.session_state.page == 'results':
        st.markdown("<h2 style='text-align: center; font-family: Outfit, sans-serif;'>📋 Analysis Report</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-text'>Here are the classification findings from our AI model</p>", unsafe_allow_html=True)
        
        if st.session_state.uploaded_file is not None:
            image = Image.open(st.session_state.uploaded_file).convert("RGB")
            col1, col2 = st.columns([1, 1], gap="medium")

            with col1:
                st.markdown("#### 🖼️ Image Preview")
                st.image(image, use_container_width=True)

            with col2:
                with st.spinner("🧠 Scanning visual patterns..."):
                    pred_class, score, raw_label, details_list, raw_prob = classify_image(image, invert_outputs)

                if pred_class == "Affected":
                    st.markdown('<div class="result-card result-affected"><p class="card-title">⚠️ Status: AFFECTED</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-card result-not-affected"><p class="card-title">✅ Status: NOT AFFECTED</p></div>', unsafe_allow_html=True)

                st.metric(label="🎯 Confidence Score", value=f"{score:.2f}%")
                st.progress(min(int(score), 100))

                st.markdown("##### 📈 Class Probability Breakdown:")
                for cat_name, cat_score in details_list:
                    st.write(f"**{cat_name}**: `{cat_score:.1f}%`")
                    st.progress(min(int(cat_score), 100))

                with st.expander("🔬 View Technical Details"):
                    st.write(f"🏷️ **Raw Model Probability (Sigmoid):** `{raw_prob:.4f}`")
                    st.write(f"🏷️ **Primary Category:** `{raw_label}`")
                    st.write(f"🏷️ **Diagnostic Classification:** `{pred_class}`")
                    st.info("💡 If your classification labels appear inverted, use the **Model Diagnostic Settings** toggle in the sidebar to flip them.")

            def go_to_upload():
                st.session_state.page = 'upload'
                st.session_state.uploaded_file = None
                st.session_state.selected_sample_url = None
            st.button("🔄 Back to Selection", on_click=go_to_upload)
        else:
            st.warning("No image found!")
            def go_to_upload():
                st.session_state.page = 'upload'
            st.button("⬅️ Back to Upload Page", on_click=go_to_upload)


# =========================================================
# PAGE 3: ABOUT PAGE
# =========================================================
elif nav_choice == "ℹ️ About":
    st.markdown("### ℹ️ About the Model & Technology")
    st.markdown(
        """
        #### 🤖 Model Architecture: Sequential CNN
        This system leverages a custom **Sequential Convolutional Neural Network (CNN)** featuring three cascaded 2D Convolution and Max-Pooling layers followed by a high-density decision head. 
        * **Input Shape:** `(224, 224, 3)`
        * **Layers:** Conv2D (32) $\rightarrow$ MaxPool $\rightarrow$ Conv2D (64) $\rightarrow$ MaxPool $\rightarrow$ Conv2D (128) $\rightarrow$ MaxPool $\rightarrow$ Flatten $\rightarrow$ Dense (128) $\rightarrow$ Dropout (0.5) $\rightarrow$ Dense (1, Sigmoid).

        #### 📚 Performance & Metrics
        * **Validation & Test Accuracy:** `97.0%`
        * **Framework:** TensorFlow runtime engine / Python 3.12
        """
    )
    
    st.markdown("---")
    st.markdown("### 👩‍💻 Developer Details")
    st.markdown(
        """
        * **Name:** Sristi Sarkar
        * **Contact:** 
          * **Email:** `emailsristisarkar@gmail.com`
          * **Phone:** `+91 8240580651`
        """
    )
st.caption("⚠️ **Disclaimer:** **AI Powered & Verified detection model:** Built by Sristi Sarkar for research demonstration. Results depend on image clarity and lighting. Use responsibly for screening purposes.")
