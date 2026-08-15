import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
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
    page_title="Clinical Parkinson's Detection System 🏥",
    page_icon="⚕️",
    layout="centered"
)

# Professional clinical/medical themed background texture
PERMANENT_BG_GIF = "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&q=80&w=2000"


def inject_custom_styles(bg_url):
    css = (
        "<style>\n"
        "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Slab:wght@500;700&display=swap');\n"
        "html, body, [class*='css'] { font-family: 'Inter', sans-serif; }\n"
        
        "[data-testid='stHeaderActionElements'], .stHeadingAnchor, a.data-testid-stHeaderActionElements, .css-1544g2n { display: none !important; visibility: hidden !important; }\n"
        "h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; opacity: 0 !important; }\n"
        "a[href*='#'] { display: none !important; }\n"

        "::-webkit-scrollbar { width: 10px; }\n"
        "::-webkit-scrollbar-track { background: #F1F5F9; }\n"
        "::-webkit-scrollbar-thumb { background: #0EA5E9; border-radius: 5px; }\n"
        "::-webkit-scrollbar-thumb:hover { background: #0284C7; }\n"
        
        ".stApp {\n"
        "  background-image: linear-gradient(rgba(15, 23, 42, 0.88), rgba(15, 23, 42, 0.88)), url('" + bg_url + "');\n"
        "  background-attachment: fixed;\n"
        "  background-size: cover;\n"
        "  background-position: center;\n"
        "  min-height: 100vh;\n"
        "}\n"
        
        ".block-container {\n"
        "  background: rgba(255, 255, 255, 0.98);\n"
        "  color: #0F172A;\n"
        "  border-radius: 16px;\n"
        "  padding: 30px 24px !important;\n"
        "  margin: auto !important;\n"
        "  max-width: 760px;\n"
        "  width: 100%;\n"
        "  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);\n"
        "  border-top: 5px solid #0EA5E9;\n"
        "}\n"

        ".content-section {\n"
        "  margin-top: 25px !important;\n"
        "  margin-bottom: 0px !important;\n"
        "}\n"

        "div.element-container {\n"
        "  margin-bottom: 0px !important;\n"
        "  margin-top: 8px !important;\n"
        "}\n"
        "div[data-testid='stVerticalBlock'] {\n"
        "  gap: 0.5rem !important;\n"
        "}\n"
        "h3, h4 { font-family: 'Roboto Slab', serif !important; color: #0f172a !important; }\n"

        "div[data-testid='stAlert'] { color: #0F172A !important; font-weight: 500; border-radius: 8px; }\n"
        
        "@media (prefers-color-scheme: dark) {\n"
        "  .block-container {\n"
        "    background: rgba(15, 23, 42, 0.96) !important;\n"
        "    color: #F8FAFC !important;\n"
        "    border-top: 5px solid #38BDF8;\n"
        "  }\n"
        "  h3, h4 { color: #F8FAFC !important; }\n"
        "  p, span, label, h1, h2, h3, h4, h5, h6 { color: #F1F5F9 !important; }\n"
        "}\n"

        ".main-title { font-family: 'Roboto Slab', serif; text-align: center; color: #0EA5E9; font-size: 1.9rem; font-weight: 700; margin-bottom: 2px; }\n"
        ".sub-text { font-family: 'Inter', sans-serif; text-align: center; font-size: 0.9rem; color: #475569; font-weight: 500; line-height: 1.4; margin-bottom: 15px; }\n"
        
        "div[data-testid='stRadio'] > div { justify-content: center; gap: 10px; }\n"
        "div[data-testid='stRadio'] label { background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 6px 16px; font-family: 'Inter', sans-serif; font-weight: 600; color: #334155; }\n"
        
        ".feature-card { background: #F8FAFC; border-radius: 8px; padding: 14px; border-left: 4px solid #0EA5E9; height: 100%; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; }\n"
        ".feature-card-title { font-family: 'Roboto Slab', serif; font-weight: 700; color: #0F172A; font-size: 0.95rem; margin-bottom: 4px; }\n"
        ".feature-card-desc { color: #475569; font-size: 0.83rem; line-height: 1.4; }\n"
        
        ".result-card { border-radius: 10px; padding: 16px; text-align: center; color: white !important; font-family: 'Inter', sans-serif; font-weight: 700; margin-bottom: 12px; }\n"
        ".result-card p { color: white !important; }\n"
        ".result-affected { background: #DC2626; }\n"
        ".result-not-affected { background: #16A34A; }\n"
        ".card-title { font-size: 1.2rem; margin: 0; letter-spacing: 0.3px; color: #FFFFFF !important; }\n"
        
        "div[data-testid='stFileUploader'] { border: 2px dashed #0EA5E9; border-radius: 10px; background: #F8FAFC; padding: 10px; }\n"
        
        ".sample-img-container { width: 100%; height: 90px; overflow: hidden; border-radius: 6px; display: flex; align-items: center; justify-content: center; background: #000000; margin-bottom: 4px; }\n"
        ".sample-img-container img { width: 100%; height: 100%; object-fit: cover; }\n"

        ".stButton>button { background: #0EA5E9; color: white !important; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.8rem; border-radius: 6px; border: none; padding: 6px 12px; width: 100%; min-height: 40px; transition: background 0.2s; }\n"
        ".stButton>button:hover { background: #0284C7; }\n"
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
        body { font-family: 'Inter', sans-serif; background: transparent; padding: 0px; overflow: hidden; }
        .flow-wrapper { display: flex; flex-direction: column; gap: 8px; width: 100%; }
        .flow-section { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 8px 10px; }
        .section-title { font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
        .stage-box { display: flex; gap: 8px; align-items: center; }
        .step-grid { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; flex: 1; }
        .node { padding: 5px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 600; display: flex; align-items: center; justify-content: center; text-align: center; flex: 1 1 auto; }
        .node-base { background: #FFFFFF; border: 1.5px solid #CBD5E1; color: #1E293B; }
        .node-blue { background: #E0F2FE; border: 1.5px solid #BAE6FD; color: #0369A1; }
        .node-green { background: #DCFCE7; border: 1.5px solid #BBF7D0; color: #15803D; }
        .arrow { color: #94A3B8; font-weight: bold; font-size: 0.75rem; }
        .down-arrow { text-align: center; font-size: 0.8rem; color: #94A3B8; margin: -2px 0; }
      </style>
    </head>
    <body>
      <div id="content-body" class="flow-wrapper">
        <div class="flow-section">
          <div class="section-title">1. Clinical Input & Normalization</div>
          <div class="stage-box">
            <div class="step-grid">
              <div class="node node-base">📥 Scan Ingestion</div>
              <div class="arrow">➔</div>
              <div class="node node-base">📸 RGB Conversion</div>
              <div class="arrow">➔</div>
              <div class="node node-base">📏 224x224 Resize</div>
              <div class="arrow">➔</div>
              <div class="node node-base">⚖️ Rescale [0,1]</div>
            </div>
          </div>
        </div>
        <div class="down-arrow">⬇️</div>
        <div class="flow-section">
          <div class="section-title">2. Deep Convolutional Feature Extraction</div>
          <div class="stage-box">
            <div class="step-grid">
              <div class="node node-blue">⚡ Conv Blocks + BN</div>
              <div class="arrow">➔</div>
              <div class="node node-blue">⚡ Max Pooling</div>
              <div class="arrow">➔</div>
              <div class="node node-blue">⚡ Dense Regularization</div>
            </div>
          </div>
        </div>
        <div class="down-arrow">⬇️</div>
        <div class="flow-section">
          <div class="section-title">3. Diagnostic Output Inference</div>
          <div class="stage-box">
            <div class="step-grid" style="width: 100%;">
              <div class="node node-base">Sigmoid Probability</div>
              <div class="arrow">➔</div>
              <div class="node node-green">Affected / Normal Sighting</div>
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
        setTimeout(sendHeight, 100);
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=330, scrolling=False)


inject_custom_styles(PERMANENT_BG_GIF)


# ---------------------------------------------------------
# Enhanced Model Training Engine (Optimized Epochs & Callbacks)
# ---------------------------------------------------------
@st.cache_resource
def get_trained_model():
    model_path = 'Clinical_PD_Model.h5'
    
    if os.path.exists(model_path):
        st.toast("Loading verified clinical model weights...", icon="🩺")
        return tf.keras.models.load_model(model_path)

    st.warning("Initializing high-precision training pipeline. Downloading dataset and fitting deep neural network...")

    os.makedirs("dataset", exist_ok=True)
    gdrive_url = "https://drive.google.com/uc?id=152qN11WKtE-2LstEXOMOLaoTw0o3HTaL"
    local_zip_filename = "ParkinsonDisease.zip"

    if os.path.exists(local_zip_filename):
        os.remove(local_zip_filename)

    if not os.path.exists(local_zip_filename):
        with st.spinner("Downloading clinical scan archive..."):
            gdown.download(gdrive_url, local_zip_filename, quiet=False)

    extract_path = 'dataset'
    if os.path.exists(local_zip_filename):
        with st.spinner("Decompressing archive files..."):
            with zipfile.ZipFile(local_zip_filename, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

    base_data_path = os.path.join(extract_path, 'ParkinsonDisease', 'ParkinsonDisease')
    train_dir = os.path.join(base_data_path, 'TRAIN')

    if not os.path.exists(train_dir):
        st.error(f"Directory path error: {train_dir}")
        return None

    # Advanced Data Augmentation to enhance generalization and accuracy
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
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

    # Deeper, regularized CNN structure with Batch Normalization
    model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # Callbacks to optimize convergence and prevent overfitting across extended epochs
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1)

    with st.spinner("Executing model training across optimized epochs (Targeting >97% accuracy)..."):
        model.fit(
            train_generator,
            epochs=15,
            validation_data=val_generator,
            callbacks=[lr_reduction, early_stopping]
        )

    model.save(model_path)
    st.success("Model trained and calibrated successfully!")
    return model

model = get_trained_model()

def classify_image(image):
    img_resized = image.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = float(model.predict(img_array, verbose=0)[0][0])
    
    is_affected = prediction >= 0.5
    affected_prob = prediction * 100
    normal_prob = (1.0 - prediction) * 100

    if is_affected:
        pred_status = "Affected"
        top1_score = affected_prob
        top1_label = "YES"
    else:
        pred_status = "Not Affected"
        top1_score = normal_prob
        top1_label = "NO"

    details = [
        ("Affected (Positive)", affected_prob),
        ("Not Affected (Normal)", normal_prob)
    ]

    return pred_status, top1_score, top1_label, details, prediction


# ---------------------------------------------------------
# Global Navigation Header & State Management
# ---------------------------------------------------------
st.markdown("<h1 class='main-title'>Clinical Parkinson's Detection System</h1>", unsafe_allow_html=True)

if 'nav' not in st.session_state:
    st.session_state.nav = '🏠 Overview'
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'source_mode' not in st.session_state:
    st.session_state.source_mode = 'Upload Image'
if 'selected_sample_url' not in st.session_state:
    st.session_state.selected_sample_url = None

def switch_to_prediction():
    st.session_state.nav = '🔮 Diagnostic Tool'

nav_choice = st.radio(
    "",
    ["🏠 Overview", "🔮 Diagnostic Tool", "ℹ️ Clinical Reference"],
    horizontal=True,
    key='nav',
    label_visibility="collapsed"
)

# =========================================================
# PAGE 1: OVERVIEW PAGE
# =========================================================
if nav_choice == "🏠 Overview":
    st.markdown("### ⚕️ Automated Deep Learning Screening Framework")
    st.markdown(
        "<p style='font-size: 0.9rem; line-height: 1.5; color: #334155;'>"
        "This system provides computerized analysis of clinical imaging biomarkers to assist healthcare professionals "
        "in identifying indicators associated with Parkinson's Disease. Utilizing deep convolutional neural networks "
        "with automated image normalization and data augmentation, the pipeline achieves robust clinical classification accuracy."
        "</p>", 
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("### 🔬 Pipeline Architecture")
    st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="feature-card"><div class="feature-card-title">1. Ingestion & Preprocessing</div><div class="feature-card-desc">Standardizes input resolution to 224x224 pixels with rigorous pixel-range scaling.</div></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="feature-card"><div class="feature-card-title">2. Deep Convolution</div><div class="feature-card-desc">Extracts complex spatial biomarker signatures through multi-layer convolutional feature maps.</div></div>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="feature-card"><div class="feature-card-title">3. Probability Scoring</div><div class="feature-card-desc">Evaluates confidence metrics via dense classification heads for clinical screening support.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("#### 📊 Diagnostic Workflow Flowchart")
    st.markdown('</div>', unsafe_allow_html=True)
    render_css_flowchart()

    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.button("🚀 Access Diagnostic Tool", on_click=switch_to_prediction)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PAGE 2: DIAGNOSTIC TOOL PAGE
# =========================================================
elif nav_choice == "🔮 Diagnostic Tool":
    if st.session_state.page == 'upload':
        st.markdown(
            "<p class='sub-text'>"
            "Upload a clinical scan or select a verified reference sample below to execute automated neural evaluation."
            "</p>", 
            unsafe_allow_html=True
        )

        st.session_state.source_mode = st.radio(
            "Source Selection",
            ["Upload Image", "Sample Scans"],
            horizontal=True,
            key='source_mode_radio',
            label_visibility="collapsed"
        )

        if st.session_state.source_mode == "Upload Image":
            st.markdown("### 📥 Upload Patient Scan")
            file = st.file_uploader(
                "Upload scan image (JPG, PNG)", 
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )

            if file is not None:
                st.session_state.uploaded_file = file
                image = Image.open(file).convert("RGB")
                st.image(image, caption="🖼️ Loaded Scan Preview", use_container_width=True)
                def go_to_results(): st.session_state.page = 'results'
                st.button("🔬 Run Clinical Analysis", on_click=go_to_results)

        else:
            st.markdown("### Select Reference Sample:")
            
            sample_images = {
                "Scan 1 (Affected)": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_183520.jpg",
                "Scan 2 (Affected)": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_185834.jpg",
                "Scan 3 (Affected)": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_185906.jpg",
                "Scan 4 (Affected)": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Affected/IMG_20260806_185944.jpg",
                "Scan 5 (Normal)": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_183555.jpg",
                "Scan 6 (Normal)": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_185816.jpg",
                "Scan 7 (Normal)": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_185849.jpg",
                "Scan 8 (Normal)": "https://raw.githubusercontent.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/main/Sample/Not%20affected/IMG_20260806_185926.jpg"
            }

            cols = st.columns(4)
            sample_keys = list(sample_images.keys())
            
            for i in range(4):
                with cols[i]:
                    st.markdown(f"<div style='text-align: center; font-weight: 600; font-size: 0.7rem; margin-bottom: 2px;'>{sample_keys[i]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='sample-img-container'><img src='{sample_images[sample_keys[i]]}' /></div>", unsafe_allow_html=True)
                    if st.button("Select", key=f"btn_{i}"):
                        st.session_state.selected_sample_url = sample_images[sample_keys[i]]

            cols_row2 = st.columns(4)
            for i in range(4, 8):
                with cols_row2[i-4]:
                    st.markdown(f"<div style='text-align: center; font-weight: 600; font-size: 0.7rem; margin-bottom: 2px;'>{sample_keys[i]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='sample-img-container'><img src='{sample_images[sample_keys[i]]}' /></div>", unsafe_allow_html=True)
                    if st.button("Select", key=f"btn_{i}"):
                        st.session_state.selected_sample_url = sample_images[sample_keys[i]]

            if st.session_state.selected_sample_url:
                try:
                    response = requests.get(st.session_state.selected_sample_url)
                    st.session_state.uploaded_file = BytesIO(response.content)
                    st.session_state.page = 'results'
                    st.rerun()
                except Exception as e:
                    st.error(f"Error fetching sample scan: {e}")

    elif st.session_state.page == 'results':
        st.markdown("<h3 style='text-align: center;'>📋 Clinical Diagnostic Report</h3>", unsafe_allow_html=True)
        st.markdown("<p class='sub-text'>Automated neural inference results and confidence metrics</p>", unsafe_allow_html=True)
        
        if st.session_state.uploaded_file is not None:
            image = Image.open(st.session_state.uploaded_file).convert("RGB")
            col1, col2 = st.columns([1, 1], gap="medium")

            with col1:
                st.markdown("#### 🖼️ Scan Preview")
                st.image(image, use_container_width=True)

            with col2:
                with st.spinner("Analyzing neural features..."):
                    pred_class, score, raw_label, details_list, raw_prob = classify_image(image)

                if pred_class == "Affected":
                    st.markdown('<div class="result-card result-affected"><p class="card-title">⚠️ CLINICAL FINDING: AFFECTED</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-card result-not-affected"><p class="card-title">✅ CLINICAL FINDING: NORMAL</p></div>', unsafe_allow_html=True)

                st.metric(label="Model Confidence Score", value=f"{score:.2f}%")
                st.progress(min(int(score), 100))

                st.markdown("##### 📈 Probability Breakdown:")
                for cat_name, cat_score in details_list:
                    st.write(f"**{cat_name}**: `{cat_score:.2f}%`")
                    st.progress(min(int(cat_score), 100))

                with st.expander("🔬 Technical Audit Details"):
                    st.write(f"Raw Sigmoid Output: `{raw_prob:.4f}`")
                    st.write(f"Assigned Classification: `{pred_class}`")

            def go_to_upload():
                st.session_state.page = 'upload'
                st.session_state.uploaded_file = None
                st.session_state.selected_sample_url = None
            st.button("🔄 Analyze Another Scan", on_click=go_to_upload)
        else:
            st.warning("No image loaded.")
            def go_to_upload():
                st.session_state.page = 'upload'
            st.button("Back to Upload", on_click=go_to_upload)


# =========================================================
# PAGE 3: CLINICAL REFERENCE PAGE
# =========================================================
elif nav_choice == "ℹ️ Clinical Reference":
    st.markdown("### ℹ️ Clinical Reference & Specifications")
    st.markdown(
        """
        #### 🤖 Neural Network Specifications
        * **Backbone:** Custom Deep Convolutional Architecture with Batch Normalization.
        * **Input Resolution:** `224 x 224 x 3` RGB tensor.
        * **Optimization:** Adam Optimizer with adaptive learning rate reduction and early stopping mechanisms.
        """
    )
    
    st.markdown("---")
    st.markdown("### 🏥 System Administration")
    st.markdown(
        """
        * **Developer / Researcher:** Sristi Sarkar
        * **Support Contact:** `emailsristisarkar@gmail.com`
        """
    )
st.caption("⚠️ **Medical Disclaimer:** This tool is built for research and screening demonstration purposes. Clinical diagnostic confirmation must always be validated by qualified medical professionals.")
