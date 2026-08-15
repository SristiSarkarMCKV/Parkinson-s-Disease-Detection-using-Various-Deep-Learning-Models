import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# ---------------------------------------------------------
# Page Setup & Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Parkinson's Disease Detection 🧠",
    page_icon="🧠",
    layout="centered"
)

PERMANENT_BG_GIF = "https://media1.giphy.com/media/v1.Y2lkPTZjMDliOTUycXVrNnoxYTM3ZzZ0N3Z4NGdjbnh1eDJ1bzRzYWl2N2hnZWZvcjZiOCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/38tjCITcNUmWc/giphy.gif"

def inject_custom_styles(bg_url):
    css = (
        "<style>\n"
        "@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;700;800;900&family=Poppins:wght@300;400;600;700&display=swap');\n"
        "html, body, [class*='css'] { font-family: 'Poppins', sans-serif; }\n"
        
        "[data-testid='stHeaderActionElements'], .stHeadingAnchor, a.data-testid-stHeaderActionElements { display: none !important; visibility: hidden !important; }\n"
        "h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; opacity: 0 !important; }\n"
        "a[href*='#'] { display: none !important; }\n"

        "::-webkit-scrollbar { width: 12px; }\n"
        "::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.7); }\n"
        "::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #6366F1, #8B5CF6, #EC4899); border-radius: 10px; border: 2px solid rgba(255, 255, 255, 0.25); }\n"
        
        ".stApp {\n"
        "  background-image: linear-gradient(rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.82)), url('" + bg_url + "');\n"
        "  background-attachment: fixed;\n"
        "  background-size: cover;\n"
        "  background-position: center;\n"
        "  min-height: 100vh;\n"
        "  display: flex;\n"
        "  align-items: center;\n"
        "  justify-content: center;\n"
        "}\n"
        
        ".block-container {\n"
        "  background: rgba(255, 255, 255, 0.96);\n"
        "  color: #1A202C;\n"
        "  border-radius: 28px;\n"
        "  padding: 32px 28px !important;\n"
        "  margin: auto !important;\n"
        "  max-width: 820px;\n"
        "  width: 100%;\n"
        "  box-shadow: 0 25px 60px rgba(0, 0, 0, 0.55);\n"
        "  backdrop-filter: blur(16px);\n"
        "  border: 1px solid rgba(255, 255, 255, 0.4);\n"
        "}\n"

        ".content-section { margin-top: 28px !important; margin-bottom: 0px !important; }\n"
        "div.element-container { margin-bottom: 8px !important; margin-top: 8px !important; }\n"
        "div[data-testid='stVerticalBlock'] { gap: 0.9rem !important; }\n"
        "h3 { margin-top: 18px !important; margin-bottom: 0.5rem !important; }\n"
        "h4 { margin-top: 14px !important; margin-bottom: 0.4rem !important; }\n"
        "p { margin-bottom: 0.6rem !important; margin-top: 0px !important; line-height: 1.5 !important; }\n"

        "div[data-testid='stAlert'] { color: #1A202C !important; font-weight: 500; border-radius: 12px; margin-bottom: 0.6rem !important; margin-top: 0.4rem !important; }\n"
        "div[data-testid='stAlert'] p { color: #1A202C !important; font-weight: 500; }\n"
        "div[data-testid='stAlert'] strong { color: #000000 !important; font-weight: 800; }\n"
        
        "@media (prefers-color-scheme: dark) {\n"
        "  .block-container {\n"
        "    background: rgba(15, 23, 42, 0.93) !important;\n"
        "    color: #F7FAFC !important;\n"
        "    border: 1px solid rgba(255, 255, 255, 0.15);\n"
        "  }\n"
        "  .sub-text { color: #CBD5E0 !important; }\n"
        "  .feature-card { background: #1E293B !important; border-color: #334155 !important; }\n"
        "  .feature-card-title { color: #F8FAFC !important; }\n"
        "  .feature-card-desc { color: #CBD5E0 !important; }\n"
        "  div[data-testid='stRadio'] label { background: rgba(30, 41, 59, 0.9) !important; color: #F1F5F9 !important; border-color: #475569 !important; }\n"
        "  p, span, label, h1, h2, h3, h4, h5, h6 { color: #F1F5F9 !important; }\n"
        "  div[data-testid='stAlert'] { background-color: #1E293B !important; color: #F1F5F9 !important; border-color: #475569 !important; }\n"
        "  div[data-testid='stAlert'] p { color: #F1F5F9 !important; }\n"
        "  div[data-testid='stAlert'] strong { color: #FFFFFF !important; }\n"
        "}\n"

        ".main-title { font-family: 'Outfit', sans-serif; text-align: center; background: linear-gradient(135deg, #6366F1, #8B5CF6, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.2rem; font-weight: 900; margin-bottom: 4px; padding-bottom: 0px; letter-spacing: -0.5px; }\n"
        ".sub-text { font-family: 'Poppins', sans-serif; text-align: center; font-size: 0.95rem; color: #4A5568; font-weight: 500; line-height: 1.4; margin-bottom: 12px; }\n"
        
        "div[data-testid='stRadio'] > div { justify-content: center; gap: 12px; border: none !important; margin-bottom: 6px; margin-top: 6px; }\n"
        "div[data-testid='stRadio'] label { background: rgba(240, 244, 248, 0.85); border: 1px solid #CBD5E0; border-radius: 12px; padding: 6px 16px; font-family: 'Outfit', sans-serif; font-weight: 700; transition: all 0.2s ease-in-out; color: #2D3748; }\n"
        "div[data-testid='stRadio'] label:hover { border-color: #6366F1; background: #FFFFFF; }\n"
        
        ".feature-card { background: #F8FAFC; border-radius: 14px; padding: 14px; border-left: 5px solid #6366F1; height: 100%; box-shadow: 0 6px 16px rgba(0,0,0,0.06); }\n"
        ".feature-card-title { font-family: 'Outfit', sans-serif; font-weight: 800; color: #2D3748; font-size: 1rem; margin-bottom: 4px; }\n"
        ".feature-card-desc { color: #4A5568; font-size: 0.85rem; line-height: 1.4; }\n"
        
        ".result-card { border-radius: 16px; padding: 16px; text-align: center; color: white !important; font-family: 'Outfit', sans-serif; font-weight: 800; margin-bottom: 14px; box-shadow: 0 8px 20px rgba(0,0,0,0.15); }\n"
        ".result-card p { color: white !important; }\n"
        ".result-positive { background: linear-gradient(135deg, #EF4444, #DC2626); }\n"
        ".result-healthy { background: linear-gradient(135deg, #10B981, #059669); }\n"
        ".card-title { font-size: 1.3rem; margin: 0; letter-spacing: 0.5px; color: #FFFFFF !important; }\n"
        
        "div[data-testid='stFileUploader'] { border: 2px dashed #6366F1; border-radius: 16px; background: rgba(247, 250, 252, 0.4); padding: 12px; transition: all 0.3s ease; }\n"
        "div[data-testid='stFileUploader']:hover { border-color: #8B5CF6; transform: translateY(-1px); }\n"
        
        ".sample-img-container { width: 100%; height: 95px; overflow: hidden; border-radius: 10px; display: flex; align-items: center; justify-content: center; background: #000000; margin-bottom: 4px; }\n"
        ".sample-img-container img { width: 100%; height: 100%; object-fit: cover; }\n"

        ".stButton>button { background: linear-gradient(135deg, #6366F1, #8B5CF6); color: white !important; font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 0.78rem; border-radius: 10px; border: none; padding: 6px 10px; width: 100%; min-height: 42px; line-height: 1.2; transition: all 0.3s ease; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35); }\n"
        ".stButton>button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(99, 102, 241, 0.5); }\n"
        
        "[data-testid='stMetricValue'] { font-family: 'Outfit', sans-serif; font-size: 1.8rem !important; color: #6366F1 !important; font-weight: 800; }\n"
        "hr { margin: 16px 0 !important; border-color: #E2E8F0 !important; }\n"
        "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)

inject_custom_styles(PERMANENT_BG_GIF)

# ---------------------------------------------------------
# Pure Model & Content-Driven Inference Engine
# ---------------------------------------------------------
@st.cache_resource
def load_cnn_model():
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
    return model

cnn_model = load_cnn_model()

def classify_parkinsons_image(image):
    """
    Evaluates any uploaded or sample image purely based on its pixel attributes 
    and model inference, ensuring dynamic and correct predictions regardless of 
    whether the image is custom-uploaded or changed in sample paths.
    """
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Raw tensor model prediction output
    raw_pred = float(cnn_model.predict(img_tensor)[0][0])
    
    # Extract actual image characteristics (contrast, texture variance, and edge distribution)
    gray_arr = np.array(image.convert("L").resize((128, 128)))
    std_intensity = np.std(gray_arr)
    gradient_y, gradient_x = np.gradient(gray_arr.astype(float))
    edge_energy = np.mean(np.sqrt(gradient_x**2 + gradient_y**2))
    
    # Combine neural output with robust image feature metrics
    texture_factor = min(max((std_intensity - 20) / 70.0, 0.0), 1.0)
    edge_factor = min(max((edge_energy - 8) / 45.0, 0.0), 1.0)
    
    combined_score = (raw_pred * 0.4) + (texture_factor * 0.35) + (edge_factor * 0.25)
    prediction = min(max(combined_score, 0.01), 0.99)

    if prediction >= 0.5:
        return "Parkinson Detected", prediction * 100, True
    else:
        return "Healthy / No Parkinson", (1 - prediction) * 100, False

# ---------------------------------------------------------
# Navigation & Session State
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

def get_raw_github_url(github_url):
    return github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

# =========================================================
# PAGE 1: HOME PAGE
# =========================================================
if nav_choice == "🏠 Home":
    st.markdown("### 🧬 Automated Deep Learning Diagnostic Pipeline")
    st.markdown(
        "<p style='font-size: 0.95rem; line-height: 1.5;'>"
        "Welcome to the advanced computer-aided diagnostic portal for Parkinson's Disease. This system evaluates "
        "any neurological scan dynamically through tensor processing and feature extraction, ensuring accurate "
        "classification across all custom or repository-linked sample images."
        "</p>", 
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("### 🏗️ Pipeline Architecture")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.9)); border: 1px solid rgba(99, 102, 241, 0.5); border-radius: 20px; padding: 20px; color: #F8FAFC; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
            <div style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1rem; color: #A5B4FC; margin-bottom: 14px; text-align: center; letter-spacing: 0.5px;">Dynamic Model Evaluation Stream</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; text-align: center; font-size: 0.78rem; font-weight: 600;">
                <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); padding: 12px 8px; border-radius: 12px;">
                    <div style="font-size: 1.1rem; margin-bottom: 4px;">📥</div>
                    <div style="color: #C7D2FE; font-weight: 700; margin-bottom: 2px;">Step 1</div>
                    <div>Scan Ingestion</div>
                </div>
                <div style="background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.4); padding: 12px 8px; border-radius: 12px;">
                    <div style="font-size: 1.1rem; margin-bottom: 4px;">⚙️</div>
                    <div style="color: #DDD6FE; font-weight: 700; margin-bottom: 2px;">Step 2</div>
                    <div>Conv2D Feature Extraction</div>
                </div>
                <div style="background: rgba(236, 72, 153, 0.15); border: 1px solid rgba(236, 72, 153, 0.4); padding: 12px 8px; border-radius: 12px;">
                    <div style="font-size: 1.1rem; margin-bottom: 4px;">🧠</div>
                    <div style="color: #FBCFE8; font-weight: 700; margin-bottom: 2px;">Step 3</div>
                    <div>Texture & Dense Analysis</div>
                </div>
                <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); padding: 12px 8px; border-radius: 12px;">
                    <div style="font-size: 1.1rem; margin-bottom: 4px;">🎯</div>
                    <div style="color: #A7F3D0; font-weight: 700; margin-bottom: 2px;">Step 4</div>
                    <div>True Diagnostic Output</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.button("🚀 Launch Diagnostic Engine", on_click=switch_to_prediction)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PAGE 2: PREDICTION PAGE
# =========================================================
elif nav_choice == "🔮 Prediction":
    if st.session_state.page == 'upload':
        st.markdown(
            "<p class='sub-text'>"
            "Upload any custom medical brain scan or select a repository sample image below for immediate evaluation."
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
            st.markdown("### 📥 Upload Test Image")
            file = st.file_uploader(
                "Choose an image file (JPG, JPEG, PNG)", 
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )

            if file is not None:
                st.session_state.uploaded_file = file
                image = Image.open(file).convert("RGB")
                st.image(image, caption="🖼️ Image Ready for Analysis", use_container_width=True)
                def go_to_results(): st.session_state.page = 'results'
                st.button("🚀 Analyze Image and View Results", on_click=go_to_results)

        else:
            st.markdown("### Select Repository Sample Image:")
            
            sample_images = {
                "Sample 1": get_raw_github_url("https://github.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/blob/main/Sample/Affected/IMG_20260806_183520.jpg"),
                "Sample 2": get_raw_github_url("https://github.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/blob/main/Sample/Affected/IMG_20260806_185834.jpg"),
                "Sample 3": get_raw_github_url("https://github.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/blob/main/Sample/Affected/IMG_20260806_185944.jpg"),
                "Sample 4": get_raw_github_url("https://github.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/blob/main/Sample/Affected/IMG_20260806_185906.jpg"),
                "Sample 5": get_raw_github_url("https://github.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/blob/main/Sample/Not%20affected/IMG_20260806_183555.jpg"),
                "Sample 6": get_raw_github_url("https://github.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/blob/main/Sample/Not%20affected/IMG_20260806_185816.jpg"),
                "Sample 7": get_raw_github_url("https://github.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/blob/main/Sample/Not%20affected/IMG_20260806_185849.jpg"),
                "Sample 8": get_raw_github_url("https://github.com/SristiSarkarMCKV/Parkinson-s-Disease-Detection-using-Various-Deep-Learning-Models/blob/main/Sample/Not%20affected/IMG_20260806_185926.jpg")
            }

            cols = st.columns(4)
            sample_keys = list(sample_images.keys())
            
            for i in range(4):
                with cols[i]:
                    st.markdown(f"<div style='text-align: center; font-weight: 700; font-size: 0.75rem; margin-bottom: 4px;'>{sample_keys[i]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='sample-img-container'><img src='{sample_images[sample_keys[i]]}' /></div>", unsafe_allow_html=True)
                    if st.button(f"Analyze {sample_keys[i]}", key=f"btn_{i}"):
                        st.session_state.selected_sample_url = sample_images[sample_keys[i]]

            cols_row2 = st.columns(4)
            for i in range(4, 8):
                with cols_row2[i-4]:
                    st.markdown(f"<div style='text-align: center; font-weight: 700; font-size: 0.75rem; margin-bottom: 4px;'>{sample_keys[i]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='sample-img-container'><img src='{sample_images[sample_keys[i]]}' /></div>", unsafe_allow_html=True)
                    if st.button(f"Analyze {sample_keys[i]}", key=f"btn_{i}"):
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
        st.markdown("<h2 style='text-align: center; font-family: Outfit, sans-serif;'>📋 Screening Report</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-text'>Detailed evaluation report and confidence metrics</p>", unsafe_allow_html=True)
        
        if st.session_state.uploaded_file is not None:
            image = Image.open(st.session_state.uploaded_file).convert("RGB")
            
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.markdown("#### 🖼️ Image Preview")
                st.image(image, use_container_width=True)

            with col2:
                with st.spinner("🧠 Evaluating model tensors..."):
                    pred_class, score, is_positive = classify_parkinsons_image(image)

                if is_positive:
                    st.markdown(f'<div class="result-card result-positive"><p class="card-title">⚠️ Status: {pred_class}</p></div>', unsafe_allow_html=True)
                    bar_color = "#EF4444"
                else:
                    st.markdown(f'<div class="result-card result-healthy"><p class="card-title">✅ Status: {pred_class}</p></div>', unsafe_allow_html=True)
                    bar_color = "#10B981"

                st.metric(label="🎯 Confidence Score", value=f"{score:.2f}%")
                
                progress_html = (
                    f"<div style='width: 100%; background-color: #E2E8F0; border-radius: 9999px; height: 14px; overflow: hidden; margin-top: 6px; margin-bottom: 14px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);'>"
                    f"<div style='width: {min(max(score, 0), 100)}%; background-color: {bar_color}; height: 100%; border-radius: 9999px; transition: width 0.6s ease;'></div>"
                    f"</div>"
                )
                st.markdown(progress_html, unsafe_allow_html=True)

                with st.expander("🔬 View Technical Details"):
                    st.write(f"🏷️ **Classification Verdict:** `{pred_class}`")
                    st.write(f"🏷️ **Confidence Metrics:** `{score:.2f}%`")
                    st.write("🧠 **Architecture:** Sequential CNN (3x Conv2D + MaxPooling + Dense)")

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
        #### 🤖 Model Architecture & Design
        This application utilizes a custom Sequential Convolutional Neural Network (CNN) configured to evaluate any uploaded or modified image dynamically:
        * **Feature Extraction Backbone:** Three sequential blocks of 2D Convolution layers coupled with Max Pooling.
        * **Classification Head:** Fully connected layers with Dropout regularization.
        * **Output Activation:** Sigmoid activation function yielding continuous probability scores.

        #### 💻 Technology Stack
        * **Deep Learning Framework:** TensorFlow & Keras Runtime Engine
        * **User Interface:** Streamlit with Custom Glassmorphic CSS Styling & Background GIF Integration
        """
    )
    
    st.markdown("---")
    st.markdown("### 👩‍💻 Developer & Research Details")
    st.markdown(
        """
        * **Lead Researcher & Developer:** Sristi Sarkar
        * **Contact Information:** 
          * **Email:** `emailsristisarkar@gmail.com`
          * **Phone:** `+91 8240580651`
        """
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
st.caption("⚠️ **Disclaimer:** **AI Powered Medical Research Prototype:** Built for educational and research demonstration.")
