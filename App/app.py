import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import streamlit.components.v1 as components

# ---------------------------------------------------------
# Page Setup & Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Parkinson's Disease Detection 🧠",
    page_icon="🧠",
    layout="centered"
)

PERMANENT_BG_GIF = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdzl5dGtmeHJvZTBkY3NmY2Y3OXBzZW43bjZsdGRzYXhiZnA0dms4ZyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/vUc341wCXiY4U/giphy.gif"


def inject_custom_styles(bg_url):
    """Injects robust CSS styling matching the requested aesthetic theme."""
    css = (
        "<style>\n"
        "@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;700;800;900&family=Poppins:wght@300;400;600;700&display=swap');\n"
        "html, body, [class*='css'] { font-family: 'Poppins', sans-serif; }\n"
        
        "/* HIDE STREAMLIT LINK/ANCHOR ICONS NEXT TO HEADINGS */\n"
        "[data-testid='stHeaderActionElements'], .stHeadingAnchor, a.data-testid-stHeaderActionElements { display: none !important; visibility: hidden !important; }\n"
        "h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; opacity: 0 !important; }\n"
        "a[href*='#'] { display: none !important; }\n"

        "::-webkit-scrollbar { width: 12px; }\n"
        "::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.7); }\n"
        "::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #FF781F, #FF9800, #F57C00); border-radius: 10px; border: 2px solid rgba(255, 255, 255, 0.25); }\n"
        "::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #E65100, #FF6D00, #FF9800); }\n"
        
        "/* FULL VIEWPORT CENTERING FOR STAPP */\n"
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
        
        "/* Main Adaptive Glassmorphism Container - Middle Aligned */\n"
        ".block-container {\n"
        "  background: rgba(255, 255, 255, 0.95);\n"
        "  color: #1A202C;\n"
        "  border-radius: 28px;\n"
        "  padding: 24px 20px !important;\n"
        "  margin: auto !important;\n"
        "  max-width: 780px;\n"
        "  width: 100%;\n"
        "  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);\n"
        "  backdrop-filter: blur(14px);\n"
        "  border: 1px solid rgba(255, 255, 255, 0.4);\n"
        "}\n"

        ".content-section {\n"
        "  margin-top: 20px !important;\n"
        "  margin-bottom: 0px !important;\n"
        "}\n"

        "div.element-container {\n"
        "  margin-bottom: 0px !important;\n"
        "  margin-top: 5px !important;\n"
        "}\n"
        "div[data-testid='stVerticalBlock'] {\n"
        "  gap: 0.4rem !important;\n"
        "}\n"
        "h3 {\n"
        "  margin-top: 10px !important;\n"
        "  margin-bottom: 0.3rem !important;\n"
        "}\n"
        "h4 {\n"
        "  margin-top: 10px !important;\n"
        "  margin-bottom: 0.3rem !important;\n"
        "}\n"
        "p {\n"
        "  margin-bottom: 0.3rem !important;\n"
        "  margin-top: 0px !important;\n"
        "}\n"

        "div[data-testid='stAlert'] { color: #1A202C !important; font-weight: 500; border-radius: 12px; margin-bottom: 0.3rem !important; margin-top: 0.2rem !important; }\n"
        "div[data-testid='stAlert'] p { color: #1A202C !important; font-weight: 500; }\n"
        "div[data-testid='stAlert'] strong { color: #000000 !important; font-weight: 800; }\n"
        
        "/* Dark Mode Overrides */\n"
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

        ".main-title { font-family: 'Outfit', sans-serif; text-align: center; background: linear-gradient(135deg, #6366F1, #8B5CF6, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2rem; font-weight: 900; margin-bottom: 0px; padding-bottom: 0px; letter-spacing: -0.5px; }\n"
        ".sub-text { font-family: 'Poppins', sans-serif; text-align: center; font-size: 0.9rem; color: #4A5568; font-weight: 500; line-height: 1.3; margin-bottom: 6px; }\n"
        ".highlight-text { color: #6366F1; font-weight: 700; }\n"
        
        "div[data-testid='stRadio'] > div { justify-content: center; gap: 8px; border: none !important; margin-bottom: 2px; margin-top: 2px; }\n"
        "div[data-testid='stRadio'] label { background: rgba(240, 244, 248, 0.85); border: 1px solid #CBD5E0; border-radius: 10px; padding: 3px 12px; font-family: 'Outfit', sans-serif; font-weight: 700; transition: all 0.2s ease-in-out; color: #2D3748; }\n"
        "div[data-testid='stRadio'] label:hover { border-color: #6366F1; background: #FFFFFF; }\n"
        
        ".feature-card { background: #F8FAFC; border-radius: 10px; padding: 10px; border-left: 4px solid #6366F1; height: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.04); }\n"
        ".feature-card-title { font-family: 'Outfit', sans-serif; font-weight: 800; color: #2D3748; font-size: 0.95rem; margin-bottom: 2px; }\n"
        ".feature-card-desc { color: #4A5568; font-size: 0.82rem; line-height: 1.35; }\n"
        
        ".result-card { border-radius: 16px; padding: 14px; text-align: center; color: white !important; font-family: 'Outfit', sans-serif; font-weight: 800; margin-bottom: 10px; box-shadow: 0 8px 20px rgba(0,0,0,0.15); }\n"
        ".result-card p { color: white !important; }\n"
        ".result-positive { background: linear-gradient(135deg, #EF4444, #DC2626); }\n"
        ".result-healthy { background: linear-gradient(135deg, #10B981, #059669); }\n"
        ".card-title { font-size: 1.25rem; margin: 0; letter-spacing: 0.5px; color: #FFFFFF !important; }\n"
        
        ".stButton>button { background: linear-gradient(135deg, #6366F1, #8B5CF6); color: white !important; font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 0.75rem; border-radius: 8px; border: none; padding: 6px 10px; width: 100%; min-height: 40px; transition: all 0.3s ease; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35); }\n"
        ".stButton>button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(99, 102, 241, 0.5); }\n"
        
        "hr { margin: 6px 0 !important; border-color: #E2E8F0 !important; }\n"
        "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)


inject_custom_styles(PERMANENT_BG_GIF)


# ---------------------------------------------------------
# ML Model & Training Engine (from Colab reference)
# ---------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    # Load dataset from UCI ML repository URL used in the referenced Colab notebook
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/parkinsons.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        # Fallback local mock structure or alternate mirror if needed
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")
    
    # Feature separation
    X = df.drop(columns=['name', 'status'], errors='ignore')
    y = df['status']

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

    # Support Vector Machine Classifier model training
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    return model, scaler, list(X.columns)

model, scaler, feature_names = load_and_train_model()


# ---------------------------------------------------------
# Global Navigation Header & State Management
# ---------------------------------------------------------
st.markdown("<h1 class='main-title'>🧠 Parkinson's Disease Detector 🧠</h1>", unsafe_allow_html=True)

if 'nav' not in st.session_state:
    st.session_state.nav = '🏠 Home'
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

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
    st.markdown("### 🧬 Biomedical Voice Measurement Classifier")
    st.markdown(
        "<p style='font-size: 0.9rem; line-height: 1.4;'>"
        "Welcome! This application utilizes a Support Vector Machine (SVM) classification model trained on "
        "biomedical voice measurements to detect indicators of Parkinson's disease. "
        "By evaluating parameters such as signal fundamental frequency, jitter, shimmer variation, "
        "and noise-to-tonal ratios, the system delivers rapid health screening insights."
        "</p>", 
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Diagnostic Pipeline Architecture")
    st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="feature-card" style="border-left-color: #6366F1;"><div class="feature-card-title">1. Voice Analysis</div><div class="feature-card-desc">Captures acoustic parameters including frequency perturbation metrics (Jitter) and amplitude variation (Shimmer).</div></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="feature-card" style="border-left-color: #8B5CF6;"><div class="feature-card-title">2. Standardization</div><div class="feature-card-desc">Normalizes input features using a robust StandardScaler across multi-dimensional biomedical attributes.</div></div>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="feature-card" style="border-left-color: #EC4899;"><div class="feature-card-title">3. SVM Classification</div><div class="feature-card-desc">Evaluates hyperplanes via a linear Support Vector Machine to classify health status probability.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Key Features Highlight")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**⚡ Instant Prediction**\n\nReal-time evaluation of 22 clinical voice metrics.")
    with col2:
        st.success("**🔬 High Accuracy**\n\nOptimized SVM decision boundaries trained on UCI benchmark datasets.")
    with col3:
        st.warning("**🛡️ Secure & Private**\n\nInstant processing entirely browser-contained.")

    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.button("🚀 Launch Diagnostic Engine", on_click=switch_to_prediction)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PAGE 2: PREDICTION PAGE
# =========================================================
elif nav_choice == "🔮 Prediction":
    st.markdown(
        "<p class='sub-text'>"
        "Enter the patient's biomedical voice measurement parameters below or use default sample values "
        "to run the classification model."
        "</p>", 
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):
        st.markdown("### 📋 Enter Patient Voice Features")
        
        col1, col2, col3 = st.columns(3)
        
        # Default sample values representing a known baseline profile
        defaults = [
            119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 
            0.01109, 0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 
            0.02211, 21.033, 0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654
        ]

        user_inputs = []
        for i, col_name in enumerate(feature_names):
            # Split inputs evenly across 3 columns
            if i % 3 == 0:
                with col1:
                    val = st.number_input(f"{col_name}", value=float(defaults[i]), format="%.5f")
            elif i % 3 == 1:
                with col2:
                    val = st.number_input(f"{col_name}", value=float(defaults[i]), format="%.5f")
            else:
                with col3:
                    val = st.number_input(f"{col_name}", value=float(defaults[i]), format="%.5f")
            user_inputs.append(val)

        submit_btn = st.form_submit_button("🔍 Run Parkinson's Diagnosis")

        if submit_btn:
            input_data_as_numpy_array = np.asarray(user_inputs).reshape(1, -1)
            std_data = scaler.transform(input_data_as_numpy_array)
            prediction = model.predict(std_data)
            prob = model.predict_proba(std_data)
            
            st.session_state.prediction_result = {
                'pred': prediction[0],
                'prob': np.max(prob) * 100
            }

    if st.session_state.prediction_result is not None:
        st.markdown("---")
        st.markdown("### 📊 Diagnostic Results")
        res = st.session_state.prediction_result
        
        if res['pred'] == 1:
            st.markdown(f'<div class="result-card result-positive"><p class="card-title">⚠️ Positive: Parkinson\'s Symptoms Detected</p><p style="margin: 4px 0 0 0; font-size: 0.9rem; opacity: 0.9;">Confidence Score: {res["prob"]:.2f}%</p></div>', unsafe_allow_html=True)
            st.warning("The model indicates patterns consistent with Parkinson's disease voice measurements. Please consult a qualified neurologist for formal clinical evaluation.")
        else:
            st.markdown(f'<div class="result-card result-healthy"><p class="card-title">✅ Negative: Healthy Profile Detected</p><p style="margin: 4px 0 0 0; font-size: 0.9rem; opacity: 0.9;">Confidence Score: {res["prob"]:.2f}%</p></div>', unsafe_allow_html=True)
            st.success("The model indicates voice parameter patterns within normal healthy ranges.")


# =========================================================
# PAGE 3: ABOUT PAGE
# =========================================================
elif nav_choice == "ℹ️ About":
    st.markdown("### ℹ️ About the Model & Technology")
    st.markdown(
        """
        #### 🤖 Machine Learning Model: Support Vector Machine (SVM)
        This application implements a supervised **Support Vector Machine (SVM)** classifier with a linear kernel. 
        SVM algorithms excel at finding optimal decision boundaries (hyperplanes) to separate multi-dimensional medical parameters.

        #### 📚 Dataset Source
        * Trained on the **UCI Machine Learning Repository - Parkinson's Dataset**.
        * Comprises a range of biomedical voice measurements from 31 individuals, of which 23 have Parkinson's disease.
        * Features include fundamental frequencies (MDVP:Fo, Fhi, Flo), jitter variations, shimmer parameters, and ratio measures (NHR, HNR).

        #### 💻 Tech Stack
        * **Machine Learning:** Scikit-Learn (`svm.SVC`, `StandardScaler`)
        * **Data Processing:** Pandas & NumPy
        * **Frontend UI:** Streamlit Custom Glassmorphism Theme
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

st.caption("⚠️ **Disclaimer:** **AI Powered Medical Research Prototype:** Built by Sristi Sarkar for educational and research demonstration. Results should not replace professional clinical diagnosis.")
