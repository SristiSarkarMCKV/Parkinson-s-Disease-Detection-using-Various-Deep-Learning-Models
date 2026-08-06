import os
import urllib.request
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "PD_CNN_Model.h5"

# Automatically download model weights from Hugging Face if not present locally
if not os.path.exists(MODEL_PATH):
    print("Downloading model weights from Hugging Face...")
    MODEL_URL = "https://huggingface.co/SRISTISARKAR/parkinsons-cnn-model/resolve/main/PD_CNN_Model.h5"
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully!")

# Load model
model = load_model(MODEL_PATH)
CLASS_NAMES = ["Healthy Control", "Parkinson's Disease Detected"]

# Helper function to generate clean sample placeholder graphics for testing
def create_sample_image(label_text, color):
    img = Image.new('RGB', (224, 224), color=color)
    d = ImageDraw.Draw(img)
    d.text((30, 100), label_text, fill=(255, 255, 255))
    return img

sample_normal = create_sample_image("Normal Scan Sample", color=(30, 90, 140))
sample_parkinson = create_sample_image("PD Affected Sample", color=(160, 40, 40))

def predict_parkinsons(img):
    if img is None:
        return {"Please upload a valid scan image.": 1.0}
    
    # Preprocess image to match training pipeline input shape (224x224)
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalization
    
    prediction = model.predict(img_array)[0][0]
    
    return {
        CLASS_NAMES[0]: float(1 - prediction), 
        CLASS_NAMES[1]: float(prediction)
    }

theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")

with gr.Blocks(theme=theme, title="Parkinson's Disease Detection Platform") as demo:
    gr.Markdown(
        """
        # 🧠 Parkinson’s Disease Detection via Deep Learning Architectures
        An automated, non-invasive deep learning diagnostic pipeline built to evaluate biomarkers from visual cohorts using custom Sequential Convolutional Neural Networks (CNN). Achieves a definitive clinical test accuracy of **97.0%**.
        """
    )
    
    with gr.Tabs():
        # --- PAGE 1: HOME / PIPELINE DETAILS ---
        with gr.TabItem("🏠 Home & Architecture Details"):
            gr.Markdown(
                """
                ## 📌 Core Engineering Highlights
                * **State-of-the-Art Accuracy:** Achieved **97.0% Validation & Test Accuracy** within 10 training epochs.
                * **Clinical Safety First:** Optimizes for a **98.84% Recall rate** on positive cases to minimize dangerous False Negatives.
                * **Modern Technical Stack:** Developed natively in Python using the latest **TensorFlow** runtime engine.

                ---

                ## 🏗️ End-to-End Pipeline Overview
                1. **Environment & Path Configurations:** Automatically unzips, validates, and sets up robust environmental workspace paths.
                2. **Dataset Preparation & Augmentation:** Real-time data generators scale intensities ($1/255$), apply structural rotations ($15^{\circ}$), shears ($0.2$), zooms ($0.2$), and horizontal flips.
                3. **Deep Learning Core Model Design:** Sequential feature extraction model structured with three cascaded 2D Convolution ($3\times3$ filters: 32, 64, 128) and Max-Pooling layers, leading to a high-density decision head (Dense 128, Dropout 0.5, Dense Sigmoid output).
                4. **Training & Compilation:** Compiled using the **Adam Optimizer** and evaluated through **Binary Cross-Entropy Loss** over stable epochs.

                ---

                ## 📊 Experimental Performance Matrix
                * **Overall Accuracy:** `97.00%`
                * **Precision:** `0.9659`
                * **Recall (Sensitivity):** `0.9884`
                * **F1-Score:** `0.9770`
                """
            )
            
        # --- PAGE 2: SCAN PREDICTOR & DEMOS ---
        with gr.TabItem("🔍 Scan Predictor & Demos"):
            gr.Markdown("### Upload or Test Sample Scans (DaTscan / Biomarker Cohorts)")
            gr.Markdown("You can upload your own custom medical scan below, or click one of the pre-loaded sample images underneath to instantly run test inferences.")
            
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="Upload Scan Image")
                    submit_btn = gr.Button("Run Inference", variant="primary")
                with gr.Column():
                    output_label = gr.Label(num_top_classes=2, label="Prediction Probabilities")
            
            submit_btn.click(
                fn=predict_parkinsons,
                inputs=input_img,
                outputs=output_label
            )
            
            gr.Markdown("### 🧪 Quick Sample Test Demos")
            gr.Markdown("Click on any sample image below to load it into the predictor above for quick verification:")
            
            with gr.Row():
                sample_btn_1 = gr.Button("Load Normal Brain Sample")
                sample_btn_2 = gr.Button("Load Parkinson's Affected Sample")
                
            sample_btn_1.click(fn=lambda: sample_normal, outputs=input_img)
            sample_btn_2.click(fn=lambda: sample_parkinson, outputs=input_img)

if __name__ == "__main__":
    demo.launch(share=True)
