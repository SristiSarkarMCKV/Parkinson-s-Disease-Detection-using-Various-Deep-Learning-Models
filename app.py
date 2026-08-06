import os
import urllib.request
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "PD_CNN_Model.h5"

# Automatically download the model weights if they aren't present locally
if not os.path.exists(MODEL_PATH):
    print("Model weights not found locally. Downloading from cloud storage...")
    # Replace the URL below with your actual direct download link for PD_CNN_Model.h5
    MODEL_URL = "YOUR_DIRECT_MODEL_DOWNLOAD_URL_HERE"
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully!")

# Load the model
model = load_model(MODEL_PATH)

CLASS_NAMES = ["Healthy Control", "Parkinson's Disease Detected"]

def predict_parkinsons(img):
    if img is None:
        return "Please upload a valid scan image."
    
    # Match the 224x224 input shape
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
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
        # 🧠 Parkinson's Disease Detection Platform
        *Non-invasive diagnostic evaluation tool powered by Convolutional Neural Networks (CNN).*
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("🏠 Pipeline Overview"):
            gr.Markdown(
                """
                ### Core Architecture Highlights
                * **Model:** 3-tier cascaded 2D Convolution and Max-Pooling network.
                * **Validation Performance:** Optimized on structured medical cohorts using data augmentations.
                """
            )
            
        with gr.TabItem("🔍 Scan Predictor"):
            gr.Markdown("### Upload a Patient Scan (Spiral/Wave)")
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

if __name__ == "__main__":
    demo.launch()
