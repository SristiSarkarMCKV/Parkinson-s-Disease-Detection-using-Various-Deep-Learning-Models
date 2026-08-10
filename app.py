import os
import urllib.request
import io
import gradio as gr
import numpy as np
from PIL import Image
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

def load_image_from_url(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")
    except Exception as e:
        print(f"Error loading image {url}: {e}")
        return Image.new('RGB', (224, 224), color=(50, 50, 50))

# Pre-load sample images into PIL format for the gallery
affected_imgs = [load_image_from_url(url) for url in AFFECTED_URLS]
unaffected_imgs = [load_image_from_url(url) for url in UNAFFECTED_URLS]
all_sample_imgs = unaffected_imgs + affected_imgs

def predict_parkinsons(img):
    if img is None:
        return {"Please provide a valid scan image.": 1.0}
    
    # Standard Model Inference
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  
    
    prediction = model.predict(img_array)[0][0]
    
    return {
        CLASS_NAMES[0]: float(1 - prediction), 
        CLASS_NAMES[1]: float(prediction)
    }

def toggle_source(source_choice):
    if source_choice == "Upload Image":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")

with gr.Blocks(theme=theme, title="Parkinson's Disease Diagnostic Platform") as demo:
    gr.Markdown(
        """
        # 🧠 🩺 Parkinson's Disease Diagnostic Platform
        *Automated deep learning biometric and image classification engine.*
        """
    )
    
    with gr.Tabs():
        # --- HOME PAGE ---
        with gr.TabItem("🏠 Home"):
            gr.Markdown(
                """
                ## 🧬 Automated Deep Learning Recognition Engine
                
                Welcome! This application utilizes state-of-the-art Deep Computer Vision to instantly analyze and evaluate biomarker visual cohorts. Built on top of a custom **Sequential Convolutional Neural Network (CNN)**, the framework processes structural intensity inputs to classify medical scans into distinct diagnostic categories:
                * 🟢 **Healthy Control** or 🔴 **Parkinson's Disease Detected**.

                ---

                ### 📌 Core Engineering Highlights
                * **State-of-the-Art Accuracy:** Achieved **97.0% Validation & Test Accuracy** within 10 training epochs.
                * **Clinical Safety First:** Optimized for a **98.84% Recall rate** on positive cases to minimize dangerous False Negatives.
                * **Modern Technical Stack:** Developed natively in Python using the latest **TensorFlow** runtime engine.
                """
            )
            
        # --- PREDICTION PAGE ---
        with gr.TabItem("🔮 Prediction"):
            gr.Markdown("### 🔍 Interactive Scan Predictor")
            gr.Markdown("Select a source below to either upload your own scan image or pick from the pre-loaded clinical sample database.")
            
            source_radio = gr.Radio(
                choices=["Upload Image", "Sample Images"],
                value="Upload Image",
                label="Source"
            )
            
            with gr.Row():
                with gr.Column():
                    # Upload component container
                    upload_container = gr.Column(visible=True)
                    with upload_container:
                        input_img_upload = gr.Image(type="pil", label="Upload Scan Image")
                    
                    # Sample gallery container
                    sample_container = gr.Column(visible=False)
                    with sample_container:
                        gr.Markdown("#### Click a sample image below to load it:")
                        gallery = gr.Gallery(
                            value=all_sample_imgs,
                            label="GitHub Sample Scans", 
                            columns=4, 
                            rows=2, 
                            object_fit="contain", 
                            height="auto"
                        )
                        input_img_sample = gr.Image(type="pil", label="Selected Sample Image", interactive=False)
                    
                    submit_btn = gr.Button("Run Inference", variant="primary")
                
                with gr.Column():
                    output_label = gr.Label(num_top_classes=2, label="Prediction Probabilities")
            
            # Switch view based on radio selection
            source_radio.change(
                fn=toggle_source,
                inputs=source_radio,
                outputs=[upload_container, sample_container]
            )
            
            # When user clicks a gallery image, populate the sample image display box
            def select_sample(evt: gr.SelectData):
                return all_sample_imgs[evt.index]
                
            gallery.select(fn=select_sample, outputs=input_img_sample)
            
            # Handle prediction submission depending on active source mode
            def handle_prediction(source_val, up_img, samp_img):
                target_img = up_img if source_val == "Upload Image" else samp_img
                return predict_parkinsons(target_img)

            submit_btn.click(
                fn=handle_prediction,
                inputs=[source_radio, input_img_upload, input_img_sample],
                outputs=output_label
            )

        # --- ABOUT PAGE ---
        with gr.TabItem("ℹ️ About"):
            gr.Markdown(
                """
                ## ⚙️ Classification System Architecture & Workflow
                * **Dataset & Augmentation:** Real-time data generators scale pixel intensities ($1/255$), apply structural transformations ($15^{\circ}$ rotation, $0.2$ shear, $0.2$ zoom, and horizontal flips) to protect against overfitting.
                * **Model Design:** Cascaded 2D Convolution and Max-Pooling blocks leading into dense operational layers with dropout regulation ($0.5$).
                * **Performance Metrics:** 
                  * **Accuracy:** `97.00%`
                  * **Precision:** `0.9659`
                  * **Recall:** `98.84%`
                  * **F1-Score:** `0.9770`
                """
            )

if __name__ == "__main__":
    demo.launch()
