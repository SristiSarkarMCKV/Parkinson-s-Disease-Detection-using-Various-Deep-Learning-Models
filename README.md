# Parkinson’s Disease Detection via Deep Learning Architectures
An automated, non-invasive deep learning diagnostic pipeline built to evaluate biomarkers from visual cohorts using custom Sequential Convolutional Neural Networks (CNN). Leveraging state-of-the-art intensity range normalizations and real-time data augmentations, this framework achieves a definitive clinical test accuracy of **97.0%**.

---

## 📌 Core Engineering Highlights
* **State-of-the-Art Accuracy:** Achieved **97.0% Validation & Test Accuracy** within 10 training epochs.
* **Clinical Safety First:** Optimizes for a **98.84% Recall rate** on positive cases to minimize dangerous False Negatives.
* **Modern Technical Stack:** Developed natively in Python 3.12 using the latest **TensorFlow 2.19.0** runtime engine.

---

## 🏗️ End-to-End Pipeline Steps

The repository maps directly to a rigorous 11-step execution architecture:

### 1. Environment & Path Configurations
Automatically unzips and validates the structural integrity of the workspace. Dynamically maps nested internal layouts into robust environmental paths:
```python
base_data_path = os.path.join(extract_path, 'ParkinsonDisease', 'ParkinsonDisease')
train_dir = os.path.join(base_data_path, 'TRAIN')
test_dir = os.path.join(base_data_path, 'TEST')

```

### 2. Dataset Preparation & Augmentation Engine

Uses real-time data generators to scale intensities, apply structural transformations to protect against overfitting and split training data with a 20% validation anchor:

* **Train Set:** 413 images
* **Validation Set:** 103 images
* **Test Set:** 129 images
* **Transformations:** Rescale ($1/255$), Rotation ($15^{\circ}$), Shear ($0.2$), Zoom ($0.2$), Horizontal Flip.

### 3. Deep Learning Core Model Design

A sequential feature extraction model structured with three cascaded 2D Convolution and Max-Pooling layers, leading to a high-density decision head:

```
Input (224, 224, 3) 
   │
   ├──> Conv2D (32 filters, 3x3, ReLU) ──> MaxPooling2D (2x2)
   ├──> Conv2D (64 filters, 3x3, ReLU) ──> MaxPooling2D (2x2)
   ├──> Conv2D (128 filters, 3x3, ReLU) ──> MaxPooling2D (2x2)
   │
   └──> Flatten (86,528 features) ──> Dense (128, ReLU) ──> Dropout (0.5) ──> Dense (1, Sigmoid)

```

### 4. Neural Network Training

Compiled using the **Adam Optimizer** and evaluated through **Binary Cross-Entropy Loss**. Trained over 10 stable epochs with a mini-batch constraints size of 32.

---

## 📊 Model Summary & Parameter Footprint

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 222, 222, 32)   │           896 │
│ max_pooling2d (MaxPooling2D)    │ (None, 111, 111, 32)   │             0 │
│ conv2d_1 (Conv2D)               │ (None, 109, 109, 64)   │        18,496 │
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 54, 54, 64)     │             0 │
│ conv2d_2 (Conv2D)               │ (None, 52, 52, 128)    │        73,856 │
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 26, 26, 128)    │             0 │
│ flatten (Flatten)               │ (None, 86528)          │             0 │
│ dense (Dense)                   │ (None, 128)            │    11,075,712 │
│ dropout (Dropout)               │ (None, 128)            │             0 │
│ dense_1 (Dense)                 │ (None, 1)              │           129 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 11,169,089 (42.61 MB)
 Trainable params: 11,169,089 (42.61 MB)

```

---

## 📈 Experimental Performance Results

### Classification Report Matrix

Evaluated over 129 completely unseen target samples containing distinct categorical classes (`YES` vs. `NO`).

```
              precision    recall  f1-score   support

          NO       0.98      0.93      0.95        43
         YES       0.97      0.99      0.98        86

    accuracy                           0.97       129
   macro avg       0.97      0.96      0.96       129
weighted avg       0.97      0.97      0.97       129

```

### Core Diagnostic Metrics

* **Overall Accuracy:** `97.00%`
* **Precision:** `0.9659`
* **Recall (Sensitivity):** `0.9884`
* **F1-Score:** `0.9770`
<img width="505" height="470" alt="image" src="https://github.com/user-attachments/assets/008f81aa-5573-4cbd-b4bb-33e508361962" />


---

## 🚀 Model Deployment & Single-Image Inference

The framework ships with an integrated, standalone prediction pipeline (`Step 11`) to simulate real-world clinical usage. It ingests an un-scanned test matrix, maps internal generator classes dynamically and visualizes a structural verdict complete with confidence weights.


### Execution Interface Logs

```bash
Classes found in test directory: ['YES', 'NO']
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 202ms/step
Prediction: Parkinson Detected | Confidence Score: 0.9842

```
<img width="389" height="432" alt="image" src="https://github.com/user-attachments/assets/e7a2d36e-d962-41e6-8558-2b984abd2918" />

```
