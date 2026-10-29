# 🧠 Brain Tumor Detection using Deep Learning

A deep learning project that classifies brain MRI images into four categories using custom CNN models and a ResNet baseline, with a **Streamlit** web app for real-time predictions.

> ⚠️ **Disclaimer:** This project is for educational purpose only. It is **not** a medical device and must not be used for clinical diagnosis.

### deployment link-
 https://braintumordetectioncustomcnn.streamlit.app/
 
**NOTE**- **.pth files are not added to repository due to large size.**
---

## 📌 Overview

The system takes a brain MRI image as input and predicts one of four classes:

| Class | Description |
|---|---|
| `glioma` | Tumor arising from glial cells |
| `meningioma` | Tumor arising from the meninges |
| `pituitary` | Tumor in the pituitary gland |
| `notumor` | Healthy scan, no tumor detected |

---

## 📂 Dataset

- **Classes:** 4 (glioma, meningioma, notumor, pituitary)
- **Training images:** 1,400 per class (5,600 total)
- **Input:** MRI images resized to a fixed resolution (`224 × 224`) 

```
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

##  Models

Three versions were built and compared:

| # | Model | Architecture | Normalization | Training Strategy | Accuracy |
|---|---|---|---|---|---|
| 1 | **ResNet** | Transfer learning (ResNet) | ImageNet mean/std | Fine-tuning | **85%** |
| 2 | **Custom CNN (v1)** | Custom CNN | Standard mean/std values | Fixed epochs | **90.69%** |
| 3 | **Custom CNN (v2 / `new_model`)** | Custom CNN | Mean/std **calculated from the dataset** | Validation split + Early Stopping | **87.69%** |

### Model notes

- **Custom CNN (v1):** Uses standard normalization values. Highest test accuracy among the custom models.
- **Custom CNN (v2):** Uses dataset-specific mean and standard deviation, a dedicated validation set, and early stopping to reduce overfitting. The reported accuracy comes from a more rigorous training/evaluation setup, so it is a more conservative estimate.

### Model used in the app

The Streamlit app uses the **Custom CNN (v1) {model}– 90.69% accuracy**.

---

## 🛠️ Tech Stack

- Python 3.9+
- PyTorch / TensorFlow
- NumPy, Pandas, Matplotlib, scikit-learn
- Pillow / OpenCV
- Streamlit

---

## 📁 Project Structure

```
brain-tumor-detection/
├── app.py                  # Streamlit application
├── models/
│   ├── brain_tumor_resnet.pth
│   ├── brain_tumor_cnn.pth
│   └── brain_tumor_v2.pth
├── notebooks/
│   ├── resnet_model.ipynb
│   ├── model.ipynb
│   └── new_model.ipynb
├── requirements.txt
└── README.md
```

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate     
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload an MRI image, and view the predicted class with confidence scores.

---


## 🖥️ Streamlit App Features

- Upload an MRI image (`.jpg`, `.jpeg`, `.png`)
- Instant tumor-type prediction
- Class-wise confidence scores
- Simple, clean interface


---

## 📊 Results

| Model | Test Accuracy |
|---|---|
| ResNet | 85% |
| Custom CNN v1 (standard normalization) | 90.69% |
| Custom CNN v2 (calculated mean/std + early stopping + validation) | 87.69% |



---

## 🔮 Future Improvements

- Data augmentation (rotation, flip, contrast)
- Hyperparameter tuning and learning-rate scheduling
- Grad-CAM visualizations for explainability
- Cross-validation for more reliable evaluation
- Deployment on Streamlit Cloud / Hugging Face Spaces

---

## 👤 Author

**Jeevika H.** 
