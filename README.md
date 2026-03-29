# 🔢 Advanced MNIST Handwritten Digit Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-~99.5%25-10B981?style=for-the-badge)

A production-quality handwritten digit recognition system powered by a deep **Convolutional Neural Network (CNN)** trained on the MNIST dataset. Features a premium **glassmorphism** Streamlit web interface with a real-time drawing canvas.

</div>

---

## 📸 Features

- 🎨 **Interactive drawing canvas** — draw any digit (0–9) with your mouse
- ⚡ **Real-time inference** — CNN predicts instantly as you draw
- 📊 **Confidence visualization** — animated probability bars for all 10 digits
- 🕑 **Prediction history** — session tracking of last 10 predictions
- 📈 **Evaluation dashboard** — confusion matrix + accuracy/loss training curves
- 🧠 **Architecture explorer** — detailed CNN layer breakdown

---

## 🏗️ CNN Architecture

```
Input (28×28×1)
     │
┌────▼────────────────────────────────────────────────┐
│  BLOCK 1 — Feature Extraction (Low-level)           │
│  Conv2D(32, 3×3) → BatchNorm → ReLU                │
│  Conv2D(32, 3×3) → BatchNorm → ReLU                │
│  MaxPooling(2×2) → Dropout(0.25)                   │
└────┬────────────────────────────────────────────────┘
     │
┌────▼────────────────────────────────────────────────┐
│  BLOCK 2 — Feature Extraction (Mid-level)           │
│  Conv2D(64, 3×3) → BatchNorm → ReLU                │
│  Conv2D(64, 3×3) → BatchNorm → ReLU                │
│  MaxPooling(2×2) → Dropout(0.25)                   │
└────┬────────────────────────────────────────────────┘
     │
┌────▼────────────────────────────────────────────────┐
│  BLOCK 3 — High-level Feature Abstraction           │
│  Conv2D(128, 3×3) → BatchNorm → ReLU               │
│  Conv2D(128, 3×3) → BatchNorm → ReLU               │
│  MaxPooling(2×2) → Dropout(0.40)                   │
└────┬────────────────────────────────────────────────┘
     │
┌────▼────────────────────────────────────────────────┐
│  CLASSIFIER HEAD                                    │
│  Flatten → Dense(256) → BatchNorm → Dropout(0.50)  │
│  Dense(10) → Softmax                               │
└─────────────────────────────────────────────────────┘
     │
Output: [p0, p1, ..., p9] — class probabilities
```

### Key Design Decisions

| Technique | Purpose |
|-----------|---------|
| **BatchNormalization** | Stabilizes training, allows higher learning rates |
| **Dropout (0.25→0.50)** | Prevents overfitting; increases from shallow→deep layers |
| **He Normal Init** | Optimal weight init for ReLU activations |
| **Label Smoothing** | Prevents overconfident predictions, improves generalization |
| **Data Augmentation** | Rotation/zoom/shifts simulate natural handwriting variation |
| **Cosine Annealing LR** | Smooth learning rate decay for better convergence |

---

## 📦 Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| **TensorFlow / Keras** | ≥2.13 | CNN model definition, training, inference |
| **NumPy** | ≥1.24 | Numerical operations, array manipulation |
| **OpenCV** | ≥4.8 | Image preprocessing (resize, denoise, crop) |
| **Matplotlib** | ≥3.7 | Training accuracy/loss curve plots |
| **Seaborn** | ≥0.12 | Confusion matrix heatmap |
| **Scikit-learn** | ≥1.3 | Evaluation metrics (confusion matrix, F1-score) |
| **Streamlit** | ≥1.28 | Web interface framework |
| **streamlit-drawable-canvas** | ≥0.9.3 | Interactive in-browser drawing widget |
| **Pillow** | ≥10.0 | Image format conversion |
| **Pandas** | ≥2.0 | Data manipulation |

---

## 🚀 Quick Start

### Option 1 — Automated Setup (Windows)

```batch
# Double-click setup.bat OR run in terminal:
setup.bat
```

This script will:
1. Create a Python virtual environment
2. Install all dependencies
3. Train the CNN model (~5–10 min)
4. Launch the Streamlit web app

### Option 2 — Manual Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train_model.py

# 4. Launch the web app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
minorproject2/
├── train_model.py          # CNN training script (data aug + evaluation)
├── app.py                  # Premium Streamlit web application
├── requirements.txt        # Python package dependencies
├── setup.bat               # One-click Windows setup script
├── README.md               # This file
├── models/
│   └── mnist_cnn.keras     # Saved best CNN model (generated after training)
└── assets/
    ├── accuracy_loss_plot.png   # Training curves (generated after training)
    ├── confusion_matrix.png     # 10×10 confusion matrix (generated)
    └── training_metadata.json  # Test accuracy, loss, hyperparameters
```

---

## 🔧 Data Augmentation Pipeline

During training, each batch applies random transforms to increase robustness:

```python
ImageDataGenerator(
    rotation_range   = 10,    # ±10° rotation
    width_shift_range  = 0.10, # ±10% horizontal shift
    height_shift_range = 0.10, # ±10% vertical shift
    zoom_range       = 0.10,  # ±10% zoom
    shear_range      = 0.05,  # ±5° shear
    fill_mode        = 'nearest'
)
```

---

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~99.8% |
| Validation Accuracy | ~99.5% |
| Test Accuracy | ~99.3–99.5% |
| Test Loss | ~0.025 |

---

## 💡 Usage Tips

- **Draw large**: Fill most of the canvas for best accuracy
- **Draw centered**: The preprocessing centers the digit automatically
- **Use thick strokes**: Set brush size to 18–25 for MNIST-like strokes
- **Clear between digits**: Use the "Clear Canvas" button between drawings

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
