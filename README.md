# 🔢 Advanced MNIST Handwritten Digit Recognition

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://handwritten-digit-recognizer-hrfdvfspcmvznh8tfx4mfm.streamlit.app/)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-99.6%25-10B981?style=for-the-badge)](#)

*A production-grade, deep Convolutional Neural Network (CNN) trained on the MNIST dataset, wrapped in a beautiful, highly interactive real-time web application.*

**[🔴 PLAY WITH THE LIVE APP HERE!](https://handwritten-digit-recognizer-hrfdvfspcmvznh8tfx4mfm.streamlit.app/)**

</div>

---

## ✨ Features overview

* **🎨 Real-Time Drawing Canvas** — Draw a digit on the interactive chalkboard and watch the model predict it instantly.
* **📷 Smart Image Uploads** — Upload photos of handwritten digits. Uses Otsu's thresholding and aspect-ratio-preserving Computer Vision to extract digits from real-world photos.
* **⚡ Blazing Fast CNN** — A highly optimized 3-block Deep CNN with BatchNorm, Dropout, and Data Augmentation achieving >99.6% accuracy.
* **📊 Probability Visualization** — Dynamic gradient bars showcasing the model's confidence across all 10 possible classes.
* **🔮 Premium UI/UX** — A custom-styled, dark-mode *glassmorphism* dashboard built purely with Streamlit.

---

## 🚀 Experience The Live Application

No installation required! You can interact with the pre-trained neural network right now in your browser:

### 👉 **[Live Demo: Handwritten Digit Recognizer](https://handwritten-digit-recognizer-hrfdvfspcmvznh8tfx4mfm.streamlit.app/)** 👈

---

## 🏗️ Technical Architecture

The core of this project is a custom Convolutional Neural Network built with `tf.keras`.

```text
Input Image (28×28 Grayscale)
      │
┌─────▼─────────────────────────────────────────────────┐
│ BLOCK 1: Low-Level Features                           │
│ 2x [Conv2D (32 filters) → BatchNorm → ReLU]           │
│ MaxPooling2D → Dropout (0.25)                         │
└─────┬─────────────────────────────────────────────────┘
      │
┌─────▼─────────────────────────────────────────────────┐
│ BLOCK 2: Mid-Level Features                           │
│ 2x [Conv2D (64 filters) → BatchNorm → ReLU]           │
│ MaxPooling2D → Dropout (0.25)                         │
└─────┬─────────────────────────────────────────────────┘
      │
┌─────▼─────────────────────────────────────────────────┐
│ BLOCK 3: High-Level Abstractions                      │
│ 2x [Conv2D (128 filters) → BatchNorm → ReLU]          │
│ MaxPooling2D → Dropout (0.40)                         │
└─────┬─────────────────────────────────────────────────┘
      │
┌─────▼─────────────────────────────────────────────────┐
│ CLASSIFIER HEAD                                       │
│ Flatten → Dense (256) → BatchNorm → Dropout (0.50)    │
│ Dense (10) → Softmax Activation                       │
└───────────────────────────────────────────────────────┘
```

> **Optimization Highlights:** 
> * **Cosine Annealing** Learning Rate Scheduler
> * **Label Smoothing** (0.1) for better generalization against noisy strokes.
> * **Data Augmentation** (Rotation, Zoom, Shifts) to simulate human handwriting imperfections.

---

## 💻 Local Installation & Setup

Want to run it on your own machine, train your own custom CNN, or modify the code? Follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer
```

### 2. Set up the Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the App
*If the `models/mnist_cnn.keras` file is already present, you can skip training.*
```bash
streamlit run app.py
```

### (Optional) Retrain the Model
To train the CNN from scratch and generate new accuracy/loss graphs:
```bash
python train_model.py
```

---

## 🛠️ Tech Stack & Libraries

| Category              | Technologies used                                                                 |
| --------------------- | --------------------------------------------------------------------------------- |
| **Deep Learning**     | `TensorFlow 2.x`, `Keras`                                                         |
| **Computer Vision**   | `OpenCV` (`opencv-python-headless`), `Pillow`, `NumPy`                            |
| **Web Interface**     | `Streamlit`, `streamlit-drawable-canvas`                                          |
| **Data Analytics**    | `Scikit-learn`, `Matplotlib`, `Seaborn`, `Pandas`                                 |

---

## 💡 Usage Tips for the Best Results
* **Thickness matters:** Use the sidebar slider to keep the brush size thick (around `22-26px`).
* **Center your digits:** Draw in the middle of the canvas, just like standard MNIST numbers.
* **Upload clear backgrounds:** When uploading photos, plain white paper with a thick black marker yields a 100% correct prediction rate.

<br>
<div align="center">
  <i>Created with ❤️</i>
</div>

