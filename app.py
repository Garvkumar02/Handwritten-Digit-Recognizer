"""
Advanced MNIST Digit Recognition — Premium Streamlit Web App
=============================================================
Features:
  - Glassmorphism dark-theme UI
  - Real-time drawing canvas with mouse
  - CNN prediction with animated confidence bars
  - Prediction history with session tracking
  - Evaluation plots (confusion matrix, accuracy/loss)
  - Model architecture overview
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
import json
import os
import time

# ─── Page Config — MUST be first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="Digit Recognizer · MNIST CNN",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Advanced MNIST CNN Digit Recognizer — Built with TensorFlow & Streamlit"
    }
)

# ─── Lazy imports (avoid top-level failures) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained CNN model (cached across reruns)."""
    import tensorflow as tf
    model_path = "models/mnist_cnn.keras"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Base & fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}

    /* ── Dark glassmorphism background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1225 40%, #111827 70%, #0a0e1a 100%);
        background-attachment: fixed;
    }

    /* Animated gradient orbs */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background:
            radial-gradient(ellipse at 20% 20%, rgba(99,102,241,0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(16,185,129,0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(244,63,94,0.04) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }

    /* ── Glass card component ── */
    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 24px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
        margin-bottom: 16px;
    }

    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 32px 0 24px 0;
        position: relative;
    }
    .app-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #a78bfa 70%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .app-subtitle {
        color: rgba(255,255,255,0.45);
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    .header-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 100px;
        padding: 4px 16px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #a78bfa;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    /* ── Section headers ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: rgba(255,255,255,0.9);
        letter-spacing: 0.3px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(99,102,241,0.4), transparent);
        margin-left: 8px;
    }

    /* ── Prediction display ── */
    .prediction-box {
        text-align: center;
        padding: 28px 20px;
        background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08));
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 20px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa);
    }
    .predicted-digit {
        font-size: 6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        font-family: 'JetBrains Mono', monospace;
        text-shadow: none;
    }
    .prediction-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.4);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 6px;
    }
    .confidence-text {
        font-size: 1.3rem;
        font-weight: 700;
        color: #10b981;
        margin-top: 8px;
    }

    /* ── Confidence bar ── */
    .conf-bar-container {
        margin: 4px 0;
    }
    .conf-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: rgba(255,255,255,0.6);
        margin-bottom: 3px;
        font-family: 'JetBrains Mono', monospace;
    }
    .conf-bar-track {
        height: 7px;
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    /* ── Stat chips ── */
    .stat-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 0.82rem;
        color: rgba(255,255,255,0.7);
        margin: 4px;
    }
    .stat-chip-value {
        font-weight: 700;
        color: #a78bfa;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── History item ── */
    .history-item {
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 10px 14px;
        margin: 6px 0;
        transition: all 0.2s ease;
    }
    .history-item:hover {
        background: rgba(99,102,241,0.08);
        border-color: rgba(99,102,241,0.25);
    }
    .history-digit-badge {
        width: 38px; height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(139,92,246,0.25));
        border: 1px solid rgba(99,102,241,0.3);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem; font-weight: 800;
        color: #c4b5fd;
        font-family: 'JetBrains Mono', monospace;
        flex-shrink: 0;
    }
    .history-conf {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.4);
    }
    .history-conf-val {
        font-weight: 600;
        color: #10b981;
    }

    /* ── Arch block ── */
    .arch-layer {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid;
        border-radius: 0 12px 12px 0;
        padding: 10px 16px;
        margin: 6px 0;
        font-size: 0.83rem;
        color: rgba(255,255,255,0.75);
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Streamlit widget overrides ── */
    .stSlider > div > div > div { background: rgba(99,102,241,0.3) !important; }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
    }
    [data-testid="stSidebar"] {
        background: rgba(10,14,26,0.8) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    [data-testid="stMetricValue"] {
        color: #a78bfa !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: rgba(255,255,255,0.5) !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #a78bfa !important;
        border-bottom-color: #6366f1 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
        gap: 16px !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_image(image_data: np.ndarray, source: str = "canvas") -> np.ndarray | None:
    """Convert RGBA image → 28×28 grayscale tensor for CNN inference."""
    if image_data is None:
        return None

    # 1. Convert to grayscale
    if image_data.shape[-1] == 4:
        if source == "canvas":
            # Just drop alpha, use RGB
            rgb = image_data[:, :, :3]
            img_gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            # Threshold out the background (canvas uses dark theme so bg is ~20, text ~255)
            _, img_gray = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
        else:
            # Upload with alpha. Composite over white background.
            alpha = image_data[:, :, 3] / 255.0
            rgb = image_data[:, :, :3]
            bg = np.ones_like(rgb, dtype=np.uint8) * 255
            blended = (rgb * alpha[..., None] + bg * (1 - alpha[..., None])).astype(np.uint8)
            img_gray = cv2.cvtColor(blended, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # 2. Invert and clean for uploads
    if source == "upload":
        # Use Otsu's thresholding directly on the original grayscale to separate
        # the background from the foreground accurately.
        if np.mean(img_gray) > 127:
            # Light background -> dark/colored regions become white (255)
            _, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Dark background -> light regions become white (255)
            _, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        # Morphological closing to fill small holes (like internal highlights in bubbly text)
        kernel = np.ones((5, 5), np.uint8)
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)

    # Check if there's anything drawn
    if cv2.countNonZero(img_gray) < 5:
        return None

    # 3. Find bounding box
    coords = cv2.findNonZero(img_gray)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)

    # Crop out the digit
    digit_roi = img_gray[y:y+h, x:x+w]
    if digit_roi.size == 0 or w == 0 or h == 0:
         return None

    # 4. Resize to 20x20 preserving aspect ratio (MNIST standard)
    scale = 20.0 / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5. Pad to 28x28 (center the digit)
    pad_y = (28 - new_h) // 2
    pad_x = (28 - new_w) // 2
    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    # (Optional) Apply slight dilation for uploads to thicken pen strokes
    if source == "upload":
        kernel = np.ones((2,2), np.uint8)
        padded = cv2.dilate(padded, kernel, iterations=1)

    # Normalize mapped to 0...1
    padded = padded.astype(np.float32) / 255.0
    return padded.reshape(1, 28, 28, 1)


# ─── Confidence bar HTML ──────────────────────────────────────────────────────
def confidence_bar_html(digit: int, prob: float, is_top: bool) -> str:
    pct   = prob * 100
    color = ("#6366f1" if is_top else
             "#10b981" if pct > 15 else
             "#374151")
    fill  = f"linear-gradient(90deg, {color}, {'#8b5cf6' if is_top else color})"
    bold  = "font-weight:700; color:#e2e8f0;" if is_top else ""
    return f"""
    <div class="conf-bar-container">
      <div class="conf-bar-label">
        <span style="{bold}">Digit  {digit}</span>
        <span style="{'color:#a78bfa;font-weight:700;' if is_top else ''}">{pct:.1f}%</span>
      </div>
      <div class="conf-bar-track">
        <div class="conf-bar-fill" style="width:{pct:.1f}%; background:{fill};"></div>
      </div>
    </div>
    """


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    inject_css()

    # ── Session state init ────────────────────────────────────────────────────
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "total_predictions" not in st.session_state:
        st.session_state.total_predictions = 0
    if "brush_size" not in st.session_state:
        st.session_state.brush_size = 22
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = "canvas_0"

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model()

    # ── Load metadata ─────────────────────────────────────────────────────────
    metadata = {}
    if os.path.exists("assets/training_metadata.json"):
        with open("assets/training_metadata.json") as f:
            metadata = json.load(f)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <div class="header-badge">🧠 MNIST CNN · TensorFlow 2.x</div>
        <div class="app-title">Digit Recognizer</div>
        <div class="app-subtitle">Draw a digit · Get instant AI prediction</div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model not found. Please run `python train_model.py` first.", icon="⚠️")
        st.code("python train_model.py", language="bash")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding: 8px 0 24px 0;">
            <div style="font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#6366f1,#a78bfa);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;letter-spacing:-0.5px;">⚙ Controls</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Drawing Settings**")
        brush_size = st.slider("Brush Size", 8, 40, st.session_state.brush_size, 1, key="brush_slider")
        st.session_state.brush_size = brush_size

        stroke_color = "#FFFFFF"

        st.markdown("---")
        st.markdown("**Model Stats**")

        acc     = metadata.get("test_accuracy", 0)
        loss    = metadata.get("test_loss", 0)
        ep      = metadata.get("epochs_trained", 0)
        best_va = metadata.get("best_val_acc", 0)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Acc", f"{acc*100:.2f}%")
            st.metric("Epochs",   str(ep))
        with col2:
            st.metric("Test Loss", f"{loss:.4f}")
            st.metric("Val Acc",  f"{best_va*100:.2f}%")

        st.markdown("---")
        st.markdown("**Session**")
        tot = st.session_state.total_predictions
        st.metric("Total Predictions", tot)

        if st.button("🗑 Clear History / Canvas", key="clear_hist"):
            st.session_state.prediction_history = []
            st.session_state.total_predictions  = 0
            st.session_state.canvas_key = f"canvas_{time.time()}"
            st.rerun()

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab_draw, tab_upload = st.tabs(["✏️ Draw Digit", "📷 Upload Image"])

    # ════════════════════════════════════════════════════════════════════════════
    # Tab 1 — Drawing Canvas
    # ════════════════════════════════════════════════════════════════════════════
    with tab_draw:
        try:
            from streamlit_drawable_canvas import st_canvas
        except ImportError:
            st.error("Please install: `pip install streamlit-drawable-canvas`")
            st.stop()

        col_canvas, col_result = st.columns([1.15, 1], gap="large")

        with col_canvas:
            st.markdown('<div class="section-title">🎨 Draw a Digit</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="color:rgba(255,255,255,0.35);font-size:0.8rem;margin:-8px 0 12px 0;">
            Draw any digit 0–9 in the canvas below. Try to fill the space!
            </div>
            """, unsafe_allow_html=True)

            canvas_result = st_canvas(
                fill_color   = "rgba(0,0,0,0)",
                stroke_width = st.session_state.brush_size,
                stroke_color = stroke_color,
                background_color = "#111827",
                height       = 340,
                width        = 340,
                drawing_mode = "freedraw",
                key          = st.session_state.canvas_key,
                display_toolbar = False,
            )

            c1, c2 = st.columns(2)
            with c1:
                predict_btn = st.button("🔍 Predict Digit", key="predict_btn", use_container_width=True)
            with c2:
                clear_btn = st.button("🗑 Clear Canvas", key="clear_canvas_btn", use_container_width=True)

            if clear_btn:
                st.session_state.canvas_key = f"canvas_{time.time()}"
                st.rerun()

            st.markdown("""
            <div style="color:rgba(255,255,255,0.2);font-size:0.75rem;margin-top:10px;text-align:center;">
            💡 Tip: Draw large, centered digits for best accuracy
            </div>
            """, unsafe_allow_html=True)

        with col_result:
            st.markdown('<div class="section-title">🎯 Prediction</div>', unsafe_allow_html=True)

            # Run inference
            prediction_probs = None
            predicted_digit  = None
            confidence       = 0.0

            image_data = canvas_result.image_data if canvas_result is not None else None

            if image_data is not None and image_data.max() > 10:
                processed = preprocess_image(image_data, source="canvas")
                if processed is not None:
                    preds            = model.predict(processed, verbose=0)[0]
                    predicted_digit  = int(np.argmax(preds))
                    confidence       = float(preds[predicted_digit])
                    prediction_probs = preds

                    # Auto-record on predict button
                    if predict_btn:
                        st.session_state.total_predictions += 1
                        st.session_state.prediction_history.insert(0, {
                            "digit":      predicted_digit,
                            "confidence": confidence,
                            "n":          st.session_state.total_predictions
                        })
                        if len(st.session_state.prediction_history) > 10:
                            st.session_state.prediction_history.pop()

            # Display prediction
            if predicted_digit is not None:
                conf_color = "#10b981" if confidence > 0.85 else ("#f59e0b" if confidence > 0.55 else "#ef4444")
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="predicted-digit">{predicted_digit}</div>
                    <div class="prediction-label">Predicted Digit</div>
                    <div class="confidence-text" style="color:{conf_color};">{confidence*100:.1f}% confident</div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bars
                st.markdown('<div class="section-title">📈 All Probabilities</div>', unsafe_allow_html=True)
                sorted_idx = np.argsort(prediction_probs)[::-1]
                bars_html  = "".join(
                    confidence_bar_html(int(i), float(prediction_probs[i]), i == predicted_digit)
                    for i in sorted_idx
                )
                st.markdown(bars_html, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                            height:280px;color:rgba(255,255,255,0.2);text-align:center;">
                    <div style="font-size:4rem;margin-bottom:16px;">✏️</div>
                    <div style="font-size:1rem;font-weight:500;">Draw a digit to see the prediction</div>
                    <div style="font-size:0.8rem;margin-top:8px;">The model will update in real-time</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Prediction History ────────────────────────────────────────────────
        if st.session_state.prediction_history:
            st.markdown("---")
            st.markdown('<div class="section-title">🕑 Prediction History</div>', unsafe_allow_html=True)
            hist_cols = st.columns(min(5, len(st.session_state.prediction_history)))
            for i, item in enumerate(st.session_state.prediction_history[:5]):
                with hist_cols[i]:
                    conf_c = "#10b981" if item["confidence"] > 0.85 else "#f59e0b"
                    st.markdown(f"""
                    <div class="history-item" style="flex-direction:column;text-align:center;padding:14px 8px;">
                        <div class="history-digit-badge" style="width:48px;height:48px;font-size:1.6rem;margin:0 auto 8px auto;">
                            {item["digit"]}
                        </div>
                        <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">#{item["n"]}</div>
                        <div style="font-size:0.8rem;color:{conf_c};font-weight:700;margin-top:2px;">
                            {item["confidence"]*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # Tab 2 — Upload Image
    # ════════════════════════════════════════════════════════════════════════════
    with tab_upload:
        st.markdown('<div class="section-title">📷 Upload an Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a picture of a handwritten digit (dark digit on light bg is best)", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            col_img, col_pred = st.columns([1, 1], gap="large")
            with col_img:
                image = Image.open(uploaded_file).convert('RGBA')
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
            with col_pred:
                st.markdown('<div class="section-title">🎯 Prediction</div>', unsafe_allow_html=True)
                
                img_array = np.array(image)
                processed = preprocess_image(img_array, source="upload")
                
                if processed is not None:
                    preds            = model.predict(processed, verbose=0)[0]
                    predicted_digit  = int(np.argmax(preds))
                    confidence       = float(preds[predicted_digit])
                    prediction_probs = preds
                    
                    # Debug: Show the model's actual 28x28 input
                    st.markdown('<div style="color:rgba(255,255,255,0.4);font-size:0.7rem;">Model Input (28x28)</div>', unsafe_allow_html=True)
                    st.image(processed[0, :, :, 0], width=84, clamp=True)
                    
                    if f"uploaded_{uploaded_file.name}" not in st.session_state:
                        st.session_state[f"uploaded_{uploaded_file.name}"] = True
                        st.session_state.total_predictions += 1
                        st.session_state.prediction_history.insert(0, {
                            "digit":      predicted_digit,
                            "confidence": confidence,
                            "n":          st.session_state.total_predictions
                        })
                        if len(st.session_state.prediction_history) > 10:
                            st.session_state.prediction_history.pop()

                    conf_color = "#10b981" if confidence > 0.85 else ("#f59e0b" if confidence > 0.55 else "#ef4444")
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="predicted-digit">{predicted_digit}</div>
                        <div class="prediction-label">Predicted Digit</div>
                        <div class="confidence-text" style="color:{conf_color};">{confidence*100:.1f}% confident</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="section-title">📈 All Probabilities</div>', unsafe_allow_html=True)
                    sorted_idx = np.argsort(prediction_probs)[::-1]
                    bars_html  = "".join(
                        confidence_bar_html(int(i), float(prediction_probs[i]), i == predicted_digit)
                        for i in sorted_idx
                    )
                    st.markdown(bars_html, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                                height:280px;color:rgba(255,255,255,0.4);text-align:center;">
                        <div style="font-size:3rem;margin-bottom:16px;">⚠️</div>
                        <div style="font-size:1rem;font-weight:500;">No clear digit found</div>
                        <div style="font-size:0.8rem;margin-top:8px;">Ensure it's a dark digit on a clear background</div>
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
