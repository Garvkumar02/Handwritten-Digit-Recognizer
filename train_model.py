"""
Advanced MNIST CNN Training Script
===================================
Deep Convolutional Neural Network for handwritten digit recognition.
Features:
  - Multi-block CNN with BatchNormalization and Dropout
  - Data augmentation (rotation, zoom, shifts)
  - Cosine annealing learning rate schedule
  - Label smoothing loss
  - Confusion matrix and accuracy/loss plots
  - Saves model to models/mnist_cnn.keras
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    LearningRateScheduler
)

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── Constants ───────────────────────────────────────────────────────────────
NUM_CLASSES    = 10
IMG_SIZE       = 28
BATCH_SIZE     = 128
EPOCHS         = 30
LABEL_SMOOTHING = 0.1

os.makedirs("models",  exist_ok=True)
os.makedirs("assets",  exist_ok=True)

print("=" * 60)
print("  Advanced MNIST CNN Training")
print("=" * 60)

# ─── 1. Load & Preprocess Data ───────────────────────────────────────────────
print("\n[1/6] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape → (N, 28, 28, 1) and normalize to [0, 1]
x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0

# One-hot encode labels
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

print(f"  Train: {x_train.shape[0]} samples")
print(f"  Test : {x_test.shape[0]} samples")

# ─── 2. Data Augmentation ────────────────────────────────────────────────────
print("\n[2/6] Setting up data augmentation pipeline...")
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    shear_range=0.05,
    fill_mode='nearest',
    validation_split=0.1
)
datagen.fit(x_train)

train_gen = datagen.flow(
    x_train, y_train_cat,
    batch_size=BATCH_SIZE,
    subset='training',
    seed=SEED
)
val_gen = datagen.flow(
    x_train, y_train_cat,
    batch_size=BATCH_SIZE,
    subset='validation',
    seed=SEED
)

# ─── 3. Build CNN Model ───────────────────────────────────────────────────────
print("\n[3/6] Building deep CNN architecture...")

def build_model(input_shape=(28, 28, 1), num_classes=10):
    inputs = keras.Input(shape=input_shape, name="input_image")

    # ── Convolutional Block 1 ──────────────────────────────────────────
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu',
                      kernel_initializer='he_normal', name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu',
                      kernel_initializer='he_normal', name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.MaxPooling2D((2,2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # ── Convolutional Block 2 ──────────────────────────────────────────
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu',
                      kernel_initializer='he_normal', name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu',
                      kernel_initializer='he_normal', name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.MaxPooling2D((2,2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # ── Convolutional Block 3 ──────────────────────────────────────────
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu',
                      kernel_initializer='he_normal', name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu',
                      kernel_initializer='he_normal', name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.MaxPooling2D((2,2), name="pool3")(x)
    x = layers.Dropout(0.40, name="drop3")(x)

    # ── Dense Head ────────────────────────────────────────────────────
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation='relu',
                     kernel_initializer='he_normal', name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.50, name="drop_dense")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MNIST_CNN")
    return model

model = build_model()
model.summary()

# ─── 4. Compile ──────────────────────────────────────────────────────────────
print("\n[4/6] Compiling model...")

def cosine_annealing_schedule(epoch, lr, epochs=EPOCHS, min_lr=1e-6, max_lr=1e-3):
    """Cosine annealing learning rate schedule."""
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * epoch / epochs))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint(
        "models/mnist_cnn.keras",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    LearningRateScheduler(cosine_annealing_schedule, verbose=0)
]

# ─── 5. Train ─────────────────────────────────────────────────────────────────
print("\n[5/6] Training model...")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# ─── 6. Evaluate & Visualize ─────────────────────────────────────────────────
print("\n[6/6] Evaluating model and generating plots...")

# Load best saved model
best_model = keras.models.load_model("models/mnist_cnn.keras")
test_loss, test_acc = best_model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\n  ✓ Test Accuracy : {test_acc*100:.2f}%")
print(f"  ✓ Test Loss     : {test_loss:.4f}")

# Predictions
y_pred_probs = best_model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# ── A) Accuracy & Loss Plot ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#c9d1d9')
    ax.xaxis.label.set_color('#c9d1d9')
    ax.yaxis.label.set_color('#c9d1d9')
    ax.title.set_color('#e6edf3')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# Accuracy
accuracy_history = history.history.get('accuracy', history.history.get('acc', []))
val_accuracy_history = history.history.get('val_accuracy', history.history.get('val_acc', []))
epochs_range = range(1, len(accuracy_history) + 1)

axes[0].plot(epochs_range, accuracy_history,     color='#58a6ff', linewidth=2.5, label='Train Accuracy', marker='o', markersize=4)
axes[0].plot(epochs_range, val_accuracy_history, color='#3fb950', linewidth=2.5, label='Val Accuracy',   marker='s', markersize=4, linestyle='--')
axes[0].set_title(f'Model Accuracy (Best: {max(val_accuracy_history)*100:.2f}%)', fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(framealpha=0.3, facecolor='#30363d', edgecolor='#484f58', labelcolor='#e6edf3')
axes[0].grid(True, alpha=0.2, color='#30363d')

# Loss
axes[1].plot(epochs_range, history.history['loss'],     color='#f78166', linewidth=2.5, label='Train Loss', marker='o', markersize=4)
axes[1].plot(epochs_range, history.history['val_loss'], color='#ffa657', linewidth=2.5, label='Val Loss',   marker='s', markersize=4, linestyle='--')
axes[1].set_title(f'Model Loss (Final: {test_loss:.4f})', fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(framealpha=0.3, facecolor='#30363d', edgecolor='#484f58', labelcolor='#e6edf3')
axes[1].grid(True, alpha=0.2, color='#30363d')

plt.tight_layout(pad=3.0)
plt.savefig('assets/accuracy_loss_plot.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  ✓ Saved: assets/accuracy_loss_plot.png")

# ── B) Confusion Matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

mask = np.eye(10, dtype=bool)
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            ax=ax, linewidths=0.5, linecolor='#30363d',
            cbar_kws={'label': 'Prediction %'},
            annot_kws={'size': 11, 'weight': 'bold'})

# Highlight errors
error_cm = cm_pct.copy()
np.fill_diagonal(error_cm, 0)

ax.set_title(f'Confusion Matrix — Test Accuracy: {test_acc*100:.2f}%',
             fontsize=16, fontweight='bold', color='#e6edf3', pad=20)
ax.set_xlabel('Predicted Label', fontsize=13, color='#c9d1d9', labelpad=10)
ax.set_ylabel('True Label',      fontsize=13, color='#c9d1d9', labelpad=10)
ax.tick_params(colors='#c9d1d9', labelsize=12)
ax.set_xticklabels(range(10), fontsize=12, color='#c9d1d9')
ax.set_yticklabels(range(10), fontsize=12, color='#c9d1d9', rotation=0)

cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_color('#c9d1d9')
cbar.ax.tick_params(colors='#c9d1d9')

plt.tight_layout()
plt.savefig('assets/confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  ✓ Saved: assets/confusion_matrix.png")

# ── C) Classification Report ─────────────────────────────────────────────────
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# Save training metadata
import json
metadata = {
    "test_accuracy":  float(test_acc),
    "test_loss":      float(test_loss),
    "best_val_acc":   float(max(val_accuracy_history)),
    "epochs_trained": len(accuracy_history),
    "architecture": {
        "blocks": 3,
        "filters": [32, 64, 128],
        "dense_units": 256,
        "dropout_rates": [0.25, 0.25, 0.40, 0.50],
        "optimizer": "Adam",
        "label_smoothing": LABEL_SMOOTHING
    }
}
with open("assets/training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("  ✓ Saved: assets/training_metadata.json")

print("\n" + "=" * 60)
print(f"  Training Complete!")
print(f"  Final Test Accuracy: {test_acc*100:.2f}%")
print(f"  Model saved to:      models/mnist_cnn.keras")
print("=" * 60)
print("\n  Run the app with:")
print("  streamlit run app.py")
