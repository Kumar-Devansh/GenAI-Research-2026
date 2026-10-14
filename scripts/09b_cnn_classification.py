"""
STEP 8B: COMPUTER VISION - CNN FOR SKY/CLOUD CLASSIFICATION
===========================================================

This script trains a Convolutional Neural Network (CNN) to classify
sky images into 3 categories:

* clear_sky
* partly_cloudy
* heavy_clouds

What does a CNN do? (Simple explanation)

* A normal Neural Network looks at each pixel of an image independently.
* A CNN uses filters/kernels that scan small patches (such as 3x3
  pixels). This helps it recognize SHAPES, EDGES, and TEXTURES
  (such as cloud boundaries and texture patterns).
* Therefore, CNNs are much better suited for image classification
  than standard Neural Networks.
  """


from common import DATA_DIR, MODEL_DIR, OUTPUT_DIR, prepare_dirs
import numpy as np
import pandas as pd
import os
import glob
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

# ---- ROBUST PATH HANDLING ----
IMG_DIR = DATA_DIR / "sky_images"
prepare_dirs(OUTPUT_DIR, MODEL_DIR)

CATEGORIES = ["clear_sky", "partly_cloudy", "heavy_clouds"]
IMG_SIZE = 64

# ---- IMAGES LOAD  ----
print("Loading images...")
X, y = [], []
for label_idx, category in enumerate(CATEGORIES):
    filepaths = sorted(glob.glob(os.path.join(IMG_DIR, category, "*.png")))
    for fp in filepaths:
        img = cv2.imread(fp)
        if img is None:
            print(f"Skipping unreadable image: {fp}")
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label_idx)

if not X or len(set(y)) != len(CATEGORIES):
    raise ValueError(f"Expected readable images for all classes under {IMG_DIR}")
X = np.array(X, dtype=np.float32) / 255.0  # Normalize: 0-255 -> 0-1
y = np.array(y)

print(f"Total images loaded: {len(X)}")
print(f"Image shape: {X[0].shape}")

# ---- TRAIN-TEST SPLIT ----
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1875, random_state=42, stratify=y_trainval
)  # ~15% validation of the complete data

y_train_cat = to_categorical(y_train, num_classes=len(CATEGORIES))
y_val_cat = to_categorical(y_val, num_classes=len(CATEGORIES))
y_test_cat = to_categorical(y_test, num_classes=len(CATEGORIES))

print(f"Training images: {len(X_train)}, Testing images: {len(X_test)}")

# ---- Create CNN Model ----
model = Sequential([
    # Layer 1: 32 filters, 3x3 size - detect basic edges/textures 
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),  # Reduces the image size by half while preserving important features

    # Layer 2: 64 filters - more complex patterns (cloud shapes)
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    # Layer 3: 128 filters - even more complex patterns
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),  # convert 2D image data into 1D 
    Dense(128, activation="relu"),
    Dropout(0.3),  # Helps prevent overfitting by randomly turning off 30% of neurons during training
    Dense(len(CATEGORIES), activation="softmax"),  # 3 probabilities of classes (clear, partly, heavy)
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("\nModel Architecture:")
model.summary()

# ---- TRAINING ----
print("\nTraining CNN...")
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    verbose=0,
)
print(f"Training stopped at epoch {len(history.history['loss'])} (early stopping)")

# ---- EVALUATION ----
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"Test Loss: {test_loss:.4f}")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

# ---- SAVE MODEL ----
model.save(MODEL_DIR / "cnn_sky_classifier.keras")
pd.DataFrame([{"Accuracy": float(test_acc), "Test images": len(y_test), "Classes": len(CATEGORIES)}]).to_csv(
    OUTPUT_DIR / "cnn_classification_comparison.csv", index=False
)
print(f"Model saved: {MODEL_DIR / 'cnn_sky_classifier.keras'}")

# =========================================================
# VISUALIZATION
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Training curves
axes[0, 0].plot(history.history["accuracy"], label="Train Accuracy")
axes[0, 0].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[0, 0].set_title("CNN Training Accuracy")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].legend()

axes[0, 1].plot(history.history["loss"], label="Train Loss")
axes[0, 1].plot(history.history["val_loss"], label="Validation Loss")
axes[0, 1].set_title("CNN Training Loss")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].legend()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[1, 0].imshow(cm, cmap="Blues")
axes[1, 0].set_xticks(range(len(CATEGORIES)))
axes[1, 0].set_yticks(range(len(CATEGORIES)))
axes[1, 0].set_xticklabels(CATEGORIES, rotation=30)
axes[1, 0].set_yticklabels(CATEGORIES)
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")
axes[1, 0].set_title("Confusion Matrix")
for i in range(len(CATEGORIES)):
    for j in range(len(CATEGORIES)):
        axes[1, 0].text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")

# 3. Sample predictions
axes[1, 1].axis("off")
sample_idx = np.random.choice(len(X_test), 6, replace=False)
sample_text = "Sample Predictions:\n\n"
for idx in sample_idx:
    actual = CATEGORIES[y_test[idx]]
    predicted = CATEGORIES[y_pred[idx]]
    correct = "✓" if actual == predicted else "✗"
    sample_text += f"{correct} Actual: {actual:15s} Predicted: {predicted}\n"
axes[1, 1].text(0.05, 0.9, sample_text, transform=axes[1, 1].transAxes,
                 fontsize=11, va="top", family="monospace")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cnn_classification_results.png", dpi=150, bbox_inches="tight")
print(f"\nGraphs saved: {OUTPUT_DIR / 'cnn_classification_results.png'}")

print("""
NOTE FOR YOUR RESEARCH PAPER:

-This CNN model was trained on SYNTHETIC images instead of real
  satellite images such as Sentinel-2/MODIS, because downloading real
  data requires Google Earth Engine authentication.
- The METHODOLOGY will remain the same with real data; only the image
  folder path needs to be changed to the real satellite tiles.
- This is a limitation that should be clearly mentioned in the paper.
  """)

