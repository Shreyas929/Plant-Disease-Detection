import tensorflow as tf
import numpy as np
import json
import os
import gradio as gr
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input  # ✅ EfficientNet preprocessing

# --- CONFIG ---
IMG_SIZE = 260
MODEL_PATH = "efficientnet.h5"
LABEL_PATH = "class_names.json"  # Assumes class labels are stored as a list

# --- LOAD MODEL ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# --- LOAD LABELS ---
if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(f"❌ Label file not found: {LABEL_PATH}")
with open(LABEL_PATH, "r") as f:
    class_indices = json.load(f)

# ✅ Convert list to index-to-label mapping
index_to_label = {i: label for i, label in enumerate(class_indices)}

# --- PREDICTION FUNCTION ---
def predict_leaf_disease_from_image(img):
    if img is None:
        return "❌ No image uploaded."

    # Preprocess the image
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # EfficientNet preprocessing

    # Predict
    prediction = model.predict(img_array)[0]

    # Top-3 predictions
    top3_indices = prediction.argsort()[-3:][::-1]
    result_str = "🏆 Top 3 Predictions:\n"
    for i in top3_indices:
        label = index_to_label.get(i, f"class_{i}")
        confidence = prediction[i] * 100
        result_str += f"{label}: {confidence:.2f}%\n"

    # Final prediction
    predicted_index = top3_indices[0]
    predicted_label = index_to_label[predicted_index]
    confidence = prediction[predicted_index] * 100

    result_str += f"\n✅ Predicted: {predicted_label} ({confidence:.2f}% confidence)"
    return result_str

# --- GRADIO INTERFACE ---
iface = gr.Interface(
    fn=predict_leaf_disease_from_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🍃 Plant Leaf Disease Classifier",
    description="Upload a leaf image to classify the plant disease using an EfficientNet model. Only top 3 predictions will be shown."
)

iface.launch()
