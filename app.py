import os
import numpy as np
from pathlib import Path
from typing import Dict
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# For traffic model
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
VEHICLE_MODEL_PATH = "vehicle.h5"
TRAFFIC_MODEL_NAME = "prithivMLmods/Traffic-Density-Classification"
TARGET_SIZE = (224, 224)
SCALE = 1.0 / 255.0
EMERGENCY_THRESHOLD = 0.3   # prob >= this = emergency
EMERGENCY_GREEN_TIME = 120

TRAFFIC_LABELS = {
    0: "high-traffic",
    1: "low-traffic",
    2: "medium-traffic",
    3: "no-traffic"
}

TRAFFIC_GREEN_MAPPING = {
    "high-traffic": 80,
    "medium-traffic": 60,
    "low-traffic": 30,
    "no-traffic": 0
}

# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------

def load_vehicle_model():
    try:
        return tf.keras.models.load_model(VEHICLE_MODEL_PATH)
    except:
        import tensorflow_hub as hub
        return tf.keras.models.load_model(
            VEHICLE_MODEL_PATH,
            custom_objects={"KerasLayer": hub.KerasLayer}
        )

print("🚗 Loading vehicle emergency model...")
vehicle_model = load_vehicle_model()
print("Vehicle model loaded.")

print("🚦 Loading traffic density model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(TRAFFIC_MODEL_NAME)
traffic_model = AutoModelForImageClassification.from_pretrained(TRAFFIC_MODEL_NAME).to(device)
traffic_model.eval()
print("Traffic model loaded.")

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def preprocess_vehicle(img: Image.Image) -> np.ndarray:
    img = img.resize(TARGET_SIZE)
    arr = np.asarray(img, dtype=np.float32) * SCALE
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_vehicle(img: Image.Image) -> float:
    x = preprocess_vehicle(img)
    pred = vehicle_model.predict(x)
    prob = float(np.squeeze(pred))
    if not (0 <= prob <= 1):
        prob = 1.0 / (1.0 + np.exp(-prob))
    return prob

def predict_traffic(img: Image.Image) -> Dict[str, float]:
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = traffic_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()

    return {TRAFFIC_LABELS[i]: probs[i] for i in range(4)}

# -------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------

app = FastAPI(title="Integrated Traffic Controller API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
    except:
        raise HTTPException(400, "Invalid image file")

    # --- Step 1: Emergency Classification ---
    vehicle_prob = predict_vehicle(img)
    is_emergency = vehicle_prob >= EMERGENCY_THRESHOLD

    if is_emergency:
        return {
            "vehicle_prob": vehicle_prob,
            "vehicle_label": "emergency",
            "traffic_label": None,
            "green_time": EMERGENCY_GREEN_TIME
        }

    # --- Step 2: Traffic Density Classification ---
    traffic_probs = predict_traffic(img)

    top_label = max(traffic_probs, key=traffic_probs.get)
    green_time = TRAFFIC_GREEN_MAPPING.get(top_label, 0)

    return {
        "vehicle_prob": vehicle_prob,
        "vehicle_label": "not_emergency",
        "traffic_label": top_label,
        "traffic_probabilities": traffic_probs,
        "green_time": green_time
    }

# -------------------------------------------------------
# AUTO START FASTAPI ON PORT 8000
# -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting FastAPI server at http://0.0.0.0:8000 ...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
