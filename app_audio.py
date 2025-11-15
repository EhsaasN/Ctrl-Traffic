import os
import io
import pickle
import traceback
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# For traffic model
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# For audio endpoint
import librosa
import soundfile as sf

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

# Audio model config (update filenames if different)
AUDIO_MODEL_PATH = "audio_cnn.h5"            # your trained audio model
LABEL_ENCODER_PATHS = ["labelencoder.pkl", "label_encoder.pkl"]
# Feature extraction params (match training)
N_MFCC = 80
RES_TYPE = "kaiser_fast"
DEFAULT_AUDIO_LABELS = ["firetruck", "ambulance", "road_noise"]

# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------

def load_vehicle_model():
    try:
        return tf.keras.models.load_model(VEHICLE_MODEL_PATH)
    except Exception:
        try:
            import tensorflow_hub as hub
            return tf.keras.models.load_model(
                VEHICLE_MODEL_PATH,
                custom_objects={"KerasLayer": hub.KerasLayer}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load vehicle model: {e}")

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
# HELPERS (image models)
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

app = FastAPI(title="Integrated Traffic & Audio Controller API")

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
    except Exception:
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
# AUDIO MODEL LOADING + HELPERS (added endpoint)
# -------------------------------------------------------
audio_model = None
audio_labels: List[str] = DEFAULT_AUDIO_LABELS.copy()
audio_model_loaded = False
audio_load_error: Optional[str] = None

def try_load_label_encoder() -> Optional[List[str]]:
    for p in LABEL_ENCODER_PATHS:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    le = pickle.load(f)
                if hasattr(le, "classes_"):
                    return [str(x) for x in le.classes_]
            except Exception:
                pass
    return None

def load_audio_model(path: str):
    try:
        m = tf.keras.models.load_model(path)
        return m
    except Exception:
        try:
            import tensorflow_hub as hub
            m = tf.keras.models.load_model(path, custom_objects={"KerasLayer": hub.KerasLayer})
            return m
        except Exception as e:
            raise RuntimeError(f"Audio model load failed: {e}")

def extract_features_from_bytes(wav_bytes: bytes) -> np.ndarray:
    """
    Use soundfile (sf) to read bytes and compute MFCC mean (n_mfcc=N_MFCC),
    then reshape to (1, N_MFCC, 1) to match training.
    """
    # load audio from bytes using soundfile
    try:
        audio, sr = sf.read(io.BytesIO(wav_bytes))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
    except Exception:
        # fallback to librosa (it will handle various formats)
        audio, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, res_type=RES_TYPE)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfccs_mean = np.mean(mfccs.T, axis=0)  # shape (N_MFCC,)
    arr = mfccs_mean.reshape(1, -1, 1).astype(np.float32)  # (1,80,1)
    return arr

def predict_audio_from_bytes(wav_bytes: bytes) -> Dict:
    """
    Returns dict: {"predictions": {label:prob,...}, "top_label":label, "top_prob":float}
    """
    global audio_model, audio_labels
    if audio_model is None:
        raise RuntimeError("Audio model not loaded")

    X = extract_features_from_bytes(wav_bytes)  # (1,80,1)
    preds = audio_model.predict(X)
    preds = np.asarray(preds).squeeze()
    # ensure probabilities
    if preds.ndim == 0:
        probs = np.array([preds])
    else:
        probs = np.array(preds)

    # normalize if necessary (softmax)
    if not np.all((probs >= 0.0) & (probs <= 1.0)) or not np.isclose(probs.sum(), 1.0, atol=1e-3):
        e = np.exp(probs - np.max(probs))
        probs = e / e.sum()

    # map to labels
    L = min(len(probs), len(audio_labels))
    probs_map = {audio_labels[i]: float(probs[i]) for i in range(L)}
    # pick top
    if probs_map:
        top_label, top_prob = max(probs_map.items(), key=lambda x: x[1])
    else:
        top_label, top_prob = None, None

    return {"predictions": probs_map, "top_label": top_label, "top_prob": top_prob}

# Startup load audio model (non-blocking behavior is fine; we load at startup)
def startup_load_audio():
    global audio_model, audio_labels, audio_model_loaded, audio_load_error
    if not os.path.exists(AUDIO_MODEL_PATH):
        audio_load_error = f"Audio model file not found at '{AUDIO_MODEL_PATH}'. Please place your audio model there."
        audio_model_loaded = False
        print(audio_load_error)
        return

    try:
        audio_model = load_audio_model(AUDIO_MODEL_PATH)
    except Exception as e:
        audio_load_error = f"Failed to load audio model: {e}\n{traceback.format_exc()}"
        audio_model_loaded = False
        print(audio_load_error)
        return

    le_labels = try_load_label_encoder()
    if le_labels:
        audio_labels = le_labels
    else:
        audio_labels = DEFAULT_AUDIO_LABELS.copy()

    audio_model_loaded = True
    print("Audio model loaded and ready. Labels:", audio_labels)

# call startup
startup_load_audio()

# -------------------------------------------------------
# AUDIO ENDPOINT
# -------------------------------------------------------
@app.get("/health")
def health():
    return {
        "vehicle_model_loaded": True,
        "traffic_model_loaded": True,
        "audio_model_loaded": bool(audio_model_loaded),
        "audio_load_error": audio_load_error
    }

# -------------------------------------------------------
# AUDIO ENDPOINT (updated: includes green_time logic)
# -------------------------------------------------------
@app.post("/infer-audio")
async def infer_audio(file: UploadFile = File(...)):
    """
    POST an audio file (wav) as multipart/form-data 'file' to run inference.
    Returns JSON: filename, top_label, top_prob, predictions dict, green_time
    """
    if not audio_model_loaded:
        raise HTTPException(status_code=503, detail={"audio_model_loaded": False, "error": audio_load_error})

    # read bytes
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {e}")

    try:
        # run audio model
        res = predict_audio_from_bytes(contents)
        top_label = res.get("top_label")
        top_prob = res.get("top_prob")

        # Decision: if ambulance/firetruck -> emergency green override (120)
        if top_label in ("ambulance", "firetruck"):
            green_time = EMERGENCY_GREEN_TIME
            reason = "emergency_audio_detected"
        else:
            # Non-emergency audio (e.g. road_noise) -> treat as no immediate emergency
            # Here we choose to return 0 green time (no override). If you prefer a different mapping,
            # replace this with logic to call traffic model or other heuristics.
            green_time = 0
            reason = "non_emergency_audio"

        return {
            "filename": file.filename,
            "top_label": top_label,
            "top_prob": top_prob,
            "predictions": res.get("predictions"),
            "green_time": green_time,
            "decision_reason": reason
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {e}\n{traceback_str}")

# -------------------------------------------------------
# AUTO START FASTAPI ON PORT 8000
# -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting FastAPI server at http://0.0.0.0:8000 ...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
