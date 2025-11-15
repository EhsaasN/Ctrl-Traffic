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

# --- NEW IMPORTS FOR VIDEO ENDPOINT ---
import cv2
import httpx
import tempfile
import shutil
import asyncio
import math
# --- END NEW IMPORTS ---


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
VEHICLE_MODEL_PATH = "vehicle.h5"
TRAFFIC_MODEL_NAME = "prithivMLmods/Traffic-Density-Classification"
TARGET_SIZE = (224, 224)
SCALE = 1.0 / 255.0
EMERGENCY_THRESHOLD = 0.32  # prob >= this = emergency
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
# --- NEW VIDEO ENDPOINT AND HELPERS ---
# -------------------------------------------------------

async def _call_infer_endpoint(client: httpx.AsyncClient, frame_path: str, frame_timestamp: float) -> Dict:
    """Helper to call the /infer endpoint for a single frame file."""
    try:
        with open(frame_path, "rb") as f:
            files = {'file': (os.path.basename(frame_path), f, 'image/jpeg')}
            
            # Note: Assumes server is running on localhost:8000 (where this app runs)
            # Use 127.0.0.1 which is often more reliable than localhost for loopback
            response = await client.post("http://127.0.0.1:8000/infer", files=files, timeout=30.0)
            
            if response.status_code == 200:
                return {
                    "frame_timestamp_s": frame_timestamp,
                    "frame_file": os.path.basename(frame_path),
                    "status": "success",
                    "result": response.json()
                }
            else:
                return {
                    "frame_timestamp_s": frame_timestamp,
                    "frame_file": os.path.basename(frame_path),
                    "status": "error",
                    "detail": response.text
                }
    except Exception as e:
        return {
            "frame_timestamp_s": frame_timestamp,
            "frame_file": os.path.basename(frame_path),
            "status": "exception",
            "detail": str(e)
        }


@app.post("/infer-video")
async def infer_video(file: UploadFile = File(...)):
    """
    POST a video file. This endpoint will:
    1. Save the video temporarily.
    2. Extract one frame every 5 seconds.
    3. Send each frame to the /infer endpoint concurrently.
    4. Return a collected list of results.
    """
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, file.filename)
    
    try:
        # 1. Save video file temporarily
        try:
            contents = await file.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Uploaded video file is empty")
            with open(video_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded video: {e}")

        # 2. Open video and get properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file. Invalid format?")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            # Fallback or error if FPS is 0 (e.g., for some streamed formats)
            print("Warning: Video FPS reported as 0. Defaulting to 25 FPS.")
            fps = 25 # Use a reasonable default
        
        frame_interval = int(fps * 5) # 5 seconds
        if frame_interval == 0: 
            frame_interval = 1 # Avoid division by zero if video is < 1s

        # 3. Extract frames
        frame_files = [] # List of tuples: (path, timestamp_in_seconds)
        frame_count = 0
        
        print(f"Processing video: {file.filename} (FPS: {fps}, Frame Interval: {frame_interval})")

        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            # Check if this is a frame we want to save
            if frame_count % frame_interval == 0:
                timestamp_s = round(frame_count / fps, 2)
                frame_filename = f"frame_at_{timestamp_s}s.jpg"
                frame_path = os.path.join(temp_dir, frame_filename)
                
                if cv2.imwrite(frame_path, frame):
                    frame_files.append((frame_path, timestamp_s))
                else:
                    print(f"Warning: Could not write frame {frame_filename}")

            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(frame_files)} frames to process.")

        if not frame_files:
            return {"message": "Video processed, but no frames were extracted (video might be shorter than 5 seconds).", "results": []}

        # 4. Call /infer for each frame concurrently
        tasks = []
        async with httpx.AsyncClient() as client:
            for path, timestamp in frame_files:
                tasks.append(_call_infer_endpoint(client, path, timestamp))
            
            results = await asyncio.gather(*tasks)
        
        return {"message": f"Processed {len(results)} frames from the video.", "results": results}

    except HTTPException:
        raise # Re-raise known HTTP exceptions
    except Exception as e:
        traceback.print_exc() # Log the full error
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    finally:
        # 5. Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# -------------------------------------------------------
# --- END NEW VIDEO ENDPOINT ---
# -------------------------------------------------------


# -------------------------------------------------------
# AUDIO MODEL LOADING + HELPERS (Existing Code)
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
# HEALTH & AUDIO ENDPOINTS (Existing Code)
# -------------------------------------------------------
@app.get("/health")
def health():
    return {
        "vehicle_model_loaded": True,
        "traffic_model_loaded": True,
        "audio_model_loaded": bool(audio_model_loaded),
        "audio_load_error": audio_load_error
    }

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