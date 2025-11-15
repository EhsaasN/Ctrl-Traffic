#!/usr/bin/env python3
"""
run_both_models.py

Integrates:
 - TensorFlow emergency classifier (vehicle.h5)
 - HuggingFace / Transformers traffic density classifier (prithivMLmods/Traffic-Density-Classification)

Reads images from ./test_images and prints per-image:
  - vehicle result (probability)
  - if not emergency: traffic density label + probability
  - final green_time decision (seconds)

Requirements:
  pip install tensorflow torch transformers pillow
  (install tensorflow-hub if your .h5 needs it)
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image
import tensorflow as tf

# Transformer/torch imports deferred to function to allow graceful fallback
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
VEHICLE_MODEL_PATH = "vehicle.h5"
TRAFFIC_MODEL_NAME = "prithivMLmods/Traffic-Density-Classification"  # HF id from your code
TEST_FOLDER = Path("test_images")
TARGET_SIZE = (224, 224)        # for the TF vehicle classifier
SCALE = 1.0 / 255.0
EMERGENCY_THRESHOLD = 0.25      # you used 0.25 in your snippet; change if desired
EMERGENCY_GREEN_TIME = 120      # override green time when emergency detected
# Traffic label -> green time (as requested)
TRAFFIC_GREEN_MAPPING = {
    "high-traffic": 80,
    "medium-traffic": 60,
    "low-traffic": 30,
    "no-traffic": 0
}

# Label mapping used by the traffic model (from your FastAPI code)
TRAFFIC_LABELS = {0: "high-traffic", 1: "low-traffic", 2: "medium-traffic", 3: "no-traffic"}


# -----------------------------------------------------------------------------
# Vehicle classifier helpers (TF)
# -----------------------------------------------------------------------------
def load_vehicle_model(path: str) -> tf.keras.Model:
    try:
        model = tf.keras.models.load_model(path)
        print(f"Loaded vehicle model from {path} (default loader).")
        return model
    except Exception as e:
        # try TF-Hub fallback if user used hub.KerasLayer when saving / building
        try:
            import tensorflow_hub as hub  # may raise ImportError
            model = tf.keras.models.load_model(path, custom_objects={"KerasLayer": hub.KerasLayer})
            print(f"Loaded vehicle model from {path} with tensorflow_hub.KerasLayer fallback.")
            return model
        except Exception as e2:
            raise RuntimeError(f"Failed to load vehicle model:\n default error: {e}\n fallback error: {e2}")

def preprocess_for_vehicle(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = img.resize(TARGET_SIZE)
    arr = np.asarray(img, dtype=np.float32) * SCALE
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr

def predict_vehicle_prob(model: tf.keras.Model, img_path: Path) -> float:
    x = preprocess_for_vehicle(img_path)
    preds = model.predict(x)
    preds = np.asarray(preds)
    # handle common shapes:
    # - (1,1) or (1,) -> sigmoid prob
    # - (1,2) -> softmax -> use index 1 as emergency prob
    if preds.ndim == 2 and preds.shape[1] == 2:
        prob = float(preds[0, 1])
    else:
        # scalar output(s)
        prob = float(np.squeeze(preds))
        # If preds are logits outside [0,1], apply sigmoid (but your model usually returns 0..1)
        if not (0.0 <= prob <= 1.0):
            prob = 1.0 / (1.0 + np.exp(-prob))
    return prob


# -----------------------------------------------------------------------------
# Traffic density model helpers (transformers + torch)
# -----------------------------------------------------------------------------
def try_load_traffic_model(model_name: str):
    """
    Returns (processor, model, device) or raises an exception.
    """
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch
    except Exception as e:
        raise RuntimeError(f"Missing required packages for traffic model. Install 'transformers' and 'torch'. Error: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading traffic model processor and weights from:", model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = None
    # Prefer AutoModelForImageClassification; some models may need special classes but try this first
    try:
        model = AutoModelForImageClassification.from_pretrained(model_name)
    except Exception as e:
        # fallback to AutoModel if necessary
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name)
        print("Loaded generic AutoModel (outputs may differ).")

    model.to(device)
    model.eval()
    print("Traffic model loaded onto device:", device)
    return processor, model, device

def predict_traffic(processor, model, device, pil_img: Image.Image) -> Dict[str, float]:
    """
    Given a PIL.Image image, returns a dict mapping label->probability for the traffic model.
    """
    import torch
    # prepare
    inputs = processor(images=pil_img, return_tensors="pt")
    # move tensors to device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            try:
                inputs[k] = v.to(device)
            except Exception:
                pass
    with torch.no_grad():
        outputs = model(**inputs)
        # extract logits (common attribute for HF classification models)
        logits = None
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            logits = outputs[0]
        else:
            raise RuntimeError("Traffic model outputs do not contain logits; cannot compute probabilities.")
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().tolist()
        if isinstance(probs, float):
            probs = [probs]
        # map to labels according to TRAFFIC_LABELS
        preds = {TRAFFIC_LABELS[i]: float(probs[i]) for i in range(len(probs))}
        return preds


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------
def run_all():
    # load vehicle model
    if not Path(VEHICLE_MODEL_PATH).exists():
        raise SystemExit(f"vehicle model file not found at: {VEHICLE_MODEL_PATH}")
    vehicle_model = load_vehicle_model(VEHICLE_MODEL_PATH)

    # attempt to load traffic model; if it fails we'll still run vehicle classifier only
    traffic_loaded = False
    processor = model = device = None
    try:
        processor, model, device = try_load_traffic_model(TRAFFIC_MODEL_NAME)
        traffic_loaded = True
    except Exception as e:
        print(f"Warning: failed to load traffic model ({e}). Continuing with vehicle-only predictions.")
        traffic_loaded = False

    # check test folder
    if not TEST_FOLDER.exists():
        print(f"Create a folder named '{TEST_FOLDER}' and add images (jpg/png).")
        return

    images = sorted([p for p in TEST_FOLDER.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not images:
        print(f"No images found in '{TEST_FOLDER}'.")
        return

    # Process images
    for img_path in images:
        try:
            # 1) vehicle classifier
            vprob = predict_vehicle_prob(vehicle_model, img_path)
            is_emergency = (vprob >= EMERGENCY_THRESHOLD)

            if is_emergency:
                # Emergency detected: set override green time
                green_time = EMERGENCY_GREEN_TIME
                traffic_info = None
                print(f"{img_path.name}  -->  EMERGENCY (vehicle_prob={vprob:.4f})  -->  green_time={green_time} (EMERGENCY OVERRIDE)")
            else:
                # Not emergency: call traffic model if available
                if not traffic_loaded:
                    # fallback if traffic model unavailable
                    green_time = None
                    print(f"{img_path.name}  -->  not_emergency (vehicle_prob={vprob:.4f})  -->  traffic model unavailable, no green_time computed")
                else:
                    # load PIL image for processor
                    pil = Image.open(img_path).convert("RGB")
                    preds = predict_traffic(processor, model, device, pil)  # dict label->prob
                    # Determine top label
                    top_label = max(preds.items(), key=lambda x: x[1])[0]
                    top_prob = preds[top_label]
                    # Map top_label to green_time using mapping (fall back to 0 if unknown)
                    green_time = TRAFFIC_GREEN_MAPPING.get(top_label, 0)
                    print(f"{img_path.name}  -->  not_emergency (vehicle_prob={vprob:.4f})  -->  traffic={top_label} (prob={top_prob:.4f})  -->  green_time={green_time}")
        except Exception as e:
            print(f"{img_path.name}  -->  ERROR during processing: {e}")


if __name__ == "__main__":
    run_all()
