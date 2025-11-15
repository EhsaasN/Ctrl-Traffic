import os
import traceback
import base64
from io import BytesIO
from typing import Optional, Dict

import asyncio
import torch
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Deferred HF imports (allow clearer errors)
MODEL_NAME = os.getenv("MODEL_NAME", "prithivMLmods/Traffic-Density-Classification")

app = FastAPI(title="TRAFFIX ML Worker (FastAPI)")

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev ONLY
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Runtime device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Runtime device:", device)

# Globals to hold model & processor
processor = None
model = None
_model_loaded = False

# Labels (from model card)
LABELS = {0: "high-traffic", 1: "low-traffic", 2: "medium-traffic", 3: "no-traffic"}

# Async httpx client for downloads
httpx_client = httpx.AsyncClient(timeout=30.0)

def try_import_transformers():
    try:
        from transformers import AutoImageProcessor
        try:
            from transformers import SiglipForImageClassification as HF_SIGLIP
        except Exception:
            HF_SIGLIP = None
        try:
            from transformers import AutoModelForImageClassification
        except Exception:
            AutoModelForImageClassification = None
        return AutoImageProcessor, HF_SIGLIP, AutoModelForImageClassification
    except Exception as e:
        print("transformers import failed:", e)
        raise

def load_model_and_processor():
    global processor, model, _model_loaded
    try:
        AutoImageProcessor, HF_SIGLIP, AutoModelForImageClassification = try_import_transformers()
        print("Loading processor from:", MODEL_NAME)
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        print("Processor loaded.")
        loaded = False
        # Try Siglip specialized class first
        if HF_SIGLIP is not None:
            try:
                print("Trying SiglipForImageClassification.from_pretrained...")
                model = HF_SIGLIP.from_pretrained(MODEL_NAME)
                loaded = True
                print("Loaded via SiglipForImageClassification.")
            except Exception as e:
                print("Siglip load failed:", e)
        # Try AutoModelForImageClassification
        if (not loaded) and (AutoModelForImageClassification is not None):
            try:
                print("Trying AutoModelForImageClassification.from_pretrained...")
                model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
                loaded = True
                print("Loaded via AutoModelForImageClassification.")
            except Exception as e:
                print("AutoModelForImageClassification failed:", e)
        # Fallback to AutoModel
        if not loaded:
            from transformers import AutoModel
            print("Trying generic AutoModel.from_pretrained...")
            model = AutoModel.from_pretrained(MODEL_NAME)
            loaded = True
            print("Loaded via AutoModel (outputs may vary).")
        # Move to device
        if model is not None:
            try:
                model.to(device)
            except Exception as e:
                print("Could not move model to device:", e)
        _model_loaded = loaded and (model is not None) and (processor is not None)
        print("Model loaded flag:", _model_loaded)
    except Exception:
        print("Exception while loading model:")
        traceback.print_exc()
        _model_loaded = False

# load on startup (blocking) — model download may take time
print("Starting model load (this may take several minutes on first run)...")
load_model_and_processor()

# Utility: run CPU-bound inference in a thread
def _inference_sync(img: Image.Image) -> Dict[str, float]:
    if not _model_loaded:
        raise RuntimeError("Model/processor not loaded yet. Check server logs.")
    inputs = processor(images=img, return_tensors="pt")
    # move tensors to device where possible
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            try:
                inputs[k] = v.to(device)
            except Exception:
                pass
    with torch.no_grad():
        outputs = model(**inputs)
        # extract logits
        logits = None
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            logits = outputs[0]
        else:
            raise RuntimeError("Model outputs do not contain logits; cannot compute probabilities.")
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().tolist()
        if isinstance(probs, float):
            probs = [probs]
        preds = {LABELS[i]: float(round(probs[i], 6)) for i in range(len(probs))}
        return preds

async def run_inference(img: Image.Image) -> Dict[str, float]:
    # run synchronous torch inference in threadpool
    return await asyncio.to_thread(_inference_sync, img)

# Pydantic model for JSON body
class InferRequest(BaseModel):
    image_url: Optional[str] = None
    image_b64: Optional[str] = None

@app.get("/health")
async def health():
    return {"model_loaded": bool(_model_loaded), "device": str(device)}

@app.post("/infer")
async def infer(file: Optional[UploadFile] = File(None), payload: Optional[InferRequest] = None, request: Request = None):
    """
    Accept either:
      - multipart/form-data with 'file' (image)
      - JSON body with {"image_url": "..."} or {"image_b64": "..."}
    """
    try:
        if not _model_loaded:
            # Model still loading or failed
            raise HTTPException(status_code=503, detail="model not ready - still loading or failed. Check server logs.")
        img = None

        # 1) Multipart path (UploadFile)
        if file is not None:
            data = await file.read()
            if not data:
                raise HTTPException(status_code=400, detail="uploaded file is empty")
            try:
                img = Image.open(BytesIO(data)).convert("RGB")
            except UnidentifiedImageError as e:
                # attempt non-convert fallback
                try:
                    img = Image.open(BytesIO(data))
                    img = img.convert("RGB")
                except Exception as e2:
                    raise HTTPException(status_code=400, detail=f"uploaded file is not a valid image: {e2}")

        else:
            # JSON payload path: accept image_b64 or image_url
            # read JSON body if not provided via dependency
            if payload is None:
                try:
                    body = await request.json()
                    payload = InferRequest(**body)
                except Exception:
                    payload = InferRequest()
            # image_b64
            if payload.image_b64:
                try:
                    raw = base64.b64decode(payload.image_b64)
                    img = Image.open(BytesIO(raw)).convert("RGB")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"invalid image_b64: {e}")
            elif payload.image_url:
                # download with httpx async (set Accept header to prefer jpeg/png)
                headers = {"Accept": "image/jpeg,image/png;q=0.9,*/*;q=0.8"}
                try:
                    resp = await httpx_client.get(payload.image_url, headers=headers)
                    if resp.status_code != 200:
                        raise HTTPException(status_code=400, detail=f"failed to download image, status {resp.status_code}")
                    try:
                        img = Image.open(BytesIO(resp.content)).convert("RGB")
                    except UnidentifiedImageError as e:
                        raise HTTPException(status_code=400, detail=f"downloaded file not a valid image: {e}")
                except httpx.RequestError as e:
                    raise HTTPException(status_code=400, detail=f"error downloading image: {e}")
            else:
                raise HTTPException(status_code=400, detail="no image provided (multipart file, or JSON image_url/image_b64)")

        # run inference (async wrapper)
        preds = await run_inference(img)
        return {"predictions": preds}

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        # return a 500 with the error detail
        raise HTTPException(status_code=500, detail=str(exc))
