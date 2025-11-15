import os
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "vehicle.h5"
TEST_FOLDER = "test_images"
TARGET_SIZE = (224, 224)
SCALE = 1.0 / 255.0

def load_model_safe(path):
    try:
        return tf.keras.models.load_model(path)
    except:
        import tensorflow_hub as hub
        return tf.keras.models.load_model(path, custom_objects={'KerasLayer': hub.KerasLayer})

model = load_model_safe(MODEL_PATH)
print("Model loaded successfully!")

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(TARGET_SIZE)
    arr = np.asarray(img, dtype=np.float32) * SCALE
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_with_floor(image_path):
    x = load_image(image_path)
    raw = model.predict(x)[0][0]         # raw sigmoid output (0–1)
    floored = np.floor(raw)              # match original code

    label = "emergency" if raw>=0.25 else "not_emergency"

    print(f"{os.path.basename(image_path)}  -->  {label}  (raw={raw:.4f}, floored={floored})")

def main():
    if not os.path.exists(TEST_FOLDER):
        print("Create test_images folder and put images inside.")
        return

    images = sorted([f for f in os.listdir(TEST_FOLDER) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not images:
        print("No images found in test_images.")
        return

    for img in images:
        predict_with_floor(os.path.join(TEST_FOLDER, img))

if __name__ == "__main__":
    main()
