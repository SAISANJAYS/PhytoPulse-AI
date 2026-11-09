from pathlib import Path
import numpy as np
from PIL import Image

from src.config import MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE
from src.image_predictor import _load_h5_model
import json

model = _load_h5_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

folder = Path("data/test_images")

for image_path in folder.rglob("*"):
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        continue

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    scores = model.predict(array, verbose=0)[0]
    top_index = int(np.argmax(scores))

    print(f"\nImage: {image_path}")
    print(f"Prediction: {class_names[top_index]}")
    print(f"Confidence: {scores[top_index] * 100:.2f}%")
    print("All scores:", np.round(scores, 4))