
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import (
    MODEL_PATH,
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
)


def _load_h5_model(model_path: Path) -> tf.keras.Model:
    """Load the candidate HDF5 model with its custom FixedDropout layer."""
    class FixedDropout(tf.keras.layers.Dropout):
        def _get_noise_shape(self, inputs):
            if getattr(self, "noise_shape", None) is None:
                return None

            symbolic_shape = tf.shape(inputs)
            return tuple(
                symbolic_shape[i] if shape is None else shape
                for i, shape in enumerate(self.noise_shape)
            )

    return tf.keras.models.load_model(
        str(model_path),
        custom_objects={"FixedDropout": FixedDropout},
        compile=False,
    )


class LeafDiseasePredictor:
    """Predict potato and tomato leaf classes from an RGB image."""

    def __init__(
        self,
        model_path=MODEL_PATH,
        class_names_path=CLASS_NAMES_PATH,
    ):
        model_path = Path(model_path)
        class_names_path = Path(class_names_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not class_names_path.exists():
            raise FileNotFoundError(
                f"Class names file not found: {class_names_path}"
            )

        self.model = _load_h5_model(model_path)

        with open(class_names_path, "r", encoding="utf-8") as file:
            names = json.load(file)

        if isinstance(names, list):
            self.class_names = names
        elif isinstance(names, dict):
            self.class_names = [
                names[key] for key in sorted(names, key=lambda k: int(k))
            ]
        else:
            raise ValueError("Class names must be a JSON list or dictionary.")

        output_classes = int(self.model.output_shape[-1])

        if len(self.class_names) != output_classes:
            raise ValueError(
                f"Model outputs {output_classes} classes, but the labels file "
                f"contains {len(self.class_names)} names."
            )

        self.image_size = tuple(IMAGE_SIZE)

        if tuple(self.model.input_shape[1:3]) != self.image_size:
            raise ValueError(
                f"Model expects image size {self.model.input_shape[1:3]}, "
                f"but IMAGE_SIZE is configured as {self.image_size}."
            )

    def predict(self, image: Image.Image) -> dict:
        """Return the predicted class, confidence, and ranked class scores."""
        image = image.convert("RGB")
        image = image.resize(self.image_size, resample=Image.BILINEAR)

        # plant_disease_model_candidate.h5 expects normalized float32 pixel values in [0, 1]
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = np.asarray(
            self.model.predict(image_array, verbose=0)[0],
            dtype=np.float32,
        )

        if predictions.ndim != 1 or len(predictions) != len(self.class_names):
            raise ValueError("Model prediction shape does not match class labels.")

        class_index = int(np.argmax(predictions))
        confidence = float(predictions[class_index])

        all_scores = sorted(
            [
                {"class_name": name, "score": float(score)}
                for name, score in zip(self.class_names, predictions)
            ],
            key=lambda item: item["score"],
            reverse=True,
        )

        return {
            "class_name": self.class_names[class_index],
            "confidence": confidence,
            "all_scores": all_scores,
            "is_confident": confidence >= CONFIDENCE_THRESHOLD,
        }
