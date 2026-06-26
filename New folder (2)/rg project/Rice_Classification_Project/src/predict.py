"""
Prediction module for the unified rice classifier pipeline.

Guarantees the same preprocessing used during training:
- BGR -> RGB
- resize to 224x224
- divide by 255.0

Also applies the requested hybrid shape correction:
- aspect ratio > 3.5 -> Basmati
- aspect ratio < 1.6 -> Arborio
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tensorflow import keras


DEFAULT_MODEL_PATH = "models/rice_classifier.h5"
DEFAULT_METADATA_PATH = "models/rice_classifier_metadata.json"
DEFAULT_CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
CONFIDENCE_THRESHOLD = 0.55


class RicePredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, metadata_path: str = DEFAULT_METADATA_PATH):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = keras.models.load_model(model_path)
        self.metadata = self._load_metadata(metadata_path)
        self.class_names = list(self.metadata.get("class_names", DEFAULT_CLASS_NAMES))
        self.image_size = int(self.metadata.get("image_size", 224))

    def _load_metadata(self, metadata_path: str) -> Dict:
        metadata_file = Path(metadata_path)
        if metadata_file.exists():
            return json.loads(metadata_file.read_text(encoding="utf-8"))
        return {
            "class_names": DEFAULT_CLASS_NAMES,
            "image_size": 224,
            "color_format": "RGB",
            "normalization": "divide_by_255",
        }

    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not read image from {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image_rgb = image_rgb.astype(np.float32) / 255.0
        batch = np.expand_dims(image_rgb, axis=0)
        return batch, image_bgr

    def _extract_aspect_ratio(self, image_bgr: np.ndarray) -> Optional[float]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect
        short_side = min(w, h)
        long_side = max(w, h)
        if short_side <= 0:
            return None
        return float(long_side / short_side)

    def _apply_shape_override(self, predicted_class: str, confidence: float, aspect_ratio: Optional[float]) -> Tuple[str, float, str]:
        if aspect_ratio is None:
            return predicted_class, confidence, "ML"
        if aspect_ratio > 3.5:
            return "Basmati", max(confidence, 0.92), "Shape"
        if aspect_ratio < 1.6:
            return "Arborio", max(confidence, 0.92), "Shape"
        return predicted_class, confidence, "ML"

    def predict_rice(self, image_path: str) -> Dict:
        batch, image_bgr = self.preprocess_image(image_path)
        probabilities = self.model.predict(batch, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_class = self.class_names[predicted_index]
        confidence = float(probabilities[predicted_index])
        aspect_ratio = self._extract_aspect_ratio(image_bgr)

        final_class, final_confidence, source = self._apply_shape_override(predicted_class, confidence, aspect_ratio)
        class_probabilities = {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(self.class_names)
        }

        is_rice = final_confidence >= CONFIDENCE_THRESHOLD
        return {
            "image_path": image_path,
            "predicted_variety": final_class if is_rice else None,
            "model_prediction": predicted_class,
            "confidence": float(final_confidence),
            "confidence_percentage": f"{final_confidence * 100:.2f}%",
            "all_probabilities": class_probabilities,
            "aspect_ratio": None if aspect_ratio is None else round(aspect_ratio, 4),
            "source": source,
            "is_rice": is_rice,
            "message": None if is_rice else "This is not a rice image. Please upload a rice image.",
        }

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        results = []
        for image_path in image_paths:
            try:
                results.append(self.predict_rice(image_path))
            except Exception as exc:
                results.append({"image_path": image_path, "error": str(exc)})
        return results


def predict_rice(image_path: str, model_path: str = DEFAULT_MODEL_PATH, metadata_path: str = DEFAULT_METADATA_PATH) -> Dict:
    predictor = RicePredictor(model_path=model_path, metadata_path=metadata_path)
    return predictor.predict_rice(image_path)
