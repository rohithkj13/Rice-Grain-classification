import os
import json
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO

# ✅ Paths
MODEL_PATH = "models/rice_model_advanced.h5"
CLASSES_CONFIG_PATH = "models/classes.json"
YOLO_MODEL_PATH = "runs/detect/train7/weights/best.pt"

TEMPERATURE = 1.2


class RicePredictor:
    def __init__(self):
        self.classifier = None
        self.yolo_model = None
        self.classes = {}
        self.is_loaded = False

    def load_model(self):
        print("🚀 Loading models...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ CNN model not found: {MODEL_PATH}")

        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"❌ YOLO model not found: {YOLO_MODEL_PATH}")

        if not os.path.exists(CLASSES_CONFIG_PATH):
            raise FileNotFoundError(f"❌ classes.json not found: {CLASSES_CONFIG_PATH}")

        # Load CNN model
        self.classifier = tf.keras.models.load_model(MODEL_PATH)
        print("✅ CNN Loaded")

        # Load YOLO model
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO Loaded")

        # Load class labels
        with open(CLASSES_CONFIG_PATH, "r") as f:
            classes_dict = json.load(f)
            self.classes = {int(k): v for k, v in classes_dict.items()}

        print("✅ Classes:", self.classes)

        self.is_loaded = True

    def preprocess(self, img):
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def classify(self, img):
        img_batch = self.preprocess(img)
        logits = self.classifier.predict(img_batch, verbose=0)[0]

        # Temperature scaling
        scaled_logits = logits / TEMPERATURE
        probs = tf.nn.softmax(scaled_logits).numpy()

        return probs

    def process_and_predict(self, image_rgb):
        if not self.is_loaded:
            self.load_model()

        # 🔍 YOLO detection
        results = self.yolo_model(image_rgb)[0]
        boxes = results.boxes

        predictions = []

        # 🔥 CASE 1: YOLO FAILED → fallback
        if boxes is None or len(boxes) == 0:
            probs = self.classify(image_rgb)
            class_id = np.argmax(probs)
            confidence = probs[class_id]

            return {
                "status": "FALLBACK",
                "predicted_class": self.classes[class_id],
                "confidence": float(confidence),
                "grain_count": 1,
                "all_predictions": [{
                    "class": self.classes[class_id],
                    "confidence": float(confidence)
                }]
            }

        # 🔥 CASE 2: YOLO detected grains
        for box in boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            roi = image_rgb[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            probs = self.classify(roi)
            class_id = np.argmax(probs)
            confidence = probs[class_id]

            predictions.append({
                "class": self.classes[class_id],
                "confidence": float(confidence)
            })

        # 🔥 Safety fallback
        if len(predictions) == 0:
            probs = self.classify(image_rgb)
            class_id = np.argmax(probs)
            confidence = probs[class_id]

            return {
                "status": "FALLBACK",
                "predicted_class": self.classes[class_id],
                "confidence": float(confidence),
                "grain_count": 1,
                "all_predictions": [{
                    "class": self.classes[class_id],
                    "confidence": float(confidence)
                }]
            }

        # 🔥 Majority voting
        class_counts = {}
        for p in predictions:
            cls = p["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        final_class = max(class_counts, key=class_counts.get)

        avg_conf = np.mean([
            p["confidence"] for p in predictions if p["class"] == final_class
        ])

        return {
            "status": "SUCCESS",
            "predicted_class": final_class,
            "confidence": float(avg_conf),
            "grain_count": len(predictions),
            "all_predictions": predictions
        }