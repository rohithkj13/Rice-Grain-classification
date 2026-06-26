import cv2
import numpy as np
import json
from collections import Counter
from ultralytics import YOLO
import tensorflow as tf


yolo_model = YOLO("runs/detect/train7/weights/best.pt")
classifier = tf.keras.models.load_model("models/rice_model_advanced.h5")

# Load YOLO model (you will train this separately)
yolo_model = YOLO("yolov8n.pt")  # replace with your trained model later

# Load classifier
classifier = tf.keras.models.load_model("models/rice_model_advanced.h5")

# Load class mapping
with open("models/classes.json", "r") as f:
    class_map = json.load(f)

def preprocess_roi(roi):
    roi = cv2.resize(roi, (224, 224))
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)
    return roi

def predict_image(image_path):
    image = cv2.imread(image_path)

    results = yolo_model(image)[0]

    predictions = []
    annotated = image.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Ignore low confidence
        if conf < 0.4:
            continue

        # Padding
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)

        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Preprocess
        roi_input = preprocess_roi(roi)

        # Predict
        pred = classifier.predict(roi_input, verbose=0)
        class_id = int(np.argmax(pred))
        class_name = class_map[str(class_id)]

        predictions.append(class_name)

        # Draw box + label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(annotated, class_name, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Majority voting
    if len(predictions) > 0:
        final_class = Counter(predictions).most_common(1)[0][0]
    else:
        final_class = "No rice detected"

    return final_class, annotated