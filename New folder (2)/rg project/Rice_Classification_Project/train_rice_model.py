#!/usr/bin/env python3
"""
Unified TensorFlow/Keras training pipeline for rice classification.

This script fixes the most common failure mode in the project:
training and prediction using different preprocessing or label mapping.

Key guarantees:
- RGB input
- Resize to 224x224
- Normalize by dividing by 255.0 before inference/training
- Saved metadata for class index mapping and preprocessing
- Transfer learning with MobileNetV2 or EfficientNetB0
- Dataset balancing through oversampling
- Class weights to reduce class bias
- Two-stage training: frozen backbone then fine-tuning
- Validation split, confusion matrix, and classification report
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2


CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class TrainingConfig:
    dataset_path: str = "Rice_Image_Dataset"
    image_size: int = 224
    batch_size: int = 32
    validation_split: float = 0.20
    frozen_epochs: int = 6
    fine_tune_epochs: int = 8
    frozen_learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    random_seed: int = 42
    model_name: str = "MobileNetV2"
    outputs_dir: str = "outputs"
    model_path: str = "models/rice_classifier.h5"
    metadata_path: str = "models/rice_classifier_metadata.json"
    oversample_train: bool = True


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the rice classifier with a unified pipeline.")
    parser.add_argument("--dataset-path", default="Rice_Image_Dataset")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.20)
    parser.add_argument("--frozen-epochs", type=int, default=6)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--frozen-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--model-name", choices=["MobileNetV2", "EfficientNetB0"], default="MobileNetV2")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--model-path", default="models/rice_classifier.h5")
    parser.add_argument("--metadata-path", default="models/rice_classifier_metadata.json")
    parser.add_argument("--disable-oversample-train", action="store_true")
    args = parser.parse_args()
    return TrainingConfig(
        dataset_path=args.dataset_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        frozen_epochs=args.frozen_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        frozen_learning_rate=args.frozen_learning_rate,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        random_seed=args.random_seed,
        model_name=args.model_name,
        outputs_dir=args.outputs_dir,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        oversample_train=not args.disable_oversample_train,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_directories(config: TrainingConfig) -> None:
    Path(config.outputs_dir).mkdir(parents=True, exist_ok=True)
    Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.metadata_path).parent.mkdir(parents=True, exist_ok=True)


def validate_image_file(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def gather_records(dataset_path: str) -> List[dict]:
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root.resolve()}")

    records: List[dict] = []
    counts = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class folder missing: {class_dir}")

        image_paths = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        valid_paths = [path for path in image_paths if validate_image_file(path)]
        counts[class_name] = len(valid_paths)
        for path in valid_paths:
            records.append(
                {
                    "path": str(path),
                    "label": class_index,
                    "class_name": class_name,
                }
            )

    print("\n[DATASET] Images per class")
    for class_name in CLASS_NAMES:
        print(f"  {class_name:<10} -> {counts[class_name]}")

    if not records:
        raise ValueError("No valid dataset images were found.")

    return records


def split_records(records: Sequence[dict], validation_split: float, seed: int) -> Tuple[List[dict], List[dict]]:
    paths = [record["path"] for record in records]
    labels = [record["label"] for record in records]
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=validation_split,
        random_state=seed,
        stratify=labels,
    )

    def rebuild(split_paths: Sequence[str], split_labels: Sequence[int]) -> List[dict]:
        return [
            {
                "path": path,
                "label": int(label),
                "class_name": CLASS_NAMES[int(label)],
            }
            for path, label in zip(split_paths, split_labels)
        ]

    train_records = rebuild(train_paths, train_labels)
    val_records = rebuild(val_paths, val_labels)

    print("\n[SPLIT]")
    print(f"  Train      -> {len(train_records)}")
    print(f"  Validation -> {len(val_records)}")
    return train_records, val_records


def oversample_records(records: Sequence[dict], seed: int) -> List[dict]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["label"])].append(dict(record))

    max_count = max(len(group) for group in grouped.values())
    rng = random.Random(seed)
    balanced: List[dict] = []
    for class_index in range(len(CLASS_NAMES)):
        group = grouped[class_index]
        if not group:
            continue
        if len(group) < max_count:
            needed = max_count - len(group)
            group = group + [dict(rng.choice(group)) for _ in range(needed)]
        balanced.extend(group)

    rng.shuffle(balanced)
    print("\n[BALANCE] Oversampled training distribution")
    balanced_counts = Counter(record["class_name"] for record in balanced)
    for class_name in CLASS_NAMES:
        print(f"  {class_name:<10} -> {balanced_counts[class_name]}")
    return balanced


def compute_class_weights(records: Sequence[dict]) -> Dict[int, float]:
    labels = np.array([record["label"] for record in records], dtype=np.int32)
    classes = np.arange(len(CLASS_NAMES))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weights = {int(class_index): float(weight) for class_index, weight in zip(classes, weights)}
    print("\n[CLASS WEIGHTS]")
    for class_index, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name:<10} -> {class_weights[class_index]:.4f}")
    return class_weights


def decode_and_preprocess(path: tf.Tensor, label: tf.Tensor, image_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [image_size, image_size, 3])
    label = tf.cast(label, tf.int32)
    return image, label


def build_dataset(records: Sequence[dict], config: TrainingConfig, training: bool) -> tf.data.Dataset:
    paths = np.array([record["path"] for record in records])
    labels = np.array([record["label"] for record in records], dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(records), seed=config.random_seed, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: decode_and_preprocess(path, label, config.image_size),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def mobilenet_adapter(x: tf.Tensor) -> tf.Tensor:
    return (x * 2.0) - 1.0


def efficientnet_adapter(x: tf.Tensor) -> tf.Tensor:
    return x


def build_model(config: TrainingConfig) -> Tuple[keras.Model, keras.Model]:
    if config.model_name == "MobileNetV2":
        base_model = MobileNetV2(
            input_shape=(config.image_size, config.image_size, 3),
            include_top=False,
            weights="imagenet",
        )
        preprocess_layer = layers.Lambda(mobilenet_adapter, name="backbone_preprocess")
    elif config.model_name == "EfficientNetB0":
        base_model = EfficientNetB0(
            input_shape=(config.image_size, config.image_size, 3),
            include_top=False,
            weights="imagenet",
        )
        preprocess_layer = layers.Lambda(efficientnet_adapter, name="backbone_preprocess")
    else:
        raise ValueError("Unsupported model_name")

    base_model.trainable = False
    augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.15),
            layers.RandomFlip("horizontal"),
            layers.RandomContrast(0.10),
            layers.RandomBrightness(0.10),
        ],
        name="augmentation",
    )

    inputs = keras.Input(shape=(config.image_size, config.image_size, 3), name="image")
    x = augmentation(inputs)
    x = preprocess_layer(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name=f"{config.model_name}_rice_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.frozen_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def build_callbacks(config: TrainingConfig) -> List[keras.callbacks.Callback]:
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=config.model_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(Path(config.outputs_dir) / "training_log.csv")),
    ]


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_validation_set(model: keras.Model, val_dataset: tf.data.Dataset, val_records: Sequence[dict], config: TrainingConfig) -> Dict[str, object]:
    labels = np.array([record["label"] for record in val_records], dtype=np.int32)
    probabilities = model.predict(val_dataset, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    cm = confusion_matrix(labels, predictions)
    report_text = classification_report(labels, predictions, target_names=CLASS_NAMES, digits=4, zero_division=0)
    report_dict = classification_report(labels, predictions, target_names=CLASS_NAMES, digits=4, zero_division=0, output_dict=True)
    accuracy = float(np.mean(predictions == labels))

    print("\n[VALIDATION]")
    print(f"  Accuracy -> {accuracy * 100:.2f}%")
    print(report_text)

    output_dir = Path(config.outputs_dir)
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(
            {
                "validation_accuracy": accuracy,
                "confusion_matrix": cm.tolist(),
                "classification_report": report_dict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "validation_accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }


def save_metadata(config: TrainingConfig, train_records: Sequence[dict], val_records: Sequence[dict], class_weights: Dict[int, float], metrics: Dict[str, object]) -> None:
    metadata = {
        "class_names": CLASS_NAMES,
        "classes": CLASS_NAMES,
        "class_to_index": {name: index for index, name in enumerate(CLASS_NAMES)},
        "image_size": config.image_size,
        "color_format": "RGB",
        "normalization": "divide_by_255",
        "model_name": config.model_name,
        "input_shape": [config.image_size, config.image_size, 3],
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "class_weights": {CLASS_NAMES[index]: weight for index, weight in class_weights.items()},
        "metrics": metrics,
    }
    Path(config.metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_pipeline(config: TrainingConfig) -> None:
    print("=" * 72)
    print("RICE CLASSIFICATION TRAINING PIPELINE")
    print("=" * 72)
    print(f"Dataset       : {Path(config.dataset_path).resolve()}")
    print(f"Model         : {config.model_name}")
    print(f"Input shape   : ({config.image_size}, {config.image_size}, 3)")
    print(f"Preprocessing : RGB -> resize -> /255.0")
    print("=" * 72)

    set_seed(config.random_seed)
    ensure_directories(config)

    records = gather_records(config.dataset_path)
    train_records, val_records = split_records(records, config.validation_split, config.random_seed)

    if config.oversample_train:
        train_records = oversample_records(train_records, config.random_seed)

    class_weights = compute_class_weights(train_records)
    train_dataset = build_dataset(train_records, config, training=True)
    val_dataset = build_dataset(val_records, config, training=False)

    model, base_model = build_model(config)
    callbacks = build_callbacks(config)

    print("\n[TRAIN] Stage 1 - frozen backbone")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.frozen_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    print("\n[TRAIN] Stage 2 - fine-tuning")
    base_model.trainable = True
    for layer in base_model.layers[: int(len(base_model.layers) * 0.7)]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.fine_tune_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.frozen_epochs + config.fine_tune_epochs,
        initial_epoch=config.frozen_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    if Path(config.model_path).exists():
        model = keras.models.load_model(config.model_path)
    else:
        model.save(config.model_path)

    metrics = evaluate_validation_set(model, val_dataset, val_records, config)
    save_metadata(config, train_records, val_records, class_weights, metrics)

    print("\n[SAVED]")
    print(f"  Model    -> {Path(config.model_path).resolve()}")
    print(f"  Metadata -> {Path(config.metadata_path).resolve()}")
    print(f"  Outputs  -> {Path(config.outputs_dir).resolve()}")


if __name__ == "__main__":
    train_pipeline(parse_args())
