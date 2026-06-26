"""
Optimized Training Pipeline for Rice Grain Classification
==========================================================
Phase 1 – Feature extraction  : freeze base, train classifier head
Phase 2 – Fine-tuning         : unfreeze top layers, low LR
Uses tf.data for fast GPU-friendly data loading + on-the-fly augmentation.
Split: 70% train / 15% val / 15% test  (stratified)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 32
PHASE1_EPOCHS   = 20      # classifier head only
PHASE2_EPOCHS   = 15      # fine-tune top layers
UNFREEZE_LAYERS = 30      # how many base layers to unfreeze in phase 2
DROPOUT_RATE    = 0.40
LABEL_SMOOTHING = 0.05
SEED            = 42
CLASS_NAMES     = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']


# ── tf.data pipeline ──────────────────────────────────────────────────────────

def _parse_image(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    raw   = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = preprocess_input(image)          # MobileNetV2 normalisation
    return image, label


def _augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    # Random rotation via tfa-free approach (crop + resize)
    image = tf.image.random_crop(
        tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 20, IMG_SIZE + 20),
        size=[IMG_SIZE, IMG_SIZE, 3]
    )
    return image, label


def build_dataset(
    paths: list,
    labels: list,
    augment: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    paths_t  = tf.constant(paths,  dtype=tf.string)
    labels_t = tf.constant(labels, dtype=tf.int32)
    labels_oh = tf.one_hot(labels_t, depth=len(CLASS_NAMES))

    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_oh))
    ds = ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=2048, seed=SEED) if augment else ds
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_file_paths(dataset_root: str) -> Tuple[list, list]:
    """
    Walk dataset_root/<ClassName>/*.jpg and return (paths, int_labels).
    """
    paths, labels = [], []
    root = Path(dataset_root)
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = root / cls
        if not cls_dir.exists():
            print(f"  ⚠  Class folder not found: {cls_dir}")
            continue
        files = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
        paths  += [str(f) for f in files]
        labels += [idx] * len(files)
        print(f"  {cls:<15}: {len(files)} images")
    return paths, labels


def split_paths(
    paths: list,
    labels: list,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[list, list, list, list, list, list]:
    """Stratified 70 / 15 / 15 split."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        paths, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=SEED,
    )
    val_adjusted = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=val_adjusted,
        stratify=y_tmp,
        random_state=SEED,
    )
    print(f"\n  Split → train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(num_classes: int = len(CLASS_NAMES)) -> keras.Model:
    """
    MobileNetV2 backbone + custom classification head.
    Base layers are frozen for Phase 1.
    """
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization(name='bn')(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     name='dense_1')(x)
    x = layers.Dropout(DROPOUT_RATE, name='drop_1')(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     name='dense_2')(x)
    x = layers.Dropout(DROPOUT_RATE * 0.6, name='drop_2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name='rice_mobilenetv2')
    return model, base


def compile_model(model: keras.Model, lr: float = 1e-3) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc'),
        ],
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(checkpoint_path: str, phase: int = 1) -> list:
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=7 if phase == 1 else 5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.4,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            f'training_phase{phase}.csv', append=False
        ),
    ]


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model: keras.Model, test_ds: tf.data.Dataset) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*60)
    print("  Classification Report")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print("Confusion Matrix:")
    print(cm)

    return {'report': report, 'confusion_matrix': cm.tolist()}


# ── Save artefacts ────────────────────────────────────────────────────────────

def save_artefacts(model: keras.Model, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / 'rice_classifier_optimized.h5'
    model.save(str(model_path))
    print(f"\n  Model saved → {model_path}")

    meta = {
        'class_names' : CLASS_NAMES,
        'image_size'  : IMG_SIZE,
        'color_format': 'RGB',
        'normalization': 'mobilenetv2_preprocess_input',
        'architecture': 'MobileNetV2_transfer_learning',
    }
    meta_path = out / 'rice_classifier_metadata.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata saved → {meta_path}")


# ── Main training entry point ─────────────────────────────────────────────────

def train(dataset_root: str, output_dir: str = 'models') -> None:
    print("\n" + "="*60)
    print("  Rice Classification – Optimized Training Pipeline")
    print("="*60)

    # 1. Load file paths
    print("\n[1/5] Loading file paths...")
    paths, labels = load_file_paths(dataset_root)
    if not paths:
        raise RuntimeError("No images found. Check dataset_root path.")

    # 2. Split
    print("\n[2/5] Splitting dataset (70 / 15 / 15)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_paths(paths, labels)

    # 3. Build tf.data pipelines
    print("\n[3/5] Building tf.data pipelines...")
    train_ds = build_dataset(X_train, y_train, augment=True)
    val_ds   = build_dataset(X_val,   y_val,   augment=False)
    test_ds  = build_dataset(X_test,  y_test,  augment=False)

    # 4. Build model
    print("\n[4/5] Building model...")
    model, base = build_model()
    compile_model(model, lr=1e-3)
    model.summary(line_length=80)

    # ── Phase 1: train classifier head ────────────────────────────────────────
    print("\n" + "-"*60)
    print("  Phase 1 – Classifier Head Training")
    print("-"*60)
    ckpt_p1 = str(Path(output_dir) / 'ckpt_phase1.h5')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=get_callbacks(ckpt_p1, phase=1),
    )

    # ── Phase 2: fine-tune top base layers ────────────────────────────────────
    print("\n" + "-"*60)
    print(f"  Phase 2 – Fine-Tuning (unfreezing last {UNFREEZE_LAYERS} base layers)")
    print("-"*60)
    for layer in base.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True
    compile_model(model, lr=1e-4)   # lower LR for fine-tuning

    ckpt_p2 = str(Path(output_dir) / 'ckpt_phase2.h5')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        callbacks=get_callbacks(ckpt_p2, phase=2),
    )

    # 5. Evaluate
    print("\n[5/5] Evaluating on test set...")
    metrics = evaluate_model(model, test_ds)

    # Save
    save_artefacts(model, output_dir)
    results_path = Path(output_dir) / 'evaluation_results.json'
    results_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"  Evaluation results → {results_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Rice Classifier – Optimized Training')
    parser.add_argument('--dataset', required=True,
                        help='Path to cleaned dataset root (one folder per class)')
    parser.add_argument('--output',  default='models',
                        help='Directory to save model and artefacts (default: models)')
    args = parser.parse_args()

    train(dataset_root=args.dataset, output_dir=args.output)
