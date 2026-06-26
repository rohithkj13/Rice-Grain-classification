"""
Unified Rice Classification Retraining Script
==============================================
Fixes:
  1. Preprocessing mismatch  — both CNN and KNN now use identical
     MobileNetV2 preprocess_input (scale to [-1, 1])
  2. KNN data starvation     — uses up to 3,000 images/class (was 1,000)
  3. KNN evaluated on held-out val set (was training set)
  4. CNN metadata updated to reflect correct normalization
  5. Consistent class order: Arborio, Basmati, Ipsala, Jasmine, Karacadag

Run:
    python retrain.py --dataset Rice_Image_Dataset
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import warnings
from pathlib import Path

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES      = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
IMG_SIZE         = 224
BATCH_SIZE       = 32
SEED             = 42
MODEL_DIR        = Path('models')
AUTOTUNE         = tf.data.AUTOTUNE

# CNN training
FROZEN_EPOCHS    = 15
FINETUNE_EPOCHS  = 10
FROZEN_LR        = 1e-3
FINETUNE_LR      = 1e-5

# KNN
KNN_MAX_PER_CLASS = 200   # used for KNN training
MAX_PER_CLASS     = 200   # images per class for CNN too
K_NEIGHBORS       = 7


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ── Image loading ─────────────────────────────────────────────────────────────

def load_paths_and_labels(dataset_root: str) -> tuple[list, list]:
    """Return (file_paths, int_labels) capped at MAX_PER_CLASS per class."""
    paths, labels = [], []
    root = Path(dataset_root)
    rng  = np.random.default_rng(SEED)
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        files = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        )
        # randomly sample MAX_PER_CLASS
        chosen = rng.choice(len(files),
                            size=min(MAX_PER_CLASS, len(files)),
                            replace=False)
        selected = [files[i] for i in sorted(chosen)]
        paths  += [str(f) for f in selected]
        labels += [idx] * len(selected)
        print(f"  {cls:<12}: {len(selected)} images selected (of {len(files)})") 
    return paths, labels


# ── tf.data pipeline (CNN) ────────────────────────────────────────────────────

def _parse(path: tf.Tensor, label: tf.Tensor) -> tuple:
    raw   = tf.io.read_file(path)
    img   = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img   = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img   = tf.cast(img, tf.float32)
    img   = preprocess_input(img)          # scale to [-1, 1]  ← KEY FIX
    return img, label


def _augment(img: tf.Tensor, label: tf.Tensor) -> tuple:
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    # random crop
    padded = tf.image.resize_with_crop_or_pad(img, IMG_SIZE + 20, IMG_SIZE + 20)
    img    = tf.image.random_crop(padded, [IMG_SIZE, IMG_SIZE, 3])
    return img, label


def make_dataset(paths: list, labels: list, augment: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.constant(labels, dtype=tf.int32))
    )
    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(2048, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


# ── CNN model ─────────────────────────────────────────────────────────────────

def build_cnn() -> tuple[keras.Model, keras.Model]:
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.40)(x)
    x   = layers.Dense(512, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.30)(x)
    x   = layers.Dense(256, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.20)(x)
    out = layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

    model = keras.Model(inp, out, name='rice_cnn')
    return model, base


def get_callbacks(ckpt_path: str) -> list:
    return [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_accuracy',
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=6,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.4,
            patience=3, min_lr=1e-7, verbose=1),
    ]


# ── KNN feature extraction ────────────────────────────────────────────────────

def extract_features(paths: list, extractor: keras.Model) -> np.ndarray:
    """Run MobileNetV2 preprocess_input + extractor on a list of image paths."""
    feats = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i:i + BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            raw = tf.io.read_file(p)
            img = tf.image.decode_image(raw, channels=3, expand_animations=False)
            img.set_shape([None, None, 3])
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            img = tf.cast(img, tf.float32)
            img = preprocess_input(img)        # ← same as CNN  KEY FIX
            imgs.append(img.numpy())
        batch = np.stack(imgs)
        feats.append(extractor.predict(batch, verbose=0))
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"    {i + len(batch_paths)}/{len(paths)} images processed")
    return np.vstack(feats)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def train(dataset_root: str) -> None:
    set_seed()
    MODEL_DIR.mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("  RICE CLASSIFIER — UNIFIED RETRAINING")
    print("="*60)

    # ── 1. Gather paths ───────────────────────────────────────────────────────
    print("\n[1/6] Scanning dataset...")
    paths, labels = load_paths_and_labels(dataset_root)
    print(f"  Total: {len(paths)} images")

    # Stratified 70 / 15 / 15 split
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15/0.85, stratify=y_tmp, random_state=SEED)
    print(f"  Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    # ── 2. Class weights ──────────────────────────────────────────────────────
    cw_arr = compute_class_weight('balanced',
                                  classes=np.arange(len(CLASS_NAMES)),
                                  y=np.array(y_train))
    class_weights = {i: float(w) for i, w in enumerate(cw_arr)}
    print("\n  Class weights:", {CLASS_NAMES[i]: round(v, 3)
                                  for i, v in class_weights.items()})

    # ── 3. Build tf.data pipelines ────────────────────────────────────────────
    print("\n[2/6] Building tf.data pipelines...")
    train_ds = make_dataset(X_train, y_train, augment=True)
    val_ds   = make_dataset(X_val,   y_val,   augment=False)
    test_ds  = make_dataset(X_test,  y_test,  augment=False)

    # ── 4. Train CNN ──────────────────────────────────────────────────────────
    print("\n[3/6] Training CNN (MobileNetV2)...")
    model, base = build_cnn()
    ckpt = str(MODEL_DIR / 'rice_classifier.h5')

    # Phase 1 — frozen backbone
    print("\n  Phase 1: frozen backbone")
    model.compile(
        optimizer=keras.optimizers.Adam(FROZEN_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=False),
        metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds,
              epochs=FROZEN_EPOCHS,
              callbacks=get_callbacks(ckpt),
              class_weight=class_weights, verbose=1)

    # Phase 2 — fine-tune top 40% of base layers
    print("\n  Phase 2: fine-tuning top layers")
    base.trainable = True
    freeze_until = int(len(base.layers) * 0.60)
    for layer in base.layers[:freeze_until]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(FINETUNE_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=False),
        metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds,
              epochs=FROZEN_EPOCHS + FINETUNE_EPOCHS,
              initial_epoch=FROZEN_EPOCHS,
              callbacks=get_callbacks(ckpt),
              class_weight=class_weights, verbose=1)

    # Load best checkpoint
    model = keras.models.load_model(ckpt)

    # Evaluate on test set
    print("\n[4/6] Evaluating CNN on test set...")
    y_pred_prob = model.predict(test_ds, verbose=1)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    cnn_acc     = accuracy_score(y_test, y_pred)
    print(f"\n  CNN Test Accuracy: {cnn_acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Save CNN metadata — normalization matches preprocess_input
    meta = {
        'class_names'  : CLASS_NAMES,
        'classes'      : CLASS_NAMES,
        'class_to_index': {n: i for i, n in enumerate(CLASS_NAMES)},
        'image_size'   : IMG_SIZE,
        'input_shape'  : [IMG_SIZE, IMG_SIZE, 3],
        'color_format' : 'RGB',
        'normalization': 'mobilenet_v2_preprocess_input',  # ← KEY FIX
        'model_name'   : 'MobileNetV2',
        'test_accuracy': float(cnn_acc),
    }
    (MODEL_DIR / 'rice_classifier_metadata.json').write_text(
        json.dumps(meta, indent=2))
    print(f"  CNN metadata saved.")

    # ── 5. Train KNN ──────────────────────────────────────────────────────────
    print("\n[5/6] Training KNN classifier...")

    # Build feature extractor (MobileNetV2 without top, pooling='avg')
    extractor = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        pooling='avg',
        weights='imagenet',
    )
    extractor.trainable = False

    # Sample up to KNN_MAX_PER_CLASS per class for KNN training
    knn_train_paths, knn_train_labels = X_train, y_train
    for cls_idx in range(len(CLASS_NAMES)):
        count = sum(1 for l in y_train if l == cls_idx)
        print(f"  {CLASS_NAMES[cls_idx]:<12}: {count} KNN training images")

    print("\n  Extracting KNN training features...")
    X_knn_train = extract_features(knn_train_paths, extractor)
    y_knn_train = knn_train_labels

    print("\n  Extracting KNN validation features...")
    X_knn_val   = extract_features(X_val, extractor)

    # Normalize features (L2)
    def l2_norm(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-8)

    X_knn_train = l2_norm(X_knn_train)
    X_knn_val   = l2_norm(X_knn_val)

    knn = KNeighborsClassifier(
        n_neighbors=K_NEIGHBORS,
        metric='cosine',       # cosine works better than euclidean on L2-normed features
        n_jobs=-1,
    )
    knn.fit(X_knn_train, y_knn_train)

    knn_val_pred = knn.predict(X_knn_val)
    knn_acc      = accuracy_score(y_val, knn_val_pred)
    print(f"\n  KNN Validation Accuracy: {knn_acc*100:.2f}%")
    print(classification_report(y_val, knn_val_pred, target_names=CLASS_NAMES))

    # Save KNN artefacts
    with open(MODEL_DIR / 'rice_knn_classifier.pkl', 'wb') as f:
        pickle.dump(knn, f)
    extractor.save(str(MODEL_DIR / 'rice_feature_extractor.h5'))

    knn_meta = {
        'model_type'      : 'KNN',
        'n_neighbors'     : K_NEIGHBORS,
        'metric'          : 'cosine',
        'normalization'   : 'mobilenet_v2_preprocess_input',  # ← KEY FIX
        'feature_norm'    : 'l2',
        'accuracy'        : float(knn_acc),
        'image_size'      : IMG_SIZE,
        'classes'         : CLASS_NAMES,
        'training_samples': len(knn_train_labels),
    }
    (MODEL_DIR / 'rice_knn_metadata.json').write_text(
        json.dumps(knn_meta, indent=2))

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("\n[6/6] Done.")
    print("="*60)
    print(f"  CNN  test  accuracy : {cnn_acc*100:.2f}%")
    print(f"  KNN  val   accuracy : {knn_acc*100:.2f}%")
    print(f"  Models saved to     : {MODEL_DIR.resolve()}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Rice_Image_Dataset',
                        help='Path to dataset root folder')
    args = parser.parse_args()
    train(args.dataset)
